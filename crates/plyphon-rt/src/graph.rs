//! A live synth instance - plyphon's port of scsynth's `Graph`.
//!
//! A `Graph` is constructed on the audio thread from a shared [`GraphDef`] (see
//! [`crate::world::World`]). It owns exactly one rt-pool allocation - its [`Region`] - holding only
//! the per-instance *mutable* state: the unit state arena, the control wires (parameters and
//! control-rate unit outputs), and the per-parameter control-bus map. The immutable plan (vtable,
//! wiring, layout) is shared via `Arc<GraphDef>`.
//!
//! Audio wire buffers and per-unit output scratch are *not* in the block: they are World-owned, fixed
//! at boot, and reused across graphs (matching scsynth's `mWireBufSpace`), threaded in via `Block`.
//!
//! The process loop avoids scsynth's aliasing raw `float*` wires while staying `unsafe`-free: it
//! carves the block into its disjoint state/control/param-map spans in one `get_disjoint_mut` call,
//! and each unit writes into the shared scratch (disjoint from its inputs), which the loop then
//! publishes into the wires.

use alloc::boxed::Box;
use alloc::sync::Arc;

use bytemuck::{cast_slice, cast_slice_mut};
use rt_alloc::{Align64, Region, RtPool};

use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::bus::Buses;
use plyphon_dsp::rate::{Rate, RateInfo};
use plyphon_dsp::wavetable::Wavetables;
use plyphon_unit::graphdef::GraphDef;
use plyphon_unit::unit::{self, DemandAccess, DoneAction, InitCtx, Inputs, Outputs, ProcessCtx};

/// The pool type the engine uses: a heap-backed rt-pool of 64-byte-aligned blocks.
pub(crate) type Pool = RtPool<Box<[Align64]>>;

/// The per-block materials the process loop draws on to assemble each unit's [`ProcessCtx`] and
/// [`InitCtx`]. Built once per control block by the [`World`](crate::world::World) and threaded
/// through the node tree. Its fields are disjoint, so a [`Graph`] can borrow the pool, the shared
/// scratch, and the buses at once.
pub(crate) struct Block<'a> {
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// Shared wavetables.
    pub wavetables: &'a Wavetables,
    /// The World's shared buses.
    pub buses: &'a mut Buses,
    /// The World's shared buffer table.
    pub buffers: &'a mut BufferTable,
    /// The current block counter.
    pub buf_counter: u64,
    /// The rt-pool holding every graph's per-instance block.
    pub pool: &'a mut Pool,
    /// World-shared audio wire scratch, reused per graph (`max_wire_bufs * block_size` f32).
    pub wire_scratch: &'a mut [f32],
    /// World-shared per-unit output scratch, reused per unit (`max_unit_outputs * block_size` f32).
    pub unit_scratch: &'a mut [f32],
}

/// A live synth instance.
pub struct Graph {
    /// The one pool allocation: `[ state arena | control wires | param maps ]`.
    block: Region,
    /// The shared, immutable compiled def.
    def: Arc<GraphDef>,
    /// Whether the one-time [`Unit::init`](plyphon_unit::unit::Unit::init) seeding pass has run (it runs on
    /// the first control block - plyphon's analogue of scsynth's `Graph_FirstCalc`).
    initialized: bool,
    /// The within-block sample offset at which this synth was created (scsynth's node `mSampleOffset`).
    /// Surfaced to its units on the first block only, so `OffsetOut` onsets sample-exactly; 0 for an
    /// immediately-created synth.
    sample_offset: usize,
}

impl Graph {
    /// Wrap a freshly allocated, initialised block and its def into a live graph, created at
    /// `sample_offset` samples into its first control block (0 unless scheduled mid-block).
    pub(crate) fn new(block: Region, def: Arc<GraphDef>, sample_offset: usize) -> Self {
        Graph {
            block,
            def,
            initialized: false,
            sample_offset,
        }
    }

    /// Consume the graph, returning its pool block so the World can `dealloc` it on the audio thread.
    pub(crate) fn into_block(self) -> Region {
        self.block
    }

    /// Compute one control block. Returns the strongest [`DoneAction`] any of its units requested.
    #[must_use]
    pub(crate) fn process(&mut self, block: &mut Block<'_>) -> DoneAction {
        let def = &*self.def;
        let bs = def.block_size();
        let layout = def.layout();

        // Carve the per-graph block into its four disjoint spans (proved disjoint once, here). The
        // calc-unit state and the demand-state arena are separate spans so a calc unit's `&mut` state
        // slot and the `&mut` demand arena (pulled re-entrantly during its `process`) never alias.
        let buf = block.pool.slice_mut(&self.block);
        let Ok([state_arena, demand_state, ctrl_bytes, pmap_bytes]) = buf.get_disjoint_mut([
            layout.state.range(),
            layout.demand_state.range(),
            layout.control.range(),
            layout.pmaps.range(),
        ]) else {
            // Unreachable: the layout's spans are contiguous and disjoint by construction.
            return DoneAction::Nothing;
        };
        let ctrl = cast_slice_mut::<u8, f32>(ctrl_bytes);
        let pmaps = cast_slice::<u8, u32>(pmap_bytes);
        // Audio wires and output scratch are World-shared (separate allocations), reused per graph.
        let audio = &mut *block.wire_scratch;
        let scratch = &mut *block.unit_scratch;

        // Apply control-bus mappings (`/n_map`): a mapped parameter takes the bus's current value.
        for (p, &bus) in pmaps.iter().enumerate() {
            if bus != u32::MAX {
                ctrl[def.param_wires()[p] as usize] = unit::control_in(block.buses, bus as usize);
            }
        }

        // On the first block only, run each unit's one-time `init` seeding pass (in topo order, just
        // before its first `process`), so state is seeded from now-live inputs.
        let first = !self.initialized;
        self.initialized = true;
        // The node's creation offset applies only to its first block (`OffsetOut` delays the onset
        // by it, then runs flush); later blocks start at the block boundary.
        let sample_offset = if first { self.sample_offset } else { 0 };
        let mut done = DoneAction::Nothing;
        for v in def.units().iter() {
            let state = &mut state_arena[v.state_offset..v.state_offset + v.state_size];
            let ins = Inputs::new(&v.inputs, &*audio, &*ctrl, bs);
            if first {
                let init_ctx = InitCtx {
                    audio: block.audio,
                    control: block.control,
                    wavetables: block.wavetables,
                    ins,
                    buses: &*block.buses,
                    buffers: &*block.buffers,
                    buf_counter: block.buf_counter,
                };
                (v.init)(state, &init_ctx);
            }
            // Scoped so the context's borrows of the scratch/buses/demand arena end before we publish.
            done = done.max({
                let mut ctx = ProcessCtx {
                    audio: block.audio,
                    control: block.control,
                    wavetables: block.wavetables,
                    ins,
                    outs: Outputs::new(&mut scratch[..], bs),
                    buses: &mut *block.buses,
                    buffers: &mut *block.buffers,
                    buf_counter: block.buf_counter,
                    sample_offset,
                    // A consumer pulls demand sources through this; non-demand units ignore it. The
                    // demand arena is disjoint from this unit's `state` slot above.
                    demand: DemandAccess::new(
                        def.demand_units(),
                        &mut *demand_state,
                        &*audio,
                        &*ctrl,
                        bs,
                    ),
                };
                (v.process)(state, &mut ctx)
            });
            // Publish this unit's scratch outputs into the wires.
            for (k, ow) in v.outputs.iter().enumerate() {
                let src = k * bs;
                match ow.rate {
                    Rate::Audio => {
                        let dst = ow.wire as usize * bs;
                        audio[dst..dst + bs].copy_from_slice(&scratch[src..src + bs]);
                    }
                    Rate::Control | Rate::Scalar => {
                        ctrl[ow.wire as usize] = scratch[src];
                    }
                    // Calc units never publish a demand-rate output (demand units are not in this loop).
                    Rate::Demand => {}
                }
            }
        }
        done
    }

    /// Set control parameter `param` to `value`. No-op if out of range. Allocation-free (RT-safe).
    pub(crate) fn set_control(&mut self, pool: &mut Pool, param: usize, value: f32) {
        if let Some(&wire) = self.def.param_wires().get(param) {
            let bytes = &mut pool.slice_mut(&self.block)[self.def.layout().control.range()];
            cast_slice_mut::<u8, f32>(bytes)[wire as usize] = value;
        }
    }

    /// The current value of control parameter `param` (the symmetric read of
    /// [`set_control`](Self::set_control), for `/s_get`/`/g_queryTree`). `None` if out of range.
    pub(crate) fn control_value(&self, pool: &Pool, param: usize) -> Option<f32> {
        let &wire = self.def.param_wires().get(param)?;
        let bytes = &pool.slice(&self.block)[self.def.layout().control.range()];
        cast_slice::<u8, f32>(bytes).get(wire as usize).copied()
    }

    /// Number of control parameters this synth exposes (scsynth's "controls").
    pub(crate) fn num_params(&self) -> usize {
        self.def.num_params()
    }

    /// Number of unit generators in this synth's def (for `/status`'s ugen count).
    pub(crate) fn num_units(&self) -> usize {
        self.def.units().len()
    }

    /// Map control parameter `param` to control `bus` (or unmap it with `None`). While mapped, the
    /// parameter takes the bus's value at the start of every block. No-op if out of range.
    pub(crate) fn map_control(&mut self, pool: &mut Pool, param: usize, bus: Option<u32>) {
        if param < self.def.num_params() {
            let bytes = &mut pool.slice_mut(&self.block)[self.def.layout().pmaps.range()];
            cast_slice_mut::<u8, u32>(bytes)[param] = bus.unwrap_or(u32::MAX);
        }
    }
}
