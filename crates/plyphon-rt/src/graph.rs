//! A live synth instance - plyphon's port of scsynth's `Graph`.
//!
//! A `Graph` is constructed on the audio thread from a shared [`GraphDef`] (see
//! [`crate::world::World`]). It owns one graph-pool allocation - its block [`Region`] - holding the
//! per-instance *mutable* state: the unit state arena, the control wires (parameters and
//! control-rate unit outputs), and the per-parameter control-bus map. Units whose memory is sized
//! from live inputs (a delay's `maxdelaytime`) allocate it separately from the World's unit pool on
//! the first block, as scsynth's constructors call `RTAlloc`; the graph holds those handles in its
//! [`AuxSlots`]. The immutable plan (vtable, wiring, layout) is shared via `Arc<GraphDef>`.
//!
//! Audio wire buffers and per-unit output scratch are *not* in the block: they are World-owned, fixed
//! at boot, and reused across graphs (matching scsynth's `mWireBufSpace`), threaded in via `Block`.
//!
//! The process loop avoids scsynth's aliasing raw `float*` wires while staying `unsafe`-free: it
//! carves the block into its disjoint state/control/param-map spans in one `get_disjoint_mut` call,
//! and each unit writes into the shared scratch (disjoint from its inputs), which the loop then
//! publishes into the wires. Allocated unit memory lives in a *separate* pool, so a unit's region is
//! borrowed from it while the block's spans stay borrowed from the graph pool.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use bytemuck::{cast_slice, cast_slice_mut};
use rt_alloc::{HeapBlocks, Region, RtPool};

use crate::command::Reply;

use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::bus::Buses;
use plyphon_dsp::fft::FftTables;
use plyphon_dsp::math;
use plyphon_dsp::rate::{Rate, RateInfo};
use plyphon_dsp::rng::Rng;
use plyphon_dsp::wavetable::Wavetables;
use plyphon_unit::graphdef::{BlockLayout, ConstructorUnit, GraphDef, LocalBufMeta};
use plyphon_unit::unit::{
    self, Aux, AuxAlloc, DemandAccess, DemandWorld, DoneAction, DoneState, InitCtx, Inputs,
    LocalBufs, LocalBus, NodeMsg, NodeMsgSink, NodeOp, NodeOpSink, Outputs, ProcessCtx, RgenId,
    Trigger, TriggerSink,
};

/// The pool type the engine uses: a heap-backed rt-pool of 64-byte-aligned blocks.
pub(crate) type Pool = RtPool<HeapBlocks>;

/// One pool-sized unit's allocation within a synth.
enum AuxSlot {
    /// Not allocated yet.
    Unused,
    /// The unit's region in the World's unit pool.
    Live(Region),
    /// The pool could not satisfy the unit's request; the unit is silenced for the rest of its life
    /// (scsynth's `ClearUnitOnMemFailed`).
    Failed,
}

/// A synth's table of unit-pool allocations: one slot per unit that sizes its memory from live
/// inputs when the synth starts ([`GraphDef::num_pool_slots`]).
///
/// Built control-side, so the audio thread never allocates it, and carried in the synth-creation
/// command. When the synth ends its regions return to the unit pool on the audio thread and the
/// table itself goes back to the NRT side to be dropped. Empty (and allocation-free) for synths
/// with no such unit.
pub struct AuxSlots(Box<[AuxSlot]>);

impl AuxSlots {
    /// A table of `len` unallocated slots. An empty table does not allocate.
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return AuxSlots(Box::new([]));
        }
        AuxSlots((0..len).map(|_| AuxSlot::Unused).collect())
    }

    /// Number of slots.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the table has no slots (dropping it frees nothing).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Return every live region to `pool`, leaving the slots unallocated. Called on the audio thread
    /// when the synth ends (scsynth's unit destructors calling `RTFree`).
    pub(crate) fn release(&mut self, pool: &mut Pool) {
        for slot in self.0.iter_mut() {
            if let AuxSlot::Live(region) = core::mem::replace(slot, AuxSlot::Unused) {
                pool.dealloc(region);
            }
        }
    }
}

/// A pool-sized unit's allocator for one call: its slot and the World's unit pool.
struct PoolSlot<'a> {
    pool: &'a mut Pool,
    slot: &'a mut AuxSlot,
}

impl AuxAlloc for PoolSlot<'_> {
    fn alloc(&mut self, bytes: usize) -> bool {
        match self.slot {
            AuxSlot::Live(_) => true,
            AuxSlot::Failed => false,
            AuxSlot::Unused => match self.pool.alloc(bytes) {
                Some(region) => {
                    *self.slot = AuxSlot::Live(region);
                    true
                }
                None => {
                    *self.slot = AuxSlot::Failed;
                    false
                }
            },
        }
    }

    fn bytes(&mut self) -> &mut [u8] {
        match &*self.slot {
            AuxSlot::Live(region) => self.pool.slice_mut(region),
            AuxSlot::Unused | AuxSlot::Failed => &mut [],
        }
    }
}

/// The per-block materials the process loop draws on to assemble each unit's [`ProcessCtx`] and
/// [`InitCtx`]. Built once per control block by the [`World`](crate::world::World) and threaded
/// through the node tree. Its fields are disjoint, so a [`Graph`] can borrow the pool, the shared
/// scratch, and the buses at once.
pub(crate) struct Block<'a> {
    /// The World's audio-rate timing - in particular its block size, which the graph divides by its
    /// own (possibly smaller) block to get the reblock tick count. Each unit's own rate comes from
    /// its [`GraphDef`], not here.
    pub audio: &'a RateInfo,
    /// Shared wavetables.
    pub wavetables: &'a Wavetables,
    /// Shared FFT plans + windows (empty without the `fft` feature).
    pub fft: &'a FftTables,
    /// The World's shared buses.
    pub buses: &'a mut Buses,
    /// The World's shared buffer table.
    pub buffers: &'a mut BufferTable,
    /// The current block counter.
    pub buf_counter: u64,
    /// The rt-pool holding every graph's per-instance block.
    pub pool: &'a mut Pool,
    /// The rt-pool holding the memory units allocate at synth start, sized from live inputs. Separate
    /// from [`pool`](Self::pool) so a unit's region can be borrowed while the block's spans are.
    pub unit_pool: &'a mut Pool,
    /// World-shared audio wire scratch, reused per graph (`max_wire_bufs * block_size` f32).
    pub wire_scratch: &'a mut [f32],
    /// World-shared per-unit output scratch, reused per unit (`max_unit_outputs * block_size` f32).
    pub unit_scratch: &'a mut [f32],
    /// The World's random streams; each graph draws from the one it points at.
    pub rgens: &'a mut [Rng],
    /// World-shared sink for triggers fired this block (`SendTrig`), drained after the tree walk.
    pub triggers: &'a mut Vec<Trigger>,
    /// Cap on triggers per block; pushes past it are dropped so the audio thread never reallocates.
    pub trigger_cap: usize,
    /// World-shared sink for `SendReply` messages emitted this block, drained after the tree walk.
    pub node_msgs: &'a mut Vec<NodeMsg>,
    /// Cap on node messages per block; pushes past it are dropped so the audio thread never reallocates.
    pub node_msg_cap: usize,
    /// World-shared sink for node ops (`Free`/`Pause` by id) emitted this block, applied after walk.
    pub node_ops: &'a mut Vec<NodeOp>,
    /// Cap on node ops per block; pushes past it are dropped so the audio thread never reallocates.
    pub node_op_cap: usize,
    /// World-shared sink for `/n_trace` dump records emitted this block (empty unless a node is being
    /// traced), drained into the reply ring after the walk.
    pub trace: &'a mut Vec<Reply>,
    /// Cap on trace records per block; pushes past it are dropped so the audio thread never reallocates.
    pub trace_cap: usize,
    /// Live synth count at the start of this block, surfaced to `NumRunningSynths`.
    pub running_synths: usize,
}

/// A live synth instance.
pub struct Graph {
    /// The one graph-pool allocation: `[ state arena | control wires | param maps | ... ]`, grown
    /// on the first block by the storage of each `LocalBuf`.
    block: Region,
    /// Samples of local-buffer storage appended to `block` so far (the length of its `local_bufs`
    /// span, in `f32`s).
    local_samples: u32,
    /// The unit-pool allocations of units that size their memory at synth start.
    aux: AuxSlots,
    /// The shared, immutable compiled def.
    def: Arc<GraphDef>,
    /// Whether the constructor pass has run: every unit's
    /// [`Unit::init`](plyphon_unit::unit::Unit::init), in SynthDef order on the first control
    /// block, before any unit's first calc (scsynth's `Graph_FirstCalc`).
    initialized: bool,
    /// The within-block sample offset at which this synth was created (scsynth's node `mSampleOffset`).
    /// Surfaced to its units on the first block only, so `OffsetOut` onsets sample-exactly; 0 for an
    /// immediately-created synth.
    sample_offset: usize,
    /// The fractional (sub-sample) part of that creation offset (scsynth's node `mSubsampleOffset`),
    /// in `[0, 1)`; surfaced to the first block only, where a `SubsampleOffset` UGen reads it. 0 for
    /// an immediately-created synth.
    subsample_offset: f32,
    /// One-shot `/n_trace` request: when set, the next [`process`](Self::process) dumps each unit's
    /// inputs/outputs and clears it (scsynth's one-block `Graph_CalcTrace`).
    trace: bool,
    /// Which of the World's random streams this synth draws from (scsynth's `mRGen`): stream 0
    /// until a `RandID` selects another.
    rgen: u32,
}

impl Graph {
    /// Wrap a freshly allocated, initialised block and its def into a live graph, created at
    /// `sample_offset` samples (plus `subsample_offset` fractional samples) into its first control
    /// block (0 unless scheduled mid-block).
    pub(crate) fn new(
        block: Region,
        aux: AuxSlots,
        def: Arc<GraphDef>,
        sample_offset: usize,
        subsample_offset: f32,
    ) -> Self {
        Graph {
            block,
            local_samples: 0,
            aux,
            def,
            initialized: false,
            sample_offset,
            subsample_offset,
            trace: false,
            rgen: 0,
        }
    }

    /// Consume the graph, returning its graph-pool block and its unit-pool allocations so the World
    /// can release both on the audio thread.
    pub(crate) fn into_parts(self) -> (Region, AuxSlots) {
        (self.block, self.aux)
    }

    /// Request a one-block `/n_trace` dump on this synth's next [`process`](Self::process).
    pub(crate) fn set_trace(&mut self) {
        self.trace = true;
    }

    /// Compute one control block for the synth with client id `node_id` (surfaced to side-effecting
    /// units like `SendTrig`). Returns the strongest [`DoneAction`] any of its units requested.
    #[must_use]
    pub(crate) fn process(&mut self, block: &mut Block<'_>, node_id: i32) -> DoneAction {
        // A one-shot `/n_trace`: dump this block's per-unit I/O, then clear (scsynth's `Graph_CalcTrace`).
        let tracing = core::mem::take(&mut self.trace);
        let rgen = &mut self.rgen;
        let num_rgens = block.rgens.len() as u32;
        let aux_slots = &mut self.aux.0;
        let def = &*self.def;
        let bs = def.block_size();
        let layout = def.layout();

        // Carve the per-graph block into its disjoint spans (proved disjoint once, here, and again
        // only when a `LocalBuf` grows the block on the first block). The calc-unit state and the
        // demand-state arena are separate spans so a calc unit's `&mut` state slot and the `&mut`
        // demand arena (pulled re-entrantly during its `process`) never alias; the `aux` arena is
        // likewise separate so a unit holds its `&mut` state and `&mut` aux slice at once.
        let Some(Spans {
            mut state_arena,
            mut demand_state,
            mut aux_arena,
            mut ctrl,
            mut pmaps,
            mut done_flags,
            mut local,
            mut lbuf_meta,
            mut amaps,
            mut lag_state,
            mut lbuf_samples,
        }) = carve(block.pool, &self.block, &layout, self.local_samples)
        else {
            // Unreachable: the layout's spans are contiguous and disjoint by construction.
            return DoneAction::Nothing;
        };
        // Audio wires and output scratch are World-shared (separate allocations), reused per graph.
        let audio = &mut *block.wire_scratch;
        let scratch = &mut *block.unit_scratch;

        // World block vs this graph's (possibly smaller) block. A reblocked def runs the whole calc
        // list `num_ticks` times per World control block, each tick a shorter `bs`-sample slice; an
        // ordinary def has `num_ticks == 1` (`bs == world_bs`), one pass. Interior wires pack at `bs`
        // and are reused each tick, so interior DSP units are unaware of reblocking; only the bus
        // boundary (`In`/`Out`/`AudioControl`) spans the full World block, sliced per tick.
        let world_bs = block.audio.block_size;
        // The oversample factor (scsynth's `Resample(n)`): the graph's sample rate over the World's,
        // an exact power of two (1 for an ordinary def). The graph ticks `factor`x more often, and the
        // boundary I/O decimates/zero-order-holds by it. `round()` is std-only (unavailable on the
        // wasm target), so route through `math::round`.
        let resample =
            math::round(def.audio_rate().sample_rate / block.audio.sample_rate).max(1.0) as usize;
        let num_ticks = (world_bs / bs).max(1) * resample;
        // The constructor pass runs on the very first tick only; tracked across ticks.
        let first_block = !self.initialized;
        self.initialized = true;
        let mut done = DoneAction::Nothing;

        for tick in 0..num_ticks {
            // The first block's first tick constructs every unit before the block's calc (below).
            let first = first_block && tick == 0;
            // The node's creation offset applies only to that very first tick (`OffsetOut` delays the
            // onset by it); later ticks/blocks start at the boundary.
            let sample_offset = if first { self.sample_offset } else { 0 };
            let subsample_offset = if first { self.subsample_offset } else { 0.0 };

            // Apply control-bus mappings (`/n_map`): a mapped parameter takes the bus's current value.
            for (p, &bus) in pmaps.iter().enumerate() {
                if bus != u32::MAX {
                    ctrl[def.param_wires()[p] as usize] =
                        unit::control_in(block.buses, bus as usize);
                }
            }
            // Fill each audio-rate param's (`AudioControl`) audio wire: from this tick's slice of a
            // mapped audio bus (`/n_mapa`) if set, otherwise from its value slot (a control wire,
            // possibly just updated by `/n_map`). Audio-rate consumers read the param as a block.
            for ap in def.audio_params() {
                let dst = ap.wire as usize * bs;
                let bus = amaps[ap.param as usize];
                if bus != u32::MAX {
                    // Read this tick's `bs / resample` World-rate samples from the bus and
                    // zero-order-hold them up to the graph wire's `bs` samples (an identity copy when
                    // not reblocked/resampled: `tick == 0`, `resample == 1`). Only a bus written
                    // *this* block is read - scsynth's `Graph_Calc` checks `mAudioBusTouched` for a
                    // mapped audio param exactly as `In.ar` does - so a stale bus reads as silence.
                    let touched =
                        unit::audio_in_touched(block.buses, bus as usize, block.buf_counter);
                    let chan = unit::audio_in(block.buses, bus as usize);
                    let world_samples = bs / resample;
                    let off = tick * world_samples;
                    if touched && chan.len() >= off + world_samples {
                        if resample == 1 {
                            // The common (non-oversampled) case: a straight copy, with no
                            // per-sample division for the compiler to grind through.
                            audio[dst..dst + bs].copy_from_slice(&chan[off..off + bs]);
                        } else {
                            for (j, slot) in audio[dst..dst + bs].iter_mut().enumerate() {
                                *slot = chan[off + j / resample];
                            }
                        }
                    } else {
                        audio[dst..dst + bs].fill(0.0);
                    }
                } else {
                    audio[dst..dst + bs].fill(ctrl[ap.value_slot as usize]);
                }
            }
            // De-zipper each lagged param (`LagControl`): one one-pole step per control tick from its
            // value slot (possibly just `/n_map`'d) into its lagged output wire. On the first tick
            // the state seeds from the live value slot - scsynth's `LagControl_Ctor`
            // (`m_y1[i] = mapin[i][0]`) runs at first calc, after `/s_new`'s control pairs or a
            // pre-first-block `/n_set`/`/n_map` landed - so the first tick holds steady at the
            // current value with no ramp from the default.
            for (li, lp) in def.lag_params().iter().enumerate() {
                let x = ctrl[lp.value_slot as usize];
                if first {
                    lag_state[li] = x;
                }
                let y = x + lp.b1 * (lag_state[li] - x);
                lag_state[li] = y;
                ctrl[lp.wire as usize] = y;
            }

            if tracing && tick == 0 {
                push_trace(
                    block.trace,
                    block.trace_cap,
                    Reply::TraceHeader { node: node_id },
                );
            }
            // scsynth's `Graph_FirstCalc`: on the first tick every unit is constructed, in SynthDef
            // order with calc and demand units interleaved, before any unit's first calc. A calc
            // unit's constructor (`Unit::init`) reads one sample of each input - the sample each
            // earlier constructor wrote - and writes one sample of each output; a demand unit's
            // constructor resets it (`next(unit, 0)`). Then every calc unit runs the block, except a
            // scalar-rate one: scsynth calculates it only in its constructor, leaving it out of the
            // calc list.
            let construct_order = if first { def.constructor_units() } else { &[] };
            for step in 0..construct_order.len() + def.units().len() {
                let construct = step < construct_order.len();
                let i = match construct_order.get(step) {
                    None => step - construct_order.len(),
                    Some(&ConstructorUnit::Calc(index)) => index as usize,
                    Some(&ConstructorUnit::Demand(index)) => {
                        let mut local_bufs = LocalBufs::new(
                            &mut *lbuf_meta,
                            &mut *lbuf_samples,
                            def.audio_rate().sample_rate,
                        );
                        let mut node_msgs =
                            NodeMsgSink::new(&mut *block.node_msgs, block.node_msg_cap);
                        let mut world = DemandWorld {
                            buffers: &mut *block.buffers,
                            local_bufs: &mut local_bufs,
                            node_id,
                            node_msgs: &mut node_msgs,
                            buf_counter: block.buf_counter,
                            rgen: &mut block.rgens[*rgen as usize],
                        };
                        DemandAccess::new(
                            def.demand_units(),
                            &mut *demand_state,
                            &*audio,
                            &*ctrl,
                            bs,
                        )
                        .init(&mut world, index as usize);
                        continue;
                    }
                };
                let v = &def.units()[i];
                if !construct && v.rate == Rate::Scalar {
                    continue;
                }
                // This unit's own rate constants (scsynth's `unit->mRate`): the graph's audio rate
                // for an `.ar` unit, its control rate for a `.kr`/`.ir` one. Both are graph-relative,
                // so reblock/resample stay exact.
                let own = match v.rate {
                    Rate::Audio => def.audio_rate(),
                    _ => def.control_rate(),
                };
                // The unit's calc length (scsynth's `inNumSamples`): a full block for an `.ar`
                // unit, one sample for a `.kr`/`.ir` one - its `Outputs` are sliced to this, so a
                // control-rate unit computes (and pays for) exactly one sample per tick.
                let calc_len = match v.rate {
                    Rate::Audio if !construct => bs,
                    _ => 1,
                };
                // scsynth's `LocalBuf_Ctor`: on the first block a `LocalBuf` reads its shape from
                // its live inputs and its storage is appended to the block, which moves the block,
                // so every span is carved again from the grown block before the unit runs.
                if construct && let Some((index, shape)) = v.local_buf {
                    let (channels, frames) = shape(&Inputs::new(&v.inputs, &*audio, &*ctrl, bs));
                    let grown = grow_local_bufs(
                        block.pool,
                        &mut self.block,
                        &layout,
                        &mut self.local_samples,
                        channels,
                        frames,
                    );
                    let Some(spans) = carve(block.pool, &self.block, &layout, self.local_samples)
                    else {
                        return DoneAction::Nothing;
                    };
                    Spans {
                        state_arena,
                        demand_state,
                        aux_arena,
                        ctrl,
                        pmaps,
                        done_flags,
                        local,
                        lbuf_meta,
                        amaps,
                        lag_state,
                        lbuf_samples,
                    } = spans;
                    // A failed allocation leaves the record without storage; the `LocalBuf` then
                    // outputs `-1` and reports done.
                    if let (Some(meta), Some(slot)) = (grown, lbuf_meta.get_mut(index as usize)) {
                        *slot = meta;
                    }
                }
                let state = &mut state_arena[v.state_offset..v.state_offset + v.state_size];
                // A unit that sizes its memory at synth start has a slot in the unit-pool table; every
                // other unit's memory (if any) is its reserved sub-slice of the block's aux arena.
                let mut slot = v.pool_slot.and_then(|s| aux_slots.get_mut(s as usize));
                // This unit's done flag, carried in from last block; written back after `process` so
                // done-ness persists. A watcher reads earlier units' flags (already written this block).
                let mut done_flag = done_flags[i];
                let ins = Inputs::new(&v.inputs, &*audio, &*ctrl, bs);
                let ins = if construct { ins.with_len(1) } else { ins };
                // `/n_trace`: dump this unit's index and its inputs' first samples (scsynth's `ZIN0`) before
                // it runs; its outputs' first samples (`ZOUT0`) follow after `process`, below.
                if tracing && !construct {
                    push_trace(
                        block.trace,
                        block.trace_cap,
                        Reply::TraceUnit {
                            index: i as i32,
                            num_inputs: v.inputs.len() as i32,
                            num_outputs: v.outputs.len() as i32,
                        },
                    );
                    for j in 0..ins.len() {
                        push_trace(
                            block.trace,
                            block.trace_cap,
                            Reply::TraceValue {
                                value: input_first(&ins, j),
                            },
                        );
                    }
                }
                if construct {
                    let init_ctx = InitCtx {
                        audio: def.audio_rate(),
                        control: def.control_rate(),
                        own,
                        wavetables: block.wavetables,
                        fft: block.fft,
                        ins,
                        buses: &*block.buses,
                        buffers: &*block.buffers,
                        local_bufs: LocalBufs::new(
                            &mut *lbuf_meta,
                            &mut *lbuf_samples,
                            def.audio_rate().sample_rate,
                        ),
                        buf_counter: block.buf_counter,
                    };
                    // scsynth's constructor `RTAlloc`: the unit sizes its memory from the same live
                    // inputs its constructor (`init`) reads, just before it.
                    if let Some(slot) = slot.as_deref_mut() {
                        let mut alloc = PoolSlot {
                            pool: &mut *block.unit_pool,
                            slot,
                        };
                        (v.alloc)(state, &init_ctx, &mut Aux::pending(&mut alloc));
                    }
                }
                // A unit whose allocation failed outputs its cleared value (zeros, or `-1` for an FFT
                // chain unit) and reports done for the rest of its life, as scsynth's
                // `ClearUnitOnMemFailed` switches its calc func to `ClearUnitOutputs`.
                let silenced = matches!(slot.as_deref(), Some(AuxSlot::Failed));
                let mut alloc;
                let aux = match slot.as_deref_mut() {
                    _ if silenced => Aux::new(&mut []),
                    Some(AuxSlot::Live(region)) => Aux::new(block.unit_pool.slice_mut(region)),
                    Some(slot) => {
                        alloc = PoolSlot {
                            pool: &mut *block.unit_pool,
                            slot,
                        };
                        Aux::pending(&mut alloc)
                    }
                    None => Aux::new(&mut aux_arena[v.aux_offset..v.aux_offset + v.aux_size]),
                };
                // A constructor that writes nothing leaves its outputs at 0.
                if construct {
                    scratch[..v.outputs.len()].fill(0.0);
                }
                // Scoped so the context's borrows of the scratch/buses/demand arena end before we publish.
                done = done.max(if silenced {
                    DoneAction::Nothing
                } else {
                    let mut ctx = ProcessCtx {
                        audio: def.audio_rate(),
                        control: def.control_rate(),
                        own,
                        wavetables: block.wavetables,
                        fft: block.fft,
                        ins,
                        outs: Outputs::new(&mut scratch[..], calc_len),
                        buses: &mut *block.buses,
                        buffers: &mut *block.buffers,
                        buf_counter: block.buf_counter,
                        tick,
                        resample_factor: resample,
                        sample_offset,
                        subsample_offset,
                        // A consumer pulls demand sources through this; non-demand units ignore it. The
                        // demand arena is disjoint from this unit's `state` slot above.
                        demand: DemandAccess::new(
                            def.demand_units(),
                            &mut *demand_state,
                            &*audio,
                            &*ctrl,
                            bs,
                        ),
                        node_id,
                        triggers: TriggerSink::new(&mut *block.triggers, block.trigger_cap),
                        node_msgs: NodeMsgSink::new(&mut *block.node_msgs, block.node_msg_cap),
                        running_synths: block.running_synths,
                        done: DoneState::new(&*done_flags, &mut done_flag),
                        node_ops: NodeOpSink::new(&mut *block.node_ops, block.node_op_cap),
                        local: LocalBus::new(&mut *local, bs),
                        // A local buffer's own sample rate is the graph's audio rate (scsynth's
                        // `FULLRATE`).
                        local_bufs: LocalBufs::new(
                            &mut *lbuf_meta,
                            &mut *lbuf_samples,
                            def.audio_rate().sample_rate,
                        ),
                        aux,
                        rgen: &mut block.rgens[*rgen as usize],
                        rgen_id: RgenId::new(&mut *rgen, num_rgens),
                    };
                    if construct {
                        (v.init)(state, &mut ctx)
                    } else {
                        (v.process)(state, &mut ctx)
                    }
                });
                if silenced || matches!(slot.as_deref(), Some(AuxSlot::Failed)) {
                    scratch[..v.outputs.len() * calc_len].fill(v.cleared_output);
                    done_flag = 1;
                }
                // Persist this unit's done flag for next block / for later units to read this block.
                done_flags[i] = done_flag;
                // `/n_trace`: dump this unit's outputs' first samples (scsynth's `ZOUT0`), read from the
                // scratch the unit just wrote, before it is published into the wires below.
                if tracing && !construct {
                    for k in 0..v.outputs.len() {
                        push_trace(
                            block.trace,
                            block.trace_cap,
                            Reply::TraceValue {
                                value: scratch[k * calc_len],
                            },
                        );
                    }
                }
                // Publish this unit's scratch outputs into the wires. Scratch is packed at the
                // unit's calc length: full blocks for an audio unit, single samples for a
                // control-rate one (whose outputs are all control wires).
                for (k, ow) in v.outputs.iter().enumerate() {
                    let src = k * calc_len;
                    match ow.rate {
                        Rate::Audio => {
                            let dst = ow.wire as usize * bs;
                            audio[dst..dst + calc_len]
                                .copy_from_slice(&scratch[src..src + calc_len]);
                        }
                        Rate::Control | Rate::Scalar => {
                            ctrl[ow.wire as usize] = scratch[src];
                        }
                        // Calc units never publish a demand-rate output (demand units are not in this loop).
                        Rate::Demand => {}
                    }
                }
            }
            if tracing && tick == 0 {
                push_trace(block.trace, block.trace_cap, Reply::TraceEnd);
            }
            // `TrigControl`: now that the units have read them, zero each trigger param's value slot,
            // so a `/n_set` is seen for exactly the control tick it lands in and reads `0` after
            // (scsynth's "output then zero the control").
            for &slot in def.trig_params() {
                ctrl[slot as usize] = 0.0;
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

    /// Map audio-rate parameter `param` to audio `bus` (or unmap it with `None`) for `/n_mapa`. While
    /// mapped, the parameter's audio wire takes the bus's block each block. No-op if out of range; a
    /// non-audio param's slot is never read, so this is effectively a no-op there too.
    pub(crate) fn map_control_audio(&mut self, pool: &mut Pool, param: usize, bus: Option<u32>) {
        if param < self.def.num_params() {
            let bytes = &mut pool.slice_mut(&self.block)[self.def.layout().amaps.range()];
            cast_slice_mut::<u8, u32>(bytes)[param] = bus.unwrap_or(u32::MAX);
        }
    }
}

/// A synth block carved into its disjoint, typed spans (see [`BlockLayout`]).
struct Spans<'p> {
    state_arena: &'p mut [u8],
    demand_state: &'p mut [u8],
    aux_arena: &'p mut [u8],
    ctrl: &'p mut [f32],
    pmaps: &'p [u32],
    done_flags: &'p mut [u32],
    local: &'p mut [f32],
    lbuf_meta: &'p mut [LocalBufMeta],
    amaps: &'p [u32],
    lag_state: &'p mut [f32],
    lbuf_samples: &'p mut [f32],
}

/// Carve `block` into its spans, with `local_samples` samples of local-buffer storage appended.
/// `None` only for a malformed layout (the spans are contiguous and disjoint by construction).
fn carve<'p>(
    pool: &'p mut Pool,
    block: &Region,
    layout: &BlockLayout,
    local_samples: u32,
) -> Option<Spans<'p>> {
    let local_bufs = layout.local_bufs.off..layout.local_bufs.off + local_samples as usize * 4;
    let [
        state_arena,
        demand_state,
        aux_arena,
        ctrl,
        pmaps,
        done_flags,
        local,
        lbuf_meta,
        amaps,
        lag_state,
        lbuf_samples,
    ] = pool
        .slice_mut(block)
        .get_disjoint_mut([
            layout.state.range(),
            layout.demand_state.range(),
            layout.aux.range(),
            layout.control.range(),
            layout.pmaps.range(),
            layout.done_flags.range(),
            layout.local.range(),
            layout.local_buf_meta.range(),
            layout.amaps.range(),
            layout.lag_state.range(),
            local_bufs,
        ])
        .ok()?;
    Some(Spans {
        state_arena,
        demand_state,
        aux_arena,
        ctrl: cast_slice_mut(ctrl),
        pmaps: cast_slice(pmaps),
        done_flags: cast_slice_mut(done_flags),
        local: cast_slice_mut(local),
        lbuf_meta: cast_slice_mut(lbuf_meta),
        amaps: cast_slice(amaps),
        lag_state: cast_slice_mut(lag_state),
        lbuf_samples: cast_slice_mut(lbuf_samples),
    })
}

/// Append a `channels x frames` local buffer to `block` - scsynth's `LocalBuf_allocBuffer`, run
/// from `LocalBuf_Ctor` with `(int)IN0(0)` channels and `(int)IN0(1)` frames. The block is
/// reallocated with room for the new storage (allocate, copy, free), and the new samples are
/// zeroed. Returns the buffer's record, or `None` - leaving `block` untouched - when the shape is
/// out of range or the pool cannot hold the grown block.
fn grow_local_bufs(
    pool: &mut Pool,
    block: &mut Region,
    layout: &BlockLayout,
    local_samples: &mut u32,
    channels: f32,
    frames: f32,
) -> Option<LocalBufMeta> {
    // A negative shape asks for no storage (scsynth would request a size no pool can hold).
    let channels = (channels as i32).max(0) as u32;
    let frames = (frames as i32).max(0) as u32;
    let offset = *local_samples;
    // Offsets stay representable in the `u32` records.
    let grown_samples = offset.checked_add(channels.checked_mul(frames)?)?;
    let bytes = (grown_samples as usize)
        .checked_mul(4)?
        .checked_add(layout.total)?;
    let grown = pool.alloc(bytes)?;
    let Some([old, new]) = pool.slices_mut([&*block, &grown]) else {
        pool.dealloc(grown);
        return None;
    };
    let (kept, fresh) = new.split_at_mut(old.len());
    kept.copy_from_slice(old);
    fresh.fill(0);
    pool.dealloc(core::mem::replace(block, grown));
    *local_samples = grown_samples;
    Some(LocalBufMeta {
        coord: 0,
        channels,
        frames,
        offset,
        live: 1,
    })
}

/// Push a `/n_trace` record, dropping it if the per-block trace sink is full (so the audio thread
/// never reallocates - a truncated dump is harmless, reset by the next node's `TraceHeader`).
fn push_trace(trace: &mut Vec<Reply>, cap: usize, record: Reply) {
    if trace.len() < cap {
        trace.push(record);
    }
}

/// The first sample of input port `i` (scsynth's `ZIN0`), read at the port's rate. A demand input has
/// no wire value, so it reads as `0.0`.
fn input_first(ins: &Inputs, i: usize) -> f32 {
    match ins.rate(i) {
        Rate::Audio => ins.audio(i).first().copied().unwrap_or(0.0),
        Rate::Control | Rate::Scalar => ins.control(i),
        Rate::Demand => 0.0,
    }
}
