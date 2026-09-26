//! Demand-rate unit generators - plyphon's port of scsynth's `DemandUGens`, plus the units from
//! other plugin files that scsynth also runs at demand rate: the math operators ([`demand_op`])
//! and `Unpack1FFT` (`UnpackFFTUGens.cpp`).
//!
//! Demand rate is the odd one out: every other rate is *pushed* (a unit's [`process`](super::Unit::process)
//! runs once per control block and writes a wire), but a demand-rate unit is *pulled* - it produces a
//! single value only when a consumer asks for one. In scsynth a consuming unit (`Demand`/`Duty`) calls
//! the source's `mCalcFunc` with `inNumSamples > 0` to "produce the next value" (`DEMANDINPUT_A`) or
//! `inNumSamples == 0` to "reset" (`RESETINPUT`); a constant input just returns its value; sources are
//! single-output, emit `NaN` to signal exhaustion, and nest (a source's input may be another source).
//!
//! plyphon keeps that pull model but splits it from the per-block calc list. Demand-rate units are
//! *not* in [`GraphDef::units`](crate::graphdef::GraphDef::units); they live in a separate
//! [`demand plan`](crate::graphdef::GraphDef::demand_units) with their state in the block's
//! `demand_state` span, and are driven on the audio thread by a consumer via [`DemandAccess`]. All of
//! this runs on the RT thread (only SynthDef compilation - the graph topology - is off-RT), and the
//! recursion is allocation-free: each pull copies the source's tiny `Pod` state onto the stack so the
//! recursive call can reborrow the rest of the arena (the graph is a DAG, so a unit never recurses
//! into its own slot).
//!
//! - [`MAX_DEMAND_STATE`] / [`MAX_DEMAND_DEPTH`] bound the stack copy and the recursion depth. A
//!   SynthDef that would exceed either is rejected at compile time (off-RT), keeping the audio thread
//!   bounded and `unsafe`-free.
//! - A unit that needs more memory than its state, sized when the SynthDef is compiled (`Dshuf`'s
//!   index table), reserves an aux region with [`demand_unit_spec_aux`] and reaches it through
//!   [`DemandCtx::aux_mut`]. The regions follow every unit's state in the same span, in demand-plan
//!   order, and a pull lends a unit its region by splitting the arena below it, so a nested pull
//!   (always into an earlier unit) never reaches a region lent further up.

pub mod dbrown;
pub mod dbufrd;
pub mod dbufwr;
pub mod dconst;
pub mod ddup;
pub mod demand_env_gen;
pub mod demand_op;
pub mod demand_ugen;
pub mod dgeom;
pub mod dibrown;
pub mod diwhite;
pub mod dpoll;
pub mod drand;
pub mod dreset;
pub mod dseq;
pub mod dser;
pub mod dseries;
pub mod dshuf;
pub mod dswitch;
pub mod duty;
pub mod dwhite;
pub mod dwrand;
pub mod dxrand;
#[cfg(feature = "fft")]
pub mod unpack1fft;

use alloc::boxed::Box;

use bytemuck::Pod;
use plyphon_dsp::buffer::{BufView, BufViewMut, BufferTable};
use plyphon_dsp::rng::Rng;

use crate::unit::{
    InputSource, Inputs, LocalBufs, MAX_LABEL, MAX_VALUES, NodeMsg, NodeMsgKind, NodeMsgSink,
    buffer_at, buffer_at_mut,
};

pub use dbrown::Dbrown;
pub use dbufrd::Dbufrd;
pub use dbufwr::Dbufwr;
pub use dconst::Dconst;
pub use ddup::Ddup;
pub use demand_env_gen::DemandEnvGen;
pub use demand_op::{DemandBinaryOp, DemandUnaryOp};
pub use demand_ugen::Demand;
pub use dgeom::Dgeom;
pub use dibrown::Dibrown;
pub use diwhite::Diwhite;
pub use dpoll::Dpoll;
pub use drand::Drand;
pub use dreset::Dreset;
pub use dseq::Dseq;
pub use dser::Dser;
pub use dseries::Dseries;
pub use dshuf::Dshuf;
pub use dswitch::{Dswitch, Dswitch1};
pub use duty::Duty;
pub use dwhite::Dwhite;
pub use dwrand::Dwrand;
pub use dxrand::Dxrand;
#[cfg(feature = "fft")]
pub use unpack1fft::Unpack1Fft;

/// The largest `Pod` state a demand-rate unit may have, in bytes. A pull copies the source's state
/// into a stack buffer this size, so the recursion can reborrow the whole demand arena without
/// aliasing. Compilation rejects a demand unit whose state is larger (off-RT), so the RT path never
/// over-runs the buffer. Comfortably fits the built-in sources (a `u32` index, a couple of `f32`s,
/// and `Dwhite`'s 16-byte `Rng`).
pub const MAX_DEMAND_STATE: usize = 64;

/// The deepest a demand graph may nest (`Dseq([Dseq([Dseq(...)])])`). Each level recurses the audio
/// thread's stack, so compilation rejects deeper graphs (off-RT) to keep the recursion bounded.
pub const MAX_DEMAND_DEPTH: usize = 16;

/// Which side of scsynth's `inNumSamples` flag a pull is: produce the next value (`> 0`) or reset
/// (`== 0`).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum Op {
    Produce,
    Reset,
    /// Construct the unit (scsynth's `*_Ctor`), once, in the constructor pass.
    Init,
}

/// The fixed stack buffer a [`pull`] copies a demand unit's state into. 16-byte aligned so the
/// `bytemuck` cast to any `Pod` demand state (alignment up to 16, e.g. `f64` in `Dseries`) succeeds.
#[repr(align(16))]
struct StateBuf([u8; MAX_DEMAND_STATE]);

/// A demand-rate unit - plyphon's `DemandUGen`. Like [`Unit`](super::Unit) its state must be [`Pod`]
/// so it can live as bytes in the rt-pool; behaviour is invoked through the [`DemandVtbl`] a
/// [`UnitDef`](crate::unit::registry::UnitDef) builds via [`demand_unit_spec`].
///
/// A source produces one value per [`produce`](DemandUnit::produce); it returns [`f32::NAN`] to
/// signal that its sequence is exhausted (scsynth's `DNAN`). [`reset`](DemandUnit::reset) restarts it
/// (and must propagate the reset to any demand-rate inputs via [`DemandCtx::reset`]).
pub trait DemandUnit: Pod {
    /// Reset internal state to the start of the sequence (scsynth's `inNumSamples == 0` branch). The
    /// default is a no-op; sequence sources zero their counters here and reset their demand inputs.
    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {}

    /// Construct the unit - scsynth's `*_Ctor`, run once in SynthDef order on the synth's first
    /// block, before any unit's first calc (see [`Unit::init`](super::Unit::init)). Most scsynth
    /// demand constructors reset the unit (`next(unit, 0)`), and so does the default; one that does
    /// something else overrides it.
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        self.reset(ctx);
    }

    /// Produce the next value (scsynth's `inNumSamples > 0` branch), advancing state. Returns
    /// [`f32::NAN`] once the sequence is exhausted.
    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32;
}

/// A type-erased "produce next value" function over a demand unit's pool-resident state bytes.
pub type ProduceFn = fn(&mut [u8], &mut DemandCtx<'_>) -> f32;

/// A type-erased "reset" function over a demand unit's pool-resident state bytes.
pub type ResetFn = fn(&mut [u8], &mut DemandCtx<'_>);

fn produce_thunk<T: DemandUnit>(bytes: &mut [u8], ctx: &mut DemandCtx<'_>) -> f32 {
    bytemuck::from_bytes_mut::<T>(bytes).produce(ctx)
}

fn reset_thunk<T: DemandUnit>(bytes: &mut [u8], ctx: &mut DemandCtx<'_>) {
    bytemuck::from_bytes_mut::<T>(bytes).reset(ctx);
}

fn init_thunk<T: DemandUnit>(bytes: &mut [u8], ctx: &mut DemandCtx<'_>) {
    bytemuck::from_bytes_mut::<T>(bytes).init(ctx);
}

/// One demand unit's compiled record: its pull/reset/seed vtable, resolved input wiring, and state
/// slot in the demand arena - the demand-plan analogue of [`UnitVtbl`](crate::graphdef::UnitVtbl).
pub struct DemandVtbl {
    /// Produce the next value.
    pub produce: ProduceFn,
    /// Reset to the start of the sequence.
    pub reset: ResetFn,
    /// Construct the unit, once, in the constructor pass.
    pub init: ResetFn,
    /// Resolved input sources, in order (constants, wires, or nested demand units).
    pub inputs: Box<[InputSource]>,
    /// Byte offset of this unit's state within the demand-state span.
    pub state_offset: usize,
    /// Exactly `size_of::<T>()` - the bytes this unit's state occupies (`<= MAX_DEMAND_STATE`).
    pub state_size: usize,
    /// Byte offset of this unit's aux region within the demand-state span. The regions follow every
    /// unit's state, in demand-plan order, so it is at or past the end of every earlier unit's region.
    pub aux_offset: usize,
    /// Bytes of this unit's aux region (`0` for most units).
    pub aux_size: usize,
}

/// A built demand unit: its vtable plus the initial state image. Produced off the audio thread by a
/// [`UnitDef`](crate::unit::registry::UnitDef) (via [`demand_unit_spec`]) and baked into a
/// [`GraphDef`](crate::graphdef::GraphDef).
pub struct BuiltDemandUnit {
    /// Produce-next function.
    pub produce: ProduceFn,
    /// Reset function.
    pub reset: ResetFn,
    /// Constructor.
    pub init: ResetFn,
    /// `size_of::<T>()`.
    pub size: usize,
    /// `align_of::<T>()`.
    pub align: usize,
    /// Initial state bytes to copy into the demand arena when a synth is built on-RT.
    pub init_bytes: Box<[u8]>,
    /// Bytes of aux memory the unit reserves (see [`demand_unit_spec_aux`]); `0` for most units.
    pub aux_bytes: usize,
    /// Alignment of the aux region (`1` when `aux_bytes == 0`).
    pub aux_align: usize,
}

/// Build a [`BuiltDemandUnit`] from an initial state, monomorphising the thunks for `T` (the demand
/// analogue of [`unit_spec`](crate::unit::unit_spec)).
pub fn demand_unit_spec<T: DemandUnit>(state: T) -> BuiltDemandUnit {
    BuiltDemandUnit {
        produce: produce_thunk::<T>,
        reset: reset_thunk::<T>,
        init: init_thunk::<T>,
        size: core::mem::size_of::<T>(),
        align: core::mem::align_of::<T>(),
        init_bytes: bytemuck::bytes_of(&state).to_vec().into_boxed_slice(),
        aux_bytes: 0,
        aux_align: 1,
    }
}

/// Build a [`BuiltDemandUnit`] that also reserves `aux_bytes` of per-instance memory aligned to
/// `aux_align` - for a unit whose memory outgrows [`MAX_DEMAND_STATE`] but whose size is fixed when
/// the SynthDef is compiled (`Dshuf`'s index table, sized from its input count). The demand analogue
/// of [`unit_spec_aux`](crate::unit::unit_spec_aux). The unit reaches the region through
/// [`DemandCtx::aux_mut`]; it is zeroed when a synth is instantiated (it is part of the demand
/// arena's initial image) and persists for the synth's life.
pub fn demand_unit_spec_aux<T: DemandUnit>(
    state: T,
    aux_bytes: usize,
    aux_align: usize,
) -> BuiltDemandUnit {
    BuiltDemandUnit {
        aux_bytes,
        aux_align: aux_align.max(1),
        ..demand_unit_spec(state)
    }
}

/// The shared-world reach a demand-rate unit needs while producing - the pull-side analogue of the
/// world fields on [`ProcessCtx`](super::ProcessCtx). A consumer (`Demand`/`Duty`) builds this from
/// its own disjoint `ctx` fields and threads it into the pull; most demand sources ignore it, but
/// `Dbufrd`/`Dbufwr` reach the buffer table through it and `Dpoll` posts through it. Bundled (rather
/// than threaded as loose args) so a later addition - e.g. a trigger sink - is a one-field change.
///
/// Two lifetimes because the message sink itself borrows the World's `Vec`: `'w` is how long the
/// consumer lends its `ctx` fields for the pull, `'s` the lifetime of that inner borrow.
pub struct DemandWorld<'w, 's> {
    /// The World's shared buffer table (`Dbufrd`/`Dbufwr`), reached via [`DemandCtx::buffer`] /
    /// [`DemandCtx::buffer_mut`].
    pub buffers: &'w mut BufferTable,
    /// The synth's graph-local buffers (`LocalBuf`): [`DemandCtx::buffer`] and
    /// [`DemandCtx::buffer_mut`] resolve a past-capacity buffer number here, as the calc-side io
    /// fns do.
    pub local_bufs: &'w mut LocalBufs<'s>,
    /// The enclosing synth's node id (`Dpoll` tags its post with it).
    pub node_id: i32,
    /// Sink for host messages (`Dpoll` posts here, via [`DemandCtx::post`]).
    pub node_msgs: &'w mut NodeMsgSink<'s>,
    /// The World's current block counter, reached via [`DemandCtx::buf_counter`]. A source whose
    /// value belongs to the block rather than to the pull - `Unpack1FFT` reading one FFT frame -
    /// stamps it so repeated pulls within a block yield the same value.
    pub buf_counter: u64,
    /// The random stream the synth draws from (scsynth's `mParent->mRGen`), reached via
    /// [`DemandCtx::rgen`]: the demand randoms (`Dwhite`, `Drand`, ...) draw from it.
    pub rgen: &'w mut Rng,
}

/// What a demand unit touches while producing or resetting - the pull-side analogue of
/// [`ProcessCtx`](super::ProcessCtx). It exposes the unit's own inputs and, crucially, lets it pull
/// the *next value* of (or *reset*) any input that is itself a demand unit, recursing the pull. It
/// also carries the [`DemandWorld`] reach so a source can read/write a buffer or post a value.
pub struct DemandCtx<'a> {
    plan: &'a [DemandVtbl],
    /// The demand arena below this unit's aux region: every unit's state, and the aux regions of the
    /// earlier units a nested pull can reach.
    arena: &'a mut [u8],
    /// This unit's own aux region, split off above `arena`.
    aux: &'a mut [u8],
    inputs: &'a [InputSource],
    audio_wires: &'a [f32],
    control_wires: &'a [f32],
    block_size: usize,
    buffers: &'a mut BufferTable,
    local_bufs: LocalBufs<'a>,
    node_id: i32,
    node_msgs: NodeMsgSink<'a>,
    buf_counter: u64,
    rgen: &'a mut Rng,
}

impl DemandCtx<'_> {
    /// Number of inputs this unit has.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Whether input `k` is itself a demand-rate unit (scsynth's `ISDEMANDINPUT`). A demand input is
    /// pulled until it returns `NaN`; a non-demand input yields its value once.
    pub fn is_demand(&self, k: usize) -> bool {
        matches!(self.inputs[k], InputSource::Demand(_))
    }

    /// Demand the next value of input `k` (scsynth's `DEMANDINPUT_A`). A nested demand unit is pulled
    /// recursively; a constant or wire input just returns its current value.
    pub fn demand(&mut self, k: usize) -> f32 {
        match self.inputs[k] {
            // The recursive pull threads a fresh `DemandWorld` built from this ctx's own disjoint
            // fields (arena, buffers, node_msgs are separate borrows), so a nested source still reaches
            // the world.
            InputSource::Demand(d) => pull(
                self.plan,
                &mut *self.arena,
                self.audio_wires,
                self.control_wires,
                self.block_size,
                &mut DemandWorld {
                    buffers: &mut *self.buffers,
                    local_bufs: &mut self.local_bufs,
                    node_id: self.node_id,
                    node_msgs: &mut self.node_msgs,
                    buf_counter: self.buf_counter,
                    rgen: &mut *self.rgen,
                },
                d as usize,
                Op::Produce,
            ),
            InputSource::Constant(v) => v,
            InputSource::Control(w) => self.control_wires[w as usize],
            InputSource::Audio(w) => self.audio_wires[w as usize * self.block_size],
        }
    }

    /// Reset input `k` (scsynth's `RESETINPUT`). Only demand-rate inputs carry state to reset; a
    /// constant or wire input is a no-op.
    pub fn reset(&mut self, k: usize) {
        if let InputSource::Demand(d) = self.inputs[k] {
            pull(
                self.plan,
                &mut *self.arena,
                self.audio_wires,
                self.control_wires,
                self.block_size,
                &mut DemandWorld {
                    buffers: &mut *self.buffers,
                    local_bufs: &mut self.local_bufs,
                    node_id: self.node_id,
                    node_msgs: &mut self.node_msgs,
                    buf_counter: self.buf_counter,
                    rgen: &mut *self.rgen,
                },
                d as usize,
                Op::Reset,
            );
        }
    }

    /// The flat buffer at `index`, if one is installed - for a demand-rate reader (`Dbufrd`).
    /// RT-safe (no panic; a stream/empty/out-of-range slot yields `None`). A past-capacity `index`
    /// resolves to the synth's graph-local buffers, as in [`buffer_at`].
    pub fn buffer(&self, index: usize) -> Option<BufView<'_>> {
        buffer_at(self.buffers, &self.local_bufs, index)
    }

    /// The flat buffer at `index`, mutably - for a demand-rate writer (`Dbufwr`). RT-safe (no panic);
    /// past-capacity indices resolve to the graph-local buffers, as in [`buffer_at_mut`].
    pub fn buffer_mut(&mut self, index: usize) -> Option<BufViewMut<'_>> {
        buffer_at_mut(self.buffers, &mut self.local_bufs, index)
    }

    /// The enclosing synth's node id, for a source that tags an emitted message (`Dpoll`).
    pub fn node_id(&self) -> i32 {
        self.node_id
    }

    /// The World's current block counter (scsynth's `mBufCounter`).
    ///
    /// A pull is not a block boundary: one block may pull the same source many times, and a source
    /// reached through several consumers is pulled once per consumer. A source whose value is a
    /// property of the *block* - `Unpack1FFT` reading one bin of the current FFT frame - records
    /// this alongside its value and recomputes only when it changes, so every pull within a block
    /// sees the same frame.
    pub fn buf_counter(&self) -> u64 {
        self.buf_counter
    }

    /// The random stream the synth draws from (scsynth's `mParent->mRGen`).
    pub fn rgen(&mut self) -> &mut Rng {
        self.rgen
    }

    /// This unit's aux memory (reserved with [`demand_unit_spec_aux`]) as a slice of `Pod` elements
    /// `T`. Empty for a unit that reserved none, or whose region does not fit `T`'s size or alignment
    /// (a unit sizes and aligns its region for `T`, so that does not happen for a well-built unit).
    pub fn aux_mut<T: Pod>(&mut self) -> &mut [T] {
        bytemuck::try_cast_slice_mut(&mut *self.aux).unwrap_or(&mut [])
    }

    /// Post one polled `value` to the host (`Dpoll`): a [`NodeMsg`] of kind [`NodeMsgKind::Poll`]
    /// carrying the baked `label` and the optional `trigid` (echoed as `reply_id`). Best-effort - it
    /// is dropped if the block's message capacity is reached, like every other node message.
    pub fn post(&mut self, label: &[u8; MAX_LABEL], label_len: u32, trigid: i32, value: f32) {
        let mut values = [0.0f32; MAX_VALUES];
        values[0] = value;
        self.node_msgs.push(NodeMsg {
            node: self.node_id,
            reply_id: trigid,
            kind: NodeMsgKind::Poll,
            label: *label,
            label_len,
            values,
            num_values: 1,
        });
    }
}

/// Run one pull of demand unit `unit`: produce its next value or reset it.
///
/// The borrow-safety trick that keeps this `unsafe`-free under recursion: copy the unit's `Pod` state
/// into a stack buffer, run produce/reset against that copy while the [`DemandCtx`] holds `&mut` the
/// *whole* arena, then copy the state back. Because the active unit runs on the stack copy, a
/// recursive [`DemandCtx::demand`] can reborrow the arena and descend into a *different* slot - and
/// the graph is a DAG, so a unit never targets its own slot. Allocation-free; the buffer is fixed at
/// [`MAX_DEMAND_STATE`] and compilation guarantees `state_size <= MAX_DEMAND_STATE`.
///
/// A unit's aux region is too large to copy, so it is lent in place instead, by splitting the arena
/// at the region's start: the unit gets the region above the split, and the [`DemandCtx`] (hence any
/// nested pull) gets only the arena below it. Nothing a nested pull needs lies above the split:
/// every unit's state comes before all the aux regions, and the regions are laid out in demand-plan
/// order, while compilation only lets a demand unit take earlier demand units as inputs - so every
/// unit reachable from this one has a lower index and its region ends at or before this one begins.
/// Each nested pull splits its own (smaller) arena the same way, so the regions lent at every level
/// of the recursion are disjoint and none is reachable from below it.
// The argument list (plan, arena, the two wire arrays, block size, world, unit, op) is the genuine set
// this recursive core threads; bundling it would only obscure the reborrow at each call.
#[allow(clippy::too_many_arguments)]
fn pull(
    plan: &[DemandVtbl],
    arena: &mut [u8],
    audio_wires: &[f32],
    control_wires: &[f32],
    block_size: usize,
    world: &mut DemandWorld<'_, '_>,
    unit: usize,
    op: Op,
) -> f32 {
    let v = &plan[unit];
    let off = v.state_offset;
    let size = v.state_size;
    debug_assert!(
        size <= MAX_DEMAND_STATE,
        "demand state exceeds MAX_DEMAND_STATE"
    );
    let (arena, above) = arena.split_at_mut(v.aux_offset);
    let aux = &mut above[..v.aux_size];
    let mut buf = StateBuf([0u8; MAX_DEMAND_STATE]);
    buf.0[..size].copy_from_slice(&arena[off..off + size]);
    let out = {
        let mut ctx = DemandCtx {
            plan,
            arena: &mut *arena,
            aux,
            inputs: &v.inputs,
            audio_wires,
            control_wires,
            block_size,
            // Reborrow the world for this level; the inner `node_msgs` sink is reborrowed by value so
            // `DemandCtx` keeps a single lifetime while still being able to recurse.
            buffers: &mut *world.buffers,
            local_bufs: world.local_bufs.reborrow(),
            node_id: world.node_id,
            node_msgs: world.node_msgs.reborrow(),
            buf_counter: world.buf_counter,
            rgen: &mut *world.rgen,
        };
        match op {
            Op::Produce => (v.produce)(&mut buf.0[..size], &mut ctx),
            Op::Reset => {
                (v.reset)(&mut buf.0[..size], &mut ctx);
                0.0
            }
            Op::Init => {
                (v.init)(&mut buf.0[..size], &mut ctx);
                0.0
            }
        }
    };
    arena[off..off + size].copy_from_slice(&buf.0[..size]);
    out
}

/// The consumer-side handle to a synth's demand plan, carried in [`ProcessCtx`](super::ProcessCtx).
///
/// A consuming [`Unit`](super::Unit) (`Demand`/`Duty`) drives the demand subgraph through this:
/// [`produce`](DemandAccess::produce) pulls a source's next value, [`reset`](DemandAccess::reset)
/// resets it. For a synth with no demand units the plan is empty and these are never called. The
/// borrowed `state` is the block's `demand_state` span, disjoint from the calc units' state arena,
/// so it coexists with the calc unit's own `&mut` state slot.
pub struct DemandAccess<'a> {
    plan: &'a [DemandVtbl],
    state: &'a mut [u8],
    audio_wires: &'a [f32],
    control_wires: &'a [f32],
    block_size: usize,
}

impl<'a> DemandAccess<'a> {
    /// Build a demand handle over a synth's plan and its `demand_state` span. Used by the synth
    /// process loop.
    pub fn new(
        plan: &'a [DemandVtbl],
        state: &'a mut [u8],
        audio_wires: &'a [f32],
        control_wires: &'a [f32],
        block_size: usize,
    ) -> Self {
        DemandAccess {
            plan,
            state,
            audio_wires,
            control_wires,
            block_size,
        }
    }

    /// Pull the next value of demand unit `unit`, threading the consumer's [`DemandWorld`] reach.
    pub fn produce(&mut self, world: &mut DemandWorld<'_, '_>, unit: usize) -> f32 {
        pull(
            self.plan,
            &mut *self.state,
            self.audio_wires,
            self.control_wires,
            self.block_size,
            world,
            unit,
            Op::Produce,
        )
    }

    /// Reset demand unit `unit`, threading the consumer's [`DemandWorld`] reach.
    pub fn reset(&mut self, world: &mut DemandWorld<'_, '_>, unit: usize) {
        pull(
            self.plan,
            &mut *self.state,
            self.audio_wires,
            self.control_wires,
            self.block_size,
            world,
            unit,
            Op::Reset,
        );
    }

    /// Construct demand unit `unit` ([`DemandUnit::init`]) in the constructor pass.
    pub fn init(&mut self, world: &mut DemandWorld<'_, '_>, unit: usize) {
        pull(
            self.plan,
            &mut *self.state,
            self.audio_wires,
            self.control_wires,
            self.block_size,
            world,
            unit,
            Op::Init,
        );
    }
}

/// Demand the next value of a consumer's input `input` - scsynth's `DEMANDINPUT`. If the input is a
/// demand source it is pulled (recursing through any nested sources); otherwise the input's current
/// value is returned (a constant or wire behaves like a source that yields that value forever).
/// Returns [`f32::NAN`] when a pulled source is exhausted.
///
/// Takes the [`Inputs`], [`DemandAccess`] and [`DemandWorld`] as separate borrows (the `io`-style
/// free-fn convention) so a consumer can pull while it holds a `&mut` borrow of its output scratch -
/// these are all disjoint fields of [`ProcessCtx`](super::ProcessCtx).
pub fn demand_next(
    ins: &Inputs<'_>,
    demand: &mut DemandAccess<'_>,
    world: &mut DemandWorld<'_, '_>,
    input: usize,
) -> f32 {
    match ins.source(input) {
        InputSource::Demand(d) => demand.produce(world, d as usize),
        _ => ins.control(input),
    }
}

/// Reset a consumer's input `input` - scsynth's `RESETINPUT`. A demand-source input is reset
/// (recursing); a constant or wire input is a no-op.
pub fn demand_reset(
    ins: &Inputs<'_>,
    demand: &mut DemandAccess<'_>,
    world: &mut DemandWorld<'_, '_>,
    input: usize,
) {
    if let InputSource::Demand(d) = ins.source(input) {
        demand.reset(world, d as usize);
    }
}
