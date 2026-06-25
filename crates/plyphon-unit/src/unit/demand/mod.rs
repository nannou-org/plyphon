//! Demand-rate unit generators - plyphon's port of scsynth's `DemandUGens`.
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

pub mod demand_ugen;
pub mod dseq;
pub mod dseries;
pub mod duty;
pub mod dwhite;

use alloc::boxed::Box;

use bytemuck::Pod;

use crate::unit::{InputSource, Inputs, ReseedFn};

pub use demand_ugen::Demand;
pub use dseq::Dseq;
pub use dseries::Dseries;
pub use duty::Duty;
pub use dwhite::Dwhite;

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
    /// Re-seed any per-instance randomness, exactly as [`Unit::reseed`](super::Unit::reseed). The
    /// default is a no-op; `Dwhite`-style sources override it so two instances decorrelate.
    fn reseed(&mut self, _seed: u64) {}

    /// Reset internal state to the start of the sequence (scsynth's `inNumSamples == 0` branch). The
    /// default is a no-op; sequence sources zero their counters here and reset their demand inputs.
    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {}

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

fn demand_reseed_thunk<T: DemandUnit>(bytes: &mut [u8], seed: u64) {
    bytemuck::from_bytes_mut::<T>(bytes).reseed(seed);
}

/// One demand unit's compiled record: its pull/reset/seed vtable, resolved input wiring, and state
/// slot in the demand arena - the demand-plan analogue of [`UnitVtbl`](crate::graphdef::UnitVtbl).
pub struct DemandVtbl {
    /// Produce the next value.
    pub produce: ProduceFn,
    /// Reset to the start of the sequence.
    pub reset: ResetFn,
    /// Per-instance re-seed (no-op for non-random sources).
    pub reseed: ReseedFn,
    /// Resolved input sources, in order (constants, wires, or nested demand units).
    pub inputs: Box<[InputSource]>,
    /// Byte offset of this unit's state within the demand-state span.
    pub state_offset: usize,
    /// Exactly `size_of::<T>()` - the bytes this unit's state occupies (`<= MAX_DEMAND_STATE`).
    pub state_size: usize,
}

/// A built demand unit: its vtable plus the initial state image. Produced off the audio thread by a
/// [`UnitDef`](crate::unit::registry::UnitDef) (via [`demand_unit_spec`]) and baked into a
/// [`GraphDef`](crate::graphdef::GraphDef).
pub struct BuiltDemandUnit {
    /// Produce-next function.
    pub produce: ProduceFn,
    /// Reset function.
    pub reset: ResetFn,
    /// Per-instance re-seed function.
    pub reseed: ReseedFn,
    /// `size_of::<T>()`.
    pub size: usize,
    /// `align_of::<T>()`.
    pub align: usize,
    /// Initial state bytes to copy into the demand arena when a synth is built on-RT.
    pub init_bytes: Box<[u8]>,
}

/// Build a [`BuiltDemandUnit`] from an initial state, monomorphising the thunks for `T` (the demand
/// analogue of [`unit_spec`](crate::unit::unit_spec)).
pub fn demand_unit_spec<T: DemandUnit>(state: T) -> BuiltDemandUnit {
    BuiltDemandUnit {
        produce: produce_thunk::<T>,
        reset: reset_thunk::<T>,
        reseed: demand_reseed_thunk::<T>,
        size: core::mem::size_of::<T>(),
        align: core::mem::align_of::<T>(),
        init_bytes: bytemuck::bytes_of(&state).to_vec().into_boxed_slice(),
    }
}

/// What a demand unit touches while producing or resetting - the pull-side analogue of
/// [`ProcessCtx`](super::ProcessCtx). It exposes the unit's own inputs and, crucially, lets it pull
/// the *next value* of (or *reset*) any input that is itself a demand unit, recursing the pull.
pub struct DemandCtx<'a> {
    plan: &'a [DemandVtbl],
    arena: &'a mut [u8],
    inputs: &'a [InputSource],
    audio_wires: &'a [f32],
    control_wires: &'a [f32],
    block_size: usize,
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
            InputSource::Demand(d) => pull(
                self.plan,
                &mut *self.arena,
                self.audio_wires,
                self.control_wires,
                self.block_size,
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
                d as usize,
                Op::Reset,
            );
        }
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
fn pull(
    plan: &[DemandVtbl],
    arena: &mut [u8],
    audio_wires: &[f32],
    control_wires: &[f32],
    block_size: usize,
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
    let mut buf = StateBuf([0u8; MAX_DEMAND_STATE]);
    buf.0[..size].copy_from_slice(&arena[off..off + size]);
    let out = {
        let mut ctx = DemandCtx {
            plan,
            arena: &mut *arena,
            inputs: &v.inputs,
            audio_wires,
            control_wires,
            block_size,
        };
        match op {
            Op::Produce => (v.produce)(&mut buf.0[..size], &mut ctx),
            Op::Reset => {
                (v.reset)(&mut buf.0[..size], &mut ctx);
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

    /// Pull the next value of demand unit `unit`.
    pub fn produce(&mut self, unit: usize) -> f32 {
        pull(
            self.plan,
            &mut *self.state,
            self.audio_wires,
            self.control_wires,
            self.block_size,
            unit,
            Op::Produce,
        )
    }

    /// Reset demand unit `unit`.
    pub fn reset(&mut self, unit: usize) {
        pull(
            self.plan,
            &mut *self.state,
            self.audio_wires,
            self.control_wires,
            self.block_size,
            unit,
            Op::Reset,
        );
    }
}

/// Demand the next value of a consumer's input `input` - scsynth's `DEMANDINPUT`. If the input is a
/// demand source it is pulled (recursing through any nested sources); otherwise the input's current
/// value is returned (a constant or wire behaves like a source that yields that value forever).
/// Returns [`f32::NAN`] when a pulled source is exhausted.
///
/// Takes the [`Inputs`] and [`DemandAccess`] as separate borrows (the `io`-style free-fn convention)
/// so a consumer can pull while it holds a `&mut` borrow of its output scratch - they are disjoint
/// fields of [`ProcessCtx`](super::ProcessCtx).
pub fn demand_next(ins: &Inputs<'_>, demand: &mut DemandAccess<'_>, input: usize) -> f32 {
    match ins.source(input) {
        InputSource::Demand(d) => demand.produce(d as usize),
        _ => ins.control(input),
    }
}

/// Reset a consumer's input `input` - scsynth's `RESETINPUT`. A demand-source input is reset
/// (recursing); a constant or wire input is a no-op.
pub fn demand_reset(ins: &Inputs<'_>, demand: &mut DemandAccess<'_>, input: usize) {
    if let InputSource::Demand(d) = ins.source(input) {
        demand.reset(d as usize);
    }
}
