//! The init- and trigger-time random units - plyphon's ports of scsynth's `Rand`, `ExpRand`,
//! `TRand`, `TExpRand`, `TIRand`, `RandSeed` and `RandID` (`NoiseUGens.cpp`).
//!
//! Like every random unit, this family draws from the synth's random stream
//! ([`ProcessCtx::rgen`]), scsynth's `mParent->mRGen`: one of the World's streams, stream 0
//! unless `RandID` selects another. Synths drawing from the same stream share it, so their draws
//! interleave in node order, and a `RandSeed` re-seed restarts the stream for every synth on it -
//! the noise generators included, as in scsynth.
//!
//! The one-time draws happen in the first `process` call, which runs as the unit's constructor, in
//! SynthDef order before any unit's first calc, as scsynth's constructor draws do.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::sig;
use crate::unit::{BuiltUnit, DoneAction, Outputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::rng::Rng;

/// A uniform draw in `[lo, hi)` (scsynth's `rgen.frand() * (hi - lo) + lo`).
fn uniform(rgen: &mut Rng, lo: f32, hi: f32) -> f32 {
    rgen.next_unipolar() * (hi - lo) + lo
}

/// An exponential-distribution draw in `[lo, hi)` (scsynth's single-precision
/// `pow(hi / lo, frand()) * lo`): equal probability per octave, so `lo` must be non-zero and share
/// `hi`'s sign for a sensible result.
fn exponential(rgen: &mut Rng, lo: f32, hi: f32) -> f32 {
    math::powf(hi / lo, rgen.next_unipolar()) * lo
}

/// A uniform integer draw in `[lo, hi]` as a float (scsynth's `rgen.irand(hi - lo + 1) + lo`).
fn integer(rgen: &mut Rng, lo: f32, hi: f32) -> f32 {
    let lo = lo as i32;
    let hi = hi as i32;
    (rgen.next_irand(hi - lo + 1) + lo) as f32
}

/// Write `value` across the output at the unit's rate (a full block, or one control value).
fn hold(outs: &mut Outputs<'_>, audio: bool, value: f32) {
    if audio {
        outs.audio(0).fill(value);
    } else {
        *outs.control(0) = value;
    }
}

/// `Rand.new(lo, hi)`: one uniform draw in `[lo, hi)` at the synth's first block, held forever.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Rand {
    value: f32,
    /// `0` until the one-time draw has happened, in the first `process` (the constructor).
    primed: u32,
    /// `0`/`1`: audio-rate (a full block) vs control-rate (one value).
    audio: u32,
}

impl Unit for Rand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ProcessCtx {
            ins, outs, rgen, ..
        } = ctx;
        if self.primed == 0 {
            self.primed = 1;
            self.value = uniform(rgen, ins.control(0), ins.control(1));
        }
        hold(outs, self.audio != 0, self.value);
        DoneAction::Nothing
    }
}

/// Constructor for [`Rand`].
pub struct RandCtor;

impl UnitDef for RandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Rand {
            value: 0.0,
            primed: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `ExpRand.new(lo, hi)`: one exponential-distribution draw in `[lo, hi)` at the synth's first
/// block, held forever.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ExpRand {
    value: f32,
    primed: u32,
    audio: u32,
}

impl Unit for ExpRand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ProcessCtx {
            ins, outs, rgen, ..
        } = ctx;
        if self.primed == 0 {
            self.primed = 1;
            self.value = exponential(rgen, ins.control(0), ins.control(1));
        }
        hold(outs, self.audio != 0, self.value);
        DoneAction::Nothing
    }
}

/// Constructor for [`ExpRand`].
pub struct ExpRandCtor;

impl UnitDef for ExpRandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(ExpRand {
            value: 0.0,
            primed: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// The shared body of the `TRand` family: `(lo, hi, trig)` inputs, an initial draw on the first
/// block, and a fresh draw on every rising trigger edge, holding the value between.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TrigRand {
    value: f32,
    prev_trig: f32,
    primed: u32,
    audio: u32,
}

impl TrigRand {
    /// A fresh trigger-random body that draws on its first block; `audio` selects a full-block
    /// output vs a single control value.
    fn new(audio: bool) -> Self {
        TrigRand {
            value: 0.0,
            prev_trig: 0.0,
            primed: 0,
            audio: audio as u32,
        }
    }

    /// Run one block: the first call draws immediately and latches the current trigger level (as
    /// scsynth's constructor does, so a trigger already high at spawn does not double-fire); every
    /// call redraws on each `<= 0` to `> 0` trigger crossing.
    ///
    /// With `block_edge`, an audio-rate block compares every sample against the trigger level
    /// before the block rather than against the previous sample, so it redraws on every positive
    /// sample of a block that starts low. That is what scsynth's `TExpRand` and `TIRand` audio-rate
    /// loops do: they never update `prev` inside the loop (`TExpRand_next_a`, `TIRand_next_a` and
    /// their `_aa` forms in `NoiseUGens.cpp`), unlike `TRand`'s.
    fn run(
        &mut self,
        ctx: &mut ProcessCtx<'_>,
        block_edge: bool,
        draw: impl Fn(&mut Rng, f32, f32) -> f32,
    ) {
        let ProcessCtx {
            ins, outs, rgen, ..
        } = ctx;
        let lo = ins.control(0);
        let hi = ins.control(1);
        let trig = sig(ins, 2);
        if self.primed == 0 {
            self.primed = 1;
            self.value = draw(rgen, lo, hi);
            self.prev_trig = trig.at(0);
        }
        if self.audio != 0 {
            let block_start = self.prev_trig;
            for (i, o) in outs.audio(0).iter_mut().enumerate() {
                let t = trig.at(i);
                let prev = if block_edge {
                    block_start
                } else {
                    self.prev_trig
                };
                if prev <= 0.0 && t > 0.0 {
                    self.value = draw(rgen, lo, hi);
                }
                self.prev_trig = t;
                *o = self.value;
            }
        } else {
            let t = trig.at(0);
            if self.prev_trig <= 0.0 && t > 0.0 {
                self.value = draw(rgen, lo, hi);
            }
            self.prev_trig = t;
            *outs.control(0) = self.value;
        }
    }
}

/// `TRand.ar/kr(lo, hi, trig)`: a uniform draw in `[lo, hi)` on each rising trigger, held between.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TRand(TrigRand);

impl Unit for TRand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, false, uniform);
        DoneAction::Nothing
    }
}

/// Constructor for [`TRand`].
pub struct TRandCtor;

impl UnitDef for TRandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(TRand(TrigRand::new(ctx.rate == Rate::Audio))))
    }
}

/// `TExpRand.ar/kr(lo, hi, trig)`: an exponential-distribution draw in `[lo, hi)` on each rising
/// trigger, held between. At audio rate it redraws on every positive sample of a block that starts
/// low, as scsynth's audio-rate loop does.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TExpRand(TrigRand);

impl Unit for TExpRand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, true, exponential);
        DoneAction::Nothing
    }
}

/// Constructor for [`TExpRand`].
pub struct TExpRandCtor;

impl UnitDef for TExpRandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(TExpRand(TrigRand::new(ctx.rate == Rate::Audio))))
    }
}

/// `TIRand.ar/kr(lo, hi, trig)`: a uniform integer draw in `[lo, hi]` (as a float) on each rising
/// trigger, held between. At audio rate it redraws on every positive sample of a block that starts
/// low, as scsynth's audio-rate loop does.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TIRand(TrigRand);

impl Unit for TIRand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, true, integer);
        DoneAction::Nothing
    }
}

/// Constructor for [`TIRand`].
pub struct TIRandCtor;

impl UnitDef for TIRandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(TIRand(TrigRand::new(ctx.rate == Rate::Audio))))
    }
}

/// `RandSeed.kr(trig, seed)`: on each rising trigger edge, re-seed the random stream the synth
/// draws from with `seed` (truncated to an integer, as scsynth casts it), restarting it for every
/// synth drawing from it. A trigger already high on the first block seeds
/// immediately (scsynth's constructor behaviour). Outputs a constant `0.0`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RandSeed {
    prev_trig: f32,
    audio: u32,
}

impl Unit for RandSeed {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ProcessCtx {
            ins, outs, rgen, ..
        } = ctx;
        let trig = sig(ins, 0);
        // The calc length: a block at audio rate, one sample at control rate or in the constructor.
        let frames = if self.audio != 0 {
            outs.audio(0).len()
        } else {
            1
        };
        for i in 0..frames {
            let t = trig.at(i);
            if self.prev_trig <= 0.0 && t > 0.0 {
                // The seed input truncates to an `i32` and re-seeds the stream as scsynth's
                // `RGen::init` does, so a given seed restarts the sequence the server gives it.
                rgen.init(ins.control(1) as i32 as u32);
            }
            self.prev_trig = t;
        }
        hold(outs, self.audio != 0, 0.0);
        DoneAction::Nothing
    }
}

/// Constructor for [`RandSeed`].
pub struct RandSeedCtor;

impl UnitDef for RandSeedCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(RandSeed {
            prev_trig: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `RandID.ir/kr(id)`: whenever `id` changes, point the synth at the World's random stream numbered
/// `id`, so the units after it draw from that stream (scsynth's `RandID_next`). An `id` the World
/// does not have selects nothing, as in scsynth; a negative `id`, which scsynth converts to an
/// unsigned index without defining the result, selects nothing too. Outputs a constant `0.0`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RandID {
    /// The last `id` read (scsynth's `m_id`), `-1` until the constructor reads one.
    id: f32,
    audio: u32,
}

impl Unit for RandID {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let id = ctx.ins.control(0);
        if id != self.id {
            self.id = id;
            if id >= 0.0 {
                ctx.rgen_id.select(id as u32);
            }
        }
        hold(&mut ctx.outs, self.audio != 0, 0.0);
        DoneAction::Nothing
    }
}

/// Constructor for [`RandID`].
pub struct RandIDCtor;

impl UnitDef for RandIDCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(RandID {
            id: -1.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
