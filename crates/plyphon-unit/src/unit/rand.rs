//! The init- and trigger-time random units - plyphon's ports of scsynth's `Rand`, `ExpRand`,
//! `TRand`, `TExpRand`, `TIRand`, `RandSeed` and `RandID` (`NoiseUGens.cpp`).
//!
//! Unlike the noise generators (each with a private [`Rng`] embedded in its own state), this
//! family draws from the synth's shared random stream ([`ProcessCtx::rgen`]) - the analogue of
//! scsynth's per-graph `RGen` - so draws interleave deterministically across the units of one
//! synth and a `RandSeed` re-seed restarts every unit's sequence together.
//!
//! Scope divergence from scsynth: there, the `RGen`s live in a World-level array and `RandID`
//! repoints a synth at a numbered stream shared with other synths; here each graph instance owns
//! exactly one stream, so `RandID` keeps its shape (inputs consumed, `0.0` output) but selects
//! nothing. Cross-synth correlated randomness via a shared `RandID` stream is not expressible.
//!
//! The one-time draws happen on the first `process` call, which runs at the same topological
//! position as scsynth's constructor calc, so draw interleaving within a synth matches unit order
//! exactly as it does there.

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

/// An exponential-distribution draw in `[lo, hi)` (scsynth's `pow(hi / lo, frand()) * lo`): equal
/// probability per octave, so `lo` must be non-zero and share `hi`'s sign for a sensible result.
fn exponential(rgen: &mut Rng, lo: f32, hi: f32) -> f32 {
    math::exp(math::ln(hi / lo) * rgen.next_unipolar()) * lo
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
    /// `0` until the one-time draw has happened on the first `process`.
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
    fn run(&mut self, ctx: &mut ProcessCtx<'_>, draw: impl Fn(&mut Rng, f32, f32) -> f32) {
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
            for (i, o) in outs.audio(0).iter_mut().enumerate() {
                let t = trig.at(i);
                if self.prev_trig <= 0.0 && t > 0.0 {
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
        self.0.run(ctx, uniform);
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
/// trigger, held between.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TExpRand(TrigRand);

impl Unit for TExpRand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, exponential);
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
/// trigger, held between.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TIRand(TrigRand);

impl Unit for TIRand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, integer);
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

/// `RandSeed.kr(trig, seed)`: on each rising trigger edge, re-seed the synth's shared random
/// stream from `seed` (truncated to an integer, as scsynth casts it), restarting every
/// `Rand`-family sequence in the synth. A trigger already high on the first block seeds
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
            ins,
            outs,
            rgen,
            own,
            ..
        } = ctx;
        let trig = sig(ins, 0);
        let frames = if self.audio != 0 { own.block_size } else { 1 };
        for i in 0..frames {
            let t = trig.at(i);
            if self.prev_trig <= 0.0 && t > 0.0 {
                // The seed input truncates to an `i32` and re-keys the stream by its 32-bit
                // pattern, so equal seed values always produce equal sequences.
                **rgen = Rng::new(ins.control(1) as i32 as u32 as u64);
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

/// `RandID.ir/kr(id)`: in scsynth this repoints the synth at the World random stream numbered
/// `id`; each plyphon graph owns exactly one stream, so the unit consumes its input and outputs
/// the constant `0.0` scsynth outputs, selecting nothing.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RandID {
    audio: u32,
}

impl Unit for RandID {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        hold(&mut ctx.outs, self.audio != 0, 0.0);
        DoneAction::Nothing
    }
}

/// Constructor for [`RandID`].
pub struct RandIDCtor;

impl UnitDef for RandIDCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(RandID {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
