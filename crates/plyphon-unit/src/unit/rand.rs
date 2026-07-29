//! The init- and trigger-time random units - plyphon's ports of scsynth's `Rand`, `ExpRand`,
//! `TRand`, `TExpRand`, `TIRand`, `RandSeed` and `RandID` (`NoiseUGens.cpp`).
//!
//! Unlike the noise generators (each with a private [`Rng`] embedded in its own state), this
//! family draws from the synth's shared random stream ([`ProcessCtx::rgen`]) - the analogue of
//! scsynth's per-graph `RGen` - so draws interleave deterministically across the units of one
//! synth and a `RandSeed` re-seed restarts every *Rand-family* sequence together. That scope is
//! narrower than scsynth's: there the noise generators draw from the same graph `RGen`, so a
//! `RandSeed` restarts `WhiteNoise` and friends too, an idiom plyphon's private per-unit streams
//! do not support.
//!
//! Scope divergence from scsynth: there, the `RGen`s live in a World-level array and `RandID`
//! repoints a synth at a numbered stream shared with other synths; here each graph instance owns
//! exactly one stream, so `RandID` keeps its shape (inputs consumed, `0.0` output) but selects
//! nothing. Cross-synth correlated randomness via a shared `RandID` stream is not expressible.
//!
//! The one-time draws happen during the graph's ordered constructor pass, so draw interleaving
//! within a synth matches unit order exactly as it does in scsynth.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{DemandWorld, demand_next};
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
    math::powf(hi / lo, rgen.next_unipolar()) * lo
}

/// A uniform integer draw in `[lo, hi]` as a float (scsynth's `rgen.irand(hi - lo + 1) + lo`).
fn integer(rgen: &mut Rng, lo: f32, hi: f32) -> f32 {
    let lo = lo as i32;
    let hi = hi as i32;
    rgen.next_irand(hi.wrapping_sub(lo).wrapping_add(1))
        .wrapping_add(lo) as f32
}

/// Write `value` across the output at the unit's rate (a full block, or one control value).
fn hold(outs: &mut Outputs<'_>, audio: bool, value: f32) {
    if audio {
        outs.audio(0).fill(value);
    } else {
        *outs.control(0) = value;
    }
}

/// `Rand.new(lo, hi)`: one constructor-time uniform draw in `[lo, hi)`, held forever.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Rand {
    value: f32,
    /// `0`/`1`: audio-rate (a full block) vs control-rate (one value).
    audio: u32,
}

impl Unit for Rand {
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.value = uniform(ctx.rgen, ctx.ins.control(0), ctx.ins.control(1));
        *ctx.outs.control(0) = self.value;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        hold(&mut ctx.outs, self.audio != 0, self.value);
        DoneAction::Nothing
    }
}

/// Constructor for [`Rand`].
pub struct RandCtor;

impl UnitDef for RandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Rand {
            value: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `ExpRand.new(lo, hi)`: one constructor-time exponential-distribution draw in `[lo, hi)`, held
/// forever.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ExpRand {
    value: f32,
    audio: u32,
}

impl Unit for ExpRand {
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.value = exponential(ctx.rgen, ctx.ins.control(0), ctx.ins.control(1));
        *ctx.outs.control(0) = self.value;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        hold(&mut ctx.outs, self.audio != 0, self.value);
        DoneAction::Nothing
    }
}

/// Constructor for [`ExpRand`].
pub struct ExpRandCtor;

impl UnitDef for ExpRandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(ExpRand {
            value: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// The shared body of the `TRand` family: `(lo, hi, trig)` inputs, a constructor-time initial draw,
/// and a fresh draw on every rising trigger edge, holding the value between.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TrigRand {
    value: f32,
    prev_trig: f32,
    audio: u32,
}

impl TrigRand {
    /// A fresh trigger-random body; `audio` selects a full-block output vs one control value.
    fn new(audio: bool) -> Self {
        TrigRand {
            value: 0.0,
            prev_trig: 0.0,
            audio: audio as u32,
        }
    }

    /// Draw the initial value and latch the constructor-time trigger sample.
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>, draw: impl Fn(&mut Rng, f32, f32) -> f32) {
        self.value = draw(ctx.rgen, ctx.ins.control(0), ctx.ins.control(1));
        self.prev_trig = ctx.ins.control(2);
        *ctx.outs.control(0) = self.value;
    }

    /// Run one block, redrawing on the trigger edges selected by the source calc variant.
    ///
    /// `advance_prev` preserves the source plugins' differing audio-loop behavior: `TRand`
    /// advances the comparison sample inside the loop, while `TExpRand` and `TIRand` compare every
    /// sample against the block-entry trigger and only retain the final trigger after the loop.
    fn run(
        &mut self,
        ctx: &mut ProcessCtx<'_>,
        draw: impl Fn(&mut Rng, f32, f32) -> f32,
        advance_prev: bool,
    ) {
        let ProcessCtx {
            ins, outs, rgen, ..
        } = ctx;
        let trig = sig(ins, 2);
        if self.audio != 0 {
            let varying_bounds = ins.rate(0) == Rate::Audio;
            let lo = sig(ins, 0);
            let hi = sig(ins, 1);
            let mut last_trig = self.prev_trig;
            for (i, o) in outs.audio(0).iter_mut().enumerate() {
                let t = trig.at(i);
                if self.prev_trig <= 0.0 && t > 0.0 {
                    let frame = if varying_bounds { i } else { 0 };
                    self.value = draw(rgen, lo.at(frame), hi.at(frame));
                }
                if advance_prev {
                    self.prev_trig = t;
                }
                last_trig = t;
                *o = self.value;
            }
            self.prev_trig = last_trig;
        } else {
            let t = trig.at(0);
            if self.prev_trig <= 0.0 && t > 0.0 {
                self.value = draw(rgen, ins.control(0), ins.control(1));
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
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.0.construct(ctx, uniform);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, uniform, true);
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
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.0.construct(ctx, exponential);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, exponential, false);
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
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.0.construct(ctx, integer);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.0.run(ctx, integer, false);
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
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            rgen: &mut *ctx.rgen,
        };
        let trig = ctx.ins.control(0);
        if trig > 0.0 {
            let seed = demand_next(&ctx.ins, &mut ctx.demand, &mut world, 1, 1);
            world.rgen.reseed(seed as i32 as u32);
        }
        self.prev_trig = trig;
        *ctx.outs.control(0) = 0.0;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ProcessCtx {
            ins,
            outs,
            rgen,
            buffers,
            local_bufs,
            demand,
            node_id,
            node_msgs,
            ..
        } = ctx;
        let trig = sig(ins, 0);
        let frames = if self.audio != 0 {
            outs.audio(0).len()
        } else {
            1
        };
        let mut world = DemandWorld {
            buffers: &mut **buffers,
            local_bufs,
            node_id: *node_id,
            node_msgs,
            rgen: &mut **rgen,
        };
        for i in 0..frames {
            let t = trig.at(i);
            if self.prev_trig <= 0.0 && t > 0.0 {
                let seed = demand_next(ins, demand, &mut world, 1, frames);
                world.rgen.reseed(seed as i32 as u32);
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
