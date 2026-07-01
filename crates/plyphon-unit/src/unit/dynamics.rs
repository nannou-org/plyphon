//! Dynamics processors - plyphon's ports of scsynth's `Compander` and `DetectSilence`
//! (`FilterUGens.cpp`).
//!
//! `Compander` is a general compressor/expander/gate driven by a side-chain control signal;
//! `DetectSilence` fires a done action (and outputs `1`) once its input has stayed below a threshold
//! for a given time. `Limiter`/`Normalizer` are deferred - they need a look-ahead delay in aux
//! memory.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// `ln(0.1) == -ln(10)` - the `-20 dB`-over-`time` target scsynth uses for `Compander`'s
/// attack/release coefficients.
const LOG1: f32 = -core::f32::consts::LN_10;

/// The attack/release coefficient decaying to `-20 dB` over `time` seconds (0 for an immediate
/// response), matching scsynth's `time == 0 ? 0 : exp(log1 / (time * SR))`.
fn comp_coef(time: f32, sample_rate: f32) -> f32 {
    if time == 0.0 {
        0.0
    } else {
        math::exp(LOG1 / (time * sample_rate))
    }
}

/// `Compander.ar(in, control, thresh, slopeBelow, slopeAbove, clampTime, relaxTime)`: applies a gain
/// to `in` derived from the amplitude of `control`, giving compression/expansion above/below
/// `thresh`. `clampTime`/`relaxTime` set the attack/release of the amplitude follower.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Compander {
    clamp: f32,
    clampcoef: f32,
    relax: f32,
    relaxcoef: f32,
    gain: f32,
    prevmaxval: f32,
}

impl Unit for Compander {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let thresh = ctx.ins.control(2);
        let slope_below = ctx.ins.control(3);
        let slope_above = ctx.ins.control(4);
        let clamp = ctx.ins.control(5);
        let relax = ctx.ins.control(6);
        if clamp != self.clamp {
            self.clampcoef = comp_coef(clamp, sr);
            self.clamp = clamp;
        }
        if relax != self.relax {
            self.relaxcoef = comp_coef(relax, sr);
            self.relax = relax;
        }
        let (clampcoef, relaxcoef) = (self.clampcoef, self.relaxcoef);

        // Follow the amplitude of the side-chain control signal (fast attack, slow release).
        let mut prevmaxval = self.prevmaxval;
        for &c in ctx.ins.audio(1) {
            let val = c.abs();
            let coef = if val < prevmaxval {
                relaxcoef
            } else {
                clampcoef
            };
            prevmaxval = val + (prevmaxval - val) * coef;
        }
        self.prevmaxval = prevmaxval;

        // The target gain: `(maxval/thresh)^(slope - 1)`, with the below-thresh path zapped so tiny
        // values do not blow the gain up.
        let next_gain = if prevmaxval < thresh {
            if slope_below == 1.0 {
                1.0
            } else {
                let g = math::powf(prevmaxval / thresh, slope_below - 1.0);
                let a = g.abs();
                if a < 1e-15 {
                    0.0
                } else if a > 1e15 {
                    1.0
                } else {
                    g
                }
            }
        } else if slope_above == 1.0 {
            1.0
        } else {
            math::powf(prevmaxval / thresh, slope_above - 1.0)
        };

        // Apply the gain, ramping from the previous block's gain to `next_gain` to avoid zipper noise.
        let mut gain = self.gain;
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        let gain_slope = (next_gain - gain) / out.len().max(1) as f32;
        for (o, &x) in out.iter_mut().zip(input) {
            *o = x * gain;
            gain += gain_slope;
        }
        self.gain = gain;
        DoneAction::Nothing
    }
}

/// Constructor for [`Compander`].
pub struct CompanderCtor;

impl UnitDef for CompanderCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 7 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Compander {
            clamp: f32::NAN, // force coefficient computation on the first block
            clampcoef: 0.0,
            relax: f32::NAN,
            relaxcoef: 0.0,
            gain: 1.0,
            prevmaxval: 0.0,
        }))
    }
}

/// `DetectSilence.ar(in, amp, time, doneAction)`: outputs `0` while `in` is active; once `|in|` has
/// stayed at or below `amp` for `time` seconds (after first exceeding it), outputs `1` and fires the
/// `doneAction` once.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DetectSilence {
    counter: i32,
    fired: u32,
    audio: u32,
}

impl DetectSilence {
    const IN: usize = 0;
    const THRESH: usize = 1;
    const TIME: usize = 2;
    const DONE: usize = 3;
}

impl Unit for DetectSilence {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let thresh = ctx.ins.control(Self::THRESH);
        let end_counter = (ctx.audio.sample_rate as f32 * ctx.ins.control(Self::TIME)) as i32;
        let input = sig(&ctx.ins, Self::IN);
        let mut counter = self.counter;
        let mut completed = false;
        drive(ctx, audio_out, |i| {
            let val = input.at(i).abs();
            if val > thresh {
                counter = 0;
                0.0
            } else if counter >= 0 {
                counter += 1;
                if counter >= end_counter {
                    completed = true;
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        });
        self.counter = counter;
        if completed {
            ctx.done.mark_done();
            if self.fired == 0 {
                self.fired = 1;
                let action = if ctx.ins.len() > Self::DONE {
                    DoneAction::from_code(ctx.ins.control(Self::DONE))
                } else {
                    DoneAction::Nothing
                };
                return action;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DetectSilence`].
pub struct DetectSilenceCtor;

impl UnitDef for DetectSilenceCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DetectSilence {
            counter: -1, // wait for the first above-threshold sample before counting silence
            fired: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
