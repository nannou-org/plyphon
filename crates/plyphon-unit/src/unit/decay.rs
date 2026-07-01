//! Exponential decay followers - plyphon's ports of scsynth's `Decay` and `Decay2`.
//!
//! `Decay` turns an impulse into an exponential decay (a leaky integrator whose pole is set from a
//! `-60 dB` decay time). `Decay2` is the difference of a slow (decay) and a fast (attack) `Decay`, so
//! an impulse becomes an attack-then-decay envelope. Coefficients are recomputed when the times
//! change and held across the block. State is `f64`, flushed with `zap` (scsynth's `zapgremlins`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `ln(0.001)` - the `-60 dB` decay target scsynth uses for its smoothing coefficients.
const LOG001: f64 = -6.907_755_278_982_137;

/// The per-sample feedback coefficient that decays to `-60 dB` over `time` seconds (0 for an
/// immediate response), matching scsynth's `decayTime == 0 ? 0 : exp(log001 / (decayTime * SR))`.
/// Shared with `Ringz`, whose pole radius is the same quantity.
pub(crate) fn decay_coef(time: f32, sample_rate: f64) -> f64 {
    if time == 0.0 {
        0.0
    } else {
        math::exp(LOG001 / (time as f64 * sample_rate))
    }
}

/// `Decay.ar(in, decayTime)`: `out(i) = in(i) + b1 * out(i-1)`, with `b1` a `-60 dB`-over-`decayTime`
/// coefficient. The coefficient is a pure function of the (control-rate) `decayTime`, so it is
/// derived once per block rather than cached.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Decay {
    y1: f64,
}

impl Unit for Decay {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let b1 = decay_coef(ctx.ins.control(1), ctx.audio.sample_rate);
        let mut y1 = self.y1;
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            y1 = x as f64 + b1 * y1;
            *o = y1 as f32;
        }
        self.y1 = zap(y1);
        DoneAction::Nothing
    }
}

/// Constructor for [`Decay`].
pub struct DecayCtor;

impl UnitDef for DecayCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Decay { y1: 0.0 }))
    }
}

/// `Decay2.ar(in, attackTime, decayTime)`: `Decay(in, decayTime) - Decay(in, attackTime)`, so an
/// impulse becomes a smooth attack-then-decay envelope. Both coefficients are derived once per block
/// from the control-rate times.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Decay2 {
    y1a: f64,
    y1b: f64,
}

impl Unit for Decay2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate;
        let b1a = decay_coef(ctx.ins.control(2), sr); // decay
        let b1b = decay_coef(ctx.ins.control(1), sr); // attack
        let (mut y1a, mut y1b) = (self.y1a, self.y1b);
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            let x0 = x as f64;
            y1a = x0 + b1a * y1a;
            y1b = x0 + b1b * y1b;
            *o = (y1a - y1b) as f32;
        }
        self.y1a = zap(y1a);
        self.y1b = zap(y1b);
        DoneAction::Nothing
    }
}

/// Constructor for [`Decay2`].
pub struct Decay2Ctor;

impl UnitDef for Decay2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Decay2 { y1a: 0.0, y1b: 0.0 }))
    }
}
