//! Utility units - plyphon's ports of scsynth's `MulAdd`, `Lag`, and `Amplitude`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::rate::Rate;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};

/// `ln(0.001)` - the decay target scsynth uses for its `-60 dB time` smoothing coefficients.
const LOG001: f32 = -6.907_755;

/// A first-order smoothing coefficient: the per-sample multiplier that decays to 0.001 over `time`
/// seconds (0 for an immediate response).
fn smoothing_coef(time: f32, sample_rate: f32) -> f32 {
    if time > 0.0 {
        (LOG001 / (time * sample_rate)).exp()
    } else {
        0.0
    }
}

/// `MulAdd.ar/kr(in, mul, add)`: `in * mul + add`, a fused scale-and-offset. `in` may be audio- or
/// control-rate; `mul`/`add` are taken at control rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MulAdd {
    /// `0`/`1`: whether `in` is audio-rate.
    in_audio: u32,
}

impl MulAdd {
    const IN: usize = 0;
    const MUL: usize = 1;
    const ADD: usize = 2;
}

impl Unit for MulAdd {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mul = ctx.ins.control(Self::MUL);
        let add = ctx.ins.control(Self::ADD);
        let out = ctx.outs.audio(0);
        if self.in_audio != 0 {
            for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
                *o = x * mul + add;
            }
        } else {
            out.fill(ctx.ins.control(Self::IN) * mul + add);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`MulAdd`].
pub struct MulAddCtor;

impl UnitDef for MulAddCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(MulAdd {
            in_audio: (ctx.input_rates.first() == Some(&Rate::Audio)) as u32,
        }))
    }
}

/// `Lag.ar/kr(in, lagTime)`: a one-pole smoother that takes `lagTime` seconds to (mostly) reach a
/// new value - the standard way to de-zipper control changes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Lag {
    lag_time: f32,
    b1: f32,
    y: f32,
    /// `0`/`1`: whether `in` is audio-rate.
    in_audio: u32,
}

impl Lag {
    const IN: usize = 0;
    const TIME: usize = 1;
}

impl Unit for Lag {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Start at the input value (scsynth's `m_y1 = ZIN0(0)`) so the first block holds steady
        // instead of ramping up from zero - the coefficient is still computed lazily in `process`,
        // whose sentinel also catches later `lagTime` changes.
        self.y = ctx.ins.control(Self::IN);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let lag_time = ctx.ins.control(Self::TIME);
        if lag_time != self.lag_time {
            self.b1 = smoothing_coef(lag_time, ctx.audio.sample_rate as f32);
            self.lag_time = lag_time;
        }
        let b1 = self.b1;
        let mut y = self.y;
        let out = ctx.outs.audio(0);
        if self.in_audio != 0 {
            for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
                y = x + b1 * (y - x);
                *o = y;
            }
        } else {
            let x = ctx.ins.control(Self::IN);
            for o in out.iter_mut() {
                y = x + b1 * (y - x);
                *o = y;
            }
        }
        self.y = y;
        DoneAction::Nothing
    }
}

/// Constructor for [`Lag`].
pub struct LagCtor;

impl UnitDef for LagCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Lag {
            lag_time: -1.0, // force coefficient computation on the first block
            b1: 0.0,
            y: 0.0,
            in_audio: (ctx.input_rates.first() == Some(&Rate::Audio)) as u32,
        }))
    }
}

/// `Amplitude.ar/kr(in, attackTime, releaseTime)`: an amplitude follower tracking the peak magnitude
/// of `in`, rising with `attackTime` and falling with `releaseTime` (both default 0.01 s).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Amplitude {
    attack_time: f32,
    release_time: f32,
    attack_coef: f32,
    release_coef: f32,
    prev: f32,
}

impl Amplitude {
    const IN: usize = 0;
    const ATTACK: usize = 1;
    const RELEASE: usize = 2;
}

impl Unit for Amplitude {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sample_rate = ctx.audio.sample_rate as f32;
        let attack = if ctx.ins.len() > Self::ATTACK {
            ctx.ins.control(Self::ATTACK)
        } else {
            0.01
        };
        let release = if ctx.ins.len() > Self::RELEASE {
            ctx.ins.control(Self::RELEASE)
        } else {
            0.01
        };
        if attack != self.attack_time {
            self.attack_coef = smoothing_coef(attack, sample_rate);
            self.attack_time = attack;
        }
        if release != self.release_time {
            self.release_coef = smoothing_coef(release, sample_rate);
            self.release_time = release;
        }
        let (attack_coef, release_coef) = (self.attack_coef, self.release_coef);
        let mut prev = self.prev;
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let val = x.abs();
            // Rise quickly (attack) when the level grows, fall slowly (release) when it shrinks.
            let coef = if val < prev {
                release_coef
            } else {
                attack_coef
            };
            prev = coef * (prev - val) + val;
            *o = prev;
        }
        self.prev = prev;
        DoneAction::Nothing
    }
}

/// Constructor for [`Amplitude`].
pub struct AmplitudeCtor;

impl UnitDef for AmplitudeCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Amplitude {
            attack_time: -1.0,
            release_time: -1.0,
            attack_coef: 0.0,
            release_coef: 0.0,
            prev: 0.0,
        }))
    }
}
