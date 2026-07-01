//! Utility units - plyphon's ports of scsynth's `MulAdd`, `Sum3`/`Sum4`, `Lag`, and `Amplitude`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::sample_channel;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// `ln(0.001)` - the decay target scsynth uses for its `-60 dB time` smoothing coefficients.
const LOG001: f32 = -6.907_755;

/// A first-order smoothing coefficient: the per-sample multiplier that decays to 0.001 over `time`
/// seconds (0 for an immediate response).
fn smoothing_coef(time: f32, sample_rate: f32) -> f32 {
    if time > 0.0 {
        math::exp(LOG001 / (time * sample_rate))
    } else {
        0.0
    }
}

/// One exponential-lag step: move `prev` a fraction `b1` of the way toward `x` (a one-pole smoother).
#[inline]
fn lag_step(prev: f32, x: f32, b1: f32) -> f32 {
    x + b1 * (prev - x)
}

/// One asymmetric-lag step: smooth with `b1u` while rising toward `x`, `b1d` while falling.
#[inline]
fn lag_ud_step(prev: f32, x: f32, b1u: f32, b1d: f32) -> f32 {
    let b1 = if x > prev { b1u } else { b1d };
    x + b1 * (prev - x)
}

/// One three-stage asymmetric-lag step for `Lag3UD`. Its third stage keys off `y1a > y1b` (not
/// `y1b > y1c`) - a quirk of scsynth's `Lag3UD_next` we preserve for bit-compatibility.
#[inline]
fn lag3ud_step(y1a: &mut f32, y1b: &mut f32, y1c: &mut f32, x: f32, b1u: f32, b1d: f32) -> f32 {
    *y1a = lag_ud_step(*y1a, x, b1u, b1d);
    *y1b = lag_ud_step(*y1b, *y1a, b1u, b1d);
    let b1 = if *y1a > *y1b { b1u } else { b1d };
    *y1c = *y1b + b1 * (*y1c - *y1b);
    *y1c
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

/// `Sum3.ar/kr(in0, in1, in2)` / `Sum4.ar/kr(in0, in1, in2, in3)`: the sum of its inputs. scsynth's
/// SynthDef optimiser rewrites chains of additions into these, and `.sum`/`Mix` (and the class-library
/// macros `DynKlank`/`DynKlang`, which expand to a summed `Ringz`/`SinOsc` bank) depend on them. Each
/// input is read at its own rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Sum {
    /// How many inputs to add (3 for `Sum3`, 4 for `Sum4`).
    count: u32,
    /// `0`/`1`: whether the unit runs at audio rate.
    audio: u32,
}

impl Unit for Sum {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let n = self.count as usize;
        let ins = ctx.ins; // `Copy`; its slices are `'a`, so it coexists with the `&mut` output.
        if self.audio != 0 {
            for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
                *o = (0..n).map(|k| sample_channel(&ins, k, i)).sum::<f32>();
            }
        } else {
            *ctx.outs.control(0) = (0..n).map(|k| ins.control(k)).sum::<f32>();
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Sum`] (`Sum3`/`Sum4`), parameterized by the input count.
pub struct SumCtor(pub u32);

impl UnitDef for SumCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < self.0 as usize {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Sum {
            count: self.0,
            audio: (ctx.rate == Rate::Audio) as u32,
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

/// `Lag2.ar/kr(in, lagTime)`: two [`Lag`]s in series - a smoother, more rounded response than a single
/// `Lag`. `in` is audio- or control-rate; `lagTime` is control-rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Lag2 {
    lag_time: f32,
    b1: f32,
    y1a: f32,
    y1b: f32,
    in_audio: u32,
}

impl Lag2 {
    const IN: usize = 0;
    const TIME: usize = 1;
}

impl Unit for Lag2 {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let x = ctx.ins.control(Self::IN);
        self.y1a = x;
        self.y1b = x;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let lag = ctx.ins.control(Self::TIME);
        if lag != self.lag_time {
            self.b1 = smoothing_coef(lag, ctx.audio.sample_rate as f32);
            self.lag_time = lag;
        }
        let b1 = self.b1;
        let (mut y1a, mut y1b) = (self.y1a, self.y1b);
        let out = ctx.outs.audio(0);
        if self.in_audio != 0 {
            for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
                y1a = lag_step(y1a, x, b1);
                y1b = lag_step(y1b, y1a, b1);
                *o = y1b;
            }
        } else {
            let x = ctx.ins.control(Self::IN);
            for o in out.iter_mut() {
                y1a = lag_step(y1a, x, b1);
                y1b = lag_step(y1b, y1a, b1);
                *o = y1b;
            }
        }
        self.y1a = y1a;
        self.y1b = y1b;
        DoneAction::Nothing
    }
}

/// Constructor for [`Lag2`].
pub struct Lag2Ctor;

impl UnitDef for Lag2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Lag2 {
            lag_time: -1.0, // force coefficient computation on the first block
            b1: 0.0,
            y1a: 0.0,
            y1b: 0.0,
            in_audio: (ctx.input_rates.first() == Some(&Rate::Audio)) as u32,
        }))
    }
}

/// `Lag3.ar/kr(in, lagTime)`: three [`Lag`]s in series - even smoother than [`Lag2`].
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Lag3 {
    lag_time: f32,
    b1: f32,
    y1a: f32,
    y1b: f32,
    y1c: f32,
    in_audio: u32,
}

impl Lag3 {
    const IN: usize = 0;
    const TIME: usize = 1;
}

impl Unit for Lag3 {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let x = ctx.ins.control(Self::IN);
        self.y1a = x;
        self.y1b = x;
        self.y1c = x;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let lag = ctx.ins.control(Self::TIME);
        if lag != self.lag_time {
            self.b1 = smoothing_coef(lag, ctx.audio.sample_rate as f32);
            self.lag_time = lag;
        }
        let b1 = self.b1;
        let (mut y1a, mut y1b, mut y1c) = (self.y1a, self.y1b, self.y1c);
        let out = ctx.outs.audio(0);
        if self.in_audio != 0 {
            for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
                y1a = lag_step(y1a, x, b1);
                y1b = lag_step(y1b, y1a, b1);
                y1c = lag_step(y1c, y1b, b1);
                *o = y1c;
            }
        } else {
            let x = ctx.ins.control(Self::IN);
            for o in out.iter_mut() {
                y1a = lag_step(y1a, x, b1);
                y1b = lag_step(y1b, y1a, b1);
                y1c = lag_step(y1c, y1b, b1);
                *o = y1c;
            }
        }
        self.y1a = y1a;
        self.y1b = y1b;
        self.y1c = y1c;
        DoneAction::Nothing
    }
}

/// Constructor for [`Lag3`].
pub struct Lag3Ctor;

impl UnitDef for Lag3Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Lag3 {
            lag_time: -1.0,
            b1: 0.0,
            y1a: 0.0,
            y1b: 0.0,
            y1c: 0.0,
            in_audio: (ctx.input_rates.first() == Some(&Rate::Audio)) as u32,
        }))
    }
}

/// `LagUD.ar/kr(in, lagTimeU, lagTimeD)`: an asymmetric [`Lag`] with separate smoothing times for a
/// rising input (`lagTimeU`) and a falling one (`lagTimeD`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LagUD {
    lag_u: f32,
    lag_d: f32,
    b1u: f32,
    b1d: f32,
    y1: f32,
    in_audio: u32,
}

impl LagUD {
    const IN: usize = 0;
    const UP: usize = 1;
    const DOWN: usize = 2;

    /// Recompute both coefficients when either lag time changes.
    fn update(&mut self, ctx: &ProcessCtx<'_>) {
        let (lu, ld) = (ctx.ins.control(Self::UP), ctx.ins.control(Self::DOWN));
        if lu != self.lag_u || ld != self.lag_d {
            let sr = ctx.audio.sample_rate as f32;
            self.b1u = smoothing_coef(lu, sr);
            self.b1d = smoothing_coef(ld, sr);
            self.lag_u = lu;
            self.lag_d = ld;
        }
    }
}

impl Unit for LagUD {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.y1 = ctx.ins.control(Self::IN);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.update(ctx);
        let (b1u, b1d) = (self.b1u, self.b1d);
        let mut y1 = self.y1;
        let out = ctx.outs.audio(0);
        if self.in_audio != 0 {
            for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
                y1 = lag_ud_step(y1, x, b1u, b1d);
                *o = y1;
            }
        } else {
            let x = ctx.ins.control(Self::IN);
            for o in out.iter_mut() {
                y1 = lag_ud_step(y1, x, b1u, b1d);
                *o = y1;
            }
        }
        self.y1 = y1;
        DoneAction::Nothing
    }
}

/// Constructor for [`LagUD`].
pub struct LagUDCtor;

impl UnitDef for LagUDCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(LagUD {
            lag_u: -1.0,
            lag_d: -1.0,
            b1u: 0.0,
            b1d: 0.0,
            y1: 0.0,
            in_audio: (ctx.input_rates.first() == Some(&Rate::Audio)) as u32,
        }))
    }
}

/// `Lag2UD.ar/kr(in, lagTimeU, lagTimeD)`: two [`LagUD`]s in series.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Lag2UD {
    lag_u: f32,
    lag_d: f32,
    b1u: f32,
    b1d: f32,
    y1a: f32,
    y1b: f32,
    in_audio: u32,
}

impl Lag2UD {
    const IN: usize = 0;
    const UP: usize = 1;
    const DOWN: usize = 2;

    fn update(&mut self, ctx: &ProcessCtx<'_>) {
        let (lu, ld) = (ctx.ins.control(Self::UP), ctx.ins.control(Self::DOWN));
        if lu != self.lag_u || ld != self.lag_d {
            let sr = ctx.audio.sample_rate as f32;
            self.b1u = smoothing_coef(lu, sr);
            self.b1d = smoothing_coef(ld, sr);
            self.lag_u = lu;
            self.lag_d = ld;
        }
    }
}

impl Unit for Lag2UD {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let x = ctx.ins.control(Self::IN);
        self.y1a = x;
        self.y1b = x;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.update(ctx);
        let (b1u, b1d) = (self.b1u, self.b1d);
        let (mut y1a, mut y1b) = (self.y1a, self.y1b);
        let out = ctx.outs.audio(0);
        if self.in_audio != 0 {
            for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
                y1a = lag_ud_step(y1a, x, b1u, b1d);
                y1b = lag_ud_step(y1b, y1a, b1u, b1d);
                *o = y1b;
            }
        } else {
            let x = ctx.ins.control(Self::IN);
            for o in out.iter_mut() {
                y1a = lag_ud_step(y1a, x, b1u, b1d);
                y1b = lag_ud_step(y1b, y1a, b1u, b1d);
                *o = y1b;
            }
        }
        self.y1a = y1a;
        self.y1b = y1b;
        DoneAction::Nothing
    }
}

/// Constructor for [`Lag2UD`].
pub struct Lag2UDCtor;

impl UnitDef for Lag2UDCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Lag2UD {
            lag_u: -1.0,
            lag_d: -1.0,
            b1u: 0.0,
            b1d: 0.0,
            y1a: 0.0,
            y1b: 0.0,
            in_audio: (ctx.input_rates.first() == Some(&Rate::Audio)) as u32,
        }))
    }
}

/// `Lag3UD.ar/kr(in, lagTimeU, lagTimeD)`: three [`LagUD`]s in series. Note: scsynth's third stage
/// compares `y1a > y1b` (rather than `y1b > y1c`) when choosing its up/down coefficient - a quirk
/// preserved here for bit-compatibility.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Lag3UD {
    lag_u: f32,
    lag_d: f32,
    b1u: f32,
    b1d: f32,
    y1a: f32,
    y1b: f32,
    y1c: f32,
    in_audio: u32,
}

impl Lag3UD {
    const IN: usize = 0;
    const UP: usize = 1;
    const DOWN: usize = 2;

    fn update(&mut self, ctx: &ProcessCtx<'_>) {
        let (lu, ld) = (ctx.ins.control(Self::UP), ctx.ins.control(Self::DOWN));
        if lu != self.lag_u || ld != self.lag_d {
            let sr = ctx.audio.sample_rate as f32;
            self.b1u = smoothing_coef(lu, sr);
            self.b1d = smoothing_coef(ld, sr);
            self.lag_u = lu;
            self.lag_d = ld;
        }
    }
}

impl Unit for Lag3UD {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let x = ctx.ins.control(Self::IN);
        self.y1a = x;
        self.y1b = x;
        self.y1c = x;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.update(ctx);
        let (b1u, b1d) = (self.b1u, self.b1d);
        let (mut y1a, mut y1b, mut y1c) = (self.y1a, self.y1b, self.y1c);
        let out = ctx.outs.audio(0);
        if self.in_audio != 0 {
            for (o, &x) in out.iter_mut().zip(ctx.ins.audio(Self::IN)) {
                *o = lag3ud_step(&mut y1a, &mut y1b, &mut y1c, x, b1u, b1d);
            }
        } else {
            let x = ctx.ins.control(Self::IN);
            for o in out.iter_mut() {
                *o = lag3ud_step(&mut y1a, &mut y1b, &mut y1c, x, b1u, b1d);
            }
        }
        self.y1a = y1a;
        self.y1b = y1b;
        self.y1c = y1c;
        DoneAction::Nothing
    }
}

/// Constructor for [`Lag3UD`].
pub struct Lag3UDCtor;

impl UnitDef for Lag3UDCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Lag3UD {
            lag_u: -1.0,
            lag_d: -1.0,
            b1u: 0.0,
            b1d: 0.0,
            y1a: 0.0,
            y1b: 0.0,
            y1c: 0.0,
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
