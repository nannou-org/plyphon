//! Panning and spatialisation - plyphon's ports of scsynth's `Pan2`, `LinPan2`, `Balance2`,
//! `XFade2`, `LinXFade2` and `Rotate2` (`PanUGens.cpp`).
//!
//! The equal-power units share the same cos/sin law as `Pan2`. Following the `Pan2` convention, the
//! `pos`/`level` controls are read once per block (constant over the block) rather than
//! slope-interpolated per sample.

use core::f32::consts::{FRAC_PI_4, PI};

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// Equal-power `(left, right)` gains for `pos` in `[-1, 1]`, scaled by `level`: `pos = -1` is hard
/// left, `pos = +1` hard right (scsynth's sine-table pan law).
fn equal_power(pos: f32, level: f32) -> (f32, f32) {
    let angle = (pos.clamp(-1.0, 1.0) + 1.0) * FRAC_PI_4;
    (math::cos(angle) * level, math::sin(angle) * level)
}

/// The `level` input at `idx`, defaulting to `1.0` when the SynthDef omits it.
fn level_or_default(ctx: &ProcessCtx<'_>, idx: usize) -> f32 {
    if ctx.ins.len() > idx {
        ctx.ins.control(idx)
    } else {
        1.0
    }
}

/// `Pan2.ar(in, pos, level)`: pan a mono signal across two channels with an equal-power law - `pos`
/// runs -1 (hard left) to +1 (hard right), `level` (default 1) scales. Has two outputs (left, right);
/// `pos`/`level` are taken at control rate (constant over the block).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Pan2 {
    /// The panner is stateless; this pad just keeps its `Pod` state non-zero-sized in the pool.
    _pad: u32,
}

impl Pan2 {
    const IN: usize = 0;
    const POS: usize = 1;
    const LEVEL: usize = 2;
}

impl Unit for Pan2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (left_gain, right_gain) = equal_power(
            ctx.ins.control(Self::POS),
            level_or_default(ctx, Self::LEVEL),
        );
        let input = ctx.ins.audio(Self::IN);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(input) {
            *o = x * left_gain;
        }
        for (o, &x) in ctx.outs.audio(1).iter_mut().zip(input) {
            *o = x * right_gain;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Pan2`].
pub struct Pan2Ctor;

impl UnitDef for Pan2Ctor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Pan2 { _pad: 0 }))
    }
}

/// `LinPan2.ar(in, pos, level)`: a *linear* (not equal-power) stereo panner. Two outputs.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LinPan2 {
    _pad: u32,
}

impl Unit for LinPan2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let level = level_or_default(ctx, 2);
        let pan = ctx.ins.control(1) * 0.5 + 0.5;
        let right_gain = level * pan;
        let left_gain = level - right_gain;
        let input = ctx.ins.audio(0);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(input) {
            *o = x * left_gain;
        }
        for (o, &x) in ctx.outs.audio(1).iter_mut().zip(input) {
            *o = x * right_gain;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`LinPan2`].
pub struct LinPan2Ctor;

impl UnitDef for LinPan2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(LinPan2 { _pad: 0 }))
    }
}

/// `Balance2.ar(left, right, pos, level)`: equal-power balance of a stereo pair. Two outputs.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Balance2 {
    _pad: u32,
}

impl Unit for Balance2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (left_gain, right_gain) = equal_power(ctx.ins.control(2), level_or_default(ctx, 3));
        let left_in = ctx.ins.audio(0);
        let right_in = ctx.ins.audio(1);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(left_in) {
            *o = x * left_gain;
        }
        for (o, &x) in ctx.outs.audio(1).iter_mut().zip(right_in) {
            *o = x * right_gain;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Balance2`].
pub struct Balance2Ctor;

impl UnitDef for Balance2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Balance2 { _pad: 0 }))
    }
}

/// `XFade2.ar(inA, inB, pan, level)`: equal-power crossfade between two signals. One output.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct XFade2 {
    _pad: u32,
}

impl Unit for XFade2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (amp_a, amp_b) = equal_power(ctx.ins.control(2), level_or_default(ctx, 3));
        let a = ctx.ins.audio(0);
        let b = ctx.ins.audio(1);
        for ((o, &x), &y) in ctx.outs.audio(0).iter_mut().zip(a).zip(b) {
            *o = x * amp_a + y * amp_b;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`XFade2`].
pub struct XFade2Ctor;

impl UnitDef for XFade2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(XFade2 { _pad: 0 }))
    }
}

/// `LinXFade2.ar(inA, inB, pan)`: a *linear* crossfade between two signals. One output.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LinXFade2 {
    _pad: u32,
}

impl Unit for LinXFade2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let amp = ctx.ins.control(2).clamp(-1.0, 1.0) * 0.5 + 0.5;
        let a = ctx.ins.audio(0);
        let b = ctx.ins.audio(1);
        for ((o, &x), &y) in ctx.outs.audio(0).iter_mut().zip(a).zip(b) {
            *o = x + amp * (y - x);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`LinXFade2`].
pub struct LinXFade2Ctor;

impl UnitDef for LinXFade2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(LinXFade2 { _pad: 0 }))
    }
}

/// `Rotate2.ar(x, y, pos)`: rotates a two-channel sound field by `pos * pi` radians. Two outputs.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Rotate2 {
    _pad: u32,
}

impl Unit for Rotate2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let angle = ctx.ins.control(2) * PI;
        let (sint, cost) = (math::sin(angle), math::cos(angle));
        let x_in = ctx.ins.audio(0);
        let y_in = ctx.ins.audio(1);
        // Compute both outputs from the two inputs per sample, then write each channel.
        for ((o, &x), &y) in ctx.outs.audio(0).iter_mut().zip(x_in).zip(y_in) {
            *o = cost * x + sint * y;
        }
        for ((o, &x), &y) in ctx.outs.audio(1).iter_mut().zip(x_in).zip(y_in) {
            *o = cost * y - sint * x;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Rotate2`].
pub struct Rotate2Ctor;

impl UnitDef for Rotate2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Rotate2 { _pad: 0 }))
    }
}
