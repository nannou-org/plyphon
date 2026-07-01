//! Panning and spatialisation - plyphon's ports of scsynth's `Pan2`, `LinPan2`, `Balance2`,
//! `XFade2`, `LinXFade2`, `Rotate2`, the quad `Pan4`, the `numChans`-around-a-ring `PanAz`, and the
//! first-order-ambisonic `PanB`/`PanB2`/`BiPanB2` encoders and `DecodeB2` decoder (`PanUGens.cpp`).
//!
//! The equal-power units share the same cos/sin law as `Pan2` (scsynth's sine-table lookup, computed
//! here directly). Following the `Pan2` convention, the position/level controls are read once per block
//! (constant over the block) rather than slope-interpolated per sample. The ambisonic units encode to /
//! decode from B-format (`W`, `X`, `Y`, and `Z`) via the `sin`/`cos` of the azimuth (and elevation).

use core::f32::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, PI};

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, Outputs, ProcessCtx, Unit, unit_spec};
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

/// Multiply `input` by `amp` into output channel `ch`.
fn write_scaled(outs: &mut Outputs<'_>, ch: usize, input: &[f32], amp: f32) {
    for (o, &x) in outs.audio(ch).iter_mut().zip(input) {
        *o = x * amp;
    }
}

/// Project an out-of-square `(x, y)` onto the unit square's edge (scsynth's `Pan4` clamp).
fn project_square(mut x: f32, mut y: f32) -> (f32, f32) {
    if !(-1.0..=1.0).contains(&x) || !(-1.0..=1.0).contains(&y) {
        let xabs = x.abs();
        if y > xabs {
            x = (x + y) / y - 1.0;
            y = 1.0;
        } else if y < -xabs {
            x = (x - y) / -y - 1.0;
            y = -1.0;
        } else if y.abs() < x {
            y = (y + x) / x - 1.0;
            x = 1.0;
        } else {
            y = (y - x) / -x - 1.0;
            x = -1.0;
        }
    }
    (x, y)
}

/// `Pan4.ar(in, xpos, ypos, level)`: pan a mono signal across four channels (front-left, front-right,
/// back-left, back-right) by an equal-power law on each axis. `xpos`/`ypos` run -1 to +1.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Pan4 {
    _pad: u32,
}

impl Unit for Pan4 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let (xpos, ypos) = project_square(ins.control(1), ins.control(2));
        let level = ins.control(3);
        let (leftamp, rightamp) = equal_power(xpos, 1.0);
        let (backamp, frontamp) = equal_power(ypos, 1.0);
        let (frontamp, backamp) = (frontamp * level, backamp * level);
        let input = ins.audio(0);
        write_scaled(&mut ctx.outs, 0, input, leftamp * frontamp);
        write_scaled(&mut ctx.outs, 1, input, rightamp * frontamp);
        write_scaled(&mut ctx.outs, 2, input, leftamp * backamp);
        write_scaled(&mut ctx.outs, 3, input, rightamp * backamp);
        DoneAction::Nothing
    }
}

/// Constructor for [`Pan4`].
pub struct Pan4Ctor;

impl UnitDef for Pan4Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Pan4 { _pad: 0 }))
    }
}

/// `PanB.ar(in, azimuth, elevation, gain)`: encode a mono signal to first-order 3D ambisonic B-format
/// (`W`, `X`, `Y`, `Z`). `azimuth` and `elevation` are in units where `1` is half a turn.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PanB {
    _pad: u32,
}

impl Unit for PanB {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let a = PI * ins.control(1);
        let e = FRAC_PI_2 * ins.control(2);
        let level = ins.control(3);
        let (sina, cosa) = (-math::sin(a), math::cos(a));
        let (sinb, cosb) = (math::sin(e), math::cos(e));
        let input = ins.audio(0);
        write_scaled(&mut ctx.outs, 0, input, FRAC_1_SQRT_2 * level);
        write_scaled(&mut ctx.outs, 1, input, cosa * cosb * level);
        write_scaled(&mut ctx.outs, 2, input, sina * cosb * level);
        write_scaled(&mut ctx.outs, 3, input, sinb * level);
        DoneAction::Nothing
    }
}

/// Constructor for [`PanB`].
pub struct PanBCtor;

impl UnitDef for PanBCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PanB { _pad: 0 }))
    }
}

/// `PanB2.ar(in, azimuth, gain)`: encode a mono signal to 2D (planar) ambisonic B-format (`W`, `X`,
/// `Y`). `azimuth` is in units where `1` is half a turn.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PanB2 {
    _pad: u32,
}

impl Unit for PanB2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let a = PI * ins.control(1);
        let level = ins.control(2);
        let (sina, cosa) = (-math::sin(a), math::cos(a));
        let input = ins.audio(0);
        write_scaled(&mut ctx.outs, 0, input, FRAC_1_SQRT_2 * level);
        write_scaled(&mut ctx.outs, 1, input, cosa * level);
        write_scaled(&mut ctx.outs, 2, input, sina * level);
        DoneAction::Nothing
    }
}

/// Constructor for [`PanB2`].
pub struct PanB2Ctor;

impl UnitDef for PanB2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PanB2 { _pad: 0 }))
    }
}

/// `BiPanB2.ar(inA, inB, azimuth, gain)`: encode two anti-phase signals to 2D B-format - the sum goes
/// to `W`, and the difference `inA - inB` is panned to `X`/`Y` at `azimuth` (as if two sources face
/// opposite directions).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BiPanB2 {
    _pad: u32,
}

impl Unit for BiPanB2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let a = PI * ins.control(2);
        let level = ins.control(3);
        let (sina, cosa) = (-math::sin(a), math::cos(a));
        let (w_amp, x_amp, y_amp) = (FRAC_1_SQRT_2 * level, cosa * level, sina * level);
        let in_a = ins.audio(0);
        let in_b = ins.audio(1);
        for (o, (&a, &b)) in ctx.outs.audio(0).iter_mut().zip(in_a.iter().zip(in_b)) {
            *o = (a + b) * w_amp;
        }
        for (o, (&a, &b)) in ctx.outs.audio(1).iter_mut().zip(in_a.iter().zip(in_b)) {
            *o = (a - b) * x_amp;
        }
        for (o, (&a, &b)) in ctx.outs.audio(2).iter_mut().zip(in_a.iter().zip(in_b)) {
            *o = (a - b) * y_amp;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`BiPanB2`].
pub struct BiPanB2Ctor;

impl UnitDef for BiPanB2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(BiPanB2 { _pad: 0 }))
    }
}

/// `DecodeB2.ar(numChans, w, x, y, orientation)`: decode 2D B-format (`W`, `X`, `Y`) to `numChans`
/// speakers evenly spaced around a ring, `orientation` rotating the whole array. Each speaker mixes
/// `W`/`X`/`Y` by decode gains rotated one speaker-step from the last.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DecodeB2 {
    /// The per-speaker rotation `(cos, sin)` of the decode gains (`2*pi / numChans`).
    cosa: f32,
    sina: f32,
    /// The `W` decode gain (constant `1/sqrt(2)`) and the first speaker's `X`/`Y` gains (from
    /// `orientation`).
    w_amp: f32,
    x0: f32,
    y0: f32,
    num_channels: u32,
}

impl Unit for DecodeB2 {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let num = self.num_channels.max(1) as f32;
        let angle = 2.0 * PI / num;
        let orientation = ctx.ins.control(3);
        self.cosa = math::cos(angle);
        self.sina = math::sin(angle);
        self.w_amp = FRAC_1_SQRT_2;
        self.x0 = 0.5 * math::cos(orientation * angle);
        self.y0 = 0.5 * math::sin(orientation * angle);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let w_in = ins.audio(0);
        let x_in = ins.audio(1);
        let y_in = ins.audio(2);
        let (cosa, sina, w_amp) = (self.cosa, self.sina, self.w_amp);
        let (mut x_amp, mut y_amp) = (self.x0, self.y0);
        for ch in 0..self.num_channels as usize {
            let out = ctx.outs.audio(ch);
            for (i, o) in out.iter_mut().enumerate() {
                *o = w_in[i] * w_amp + x_in[i] * x_amp + y_in[i] * y_amp;
            }
            let x_next = x_amp * cosa + y_amp * sina;
            y_amp = y_amp * cosa - x_amp * sina;
            x_amp = x_next;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DecodeB2`].
pub struct DecodeB2Ctor;

impl UnitDef for DecodeB2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DecodeB2 {
            cosa: 0.0,
            sina: 0.0,
            w_amp: 0.0,
            x0: 0.0,
            y0: 0.0,
            num_channels: ctx.num_outputs.max(1) as u32,
        }))
    }
}

/// `PanAz.ar(numChans, in, pos, level, width, orientation)`: pan a mono signal around a ring of
/// `numChans` speakers. `pos` runs the source around the ring (2 = full circle), `width` sets how many
/// speakers it spreads over (a raised-sine window), `orientation` offsets the array.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PanAz {
    num_channels: u32,
}

impl Unit for PanAz {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let num = self.num_channels as usize;
        let level = ins.control(2);
        let width = ins.control(3).abs();
        let orientation = ins.control(4);
        let rwidth = if width > 0.0 { 1.0 / width } else { 0.0 };
        let range = num as f32 * rwidth;
        let rrange = if range > 0.0 { 1.0 / range } else { 0.0 };
        let pos = ins.control(1) * 0.5 * num as f32 + width * 0.5 + orientation;
        let input = ins.audio(0);
        for ch in 0..num {
            let mut chanpos = (pos - ch as f32) * rwidth;
            chanpos -= range * math::floor(rrange * chanpos);
            let amp = if chanpos >= 1.0 {
                0.0
            } else {
                level * math::sin(PI * chanpos)
            };
            write_scaled(&mut ctx.outs, ch, input, amp);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PanAz`].
pub struct PanAzCtor;

impl UnitDef for PanAzCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 5 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PanAz {
            num_channels: ctx.num_outputs.max(1) as u32,
        }))
    }
}
