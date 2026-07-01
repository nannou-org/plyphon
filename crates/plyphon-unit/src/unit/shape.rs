//! Range-shaping units - plyphon's ports of scsynth's `Clip`, `Wrap`, `Fold`, `ModDif`, `InRange`,
//! `InRect`, `LinExp` and `Unwrap` (`LFUGens.cpp`).
//!
//! These reshape a signal's amplitude range: clamping/wrapping/folding into `[lo, hi]`, testing
//! membership of a range or rectangle, remapping a linear input range onto an exponential output
//! range, or unwrapping a value that has been wrapped. Most are stateless per sample; each reads its
//! inputs at their declared rate via the shared `Sig` helper and produces output at the unit's rate.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;
use plyphon_dsp::{math, ops};

/// Which three-input range operation a [`RangeShaper`] applies.
#[derive(Copy, Clone)]
pub enum RangeKind {
    /// `Clip.ar(in, lo, hi)` - clamp into `[lo, hi]`.
    Clip,
    /// `Wrap.ar(in, lo, hi)` - wrap into `[lo, hi)`.
    Wrap,
    /// `Fold.ar(in, lo, hi)` - fold into `[lo, hi]`.
    Fold,
    /// `ModDif.ar(in, dif, mod)` - the modular distance between `in` and `dif` on a `mod`-wide ring.
    ModDif,
}

impl RangeKind {
    fn to_tag(self) -> u32 {
        match self {
            RangeKind::Clip => 0,
            RangeKind::Wrap => 1,
            RangeKind::Fold => 2,
            RangeKind::ModDif => 3,
        }
    }

    fn apply(tag: u32, a: f32, b: f32, c: f32) -> f32 {
        match tag {
            1 => ops::wrap(a, b, c),
            2 => ops::fold(a, b, c),
            3 => {
                // scsynth: `modhalf - |fmod(|in - dif|, mod) - modhalf|` (`%` is truncated, as `fmod`).
                let diff = (a - b).abs() % c;
                let modhalf = c * 0.5;
                modhalf - (diff - modhalf).abs()
            }
            _ => ops::clip(a, b, c),
        }
    }
}

/// A three-input range shaper (`Clip`/`Wrap`/`Fold`/`ModDif`), selected by [`RangeKind`].
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RangeShaper {
    kind: u32,
    audio: u32,
}

impl Unit for RangeShaper {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let kind = self.kind;
        let a = sig(&ctx.ins, 0);
        let b = sig(&ctx.ins, 1);
        let c = sig(&ctx.ins, 2);
        drive(ctx, audio_out, |i| {
            RangeKind::apply(kind, a.at(i), b.at(i), c.at(i))
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`RangeShaper`], parameterised by [`RangeKind`].
pub struct RangeShaperCtor(pub RangeKind);

impl UnitDef for RangeShaperCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(RangeShaper {
            kind: self.0.to_tag(),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `InRange.ar(in, lo, hi)`: `1` while `lo <= in <= hi`, else `0`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InRange {
    audio: u32,
}

impl Unit for InRange {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let lo = ctx.ins.control(1);
        let hi = ctx.ins.control(2);
        drive(ctx, audio_out, |i| {
            let x = input.at(i);
            if x >= lo && x <= hi { 1.0 } else { 0.0 }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`InRange`].
pub struct InRangeCtor;

impl UnitDef for InRangeCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(InRange {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `InRect.ar(x, y, left, top, right, bottom)`: `1` while `(x, y)` is inside the rectangle, else `0`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InRect {
    audio: u32,
}

impl Unit for InRect {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let x = sig(&ctx.ins, 0);
        let y = sig(&ctx.ins, 1);
        let left = ctx.ins.control(2);
        let top = ctx.ins.control(3);
        let right = ctx.ins.control(4);
        let bottom = ctx.ins.control(5);
        drive(ctx, audio_out, |i| {
            let (px, py) = (x.at(i), y.at(i));
            if px >= left && px <= right && py >= top && py <= bottom {
                1.0
            } else {
                0.0
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`InRect`].
pub struct InRectCtor;

impl UnitDef for InRectCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 6 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(InRect {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `LinExp.ar(in, srclo, srchi, dstlo, dsthi)`: maps a linear input range onto an exponential output
/// range (`dstlo`/`dsthi` must share a sign and be non-zero).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LinExp {
    audio: u32,
}

impl Unit for LinExp {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let srclo = ctx.ins.control(1);
        let srchi = ctx.ins.control(2);
        let dstlo = ctx.ins.control(3);
        let dsthi = ctx.ins.control(4);
        let dstratio = dsthi / dstlo;
        let rsrcrange = 1.0 / (srchi - srclo);
        let rrminuslo = rsrcrange * -srclo;
        drive(ctx, audio_out, |i| {
            dstlo * math::powf(dstratio, input.at(i) * rsrcrange + rrminuslo)
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`LinExp`].
pub struct LinExpCtor;

impl UnitDef for LinExpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 5 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(LinExp {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Unwrap.ar(in, lo, hi)`: undoes wrapping - when `in` jumps more than half the range, an accumulated
/// offset is adjusted so the output is continuous.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Unwrap {
    range: f32,
    half: f32,
    prev: f32,
    offset: f32,
    audio: u32,
}

impl Unit for Unwrap {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let input = ctx.ins.control(0);
        let mut lo = ctx.ins.control(1);
        let mut hi = ctx.ins.control(2);
        if lo > hi {
            core::mem::swap(&mut lo, &mut hi);
        }
        self.range = (hi - lo).abs();
        self.half = self.range * 0.5;
        self.prev = input;
        self.offset = if input < lo || input >= hi {
            math::floor((lo - input) / self.range) * self.range
        } else {
            0.0
        };
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let (range, half) = (self.range, self.half);
        let mut prev = self.prev;
        let mut offset = self.offset;
        drive(ctx, audio_out, |i| {
            let zin = input.at(i);
            let diff = zin - prev;
            if diff.abs() > half {
                if zin < prev {
                    offset += range;
                } else {
                    offset -= range;
                }
            }
            prev = zin;
            zin + offset
        });
        self.prev = prev;
        self.offset = offset;
        DoneAction::Nothing
    }
}

/// Constructor for [`Unwrap`].
pub struct UnwrapCtor;

impl UnitDef for UnwrapCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Unwrap {
            range: 0.0,
            half: 0.0,
            prev: 0.0,
            offset: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
