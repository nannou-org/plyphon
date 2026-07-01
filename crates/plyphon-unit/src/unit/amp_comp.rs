//! Frequency-dependent amplitude compensation - plyphon's ports of scsynth's `AmpComp` and
//! `AmpCompA` (`LFUGens.cpp`).
//!
//! Both map an input `freq` to a compensating gain so that, played across the spectrum, a source keeps
//! a more even perceived loudness. [`AmpComp`] uses a simple power law `(root/freq)^exp`; [`AmpCompA`]
//! uses an ANSI A-weighting-like equal-loudness curve, rescaled so the gain is `rootAmp` at `root` and
//! `minAmp` at the curve's minimum.
//!
//! (scsynth's `LinLin` is not a server UGen - the language expands it to `MulAdd(in, scale, offset)`,
//! which plyphon already provides.)

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// scsynth's constants for the A-weighting equal-loudness curve (`AmpCompA_calcLevel`).
const AMPCOMP_K: f64 = 3.504_138_4 * 10e15;
const AMPCOMP_C1: f64 = 20.598_997 * 20.598_997;
const AMPCOMP_C2: f64 = 107.652_65 * 107.652_65;
const AMPCOMP_C3: f64 = 737.862_23 * 737.862_23;
const AMPCOMP_C4: f64 = 12194.217 * 12194.217;
const AMPCOMP_MINLEVEL: f64 = -0.157_537_116_743_5;

/// A sign-preserving power (scsynth's `xa >= 0 ? pow(xa, xb) : -pow(-xa, xb)`).
fn signed_pow(x: f32, e: f32) -> f32 {
    if x >= 0.0 {
        math::powf(x, e)
    } else {
        -math::powf(-x, e)
    }
}

/// The A-weighting level at `freq` (scsynth's `AmpCompA_calcLevel`).
fn amp_comp_a_level(freq: f64) -> f64 {
    let r = freq * freq;
    let mut level = AMPCOMP_K * r * r * r * r;
    let n1 = AMPCOMP_C1 + r;
    let n2 = AMPCOMP_C4 + r;
    level /= n1 * n1 * (AMPCOMP_C2 + r) * (AMPCOMP_C3 + r) * n2 * n2;
    1.0 - math::sqrt(level)
}

/// Map input 0 (`freq`) through `f` to the output at the unit's rate (a control `freq` broadcasts).
fn map_freq(ctx: &mut ProcessCtx<'_>, audio: bool, f: impl Fn(f32) -> f32) {
    let ins = ctx.ins;
    if audio {
        let freq_audio = (ins.rate(0) == Rate::Audio).then(|| ins.audio(0));
        let freq_ctrl = ins.control(0);
        for (k, o) in ctx.outs.audio(0).iter_mut().enumerate() {
            *o = f(freq_audio.map_or(freq_ctrl, |s| s[k]));
        }
    } else {
        *ctx.outs.control(0) = f(ins.control(0));
    }
}

/// `AmpComp.ar/kr/ir(freq, root, exp)`: the power-law loudness compensation `(root/freq)^exp`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AmpComp {
    audio: u32,
}

impl Unit for AmpComp {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let root = ctx.ins.control(1);
        let exp = ctx.ins.control(2);
        map_freq(ctx, self.audio != 0, |freq| signed_pow(root / freq, exp));
        DoneAction::Nothing
    }
}

/// Constructor for [`AmpComp`].
pub struct AmpCompCtor;

impl UnitDef for AmpCompCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(AmpComp {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `AmpCompA.ar/kr/ir(freq, root, minAmp, rootAmp)`: the A-weighting equal-loudness compensation,
/// rescaled so the gain is `rootAmp` at `root` and `minAmp` at the curve's minimum.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct AmpCompA {
    /// The curve rescaling `level * scale + offset`, fixed from `root`/`minAmp`/`rootAmp` at init.
    scale: f64,
    offset: f64,
    audio: u32,
    _pad: u32,
}

impl Unit for AmpCompA {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let root_level = amp_comp_a_level(ctx.ins.control(1) as f64);
        let min_amp = ctx.ins.control(2) as f64;
        let root_amp = ctx.ins.control(3) as f64;
        self.scale = (root_amp - min_amp) / (root_level - AMPCOMP_MINLEVEL);
        self.offset = min_amp - self.scale * AMPCOMP_MINLEVEL;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (scale, offset) = (self.scale, self.offset);
        map_freq(ctx, self.audio != 0, |freq| {
            (amp_comp_a_level(freq as f64) * scale + offset) as f32
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`AmpCompA`].
pub struct AmpCompACtor;

impl UnitDef for AmpCompACtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(AmpCompA {
            scale: 0.0,
            offset: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
            _pad: 0,
        }))
    }
}
