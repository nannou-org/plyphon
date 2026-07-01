//! `MoogFF` - plyphon's port of scsynth's Moog-ladder VCF (Fontana 2007, the feedback form).

use core::f64::consts::PI;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `MoogFF.ar(in, freq, gain, reset)`: a Moog-ladder resonant low-pass - four cascaded one-pole stages
/// with a global feedback `gain` (0-4; self-oscillates near 4). `freq` sets the cutoff (its `tan`-based
/// coefficient is recomputed only when it changes); a positive `reset` zeroes the ladder state at the
/// block start. `gain`/`reset` are read at control rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MoogFF {
    b0: f64,
    a1: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    freq: f32,
    _pad: u32,
}

impl MoogFF {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const GAIN: usize = 2;
    const RESET: usize = 3;

    fn read(ins: &crate::unit::Inputs<'_>, i: usize, default: f32) -> f32 {
        if ins.len() > i {
            ins.control(i)
        } else {
            default
        }
    }
}

impl Unit for MoogFF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let k = Self::read(&ctx.ins, Self::GAIN, 2.0).clamp(0.0, 4.0) as f64;
        let reset = Self::read(&ctx.ins, Self::RESET, 0.0);
        if freq != self.freq {
            let t = ctx.audio.sample_dur;
            // Bilinear-transform cutoff; a negative (super-Nyquist) value collapses to 0.
            let wc_d = (2.0 * math::tan(t * PI * freq as f64) * ctx.audio.sample_rate).max(0.0);
            let twc_d = t * wc_d;
            self.b0 = twc_d / (twc_d + 2.0);
            self.a1 = (twc_d - 2.0) / (twc_d + 2.0);
            self.freq = freq;
        }
        let (b0, a1) = (self.b0, self.a1);
        let (mut s1, mut s2, mut s3, mut s4) = (self.s1, self.s2, self.s3, self.s4);
        if reset > 0.0 {
            s1 = 0.0;
            s2 = 0.0;
            s3 = 0.0;
            s4 = 0.0;
        }
        let b04 = b0 * b0 * b0 * b0;
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let acc = s4 + b0 * (s3 + b0 * (s2 + b0 * s1));
            let x = x as f64;
            let outs = (b04 * x + acc) / (1.0 + b04 * k);
            *o = outs as f32;
            let u = x - k * outs;
            let mut past = u;
            let mut future = b0 * past + s1;
            s1 = b0 * past - a1 * future;
            past = future;
            future = b0 * past + s2;
            s2 = b0 * past - a1 * future;
            past = future;
            future = b0 * past + s3;
            s3 = b0 * past - a1 * future;
            s4 = b0 * future - a1 * outs;
        }
        self.s1 = zap(s1);
        self.s2 = zap(s2);
        self.s3 = zap(s3);
        self.s4 = zap(s4);
        DoneAction::Nothing
    }
}

/// Constructor for [`MoogFF`].
pub struct MoogFFCtor;

impl UnitDef for MoogFFCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(MoogFF {
            b0: 0.0,
            a1: 0.0,
            s1: 0.0,
            s2: 0.0,
            s3: 0.0,
            s4: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            _pad: 0,
        }))
    }
}
