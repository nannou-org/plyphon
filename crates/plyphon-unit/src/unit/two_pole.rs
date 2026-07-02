//! Second-order primitive filters - plyphon's ports of scsynth's `TwoPole` and `TwoZero`.
//!
//! Both take a centre frequency and a "reson" radius and carry two samples of history. As with the
//! other filters, the coefficients are recomputed whenever `freq`/`reson` change (control rate) and
//! held constant across the block, rather than slope-interpolated per sample. State is `f64`, flushed
//! with `zap` (scsynth's `zapgremlins`).

use core::f64::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `TwoPole.ar(in, freq, radius)`: a two-pole resonant filter with poles at `radius * e^(±i*w)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TwoPole {
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    reson: f32,
}

impl TwoPole {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const RESON: usize = 2;
}

impl Unit for TwoPole {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let reson = ctx.ins.control(Self::RESON);
        if freq != self.freq || reson != self.reson {
            let w = freq as f64 * TAU / ctx.own.sample_rate;
            self.b1 = 2.0 * reson as f64 * math::cos(w);
            self.b2 = -(reson as f64 * reson as f64);
            self.freq = freq;
            self.reson = reson;
        }

        let (b1, b2) = (self.b1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        let input = ctx.ins.audio(Self::IN);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            let y0 = x as f64 + b1 * y1 + b2 * y2;
            *o = y0 as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// Constructor for [`TwoPole`].
pub struct TwoPoleCtor;

impl UnitDef for TwoPoleCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(TwoPole {
            b1: 0.0,
            b2: 0.0,
            y1: 0.0,
            y2: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            reson: f32::NAN,
        }))
    }
}

/// `TwoZero.ar(in, freq, radius)`: a two-zero filter with zeros at `radius * e^(±i*w)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TwoZero {
    b1: f64,
    b2: f64,
    x1: f64,
    x2: f64,
    freq: f32,
    reson: f32,
}

impl TwoZero {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const RESON: usize = 2;
}

impl Unit for TwoZero {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let reson = ctx.ins.control(Self::RESON);
        if freq != self.freq || reson != self.reson {
            let w = freq as f64 * TAU / ctx.own.sample_rate;
            self.b1 = -2.0 * reson as f64 * math::cos(w);
            self.b2 = reson as f64 * reson as f64;
            self.freq = freq;
            self.reson = reson;
        }

        let (b1, b2) = (self.b1, self.b2);
        let (mut x1, mut x2) = (self.x1, self.x2);
        let input = ctx.ins.audio(Self::IN);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            let x0 = x as f64;
            *o = (x0 + b1 * x1 + b2 * x2) as f32;
            x2 = x1;
            x1 = x0;
        }
        self.x1 = x1;
        self.x2 = x2;
        DoneAction::Nothing
    }
}

/// Constructor for [`TwoZero`].
pub struct TwoZeroCtor;

impl UnitDef for TwoZeroCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(TwoZero {
            b1: 0.0,
            b2: 0.0,
            x1: 0.0,
            x2: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            reson: f32::NAN,
        }))
    }
}
