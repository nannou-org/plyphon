//! `Hilbert` and `FreqShift` - plyphon's ports of scsynth's Hilbert-transform and single-sideband
//! frequency-shifter (`FilterUGens.cpp`).
//!
//! Both are built on the same self-contained 12-stage IIR all-pass phase-difference network (two
//! 6-stage cascades whose outputs are ~90 degrees apart across the audio band, after Sean Costello /
//! Bernie Hutchins) - no FFT is involved. `Hilbert` outputs the pair `[real, 90-degree-shifted]`;
//! `FreqShift` ring-modulates that analytic pair with a quadrature oscillator (the shared sine table)
//! to slide the whole spectrum up or down by `freq` Hz. The all-pass coefficients depend only on the
//! sample rate, so they are computed once at build time.

use core::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::wavetable::lookup_cycle;

/// The 12 all-pass tuning constants (6 per branch) scsynth uses; `coef = (g - 1)/(g + 1)` after scaling
/// each by `15*pi / sampleRate`.
const HILBERT_GAMMA: [f64; 12] = [
    0.3609, 2.7412, 11.1573, 44.7581, 179.6242, 798.4578, // "cos" branch
    1.2524, 5.5671, 22.3423, 89.6271, 364.7914, 2770.1114, // "sin" branch
];

/// The 12 first-order all-pass coefficients for `sample_rate`.
fn hilbert_coefs(sample_rate: f64) -> [f64; 12] {
    let gamconst = 15.0 * core::f64::consts::PI / sample_rate;
    let mut coefs = [0.0f64; 12];
    for (c, &g) in coefs.iter_mut().zip(HILBERT_GAMMA.iter()) {
        let gamma = gamconst * g;
        *c = (gamma - 1.0) / (gamma + 1.0);
    }
    coefs
}

/// Push one sample through the two 6-stage all-pass branches, returning `(cos_branch, sin_branch)` -
/// the analytic pair ~90 degrees apart - and updating the 12 filter states.
#[inline]
fn hilbert_filter(thisin: f64, coefs: &[f64; 12], y1: &mut [f64; 12]) -> (f64, f64) {
    let mut ay = thisin;
    for k in 0..6 {
        let y0 = ay - coefs[k] * y1[k];
        ay = coefs[k] * y0 + y1[k];
        y1[k] = y0;
    }
    let cos_out = ay;
    // The sin branch restarts from the raw input.
    let mut ay = thisin;
    for k in 6..12 {
        let y0 = ay - coefs[k] * y1[k];
        ay = coefs[k] * y0 + y1[k];
        y1[k] = y0;
    }
    (cos_out, ay)
}

/// Wrap a phase in cycles into `[0, 1)`.
#[inline]
fn wrap(x: f32) -> f32 {
    x - math::floor(x)
}

/// `Hilbert.ar(in)`: outputs the analytic pair `[real, imaginary]` - the input and a copy phase-shifted
/// by ~90 degrees across the band - via a 12-stage IIR all-pass network. Two outputs.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Hilbert {
    coefs: [f64; 12],
    y1: [f64; 12],
}

impl Unit for Hilbert {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let coefs = self.coefs;
        let mut y1 = self.y1;
        let input = ctx.ins.audio(0);
        for (i, &x) in input.iter().enumerate() {
            let (c, s) = hilbert_filter(x as f64, &coefs, &mut y1);
            ctx.outs.audio(0)[i] = c as f32;
            ctx.outs.audio(1)[i] = s as f32;
        }
        for (dst, y) in self.y1.iter_mut().zip(y1.iter()) {
            *dst = zap(*y);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Hilbert`].
pub struct HilbertCtor;

impl UnitDef for HilbertCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Hilbert {
            coefs: hilbert_coefs(ctx.audio.sample_rate),
            y1: [0.0; 12],
        }))
    }
}

/// `FreqShift.ar(in, freq, phase)`: a single-sideband frequency shifter - it slides every frequency of
/// `in` up (or, for negative `freq`, down) by `freq` Hz, ring-modulating the analytic pair from the
/// Hilbert network with a quadrature oscillator. `phase` offsets the modulator. `freq`/`phase` are read
/// at control rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FreqShift {
    coefs: [f64; 12],
    y1: [f64; 12],
    /// Modulator phase accumulator in cycles, kept in `[0, 1)`.
    phase: f32,
    _pad: u32,
}

impl FreqShift {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const PHASE: usize = 2;
}

impl Unit for FreqShift {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let phase_offset = ctx.ins.control(Self::PHASE) / TAU; // radians -> cycles
        let inc = freq * ctx.audio.sample_dur as f32;
        let table = ctx.wavetables.sine();
        let coefs = self.coefs;
        let mut y1 = self.y1;
        let mut phase = self.phase;
        let input = ctx.ins.audio(Self::IN);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            let (c, s) = hilbert_filter(x as f64, &coefs, &mut y1);
            let osc = phase + phase_offset;
            let sinosc = lookup_cycle(table, osc) as f64;
            let cososc = lookup_cycle(table, osc + 0.25) as f64; // sine + quarter cycle = cosine
            *o = (c * cososc + s * sinosc) as f32;
            phase = wrap(phase + inc);
        }
        self.phase = phase;
        for (dst, y) in self.y1.iter_mut().zip(y1.iter()) {
            *dst = zap(*y);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`FreqShift`].
pub struct FreqShiftCtor;

impl UnitDef for FreqShiftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(FreqShift {
            coefs: hilbert_coefs(ctx.audio.sample_rate),
            y1: [0.0; 12],
            phase: 0.0,
            _pad: 0,
        }))
    }
}
