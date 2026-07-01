//! `Formant` - plyphon's port of scsynth's `Formant` (`OscUGens.cpp`).
//!
//! A formant oscillator: a pitch-synchronous train of windowed sine grains. Each fundamental period
//! (`fundfreq`) fires one grain - a sine carrier at the formant frequency (`formfreq`) multiplied by a
//! raised-sine window whose duration is `1 / max(fundfreq, bwfreq)`, so a wider `bwfreq` narrows the
//! grain and widens the formant's spectral bandwidth. The carrier phase is re-seeded from the
//! fundamental each period, which keeps the formant peak locked to a harmonic of `fundfreq`.

use core::f64::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `Formant.ar(fundfreq, formfreq, bwfreq)`: a windowed-grain formant oscillator. All three
/// frequencies are read at control rate (scsynth's only calc variant).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Formant {
    /// Fundamental phase (cycles) - fires a grain and re-seeds the others each time it wraps.
    phase1: f64,
    /// Carrier phase (cycles) at the formant frequency.
    phase2: f64,
    /// Window phase (cycles), advanced at `max(fundfreq, bwfreq)`; the grain is silent once it passes 1.
    phase3: f64,
}

impl Formant {
    const FUNDFREQ: usize = 0;
    const FORMFREQ: usize = 1;
    const BWFREQ: usize = 2;
}

impl Unit for Formant {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sample_dur = ctx.audio.sample_dur;
        let inc1 = ctx.ins.control(Self::FUNDFREQ) as f64 * sample_dur;
        let inc2 = ctx.ins.control(Self::FORMFREQ) as f64 * sample_dur;
        let inc3 = ctx.ins.control(Self::BWFREQ) as f64 * sample_dur;
        // The window advances at the greater of the fundamental and bandwidth rates.
        let form_inc = inc1.max(inc3);

        let (mut p1, mut p2, mut p3) = (self.phase1, self.phase2, self.phase3);
        for o in ctx.outs.audio(0).iter_mut() {
            *o = if p3 < 1.0 {
                // A raised-sine window (0 -> 2 -> 0 over one cycle) times the formant-rate carrier.
                let window = math::sin(TAU * (p3 + 0.75)) + 1.0;
                let sample = window * math::sin(TAU * p2);
                p3 += form_inc;
                sample as f32
            } else {
                0.0
            };
            p1 += inc1;
            p2 += inc2;
            if p1 > 1.0 {
                p1 -= 1.0;
                if inc1 != 0.0 {
                    // Re-seed the carrier and window from the fundamental remainder, so each grain
                    // starts phase-locked to the fundamental (this is what pins the formant to a
                    // harmonic).
                    p2 = p1 * inc2 / inc1;
                    p3 = p1 * inc3 / inc1;
                }
            }
        }
        self.phase1 = p1;
        self.phase2 = p2;
        self.phase3 = p3;
        DoneAction::Nothing
    }
}

/// Constructor for [`Formant`].
pub struct FormantCtor;

impl UnitDef for FormantCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Formant {
            phase1: 0.0,
            phase2: 0.0,
            phase3: 0.0,
        }))
    }
}
