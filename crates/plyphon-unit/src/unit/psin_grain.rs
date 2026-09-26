//! `PSinGrain` - a fixed-frequency sine grain with a parabolic envelope that frees its synth when it
//! ends, plyphon's port of scsynth's `PSinGrain` (`OscUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `PSinGrain.ar(freq, dur, amp)`: one sine grain of `freq` Hz lasting `dur` seconds, under a
/// parabolic envelope peaking at `amp` - then silence, and the enclosing synth is freed (scsynth's
/// `NodeEnd`) at the end of the block the grain ends in. All three inputs are read once, when the
/// unit starts.
///
/// A direct port of scsynth's `PSinGrain_Ctor` and `PSinGrain_next`. The sine comes from the
/// resonator `y0 = b1 * y1 - y2`, with `b1 = 2 cos(w)` and the history seeded to start at phase 0;
/// the envelope level steps by a slope that itself steps by a constant curve. The constructor sets
/// this up in double precision (with the single-precision `sin`/`cos` of a `float` argument, as the
/// C++ overloads resolve), but each calc runs it in single precision, as scsynth does.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PSinGrain {
    /// The resonator coefficient `2 cos(w)`.
    b1: f64,
    /// The last output of the resonator.
    y1: f64,
    /// The output before that.
    y2: f64,
    /// The envelope level.
    level: f64,
    /// The envelope's per-sample level step.
    slope: f64,
    /// The per-sample change of `slope`.
    curve: f64,
    /// Samples of grain left (scsynth's `mCounter`).
    counter: i32,
    _pad: u32,
}

impl PSinGrain {
    const FREQ: usize = 0;
    const DUR: usize = 1;
    const AMP: usize = 2;
}

impl Unit for PSinGrain {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let dur = ctx.ins.control(Self::DUR);
        let amp = ctx.ins.control(Self::AMP);

        let w = (freq as f64 * ctx.own.radians_per_sample) as f32;
        let sdur = (ctx.own.sample_rate * dur as f64) as f32;
        let rdur = 1.0 / sdur;
        let rdur2 = rdur * rdur;

        self.level = 0.0;
        self.slope = 4.0 * (rdur - rdur2) as f64;
        self.curve = -8.0 * rdur2 as f64;
        self.counter = (sdur as f64 + 0.5) as i32;

        self.b1 = 2.0 * math::cos(w) as f64;
        self.y1 = (-math::sin(w) * amp) as f64;
        self.y2 = (-math::sin(w + w) * amp) as f64;

        // The constructor writes the resonator's first sample, unenveloped, and runs no calc.
        *ctx.outs.control(0) = (self.b1 * self.y1 - self.y2) as f32;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let out = ctx.outs.audio(0);
        let b1 = self.b1 as f32;
        let curve = self.curve as f32;
        let mut y1 = self.y1 as f32;
        let mut y2 = self.y2 as f32;
        let mut level = self.level as f32;
        let mut slope = self.slope as f32;
        let mut counter = self.counter;
        let mut done = DoneAction::Nothing;

        let mut i = 0;
        let mut remain = out.len() as i32;
        while remain > 0 {
            if counter <= 0 {
                out[i..].fill(0.0);
                remain = 0;
            } else {
                let nsmps = remain.min(counter);
                remain -= nsmps;
                counter -= nsmps;
                // scsynth unrolls a full block by three; the arithmetic is the same either way.
                for o in &mut out[i..i + nsmps as usize] {
                    let y0 = b1 * y1 - y2;
                    y2 = y1;
                    y1 = y0;
                    *o = y0 * level;
                    level += slope;
                    slope += curve;
                }
                i += nsmps as usize;
                if counter == 0 {
                    done = DoneAction::FreeSelf;
                }
            }
        }

        self.counter = counter;
        self.level = level as f64;
        self.slope = slope as f64;
        self.y1 = y1 as f64;
        self.y2 = y2 as f64;
        done
    }
}

/// Constructor for [`PSinGrain`].
pub struct PSinGrainCtor;

impl UnitDef for PSinGrainCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PSinGrain {
            b1: 0.0,
            y1: 0.0,
            y2: 0.0,
            level: 0.0,
            slope: 0.0,
            curve: 0.0,
            counter: 0,
            _pad: 0,
        }))
    }
}
