//! Second-order Butterworth filters - plyphon's port of scsynth's `LPF` and `HPF`.
//!
//! Both share a biquad recurrence; they differ only in how the coefficients are derived from the
//! cutoff frequency and in the sign of the middle output term. Coefficients are recomputed whenever
//! the (control-rate) cutoff changes. State is kept in `f64` and flushed (`zap`) to avoid denormals
//! and non-finite build-up, as scsynth does.

use std::f64::consts::{PI, SQRT_2};

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// Which Butterworth response to compute. The build-time domain; stored in [`Butter`] as a `u32` tag
/// (via [`Kind::to_tag`]) so the state is [`Pod`] and lives in the rt-pool.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Low-pass (`LPF`).
    LowPass,
    /// High-pass (`HPF`).
    HighPass,
}

impl Kind {
    /// Encode as the `u32` tag stored in [`Butter`].
    fn to_tag(self) -> u32 {
        match self {
            Kind::LowPass => 0,
            Kind::HighPass => 1,
        }
    }

    /// Decode the `u32` tag stored in [`Butter`] (any non-`1` tag is low-pass).
    fn from_tag(tag: u32) -> Kind {
        if tag == 1 {
            Kind::HighPass
        } else {
            Kind::LowPass
        }
    }

    /// Biquad coefficients `(a0, b1, b2)` for pre-warped frequency `pfreq` radians.
    fn coeffs(self, pfreq: f64) -> (f64, f64, f64) {
        match self {
            Kind::LowPass => {
                let c = 1.0 / pfreq.tan();
                let c2 = c * c;
                let sqrt2c = c * SQRT_2;
                let a0 = 1.0 / (1.0 + sqrt2c + c2);
                let b1 = -2.0 * (1.0 - c2) * a0;
                let b2 = -(1.0 - sqrt2c + c2) * a0;
                (a0, b1, b2)
            }
            Kind::HighPass => {
                let c = pfreq.tan();
                let c2 = c * c;
                let sqrt2c = c * SQRT_2;
                let a0 = 1.0 / (1.0 + sqrt2c + c2);
                let b1 = 2.0 * (1.0 - c2) * a0;
                let b2 = -(1.0 - sqrt2c + c2) * a0;
                (a0, b1, b2)
            }
        }
    }

    /// Coefficient of the middle term in the output (`+2` low-pass, `-2` high-pass).
    fn mid(self) -> f64 {
        match self {
            Kind::LowPass => 2.0,
            Kind::HighPass => -2.0,
        }
    }
}

/// A second-order Butterworth filter: `LPF.ar(in, freq)` / `HPF.ar(in, freq)`.
///
/// `Pod` state for the rt-pool: `f64` coefficients/history first, then the cached cutoff and the
/// [`Kind`] tag (`repr(C)` lays this out with no implicit padding).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Butter {
    a0: f64,
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    kind: u32,
}

impl Butter {
    const IN: usize = 0;
    const FREQ: usize = 1;
}

impl Unit for Butter {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let kind = Kind::from_tag(self.kind);
        let freq = ctx.ins.control(Self::FREQ);
        if freq != self.freq {
            let pfreq = freq as f64 * ctx.audio.sample_rate.recip() * PI;
            let (a0, b1, b2) = kind.coeffs(pfreq);
            self.a0 = a0;
            self.b1 = b1;
            self.b2 = b2;
            self.freq = freq;
        }

        let (a0, b1, b2, mid) = (self.a0, self.b1, self.b2, kind.mid());
        let (mut y1, mut y2) = (self.y1, self.y2);
        let input = ctx.ins.audio(Self::IN);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            let y0 = x as f64 + b1 * y1 + b2 * y2;
            *o = (a0 * (y0 + mid * y1 + y2)) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// Constructor for [`Butter`], parameterized by filter [`Kind`].
pub struct ButterCtor(pub Kind);

impl UnitDef for ButterCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Butter {
            a0: 0.0,
            b1: 0.0,
            b2: 0.0,
            y1: 0.0,
            y2: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            kind: self.0.to_tag(),
        }))
    }
}

/// Flush denormals and non-finite values to zero (scsynth's `zapgremlins`).
fn zap(x: f64) -> f64 {
    let a = x.abs();
    if a > 1e-15 && a < 1e15 { x } else { 0.0 }
}
