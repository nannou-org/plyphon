//! Band-limited oscillators - plyphon's ports of scsynth's `Saw`, `Pulse` and `Blip`.
//!
//! [`Saw`] and [`Pulse`] accumulate a normalised phase and suppress the aliasing of their
//! discontinuities with a PolyBLEP correction, so they stay reasonably clean across the spectrum
//! (unlike the raw [`LFSaw`]/[`LFPulse`]). [`Blip`] is a band-limited impulse train evaluated from the
//! closed-form Dirichlet kernel (a sum of `numharm` cosine harmonics), matching scsynth's DSF `Blip`
//! without its cosecant lookup table. Frequency is read at control rate (one value per block).
//!
//! [`LFSaw`]: crate::unit::lf::LFSaw
//! [`LFPulse`]: crate::unit::lf::LFPulse

use core::f64::consts::PI;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `Saw.ar(freq)`: a band-limited sawtooth, output -1 to 1.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Saw {
    phase: f32,
}

impl Unit for Saw {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(0) * ctx.audio.sample_dur as f32;
        let dt = inc.abs().max(f32::MIN_POSITIVE);
        for o in ctx.outs.audio(0).iter_mut() {
            *o = (2.0 * self.phase - 1.0) - poly_blep(self.phase, dt);
            self.phase = wrap(self.phase + inc);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Saw`].
pub struct SawCtor;

impl UnitDef for SawCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Saw { phase: 0.0 }))
    }
}

/// `Pulse.ar(freq, width)`: a band-limited pulse/square, output -1 to 1, duty `width` (default 0.5).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Pulse {
    phase: f32,
}

impl Pulse {
    const FREQ: usize = 0;
    const WIDTH: usize = 1;
}

impl Unit for Pulse {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(Self::FREQ) * ctx.audio.sample_dur as f32;
        let width = if ctx.ins.len() > Self::WIDTH {
            ctx.ins.control(Self::WIDTH).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let dt = inc.abs().max(f32::MIN_POSITIVE);
        for o in ctx.outs.audio(0).iter_mut() {
            let mut value = if self.phase < width { 1.0 } else { -1.0 };
            value += poly_blep(self.phase, dt); // rising edge at the cycle start
            value -= poly_blep(wrap(self.phase + 1.0 - width), dt); // falling edge at `width`
            *o = value;
            self.phase = wrap(self.phase + inc);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Pulse`].
pub struct PulseCtor;

impl UnitDef for PulseCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Pulse { phase: 0.0 }))
    }
}

/// `Blip.ar(freq, numharm)`: a band-limited impulse train - a normalised sum of the first `numharm`
/// cosine harmonics of `freq`. Evaluated directly from the Dirichlet kernel
/// `(sin((2N+1)*pi*p) / sin(pi*p) - 1) * 0.5/N` (`p` the phase in cycles), which equals
/// `(1/N) * sum_{k=1..N} cos(2*pi*k*p)`: it peaks at 1 at each period start and stays band-limited.
/// `numharm` is clamped to the Nyquist limit `floor(sampleRate / (2*freq))`. `freq`/`numharm` are read
/// at control rate; unlike scsynth this recomputes `N` per block without the click-hiding crossfade.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Blip {
    /// Normalised phase accumulator in cycles, kept in `[0, 1)`.
    phase: f32,
}

impl Blip {
    const FREQ: usize = 0;
    const NUMHARM: usize = 1;
}

impl Unit for Blip {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let numharm = if ctx.ins.len() > Self::NUMHARM {
            ctx.ins.control(Self::NUMHARM) as i32
        } else {
            200
        };
        // Clamp the harmonic count to the Nyquist limit so the impulse never aliases.
        let sr = ctx.audio.sample_rate as f32;
        let max_n = if freq > 0.0 {
            (sr / (2.0 * freq)) as i32
        } else {
            numharm
        };
        let n = numharm.clamp(1, max_n.max(1));
        let scale = 0.5 / n as f64;
        let two_n1 = (2 * n + 1) as f64;
        let inc = freq * ctx.audio.sample_dur as f32;

        for o in ctx.outs.audio(0).iter_mut() {
            let p = self.phase as f64;
            let denom = math::sin(PI * p);
            *o = if denom.abs() < 1e-5 {
                // The `0/0` at the period start; the kernel's limit there is exactly the peak, 1.
                1.0
            } else {
                ((math::sin(two_n1 * PI * p) / denom - 1.0) * scale) as f32
            };
            self.phase = wrap(self.phase + inc);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Blip`].
pub struct BlipCtor;

impl UnitDef for BlipCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // Start at a period boundary so the first sample is the impulse.
        Ok(unit_spec(Blip { phase: 0.0 }))
    }
}

/// The PolyBLEP residual that band-limits a unit step at phase `t`, given per-sample phase step `dt`.
fn poly_blep(t: f32, dt: f32) -> f32 {
    if t < dt {
        let t = t / dt;
        2.0 * t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + 2.0 * t + 1.0
    } else {
        0.0
    }
}

/// Wrap a phase into `[0, 1)` (assuming a single cycle's worth of drift at most).
#[inline]
fn wrap(phase: f32) -> f32 {
    if phase >= 1.0 {
        phase - 1.0
    } else if phase < 0.0 {
        phase + 1.0
    } else {
        phase
    }
}
