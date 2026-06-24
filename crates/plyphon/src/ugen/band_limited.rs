//! Band-limited oscillators - plyphon's ports of scsynth's `Saw` and `Pulse`.
//!
//! Both accumulate a normalised phase and suppress the aliasing of their discontinuities with a
//! PolyBLEP correction, so they stay reasonably clean across the spectrum (unlike the raw [`LFSaw`]/
//! [`LFPulse`]). Frequency is read at control rate (one value per block).
//!
//! [`LFSaw`]: crate::ugen::lf::LFSaw
//! [`LFPulse`]: crate::ugen::lf::LFPulse

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::ugen::registry::{BuildContext, UgenDef};
use crate::ugen::{BuiltUgen, DoneAction, ProcessCtx, Ugen, ugen_spec};

/// `Saw.ar(freq)`: a band-limited sawtooth, output -1 to 1.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Saw {
    phase: f32,
}

impl Ugen for Saw {
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

impl UgenDef for SawCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError> {
        Ok(ugen_spec(Saw { phase: 0.0 }))
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

impl Ugen for Pulse {
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

impl UgenDef for PulseCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError> {
        Ok(ugen_spec(Pulse { phase: 0.0 }))
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
