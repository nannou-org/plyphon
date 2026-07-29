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
use plyphon_dsp::rate::Rate;

/// `Saw.ar(freq)`: a band-limited sawtooth, output -1 to 1.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Saw {
    phase: f32,
}

impl Unit for Saw {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(0) * ctx.own.sample_dur as f32;
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
        let inc = ctx.ins.control(Self::FREQ) * ctx.own.sample_dur as f32;
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
        let sr = ctx.own.sample_rate as f32;
        let max_n = if freq > 0.0 {
            (sr / (2.0 * freq)) as i32
        } else {
            numharm
        };
        let n = numharm.clamp(1, max_n.max(1));
        let scale = 0.5 / n as f64;
        let two_n1 = (2 * n + 1) as f64;
        let inc = freq * ctx.own.sample_dur as f32;

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

// The following four B-spline BLIT oscillators are translated from
// `AntiAliasingOscillators.cpp`.
//
// SuperCollider is under the GNU General Public License, version 3; these extensions are released
// under the same license.
//
// AntiAliasingOscillators
// Created by Nicholas Collins on 07/08/2010.
// Copyright 2010 Nicholas M Collins. All rights reserved.
//
// UGens by Nick Collins, (c) 8 August 2010, following research work by Juhan Nam, Vesa Valimaki,
// Jonathan S. Abel, and Julius O. Smith, released under the GNU GPL as SuperCollider server
// extension plugins. `BlitB3` phase tracking revised by Nathan Ho in 2016.

/// Validate the shared fixed-shape BLIT ABI.
fn validate_blit(ctx: &BuildContext<'_>, inputs: usize) -> Result<(), BuildError> {
    if ctx.input_rates.len() != inputs {
        return Err(BuildError::WrongInputCount);
    }
    if ctx.num_outputs != 1 {
        return Err(BuildError::WrongOutputCount {
            expected: 1,
            actual: ctx.num_outputs,
        });
    }
    if ctx.special_index != 0 {
        return Err(BuildError::UnsupportedOp(ctx.special_index));
    }
    if ctx.rate != Rate::Audio
        || ctx
            .input_rates
            .iter()
            .any(|rate| !matches!(rate, Rate::Scalar | Rate::Control | Rate::Audio))
    {
        return Err(BuildError::UnsupportedUnitRate);
    }
    Ok(())
}

/// `BlitB3.ar(freq)`: third-order B-spline band-limited impulse train.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlitB3 {
    phase: f32,
}

impl Unit for BlitB3 {
    /// Runs the source constructor's single-sample pre-calculation, then restores phase zero.
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        let mut frequency = ctx.ins.control(0);
        if frequency < 0.000_001 {
            frequency = 0.000_001;
        }
        let period = (ctx.own.sample_rate / frequency as f64) as f32;
        let time = (self.phase % 1.0) * period;
        *ctx.outs.control(0) = blit_b3_pulse(time);
        self.phase = 0.0;
    }

    /// Renders one audio callback while preserving oscillator phase.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut frequency = ctx.ins.control(0);
        if frequency < 0.000_001 {
            frequency = 0.000_001;
        }
        let period = (ctx.own.sample_rate / frequency as f64) as f32;
        let phase = self.phase % 1.0;
        let mut time = phase * period;
        for output in ctx.outs.audio(0).iter_mut() {
            *output = blit_b3_pulse(time);
            time += 1.0;
            if time >= period {
                time -= period;
            }
        }
        self.phase = ((time * frequency) as f64 * ctx.own.sample_dur) as f32;
        DoneAction::Nothing
    }
}

/// Evaluate the translated third-order B-spline pulse for elapsed samples `time`.
fn blit_b3_pulse(time: f32) -> f32 {
    if time >= 4.0 {
        0.0
    } else if time >= 3.0 {
        let x = 4.0 - time;
        0.166_666_67 * x * x * x
    } else if time >= 2.0 {
        let x = time - 2.0;
        let square = x * x;
        0.666_666_7 - square + 0.5 * square * x
    } else if time >= 1.0 {
        let x = time - 2.0;
        let square = x * x;
        0.666_666_7 - square - 0.5 * square * x
    } else {
        0.166_666_67 * time * time * time
    }
}

/// Constructor for [`BlitB3`].
pub(super) struct BlitB3Ctor;

impl UnitDef for BlitB3Ctor {
    /// Validates the fixed ABI and constructs a `BlitB3` voice.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_blit(ctx, 1)?;
        Ok(unit_spec(BlitB3 { phase: 0.0 }))
    }
}

/// `BlitB3Saw.ar(freq, leak)`: integrated B-spline impulse train with DC compensation.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlitB3Saw {
    phase: f32,
    last_output: f32,
    dc_offset: f32,
}

impl Unit for BlitB3Saw {
    /// Renders one audio callback while retaining the two integration states.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let leak = ctx.ins.control(1);
        let mut phase = self.phase;
        let mut last_output = self.last_output;
        let mut dc_offset = self.dc_offset;
        for output in ctx.outs.audio(0).iter_mut() {
            phase -= 1.0;
            let mut value = if phase >= 2.0 {
                dc_offset
            } else if phase >= 1.0 {
                let temp = 2.0 - phase;
                0.166_666_67 * temp * temp * temp + dc_offset
            } else if phase >= 0.0 {
                let square = phase * phase;
                0.666_666_7 - square + 0.5 * square * phase + dc_offset
            } else if phase >= -1.0 {
                let square = phase * phase;
                0.666_666_7 - square - 0.5 * square * phase + dc_offset
            } else if phase >= -2.0 {
                let temp = 2.0 + phase;
                0.166_666_67 * temp * temp * temp + dc_offset
            } else {
                let value = dc_offset;
                let mut frequency = ctx.ins.control(0);
                if frequency < 0.000_001 {
                    frequency = 0.000_001;
                }
                let mut period = (ctx.own.sample_rate / frequency as f64) as f32;
                if period <= 4.0 {
                    period = 4.0;
                }
                dc_offset = -1.0 / period;
                phase += period;
                value
            };
            value += last_output * leak;
            *output = value;
            last_output = value;
        }
        self.phase = phase;
        self.last_output = last_output;
        self.dc_offset = dc_offset;
        DoneAction::Nothing
    }
}

/// Constructor for [`BlitB3Saw`].
pub(super) struct BlitB3SawCtor;

impl UnitDef for BlitB3SawCtor {
    /// Validates the fixed ABI and constructs a `BlitB3Saw` voice.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_blit(ctx, 2)?;
        Ok(unit_spec(BlitB3Saw {
            phase: 3.0,
            last_output: 3.0 / 5.0 - 0.5,
            dc_offset: -1.0 / 5.0,
        }))
    }
}

/// `BlitB3Square.ar(freq, leak)`: alternating integrated B-spline impulse train.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlitB3Square {
    phase: f32,
    last_output: f32,
    bipolar: f32,
}

impl Unit for BlitB3Square {
    /// Renders one audio callback while retaining phase and polarity.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let leak = ctx.ins.control(1);
        let mut phase = self.phase;
        let mut last_output = self.last_output;
        let mut bipolar = self.bipolar;
        for output in ctx.outs.audio(0).iter_mut() {
            phase -= 1.0;
            let mut value = blit_b3_bipolar_pulse(phase, bipolar);
            if phase < -2.0 {
                let mut frequency = ctx.ins.control(0);
                if frequency < 0.000_001 {
                    frequency = 0.000_001;
                }
                let mut half_period = (ctx.own.sample_rate / frequency as f64 * 0.5) as f32;
                if half_period <= 1.0 {
                    half_period = 1.0;
                }
                bipolar *= -1.0;
                phase += half_period;
                value = 0.0;
            }
            value += last_output * leak;
            *output = value;
            last_output = value;
        }
        self.phase = phase;
        self.last_output = last_output;
        self.bipolar = bipolar;
        DoneAction::Nothing
    }
}

/// Evaluate the signed B-spline pulse shared by square and triangle variants.
fn blit_b3_bipolar_pulse(phase: f32, bipolar: f32) -> f32 {
    if phase >= 2.0 {
        0.0
    } else if phase >= 1.0 {
        let temp = 2.0 - phase;
        0.166_666_67 * temp * temp * temp * bipolar
    } else if phase >= 0.0 {
        let square = phase * phase;
        (0.666_666_7 - square + 0.5 * square * phase) * bipolar
    } else if phase >= -1.0 {
        let square = phase * phase;
        (0.666_666_7 - square - 0.5 * square * phase) * bipolar
    } else if phase >= -2.0 {
        let temp = 2.0 + phase;
        0.166_666_67 * temp * temp * temp * bipolar
    } else {
        0.0
    }
}

/// Constructor for [`BlitB3Square`].
pub(super) struct BlitB3SquareCtor;

impl UnitDef for BlitB3SquareCtor {
    /// Validates the fixed ABI and constructs a `BlitB3Square` voice.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_blit(ctx, 2)?;
        Ok(unit_spec(BlitB3Square {
            phase: 3.0,
            last_output: -0.5,
            bipolar: 1.0,
        }))
    }
}

/// `BlitB3Tri.ar(freq, leak, leak2)`: twice-integrated alternating B-spline impulse train.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlitB3Tri {
    phase: f32,
    last_output: f32,
    last_output_2: f32,
    bipolar: f32,
    scale: f32,
}

impl Unit for BlitB3Tri {
    /// Renders one audio callback through both retained integration stages.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let leak = ctx.ins.control(1);
        let leak_2 = ctx.ins.control(2);
        let mut phase = self.phase;
        let mut last_output = self.last_output;
        let mut last_output_2 = self.last_output_2;
        let mut bipolar = self.bipolar;
        let mut scale = self.scale;
        for output in ctx.outs.audio(0).iter_mut() {
            phase -= 1.0;
            let mut value = blit_b3_bipolar_pulse(phase, bipolar);
            if phase < -2.0 {
                let mut frequency = ctx.ins.control(0);
                if frequency < 0.000_001 {
                    frequency = 0.000_001;
                }
                let mut half_period = (ctx.own.sample_rate / frequency as f64 * 0.5) as f32;
                if half_period <= 1.0 {
                    half_period = 1.0;
                }
                scale = 0.25;
                bipolar *= -1.0;
                phase += half_period;
                value = 0.0;
            }
            value += last_output * leak;
            last_output = value;
            value += last_output_2 * leak_2;
            last_output_2 = value;
            *output = value * scale;
        }
        self.phase = phase;
        self.last_output = last_output;
        self.last_output_2 = last_output_2;
        self.bipolar = bipolar;
        self.scale = scale;
        DoneAction::Nothing
    }
}

/// Constructor for [`BlitB3Tri`].
pub(super) struct BlitB3TriCtor;

impl UnitDef for BlitB3TriCtor {
    /// Validates the fixed ABI and constructs a `BlitB3Tri` voice.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_blit(ctx, 3)?;
        Ok(unit_spec(BlitB3Tri {
            phase: 3.0,
            last_output: -0.5,
            last_output_2: 0.0,
            bipolar: 1.0,
            scale: 4.0 / 20.0,
        }))
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
