//! Implementing and registering a *custom* unit generator.
//!
//! plyphon ships a base set of units, but a host can add its own. A custom unit is three pieces, all
//! built from items re-exported by the `plyphon` crate (so a downstream user depends only on
//! `plyphon`, plus `bytemuck` to derive the `Pod` state):
//!
//! 1. a `#[repr(C)]` [`Pod`] state struct,
//! 2. an `impl Unit` providing the per-block [`Unit::process`] (and optionally [`Unit::init`] /
//!    [`Unit::reseed`]),
//! 3. an `impl UnitDef` whose [`UnitDef::build`] turns a [`BuildContext`] into a [`BuiltUnit`] via
//!    [`unit_spec`].
//!
//! Register the `UnitDef` under a name on the engine's registry
//! ([`Controller::registry_mut`](plyphon::Controller::registry_mut)) and
//! the custom unit can be named in any `SynthDef`, exactly like a built-in.
//!
//! This example defines `Saturate` - a `tanh` soft-clip distortion - registers it alongside the base
//! set, and plays `SinOsc.ar -> Saturate -> Out` through cpal so you can hear it work.

use bytemuck::{Pod, Zeroable};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, BuildContext, BuildError, BuiltUnit, DoneAction, InitCtx, InputRef, Options, Param,
    ProcessCtx, ROOT_GROUP_ID, Rate, SynthDef, Unit, UnitDef, UnitSpec, World, engine, unit_spec,
};

// ----------------------------------------------------------------------------------------------
// The custom unit.
// ----------------------------------------------------------------------------------------------

/// `Saturate.ar(in, drive)`: a `tanh` soft-clip distortion.
///
/// Its only state is the smoothed `drive` - a one-pole lag toward the `drive` input, so changing the
/// drive does not click - plus the per-sample smoothing coefficient `b1`, computed once off the
/// audio thread in [`SaturateCtor::build`].
///
/// The state must be `#[repr(C)]` and [`Pod`] (plain-old-data): the engine stores it as raw bytes in
/// its real-time pool and reinterprets it without `unsafe`, so it can hold only `Copy` number/array
/// fields - no references, `Vec`s, `String`s, or enums.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Saturate {
    /// The smoothed drive, lerped toward the `drive` input each sample.
    drive: f32,
    /// Per-sample one-pole smoothing coefficient (`0` = follow the input immediately).
    b1: f32,
}

impl Saturate {
    /// Input 0: the audio signal to distort.
    const IN: usize = 0;
    /// Input 1: the drive amount (`>= 1`; higher is dirtier), read at control rate.
    const DRIVE: usize = 1;
}

impl Unit for Saturate {
    /// Seed state from the unit's first inputs, once, just before the first [`Unit::process`] (on the
    /// audio thread, where inputs are live). Starting the smoothed `drive` *at* the input value means
    /// the first block is already at the right amount instead of gliding up from the build default -
    /// the same trick the built-in `Lag` uses. The default `init` is a no-op, so this is optional.
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.drive = ctx.ins.control(Self::DRIVE).max(1.0);
    }

    /// Compute one control block: read `ctx.ins`, write `ctx.outs`. Like every audio-thread method it
    /// must not allocate, block, or take locks. Returns the [`DoneAction`] to apply to the enclosing
    /// synth - almost always [`DoneAction::Nothing`].
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let target = ctx.ins.control(Self::DRIVE).max(1.0);
        let b1 = self.b1;
        let mut drive = self.drive;
        let input = ctx.ins.audio(Self::IN);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            // One-pole-smooth the drive toward its target, then soft-clip. `tanh` keeps the output
            // bounded to [-1, 1] no matter how hard the signal is pushed.
            drive = target + b1 * (drive - target);
            *o = (x * drive).tanh();
        }
        // Carry the smoothed value into the next block.
        self.drive = drive;
        DoneAction::Nothing
    }
}

/// The [`UnitDef`] that builds a [`Saturate`] - plyphon's analogue of an scsynth `UnitDef`. A
/// zero-sized marker registered under a name; the engine calls [`UnitDef::build`] once, off the audio
/// thread, each time a `SynthDef` naming it is compiled.
struct SaturateCtor;

impl UnitDef for SaturateCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // The smoothing coefficient depends on the sample rate, which the build context carries, so
        // compute it here - allocation and maths off the audio thread are fine. Returning an
        // `Err(BuildError::...)` instead would reject the def (e.g. an unsupported input count).
        let b1 = smoothing_coef(0.01, ctx.audio.sample_rate as f32);
        // `unit_spec` monomorphises the calc/seed vtable for `Saturate` and snapshots this initial
        // state into the compiled def's image.
        Ok(unit_spec(Saturate { drive: 1.0, b1 }))
    }
}

/// A first-order smoothing coefficient: the per-sample multiplier that decays ~60 dB over `time`
/// seconds (`0` for an immediate response). The same formula plyphon's built-in smoothers use.
fn smoothing_coef(time: f32, sample_rate: f32) -> f32 {
    if time > 0.0 {
        (-6.907_755 / (time * sample_rate)).exp() // -6.907755 = ln(0.001)
    } else {
        0.0
    }
}

// ----------------------------------------------------------------------------------------------
// Wiring it into an engine.
// ----------------------------------------------------------------------------------------------

/// Frequency of the tone being distorted.
const FREQ: f32 = 220.0;
/// Default drive amount (overridable via the `drive` parameter).
const DRIVE: f32 = 8.0;
/// Master gain applied in the cpal callback.
const GAIN: f32 = 0.2;

/// Build a `World` playing `SinOsc.ar(FREQ) -> Saturate(drive) -> Out`, with `Saturate` registered
/// alongside the base unit set.
fn build(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // Register the custom unit *alongside* the built-ins (the controller starts with the base set).
    // After this, "Saturate" can be named in any SynthDef, just like "SinOsc" or "Out".
    controller
        .registry_mut()
        .register("Saturate", Box::new(SaturateCtor));

    // SinOsc.ar(FREQ) -> Saturate.ar(in, drive) -> Out, the saturated tone copied to every channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)]; // input 0: starting output bus channel
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 1, output: 0 }); // each channel <- Saturate's output
    }
    let def = SynthDef {
        name: "distorted".to_string(),
        params: vec![Param {
            name: "drive".to_string(),
            default: DRIVE,
        }],
        units: vec![
            // 0: the tone.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
                1,
            ),
            // 1: the custom unit, fed the tone and the `drive` parameter.
            UnitSpec::new(
                "Saturate",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }, InputRef::Param(0)],
                1,
            ),
            // 2: write to the hardware output bus.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("distorted", ROOT_GROUP_ID, AddAction::Tail);

    // The synth plays forever and never frees, so there is no NRT cleanup: drop the `Controller` and
    // `Nrt`, keep only the `World`. (Keep the `Controller` and call `set_control` to sweep `drive`.)
    world
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let config = device
        .default_output_config()
        .expect("no default output config");

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
        format => panic!("unsupported sample format: {format}"),
    }
}

/// Build and play an output stream fed by the engine `World`.
fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let mut source = build(sample_rate, channels);
    // Reused interleaved `f32` scratch buffer; the source fills it, then we convert to `T`.
    let mut scratch: Vec<f32> = Vec::new();

    let stream = device
        .build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                source.fill(&mut scratch, channels);
                for (out, sample) in output.iter_mut().zip(scratch.iter()) {
                    *out = T::from_sample(*sample * GAIN);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream");
    stream.play().expect("failed to start audio stream");

    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("playing a {FREQ} Hz sine through the custom `Saturate` unit for 10s...");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    // On the web `main` returns immediately; keep the stream (and its callback) alive.
    #[cfg(target_arch = "wasm32")]
    std::mem::forget(stream);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// The magnitude of `samples` at `freq` (a one-bin Goertzel), for checking spectral content.
    fn goertzel(samples: &[f32], freq: f32) -> f32 {
        let n = samples.len();
        let k = (0.5 + n as f32 * freq / SR).floor();
        let w = 2.0 * std::f32::consts::PI * k / n as f32;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in samples {
            let s = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
    }

    /// The custom unit runs through the real engine and actually saturates: the output stays bounded
    /// by `tanh` and grows the odd harmonics a clean sine would not have. Exercises registering a
    /// custom `UnitDef`, naming it in a `SynthDef`, and its `init`/`process` - all headlessly.
    #[test]
    fn saturate_distorts_and_stays_bounded() {
        let mut world = build(SR, 1);
        let mut out = vec![0.0f32; (SR / 4.0) as usize];
        world.fill(&mut out, 1);

        // Audible, and `tanh` keeps every sample inside [-1, 1].
        assert!(out.iter().any(|s| s.abs() > 0.1), "the synth was silent");
        assert!(
            out.iter().all(|s| s.abs() <= 1.0 + 1e-4),
            "tanh output should be bounded by 1"
        );

        // Hard-driven soft clipping turns the sine toward a square wave: a strong 3rd harmonic that a
        // clean `SinOsc` (which has none) would not produce, far above an inharmonic reference bin.
        let third = goertzel(&out, FREQ * 3.0);
        let off = goertzel(&out, FREQ * 2.5);
        assert!(
            third > 5.0 * off,
            "expected the custom unit to add harmonic distortion (3rd harmonic {third} vs {off})"
        );
    }
}
