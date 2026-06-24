//! Cycle through plyphon's oscillators through a low-pass filter, via cpal, natively and on the web.
//!
//! Every couple of seconds the control plane frees the current voice and starts the next waveform -
//! `Saw`, `Pulse`, `LFSaw`, `LFPulse`, `Impulse` - so you hear each in turn (each a 110 Hz tone
//! filtered at 1.8 kHz). As in `plyphon-example-control`, the only platform-specific part is how the
//! control plane is ticked (a thread loop natively, a timer on the web).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, Controller, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

/// The oscillators to cycle through (each is also the SynthDef name).
const WAVES: [&str; 5] = ["Saw", "Pulse", "LFSaw", "LFPulse", "Impulse"];
/// How long each waveform plays, in milliseconds.
const STEP_MS: u32 = 2_000;
/// Oscillator frequency (Hz).
const FREQ: f32 = 110.0;
/// A gentle master gain.
const GAIN: f32 = 0.2;

/// The control plane: frees the current voice and starts the next waveform each step.
struct Controls {
    controller: Controller,
    step: usize,
    current: Option<i32>,
}

impl Controls {
    fn tick(&mut self) {
        if let Some(node) = self.current.take() {
            let _ = self.controller.free(node);
        }
        let wave = WAVES[self.step % WAVES.len()];
        self.step += 1;
        self.current = self
            .controller
            .synth_new(wave, ROOT_GROUP_ID, AddAction::Tail)
            .ok();
        #[cfg(not(target_arch = "wasm32"))]
        println!("playing {wave}");
    }
}

/// `<wave>.ar(110[, width]) -> LPF(1800) -> Out`, copied to every channel.
fn voice_def(wave: &str, channels: usize) -> SynthDef {
    let osc_inputs = match wave {
        "Pulse" => vec![InputRef::Constant(FREQ), InputRef::Constant(0.3)],
        "LFPulse" => vec![
            InputRef::Constant(FREQ),
            InputRef::Constant(0.0),
            InputRef::Constant(0.3),
        ],
        _ => vec![InputRef::Constant(FREQ)],
    };
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 1, output: 0 });
    }
    SynthDef {
        name: wave.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(wave, Rate::Audio, osc_inputs, 1),
            UnitSpec::new(
                "LPF",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(1800.0),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });
    for wave in WAVES {
        controller.add_synthdef(voice_def(wave, channels));
    }
    (
        Controls {
            controller,
            step: 0,
            current: None,
        },
        world,
    )
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

fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let (mut controls, mut source) = build(sample_rate, channels);
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

    controls.tick(); // start the first waveform immediately
    run_control_plane(controls, stream);
}

#[cfg(not(target_arch = "wasm32"))]
fn run_control_plane(mut controls: Controls, _stream: cpal::Stream) {
    use std::time::Duration;
    println!("cycling through the oscillators for ~10s...");
    for _ in 0..(10_000 / STEP_MS) {
        std::thread::sleep(Duration::from_millis(u64::from(STEP_MS)));
        controls.tick();
    }
}

#[cfg(target_arch = "wasm32")]
fn run_control_plane(mut controls: Controls, stream: cpal::Stream) {
    let interval = gloo_timers::callback::Interval::new(STEP_MS, move || controls.tick());
    interval.forget();
    std::mem::forget(stream);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

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

    /// Each waveform in turn should produce the 110 Hz fundamental.
    #[test]
    fn each_waveform_plays() {
        let (mut controls, mut world) = build(SR, 1);
        for _ in WAVES {
            controls.tick();
            let mut out = vec![0.0f32; SR as usize / 8];
            world.fill(&mut out, 1);
            assert!(
                goertzel(&out, FREQ) > 3.0 * goertzel(&out, FREQ * 1.5),
                "expected the {FREQ} Hz fundamental for waveform step {}",
                controls.step - 1
            );
        }
    }
}
