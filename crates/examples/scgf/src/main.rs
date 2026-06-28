//! Load a SuperCollider SCgf-compiled SynthDef and play it, via cpal, natively and on the web.
//!
//! SuperCollider's `sclang` compiles SynthDefs to a compact binary format (SCgf). A SuperCollider
//! client ships those bytes to the server with `/d_recv`; plyphon parses the same bytes with
//! [`plyphon::synthdef::read::parse`]. Here, rather than depend on an `sclang` install, we build the
//! identical bytes with the `scgf` encoder - this is exactly what an `.scsyndef` file contains - then
//! load them through plyphon's parser and play the result. The parse/play path is the real one.
//!
//! The SynthDef has two named controls, `freq` and `amp` (SC `Control` UGens, which plyphon's
//! converter folds into parameters), so it is a small but genuine instrument rather than a fixed
//! tone. Nothing is freed, so the `Controller`/`Nrt` are dropped once the synth is queued.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{AddAction, Options, ROOT_GROUP_ID, World, engine};
use scgf::{Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen};

/// The default `freq` baked into the compiled def (Hz).
const FREQ: f32 = 330.0;
/// The default `amp` baked into the compiled def.
const AMP: f32 = 0.2;
/// A gentle master gain.
const GAIN: f32 = 1.0;

/// SuperCollider binary-operator index for multiply (see `BinaryOpUGen`).
const OP_MUL: i16 = 2;

/// The SCgf bytes for a `SinOsc.ar(freq) * amp -> Out` instrument, as `sclang` would compile them.
fn synthdef_scgf() -> Vec<u8> {
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "loaded".to_string(),
            // Constant 0.0 is reused for the oscillator phase and the output bus.
            constants: vec![0.0],
            // Parameter defaults and names (index into the `Control` UGen's outputs).
            param_values: vec![FREQ, AMP],
            param_names: vec![
                ParamName {
                    name: "freq".to_string(),
                    index: 0,
                },
                ParamName {
                    name: "amp".to_string(),
                    index: 1,
                },
            ],
            ugens: vec![
                // Control: exposes the two parameters as control-rate outputs (freq, amp).
                Ugen {
                    name: "Control".to_string(),
                    rate: Rate::Control,
                    special_index: 0,
                    inputs: vec![],
                    outputs: vec![Rate::Control, Rate::Control],
                },
                // SinOsc.ar(freq, phase = 0)
                Ugen {
                    name: "SinOsc".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Ugen { ugen: 0, output: 0 },
                        Input::Constant { index: 0 },
                    ],
                    outputs: vec![Rate::Audio],
                },
                // SinOsc * amp
                Ugen {
                    name: "BinaryOpUGen".to_string(),
                    rate: Rate::Audio,
                    special_index: OP_MUL,
                    inputs: vec![
                        Input::Ugen { ugen: 1, output: 0 },
                        Input::Ugen { ugen: 0, output: 1 },
                    ],
                    outputs: vec![Rate::Audio],
                },
                // Out.ar(0, [sig, sig]): the same signal on the first two output channels.
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 0 },
                        Input::Ugen { ugen: 2, output: 0 },
                        Input::Ugen { ugen: 2, output: 0 },
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![],
        }],
    };
    scgf::encode(&file).expect("encode SCgf")
}

/// Build a `World` playing the loaded SynthDef. The def writes two channels, so the engine always
/// has at least two output channels even on a mono device.
fn build(sample_rate: f32, channels: usize) -> World {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels.max(2),
        ..Options::default()
    });

    // Parse the compiled bytes exactly as a `/d_recv` would, then start the synth.
    let defs = plyphon::synthdef::read::parse(&synthdef_scgf()).expect("parse SCgf");
    for def in defs {
        controller.add_synthdef(def);
    }
    let _ = controller.synth_new("loaded", ROOT_GROUP_ID, AddAction::Tail);

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
    let sample_rate = config.sample_rate as f32;

    let mut source = build(sample_rate, channels);
    let mut scratch: Vec<f32> = Vec::new();

    let stream = device
        .build_output_stream(
            *config,
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
        println!("playing a SynthDef loaded from SCgf bytes ({FREQ} Hz) for 10s...");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    #[cfg(target_arch = "wasm32")]
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

    /// The loaded def should play its baked-in `FREQ`, proving the SCgf parse/instantiate path works.
    #[test]
    fn loaded_synthdef_plays_its_frequency() {
        let mut world = build(SR, 1);
        let mut out = vec![0.0f32; (SR / 4.0) as usize];
        world.fill(&mut out, 1);
        assert!(
            out.iter().any(|s| s.abs() > 0.05),
            "the loaded def was silent"
        );
        assert!(
            goertzel(&out, FREQ) > 5.0 * goertzel(&out, FREQ * 2.0),
            "expected the loaded def to play {FREQ} Hz"
        );
    }
}
