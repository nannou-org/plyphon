//! Feedback FM: oscillators modulated by their own output, through a per-synth feedback bus.
//!
//! Each voice frequency-modulates a sine with its *own* output, read back a block late through
//! `LocalIn`/`LocalOut` (scsynth's local buffers). A slow LFO sweeps each voice's modulation index,
//! so the timbre breathes from a pure tone to a bright, hollow one and back. The sine stays bounded,
//! so the feedback only reshapes the spectrum - it never runs away. Each voice has its own private
//! feedback bus, so the three voices of the chord evolve independently.
//!
//! ```text
//! LocalIn.ar(1) ─►(× depth)─► freq ─► SinOsc.ar(freq) ─┬─► LocalOut.ar  (feed back)
//!        ▲ last block's output   freq = base + idx·fb   └─► × amp ─► Out
//!        └────────────────────────────────────────────────┘   idx = LFO 0..FB_INDEX
//! ```
//! `LocalIn` is calc-ordered before `LocalOut`, so it reads the value `LocalOut` wrote on the
//! previous block - the one-block feedback delay. `SinOsc`'s frequency input is read per sample, so
//! the feedback modulates frequency at audio rate. The voices never free, so (like `example-sine`)
//! there is no NRT work; only the `World` is kept. Identical on native and web.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The chord: `(frequency Hz, modulation-index LFO rate Hz)`. Unrelated LFO rates make the voices
/// breathe out of step.
const VOICES: [(f32, f32); 3] = [
    (110.00, 0.07), // A2
    (165.00, 0.11), // E3
    (220.00, 0.13), // A3
];
/// Peak self-FM modulation index (depth = index * frequency). Kept moderate so the tone brightens
/// rather than turning chaotic.
const FB_INDEX: f32 = 0.3;
/// Per-voice amplitude (three voices sum into one output).
const AMP: f32 = 0.15;
/// Master gain applied in the cpal callback.
const GAIN: f32 = 0.8;

/// Build a `World` already playing the feedback-FM chord. Each `(freq, lfo)` spawns one `fbvoice`.
fn build(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    controller.add_synthdef(fbvoice_def(channels));
    for (freq, lfo) in VOICES {
        if let Ok(voice) = controller.synth_new("fbvoice", ROOT_GROUP_ID, AddAction::Tail) {
            let _ = controller.set_control(voice, 0, freq); // parameter 0 = freq
            let _ = controller.set_control(voice, 1, lfo); // parameter 1 = LFO rate
        }
    }

    // The voices play forever and never free, so there is no NRT cleanup: keep only the `World`.
    world
}

/// `fbvoice`: a sine frequency-modulated by its own one-block-delayed output, the modulation index
/// swept by a slow LFO. Parameters: `freq` (0) and `lfoRate` (1).
fn fbvoice_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 7, output: 0 });
    }
    SynthDef {
        name: "fbvoice".to_string(),
        params: vec![
            Param {
                name: "freq".to_string(),
                default: 220.0,
            },
            Param {
                name: "lfoRate".to_string(),
                default: 0.1,
            },
        ],
        units: vec![
            // 0: LocalIn.ar(1) - this voice's output from the previous block.
            UnitSpec::new("LocalIn", Rate::Audio, vec![], 1),
            // 1: SinOsc.kr(lfoRate) - the slow index LFO (-1..1).
            UnitSpec::new(
                "SinOsc",
                Rate::Control,
                vec![InputRef::Param(1), InputRef::Constant(0.0)],
                1,
            ),
            // 2: MulAdd.kr(lfo, FB_INDEX/2, FB_INDEX/2) - map the LFO to a 0..FB_INDEX index.
            UnitSpec::new(
                "MulAdd",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(FB_INDEX * 0.5),
                    InputRef::Constant(FB_INDEX * 0.5),
                ],
                1,
            ),
            // 3: depth = index * freq  (Hz of frequency deviation; BinaryOpUGen multiply).
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Control,
                inputs: vec![InputRef::Unit { unit: 2, output: 0 }, InputRef::Param(0)],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 4: freq = depth*feedback + base  (MulAdd.ar; freq is audio-rate -> per-sample FM).
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Param(0),
                ],
                1,
            ),
            // 5: SinOsc.ar(freq) - the self-modulated oscillator.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 4, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 6: LocalOut.ar(osc) - feed the oscillator back for next block.
            UnitSpec::new(
                "LocalOut",
                Rate::Audio,
                vec![InputRef::Unit { unit: 5, output: 0 }],
                0,
            ),
            // 7: MulAdd.ar(osc, AMP, 0) - per-voice amplitude.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 5, output: 0 },
                    InputRef::Constant(AMP),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 8: Out.ar(0, voice) on every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
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
        println!("feedback-FM chord breathing for 15s...");
        std::thread::sleep(std::time::Duration::from_secs(15));
    }
    // On the web `main` returns immediately; keep the stream (and its callback) alive.
    #[cfg(target_arch = "wasm32")]
    std::mem::forget(stream);
}
