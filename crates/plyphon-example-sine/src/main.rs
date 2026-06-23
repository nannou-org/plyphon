//! The simplest plyphon example: a continuous 440 Hz sine via cpal, natively and on the web.
//!
//! plyphon's `World` plays a one-synth graph (`SinOsc.ar -> Out`) and the cpal callback fills from
//! it. Nothing is ever freed, so there is no NRT work: the `Controller` and `Nrt` are dropped once
//! the synth is queued, leaving only the `World`. (The richer `plyphon-example-motif` shows the full
//! lifecycle - starting and freeing notes, and running the `Nrt` - and is what the website ships.)

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World, engine,
};

/// Demo frequency in Hz.
const FREQ: f32 = 440.0;
/// A gentle master gain applied to the full-scale oscillator.
const GAIN: f32 = 0.2;

/// Build a `World` already playing a continuous `FREQ`-Hz sine on every output channel.
fn build(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // SinOsc.ar(FREQ) -> Out, the oscillator copied to every channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)]; // input 0: starting bus channel
    for _ in 0..channels {
        out_inputs.push(InputRef::Ugen { ugen: 0, output: 0 });
    }
    let def = SynthDef {
        name: "sine".to_string(),
        params: vec![],
        ugens: vec![
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
                1,
            ),
            UgenSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail);

    // The queued synth plays forever and never frees, so there is no NRT cleanup to do: drop the
    // `Controller` and `Nrt` and keep only the `World`.
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
        println!("playing a {FREQ} Hz sine for 10s...");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    // On the web `main` returns immediately; keep the stream (and its callback) alive.
    #[cfg(target_arch = "wasm32")]
    std::mem::forget(stream);
}
