//! Minimal SinOsc example driven by [`cpal`], working both natively and on the web.
//!
//! The audio loop is identical on both targets - a `cpal` output stream whose callback asks the
//! engine (a `plyphon::World` playing a SinOsc, built in [`sine`]) to fill an interleaved `f32`
//! buffer via `World::fill`. The native/web split (the `engine` module) exists so each side can
//! diverge later (e.g. an AudioWorklet backend on the web) without touching the other. This keeps
//! `cpal` the uniform audio backend the engine slots into.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

mod sine;

#[cfg(not(target_arch = "wasm32"))]
#[path = "engine_native.rs"]
mod engine;
#[cfg(target_arch = "wasm32")]
#[path = "engine_web.rs"]
mod engine;

/// Master gain applied to the engine's full-scale output, to keep the demo gentle on the ears.
const GAIN: f32 = 0.2;

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

/// Build and play an output stream fed by the target's engine `World`.
fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let mut source = engine::new(sample_rate, channels);
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
        println!("playing a 440 Hz sine for 10s...");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    // On the web `main` returns immediately; keep the stream (and its callback) alive.
    #[cfg(target_arch = "wasm32")]
    std::mem::forget(stream);
}
