//! Minimal cpal example, working natively and on the web.
//!
//! A `cpal` output stream's callback asks the engine's `plyphon::World` to fill an interleaved
//! `f32` buffer (`World::fill`). The control plane - a `Controls` (a `Controller` + `Nrt`, built in
//! [`demo`]) - is kept alive and ticked off the audio thread: it starts a looping motif of
//! self-freeing notes and runs the `Nrt` to drop the freed synths and drain notifications. Ticking
//! runs on a dedicated cadence per target (a thread loop natively, a timer on the web), which is the
//! whole point: the `Nrt` is *run*, not dropped.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

mod demo;

#[cfg(not(target_arch = "wasm32"))]
#[path = "engine_native.rs"]
mod engine;
#[cfg(target_arch = "wasm32")]
#[path = "engine_web.rs"]
mod engine;

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

/// Play the demo: the `World` feeds the cpal stream, while the `Controls` are ticked off the audio
/// thread to start notes and run the NRT cleanup.
fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let (controls, mut source) = engine::new(sample_rate, channels);
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
                    *out = T::from_sample(*sample);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream");
    stream.play().expect("failed to start audio stream");

    run_control_plane(controls, stream);
}

/// Tick the control plane (`Controls`) off the audio thread for the demo's lifetime, holding the
/// stream alive meanwhile.
#[cfg(not(target_arch = "wasm32"))]
fn run_control_plane(mut controls: demo::Controls, _stream: cpal::Stream) {
    use std::time::Duration;
    println!("playing a looping motif for 10s...");
    let ticks = 10_000 / demo::TICK_MS;
    for _ in 0..ticks {
        controls.tick();
        std::thread::sleep(Duration::from_millis(u64::from(demo::TICK_MS)));
    }
}

/// On the web, `main` returns immediately, so run the control plane on a periodic timer and keep
/// both it and the audio stream alive.
#[cfg(target_arch = "wasm32")]
fn run_control_plane(mut controls: demo::Controls, stream: cpal::Stream) {
    let interval = gloo_timers::callback::Interval::new(demo::TICK_MS, move || controls.tick());
    interval.forget();
    std::mem::forget(stream);
}
