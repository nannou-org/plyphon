//! Shared cpal output-stream glue for the plyphon examples.
//!
//! Each example builds a plyphon `World` (or controller) plus an interleaved-`f32` fill closure;
//! this crate resolves the output device, opens a stream in the device's native sample format, and
//! reblocks the engine's `f32` into it (scaled by a master gain).
//!
//! On the web it targets cpal's **AudioWorklet** backend when built with the `audioworklet`
//! feature - a real audio thread, off the deprecated `ScriptProcessorNode` - and falls back to the
//! platform default host (native devices, or the legacy Web Audio host on the web) otherwise.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

/// True when this module is executing on the Web Audio worklet thread rather than the main browser
/// thread.
///
/// cpal's AudioWorklet backend re-instantiates the wasm module on the worklet thread (sharing the
/// main thread's memory). With a binary crate that re-runs `main` there, so each example must skip
/// its setup on the worklet thread - audio is driven by cpal's processor, not by `main`. The
/// worklet's `AudioWorkletGlobalScope` has no `window`, which is how we tell the threads apart.
pub fn on_worklet_thread() -> bool {
    #[cfg(all(target_arch = "wasm32", feature = "audioworklet"))]
    {
        web_sys::window().is_none()
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "audioworklet")))]
    {
        false
    }
}

/// The output host: cpal's AudioWorklet host on the web under the `audioworklet` feature, otherwise
/// the platform default. The worklet host needs a cross-origin-isolated page (`SharedArrayBuffer`).
fn output_host() -> cpal::Host {
    #[cfg(all(target_arch = "wasm32", feature = "audioworklet"))]
    {
        cpal::host_from_id(cpal::HostId::AudioWorklet)
            .expect("AudioWorklet host unavailable (page must be cross-origin isolated)")
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "audioworklet")))]
    {
        cpal::default_host()
    }
}

/// Build and start the output stream that drives an example.
///
/// `make_source` receives the resolved sample rate (Hz) and channel count and returns the
/// interleaved-`f32` fill closure (the engine's `World::fill`-style callback). Its output is scaled
/// by `gain` and converted to the device's sample format. The returned stream is already playing
/// and must be kept alive for audio to continue (see [`keep_alive`]).
pub fn play<F>(gain: f32, make_source: impl FnOnce(f64, usize) -> F) -> cpal::Stream
where
    F: FnMut(&mut [f32], usize) + Send + 'static,
{
    let host = output_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let supported = device
        .default_output_config()
        .expect("no default output config");
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate as f64;

    let fill = make_source(sample_rate, channels);
    let stream = match sample_format {
        cpal::SampleFormat::F32 => build::<f32, _>(&device, config, channels, gain, fill),
        cpal::SampleFormat::I16 => build::<i16, _>(&device, config, channels, gain, fill),
        cpal::SampleFormat::U16 => build::<u16, _>(&device, config, channels, gain, fill),
        format => panic!("unsupported sample format: {format}"),
    };
    stream.play().expect("failed to start audio stream");
    stream
}

/// Construct a typed output stream, reblocking the engine's `f32` fill into the device format `T`.
fn build<T, F>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    channels: usize,
    gain: f32,
    mut fill: F,
) -> cpal::Stream
where
    T: SizedSample + FromSample<f32>,
    F: FnMut(&mut [f32], usize) + Send + 'static,
{
    // Reused interleaved `f32` scratch; the fill closure writes it, then we convert to `T`.
    let mut scratch: Vec<f32> = Vec::new();
    device
        .build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                fill(&mut scratch, channels);
                for (out, sample) in output.iter_mut().zip(scratch.iter()) {
                    *out = T::from_sample(*sample * gain);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream")
}

/// Keep `stream` playing: block the calling thread for `native_secs` natively (then stop), or hand
/// the stream to the browser to run indefinitely on the web (where `main` returns immediately).
pub fn keep_alive(stream: cpal::Stream, native_secs: u64) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::sleep(std::time::Duration::from_secs(native_secs));
        drop(stream);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = native_secs;
        std::mem::forget(stream);
    }
}
