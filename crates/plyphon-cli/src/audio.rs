//! cpal audio-device resolution, listing, and output-stream construction.
//!
//! Shared by `server` and `play`. Real-time output mirrors the example crates: a cpal output stream
//! whose callback asks a fill closure (backed by the engine's `World`) for interleaved `f32`, then
//! converts to the device sample format. v1 is output-only; capturing hardware input for `In.ar` is
//! a follow-up (the offline `render` path already feeds `In.ar` from a WAV).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use crate::cli::AudioArgs;

/// A resolved output device and the stream config to open it with.
pub struct Audio {
    pub device: cpal::Device,
    pub config: cpal::StreamConfig,
    pub sample_format: cpal::SampleFormat,
    pub channels: usize,
    pub sample_rate: f64,
}

/// Resolve the output device and config from the audio flags, applying any requested overrides
/// (`--device`, `--sample-rate`, `--output-channels`, `--hardware-buffer-size`).
pub fn resolve(args: &AudioArgs) -> Result<Audio, String> {
    let host = cpal::default_host();
    let device = match &args.device {
        None => host
            .default_output_device()
            .ok_or("no default output device available")?,
        Some(name) => output_device_by_name(&host, name)?,
    };

    let default = device
        .default_output_config()
        .map_err(|e| format!("no default output config: {e}"))?;
    let sample_format = default.sample_format();
    let mut config: cpal::StreamConfig = default.config();
    if let Some(channels) = args.output_channels {
        config.channels = channels as u16;
    }
    if let Some(rate) = args.sample_rate {
        config.sample_rate = rate as u32;
    }
    if let Some(frames) = args.hardware_buffer_size {
        config.buffer_size = cpal::BufferSize::Fixed(frames);
    }

    let channels = config.channels as usize;
    let sample_rate = config.sample_rate as f64;
    Ok(Audio {
        device,
        config,
        sample_format,
        channels,
        sample_rate,
    })
}

/// Build and start an output stream driven by `fill` (which fills interleaved `f32`, `channels`
/// wide). The returned [`cpal::Stream`] must be kept alive for playback to continue.
pub fn play_output<F>(audio: &Audio, fill: F) -> Result<cpal::Stream, String>
where
    F: FnMut(&mut [f32], usize) + Send + 'static,
{
    let channels = audio.channels;
    let stream = match audio.sample_format {
        cpal::SampleFormat::F32 => build::<f32, _>(&audio.device, &audio.config, channels, fill)?,
        cpal::SampleFormat::I16 => build::<i16, _>(&audio.device, &audio.config, channels, fill)?,
        cpal::SampleFormat::U16 => build::<u16, _>(&audio.device, &audio.config, channels, fill)?,
        other => return Err(format!("unsupported sample format: {other}")),
    };
    stream
        .play()
        .map_err(|e| format!("starting audio stream: {e}"))?;
    Ok(stream)
}

/// Construct a typed output stream, reblocking the engine's `f32` fill into the device format `T`.
fn build<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut fill: F,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
    F: FnMut(&mut [f32], usize) + Send + 'static,
{
    // Reused interleaved `f32` scratch; the fill closure writes it, then we convert to `T`.
    let mut scratch: Vec<f32> = Vec::new();
    device
        .build_output_stream(
            *config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                fill(&mut scratch, channels);
                for (out, sample) in output.iter_mut().zip(scratch.iter()) {
                    *out = T::from_sample(*sample);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .map_err(|e| format!("building output stream: {e}"))
}

/// `plyphon devices`: list the host's output and input devices, marking the defaults.
pub fn list_devices() -> Result<(), String> {
    let host = cpal::default_host();

    let default_out = host.default_output_device().map(|d| d.to_string());
    println!("output devices:");
    for device in host.output_devices().map_err(|e| e.to_string())? {
        let name = device.to_string();
        let marker = if Some(&name) == default_out.as_ref() {
            " (default)"
        } else {
            ""
        };
        match device.default_output_config() {
            Ok(c) => println!(
                "  {name}{marker} - {} ch, {} Hz, {}",
                c.channels(),
                c.sample_rate(),
                c.sample_format()
            ),
            Err(_) => println!("  {name}{marker}"),
        }
    }

    let default_in = host.default_input_device().map(|d| d.to_string());
    println!("input devices:");
    for device in host.input_devices().map_err(|e| e.to_string())? {
        let name = device.to_string();
        let marker = if Some(&name) == default_in.as_ref() {
            " (default)"
        } else {
            ""
        };
        match device.default_input_config() {
            Ok(c) => println!(
                "  {name}{marker} - {} ch, {} Hz, {}",
                c.channels(),
                c.sample_rate(),
                c.sample_format()
            ),
            Err(_) => println!("  {name}{marker}"),
        }
    }
    Ok(())
}

/// Find an output device by name, or list-and-error if absent.
fn output_device_by_name(host: &cpal::Host, name: &str) -> Result<cpal::Device, String> {
    for device in host.output_devices().map_err(|e| e.to_string())? {
        if device.to_string() == name {
            return Ok(device);
        }
    }
    Err(format!("no output device named '{name}'"))
}
