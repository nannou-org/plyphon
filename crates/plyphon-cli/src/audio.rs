//! cpal audio-device resolution, listing, and input/output-stream construction.
//!
//! Shared by `server` and `play`. Real-time output mirrors the example crates: a cpal output stream
//! whose callback asks a fill closure (backed by the engine's `World`) for interleaved `f32`, then
//! converts to the device sample format. Live input ([`resolve_input`]/[`play_input`], server only)
//! adds a separate cpal *input* stream that converts captured samples to `f32` and pushes them into an
//! `rtrb` ring; the output side drains that ring and feeds `In.ar` via [`crate::duplex`]. cpal has no
//! duplex stream, so the two run on independent clocks and the ring is their jitter/drift buffer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::cli::AudioArgs;

/// A resolved output device and the stream config to open it with.
pub struct Audio {
    pub device: cpal::Device,
    pub config: cpal::StreamConfig,
    pub sample_format: cpal::SampleFormat,
    pub channels: usize,
    pub sample_rate: f64,
}

/// A resolved input device and the stream config to open it with (the capture twin of [`Audio`]).
/// There is no independent sample rate: input must run at the engine/output rate (no resampling).
pub struct AudioInput {
    pub device: cpal::Device,
    pub config: cpal::StreamConfig,
    pub sample_format: cpal::SampleFormat,
    pub channels: usize,
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

/// Resolve the input device and config for capturing `in_channels` at `sample_rate` (the engine rate).
///
/// The device is the named `--input-device` or the host default. Matching mirrors `render`'s file
/// input: no resampling, so the device must advertise `sample_rate` - otherwise this errors, listing
/// what it does support. The channel count is requested directly (like the output `resolve`); the
/// captured sample format is the device's preferred input format (converted to `f32` on capture).
pub fn resolve_input(
    input_device: &Option<String>,
    in_channels: usize,
    sample_rate: f64,
) -> Result<AudioInput, String> {
    let host = cpal::default_host();
    let device = match input_device {
        None => host
            .default_input_device()
            .ok_or("no default input device available")?,
        Some(name) => input_device_by_name(&host, name)?,
    };

    let default = device
        .default_input_config()
        .map_err(|e| format!("no default input config: {e}"))?;
    let sample_format = default.sample_format();

    // Require the engine rate (no resampling). Surface a helpful capability listing on mismatch.
    let rate = sample_rate as u32;
    let supports_rate = device
        .supported_input_configs()
        .map_err(|e| format!("querying input configs: {e}"))?
        .any(|r| r.min_sample_rate() <= rate && rate <= r.max_sample_rate());
    if !supports_rate {
        let mut caps = String::new();
        for r in device
            .supported_input_configs()
            .map_err(|e| format!("querying input configs: {e}"))?
        {
            caps.push_str(&format!(
                "\n  {} ch, {}-{} Hz, {}",
                r.channels(),
                r.min_sample_rate(),
                r.max_sample_rate(),
                r.sample_format()
            ));
        }
        return Err(format!(
            "input device does not support {rate} Hz (the engine rate); supported:{caps}"
        ));
    }

    let mut config: cpal::StreamConfig = default.config();
    config.channels = in_channels as u16;
    config.sample_rate = rate;
    // Leave buffer_size at the device default: the ring decouples the input stream's framing.
    Ok(AudioInput {
        device,
        config,
        sample_format,
        channels: in_channels,
    })
}

/// Build, start, and return the capture stream plus the ring [`Consumer`] the output side drains and
/// the overflow counter. `ring_capacity`/`prefill` are in samples (frames x channels); `prefill`
/// silence samples set the jitter buffer's target depth so early output callbacks do not underrun.
pub fn play_input(
    audio_in: &AudioInput,
    ring_capacity: usize,
    prefill: usize,
) -> Result<(cpal::Stream, Consumer<f32>, Arc<AtomicU64>), String> {
    let (mut producer, consumer) = RingBuffer::<f32>::new(ring_capacity);
    for _ in 0..prefill.min(ring_capacity) {
        let _ = producer.push(0.0);
    }
    let overflow = Arc::new(AtomicU64::new(0));
    let stream = match audio_in.sample_format {
        cpal::SampleFormat::F32 => build_input::<f32>(
            &audio_in.device,
            &audio_in.config,
            producer,
            overflow.clone(),
        )?,
        cpal::SampleFormat::I16 => build_input::<i16>(
            &audio_in.device,
            &audio_in.config,
            producer,
            overflow.clone(),
        )?,
        cpal::SampleFormat::U16 => build_input::<u16>(
            &audio_in.device,
            &audio_in.config,
            producer,
            overflow.clone(),
        )?,
        other => return Err(format!("unsupported input sample format: {other}")),
    };
    stream
        .play()
        .map_err(|e| format!("starting input stream: {e}"))?;
    Ok((stream, consumer, overflow))
}

/// Construct a typed input stream that converts captured `T` to interleaved `f32` and pushes it into
/// `producer`. On a full ring the unwritten tail is dropped and counted in `overflow` (RT-safe: an
/// atomic add, never a block or a log).
fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: Producer<f32>,
    overflow: Arc<AtomicU64>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    // Pre-grown reused scratch so the capture callback never reallocates in steady state.
    let mut scratch: Vec<f32> = Vec::with_capacity(config.channels as usize * 8192);
    device
        .build_input_stream(
            *config,
            move |input: &[T], _: &cpal::InputCallbackInfo| {
                scratch.clear();
                scratch.extend(input.iter().map(|&s| f32::from_sample(s)));
                let (_, dropped) = producer.push_partial_slice(&scratch);
                if !dropped.is_empty() {
                    overflow.fetch_add(dropped.len() as u64, Ordering::Relaxed);
                }
            },
            |err| eprintln!("input stream error: {err}"),
            None,
        )
        .map_err(|e| format!("building input stream: {e}"))
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

/// Find an input device by name, or error if absent (the capture twin of [`output_device_by_name`]).
fn input_device_by_name(host: &cpal::Host, name: &str) -> Result<cpal::Device, String> {
    for device in host.input_devices().map_err(|e| e.to_string())? {
        if device.to_string() == name {
            return Ok(device);
        }
    }
    Err(format!("no input device named '{name}'"))
}
