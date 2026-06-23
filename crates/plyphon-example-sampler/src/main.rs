//! Load a sample through a `BufferSource` and play it with `PlayBuf`, via cpal, native and web.
//!
//! This is the downstream side of plyphon's buffer model: the engine only installs finished
//! buffers, so *loading* sample data is the application's job, expressed by implementing
//! [`plyphon_buffers::BufferSource`]. Here we implement a small reference source inline - an
//! in-memory map of name -> WAV bytes plus a minimal WAV decoder. A real application would swap the
//! map for a filesystem read, a key-value store (e.g. `bevy_pkv`), or a network fetch; only the body
//! of `load` changes.
//!
//! The source is async (the general shape - see the crate docs), so we drive it to completion with a
//! tiny `block_on`. Because this source resolves synchronously, the future is ready on first poll; a
//! genuinely async source would be driven on a background thread (native) or `spawn_local` (web).

use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World, engine,
};
use plyphon_buffers::{BufFuture, BufferData, BufferSource, LoadError, ReadRegion};

/// A gentle master gain.
const GAIN: f32 = 0.5;

/// A reference [`BufferSource`]: an in-memory library of WAV-encoded sounds, keyed by name.
struct SampleLibrary {
    sounds: HashMap<String, Vec<u8>>,
}

impl BufferSource for SampleLibrary {
    fn load<'a>(
        &'a self,
        key: &'a str,
        _region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        // A real source would read these bytes from disk / a KV store / the network here (and could
        // honour `region`); ours just looks them up and decodes synchronously.
        let result = self
            .sounds
            .get(key)
            .ok_or_else(|| LoadError::NotFound(key.to_string()))
            .and_then(|bytes| decode_wav(bytes));
        Box::pin(async move { result })
    }
}

/// Build a `World` looping a sample loaded through a [`SampleLibrary`].
fn build(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // The "sample library": a 440 Hz tone WAV, standing in for a sound you'd load from storage.
    let mut sounds = HashMap::new();
    sounds.insert("tone".to_string(), demo_wav(sample_rate));
    let library = SampleLibrary { sounds };

    // Load it through the BufferSource and install the finished buffer.
    let data = block_on(library.load("tone", ReadRegion::all())).expect("load sample");
    controller.buffer_set(0, Box::new(data.into())).unwrap();

    // PlayBuf the loaded buffer (looping) to the output.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Ugen { ugen: 0, output: 0 });
    }
    let def = SynthDef {
        name: "player".to_string(),
        params: vec![],
        ugens: vec![
            UgenSpec::new(
                "PlayBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0), // bufnum
                    InputRef::Constant(1.0), // rate
                    InputRef::Constant(0.0), // trigger
                    InputRef::Constant(0.0), // startPos
                    InputRef::Constant(1.0), // loop
                    InputRef::Constant(0.0), // doneAction
                ],
                1,
            ),
            UgenSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("player", ROOT_GROUP_ID, AddAction::Tail);

    world
}

/// A mono 16-bit WAV holding a seamless 440 Hz tone at `sample_rate` (44 whole cycles).
fn demo_wav(sample_rate: f32) -> Vec<u8> {
    let frames = (sample_rate / 10.0) as usize; // 0.1 s => 44 cycles of 440 Hz
    let samples: Vec<f32> = (0..frames)
        .map(|i| (std::f32::consts::TAU * 44.0 * i as f32 / frames as f32).sin() * 0.6)
        .collect();
    encode_wav_pcm16(&samples, 1, sample_rate as u32)
}

/// Encode interleaved `f32` samples as a canonical 16-bit PCM WAV.
fn encode_wav_pcm16(samples: &[f32], channels: u16, sample_rate: u32) -> Vec<u8> {
    let bytes_per_sample = 2u32;
    let data_len = samples.len() as u32 * bytes_per_sample;
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample;
    let block_align = channels * bytes_per_sample as u16;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        wav.extend_from_slice(&v.to_le_bytes());
    }
    wav
}

/// Decode a canonical PCM16 or float32 WAV into interleaved `f32` samples - a minimal reference
/// decoder. A real loader would use a full decoder crate (and likely more formats).
fn decode_wav(bytes: &[u8]) -> Result<BufferData, LoadError> {
    let bad = |msg: &str| LoadError::Decode(msg.to_string());
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let u16le = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let u32le = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);

    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, sample_rate, bits)
    let mut data: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32le(&bytes[pos + 4..pos + 8]) as usize;
        let start = pos + 8;
        let body = &bytes[start..(start + size).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                fmt = Some((
                    u16le(body),
                    u16le(&body[2..]),
                    u32le(&body[4..]),
                    u16le(&body[14..]),
                ));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos = start + size + (size & 1); // chunks are word-aligned
    }

    let (format, channels, sample_rate, bits) = fmt.ok_or_else(|| bad("missing fmt chunk"))?;
    let data = data.ok_or_else(|| bad("missing data chunk"))?;
    let samples: Vec<f32> = match (format, bits) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        _ => {
            return Err(LoadError::Unsupported(format!(
                "WAV format {format}, {bits}-bit"
            )));
        }
    };
    Ok(BufferData {
        samples,
        num_channels: channels.max(1) as usize,
        sample_rate: sample_rate as f64,
    })
}

/// Drive a future to completion. Sufficient for sources that resolve synchronously (the future is
/// ready on first poll); a genuinely async source would use a real executor (a background thread
/// natively, `spawn_local` on the web).
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
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
        println!("playing a sample loaded through a BufferSource for 10s...");
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

    #[test]
    fn wav_round_trips() {
        let samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25];
        let wav = encode_wav_pcm16(&samples, 1, 48_000);
        let decoded = decode_wav(&wav).expect("decode");
        assert_eq!(decoded.num_channels, 1);
        assert_eq!(decoded.sample_rate, 48_000.0);
        assert_eq!(decoded.samples.len(), samples.len());
        for (a, b) in samples.iter().zip(&decoded.samples) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn loaded_sample_plays() {
        let mut world = build(SR, 1);
        let mut out = vec![0.0f32; SR as usize / 4];
        world.fill(&mut out, 1);
        assert!(
            out.iter().any(|s| s.abs() > 0.1),
            "the loaded sample was silent"
        );
        assert!(
            goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
            "expected the loaded 440 Hz sample"
        );
    }
}
