//! Stream a WAV from storage in chunks and play it with `DiskIn`, via cpal, native and web.
//!
//! Where `plyphon-example-sampler` decodes a whole sound into one buffer and plays it with
//! `PlayBuf`, this streams: a [`BufferStream`] decodes the WAV incrementally, a [`StreamFeeder`]
//! keeps the engine's chunk queue topped up off the audio thread, and `DiskIn` plays the queue - so
//! the engine never holds the whole decoded sound, only a small look-ahead. The stream loops the
//! phrase by seeking back to the start at the end.
//!
//! As with the sampler, the bytes are read the platform-appropriate way (filesystem natively,
//! `fetch` on the web); here the decode and feeding then run incrementally. The feeder is driven off
//! the audio thread for the demo's lifetime - a background thread natively, a timer on the web.
//!
//! To run the web build: `trunk serve crates/plyphon-example-stream/web/index.html`.

use std::future::Future;
use std::io::Cursor;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, Controller, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World,
    engine,
};
use plyphon_buffers::{BufFuture, BufferStream, LoadError, StreamFeeder, StreamInfo};

/// The streamed asset (a 330 Hz then 550 Hz phrase).
const ASSET: &str = "stream.wav";
/// A gentle master gain.
const GAIN: f32 = 0.6;
/// Chunk size and queue depth: 4 x 4096 frames is ~340 ms of look-ahead at 48 kHz.
const CHUNK_FRAMES: usize = 4096;
const NUM_CHUNKS: usize = 4;
/// How often the feeder tops the queue up, in milliseconds (well under the look-ahead).
const TICK_MS: u32 = 80;

/// A [`BufferStream`] that decodes a WAV incrementally with `hound`, looping at the end.
struct WavStream {
    reader: hound::WavReader<Cursor<Vec<u8>>>,
    channels: usize,
    sample_rate: f64,
    /// Normalisation factor for integer samples.
    scale: f32,
    float: bool,
}

impl WavStream {
    fn new(bytes: Vec<u8>) -> Result<Self, LoadError> {
        let reader = hound::WavReader::new(Cursor::new(bytes))
            .map_err(|e| LoadError::Decode(e.to_string()))?;
        let spec = reader.spec();
        Ok(WavStream {
            reader,
            channels: spec.channels.max(1) as usize,
            sample_rate: spec.sample_rate as f64,
            scale: 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32,
            float: spec.sample_format == hound::SampleFormat::Float,
        })
    }
}

fn decode_err(e: hound::Error) -> LoadError {
    LoadError::Decode(e.to_string())
}

impl BufferStream for WavStream {
    fn info(&self) -> StreamInfo {
        StreamInfo {
            num_channels: self.channels,
            sample_rate: self.sample_rate,
            total_frames: Some(self.reader.duration() as u64),
        }
    }

    fn read<'a>(&'a mut self, out: &'a mut [f32]) -> BufFuture<'a, Result<usize, LoadError>> {
        let (scale, float, channels) = (self.scale, self.float, self.channels);
        Box::pin(async move {
            let mut filled = 0;
            while filled < out.len() {
                let before = filled;
                if float {
                    for sample in self.reader.samples::<f32>() {
                        out[filled] = sample.map_err(decode_err)?;
                        filled += 1;
                        if filled == out.len() {
                            break;
                        }
                    }
                } else {
                    for sample in self.reader.samples::<i32>() {
                        out[filled] = sample.map_err(decode_err)? as f32 * scale;
                        filled += 1;
                        if filled == out.len() {
                            break;
                        }
                    }
                }
                if filled < out.len() {
                    // Reached the end of the file: loop the phrase.
                    self.reader
                        .seek(0)
                        .map_err(|e| LoadError::Io(e.to_string()))?;
                    if filled == before {
                        break; // empty file: avoid spinning
                    }
                }
            }
            Ok(filled / channels)
        })
    }

    fn seek<'a>(&'a mut self, frame: u64) -> BufFuture<'a, Result<(), LoadError>> {
        Box::pin(async move {
            self.reader
                .seek(frame as u32)
                .map_err(|e| LoadError::Io(e.to_string()))?;
            Ok(())
        })
    }
}

/// Cue a stream from `bytes`, prime its queue, and start a `DiskIn` playing it. Returns the feeder
/// (and the stream) for the caller to keep topping up off the audio thread.
async fn setup(
    mut controller: Controller,
    bytes: Vec<u8>,
    device_channels: usize,
) -> Option<(StreamFeeder, WavStream)> {
    let mut wav = match WavStream::new(bytes) {
        Ok(wav) => wav,
        Err(err) => {
            eprintln!("failed to read {ASSET}: {err}");
            return None;
        }
    };
    let channels = wav.channels;
    let producer = controller
        .buffer_cue(0, channels, wav.sample_rate, CHUNK_FRAMES, NUM_CHUNKS)
        .ok()?;
    let mut feeder = StreamFeeder::new(producer);
    if let Err(err) = feeder.fill(&mut wav).await {
        eprintln!("failed to prime the stream: {err}");
    }

    controller.add_synthdef(disk_in_def(channels, device_channels));
    let _ = controller.synth_new("stream", ROOT_GROUP_ID, AddAction::Tail);
    Some((feeder, wav))
}

/// `DiskIn.ar(streamChannels, bufnum = 0) -> Out`, mapping stream channels onto the device's.
fn disk_in_def(stream_channels: usize, device_channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for c in 0..device_channels {
        out_inputs.push(InputRef::Ugen {
            ugen: 0,
            output: (c % stream_channels) as u32,
        });
    }
    SynthDef {
        name: "stream".to_string(),
        params: vec![],
        ugens: vec![
            UgenSpec::new(
                "DiskIn",
                Rate::Audio,
                vec![InputRef::Constant(0.0)],
                stream_channels,
            ),
            UgenSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

fn build_engine(sample_rate: f32, channels: usize) -> (Controller, World) {
    let (controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels.max(1),
        ..Options::default()
    });
    (controller, world)
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

/// Start the audio stream, then load + stream the asset, feeding the queue off the audio thread.
fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let (controller, world) = build_engine(sample_rate, channels);
    let mut audio = world;
    let mut scratch: Vec<f32> = Vec::new();

    let stream = device
        .build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                audio.fill(&mut scratch, channels);
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
        let bytes = std::fs::read(asset_path()).expect("failed to read the streaming asset");
        if let Some((mut feeder, mut wav)) = block_on(setup(controller, bytes, channels)) {
            std::thread::spawn(move || {
                loop {
                    let _ = block_on(feeder.fill(&mut wav));
                    std::thread::sleep(std::time::Duration::from_millis(u64::from(TICK_MS)));
                }
            });
        }
        println!("streaming a phrase from assets/{ASSET} for 10s...");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            match fetch_bytes(ASSET).await {
                Ok(bytes) => {
                    if let Some((mut feeder, mut wav)) = setup(controller, bytes, channels).await {
                        // Once fetched, decoding from memory is synchronous, so the timer can drive
                        // the (immediately-ready) feeder directly.
                        let interval = gloo_timers::callback::Interval::new(TICK_MS, move || {
                            let _ = block_on(feeder.fill(&mut wav));
                        });
                        interval.forget();
                    }
                }
                Err(err) => eprintln!("failed to fetch {ASSET}: {err}"),
            }
        });
        std::mem::forget(stream);
    }
}

/// The path to the checked-in asset (native).
#[cfg(not(target_arch = "wasm32"))]
fn asset_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join(ASSET)
}

/// Fetch the asset's bytes over HTTP from the page's origin (web).
#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, LoadError> {
    let response = gloo_net::http::Request::get(url)
        .send()
        .await
        .map_err(|e| LoadError::Io(e.to_string()))?;
    response
        .binary()
        .await
        .map_err(|e| LoadError::Io(e.to_string()))
}

/// Drive a future to completion. The WAV stream decodes synchronously from memory, so its future is
/// ready on first poll; the web feeder timer relies on this.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
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

    /// The phrase (330 Hz then 550 Hz) should stream through in order: an early window is 330 Hz, a
    /// later one 550 Hz. Exercises the WavStream -> feeder -> DiskIn path end to end.
    #[test]
    fn streams_the_phrase_in_order() {
        let (controller, mut world) = build_engine(SR, 1);
        let bytes = std::fs::read(asset_path()).expect("asset");
        let (mut feeder, mut wav) = block_on(setup(controller, bytes, 1)).expect("setup");

        let mut out = Vec::new();
        while out.len() < (SR * 1.4) as usize {
            let _ = block_on(feeder.fill(&mut wav));
            let mut buf = vec![0.0f32; 512];
            world.fill(&mut buf, 1);
            out.extend_from_slice(&buf);
        }

        let window = |centre: f32| {
            let c = (SR * centre) as usize;
            let half = (SR * 0.05) as usize;
            &out[c - half..c + half]
        };
        let early = window(0.3); // within the 330 Hz segment
        let late = window(1.1); // within the 550 Hz segment
        assert!(out.iter().any(|s| s.abs() > 0.1), "the stream was silent");
        assert!(
            goertzel(early, 330.0) > 5.0 * goertzel(early, 550.0),
            "early phrase should be 330 Hz"
        );
        assert!(
            goertzel(late, 550.0) > 5.0 * goertzel(late, 330.0),
            "late phrase should be 550 Hz"
        );
    }
}
