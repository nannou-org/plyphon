//! Record live audio to a WAV file with `DiskOut`, drained off the audio thread - native only.
//!
//! The write-side mirror of `example-stream`. There, a `StreamFeeder` keeps `DiskIn`'s queue full
//! from a file off the audio thread; here, `DiskOut` fills a recording queue on the audio thread and a
//! [`StreamDrainer`] empties it to a WAV on a background thread - the same role as scsynth's NRT disk
//! thread. The audio thread only ever copies each block into a pre-allocated chunk and hands the chunk
//! over a wait-free ring; it never touches the filesystem, allocates, or blocks.
//!
//! A `SinOsc` tone is both played to the speakers (`Out`) and recorded (`DiskOut`), so you hear what
//! is being written. After a few seconds the stream stops, the drainer flushes the tail, and the file
//! is finalized.
//!
//! ```console
//! cargo run -p example-record-to-disk            # writes ./recording.wav
//! cargo run -p example-record-to-disk -- take.wav
//! ```

use std::future::Future;
use std::io::{Seek, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, StreamConsumer, SynthDef, UnitSpec, engine,
};
use plyphon_buffers::{
    BufFuture, BufferSink, BufferSinkStream, SaveError, StreamDrainer, StreamInfo,
};

/// A gentle master gain on playback.
const GAIN: f32 = 0.5;
/// The recorded (and played) tone, in Hz.
const FREQ: f32 = 330.0;
/// The recording is mono.
const REC_CHANNELS: usize = 1;
/// Chunk size and queue depth: 8 x 4096 frames is ~680 ms of headroom at 48 kHz, so the drainer's
/// periodic wake-ups comfortably keep up.
const CHUNK_FRAMES: usize = 4096;
const NUM_CHUNKS: usize = 8;
/// How often the drainer empties the queue, in milliseconds (well under the headroom).
const TICK_MS: u32 = 100;
/// How long to record, in seconds.
const RECORD_SECS: u64 = 4;

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "recording.wav".to_string());

    // Build the engine on the output device; `play_with` hands the controller back so we can cue the
    // recording and start the synth from this (the main) thread.
    let (stream, (mut controller, sample_rate, channels)) =
        example_audio::play_with(GAIN, |sample_rate, channels| {
            let (controller, _nrt, mut world) = engine(Options {
                sample_rate,
                output_channels: channels.max(1),
                ..Options::default()
            });
            (
                move |out: &mut [f32], ch: usize| world.fill(out, ch),
                (controller, sample_rate, channels),
            )
        });

    // Cue a mono recording buffer and start a tone that is both played (`Out`) and recorded (`DiskOut`).
    let consumer = controller
        .buffer_cue_write(0, REC_CHANNELS, sample_rate, CHUNK_FRAMES, NUM_CHUNKS)
        .expect("failed to cue the recording buffer");
    controller.add_synthdef(tone_def(channels));
    controller
        .synth_new("rec", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("failed to start the recording synth");

    // Drain the recorder to a WAV on a background thread (scsynth's NRT disk thread).
    let info = StreamInfo {
        num_channels: REC_CHANNELS,
        sample_rate,
        total_frames: None,
    };
    let stop = Arc::new(AtomicBool::new(false));
    let drainer = {
        let stop = Arc::clone(&stop);
        let path = out_path.clone();
        std::thread::spawn(move || drain_to_wav(consumer, &path, info, &stop))
    };

    println!("recording {RECORD_SECS}s of a {FREQ} Hz tone to {out_path} ...");
    example_audio::keep_alive(stream, RECORD_SECS);
    // The stream has stopped; tell the drainer to flush the tail and close the file.
    stop.store(true, Ordering::Relaxed);
    match drainer.join().expect("the drainer thread panicked") {
        Ok(frames) => println!("wrote {frames} frames to {out_path}"),
        Err(err) => eprintln!("recording failed: {err}"),
    }
}

/// Open `path` and drain the recorder into it until `stop` is set, then flush the tail and finalize
/// the file. Returns the number of frames written.
fn drain_to_wav(
    consumer: StreamConsumer,
    path: &str,
    info: StreamInfo,
    stop: &AtomicBool,
) -> Result<usize, SaveError> {
    let mut sink = block_on(FsSink.open_write(path, info))?;
    let mut drainer = StreamDrainer::new(consumer);
    let mut frames = 0;
    while !stop.load(Ordering::Relaxed) {
        frames += block_on(drainer.drain(&mut *sink))?;
        std::thread::sleep(std::time::Duration::from_millis(u64::from(TICK_MS)));
    }
    // Drain whatever the audio thread queued after the last tick, then close the file.
    frames += block_on(drainer.drain(&mut *sink))?;
    block_on(sink.close())?;
    Ok(frames)
}

/// `SinOsc.ar(FREQ) -> Out.ar(0)` (to the speakers) and `-> DiskOut.ar(bufnum=0)` (to disk).
fn tone_def(device_channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..device_channels.max(1) {
        out_inputs.push(InputRef::Unit { unit: 0, output: 0 });
    }
    SynthDef {
        name: "rec".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "DiskOut",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// A filesystem [`BufferSink`]: `key` is the output path; `open_write` creates a 32-bit-float WAV
/// there with the header taken from `info`.
struct FsSink;

impl BufferSink for FsSink {
    fn open_write<'a>(
        &'a self,
        key: &'a str,
        info: StreamInfo,
    ) -> BufFuture<'a, Result<Box<dyn BufferSinkStream>, SaveError>> {
        Box::pin(async move {
            let spec = hound::WavSpec {
                channels: info.num_channels as u16,
                sample_rate: info.sample_rate as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let writer =
                hound::WavWriter::create(key, spec).map_err(|e| SaveError::Io(e.to_string()))?;
            Ok(Box::new(WavSink {
                writer: Some(writer),
                info,
            }) as Box<dyn BufferSinkStream>)
        })
    }
}

/// A [`BufferSinkStream`] backed by a hound WAV writer, generic over the underlying sink so `main`
/// targets a file while the test targets a temp file all the same. Writes 32-bit float samples.
struct WavSink<W: Write + Seek> {
    writer: Option<hound::WavWriter<W>>,
    info: StreamInfo,
}

impl<W: Write + Seek> BufferSinkStream for WavSink<W> {
    fn info(&self) -> StreamInfo {
        self.info
    }

    fn write<'a>(&'a mut self, samples: &'a [f32]) -> BufFuture<'a, Result<usize, SaveError>> {
        Box::pin(async move {
            let writer = self
                .writer
                .as_mut()
                .ok_or_else(|| SaveError::Io("write after close".to_string()))?;
            for &s in samples {
                writer
                    .write_sample(s)
                    .map_err(|e| SaveError::Encode(e.to_string()))?;
            }
            Ok(samples.len() / self.info.num_channels.max(1))
        })
    }

    fn close<'a>(&'a mut self) -> BufFuture<'a, Result<(), SaveError>> {
        Box::pin(async move {
            if let Some(writer) = self.writer.take() {
                writer
                    .finalize()
                    .map_err(|e| SaveError::Io(e.to_string()))?;
            }
            Ok(())
        })
    }
}

/// Drive a future to completion. The WAV sink resolves synchronously (filesystem writes), so its
/// futures are ready on the first poll.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plyphon::World;

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

    /// Render the recording synth offline, drain it through the example's real `FsSink` into a temp
    /// WAV, then decode the file back and confirm it holds the recorded tone - exercises the whole
    /// DiskOut -> StreamDrainer -> WAV path the example uses, minus cpal.
    #[test]
    fn records_a_tone_to_a_wav_file() {
        let (mut controller, _nrt, mut world): (_, _, World) = engine(Options {
            sample_rate: SR as f64,
            output_channels: 1,
            ..Options::default()
        });
        let consumer = controller
            .buffer_cue_write(0, REC_CHANNELS, SR as f64, 1024, NUM_CHUNKS)
            .unwrap();
        controller.add_synthdef(tone_def(1));
        controller
            .synth_new("rec", ROOT_GROUP_ID, AddAction::Tail, &[])
            .unwrap();

        let path = std::env::temp_dir().join(format!("plyphon-rec-{}.wav", std::process::id()));
        let path = path.to_str().expect("temp path is valid utf-8").to_string();
        let info = StreamInfo {
            num_channels: REC_CHANNELS,
            sample_rate: SR as f64,
            total_frames: None,
        };
        let mut sink = block_on(FsSink.open_write(&path, info)).expect("open the wav");
        let mut drainer = StreamDrainer::new(consumer);

        // Render ~0.3 s in 512-sample blocks, draining after each so the queue never overruns.
        let mut buf = vec![0.0f32; 512];
        let mut produced = 0;
        while produced < (SR * 0.3) as usize {
            world.fill(&mut buf, 1);
            produced += buf.len();
            block_on(drainer.drain(&mut *sink)).expect("drain");
        }
        block_on(drainer.finish(&mut *sink)).expect("finish");

        let reader = hound::WavReader::open(&path).expect("reopen the wav");
        let samples: Vec<f32> = reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .expect("decode the wav");
        std::fs::remove_file(&path).ok();

        assert!(!samples.is_empty(), "no samples were recorded");
        assert!(
            samples.iter().any(|s| s.abs() > 0.1),
            "the recorded wav was silent"
        );
        assert!(
            goertzel(&samples, FREQ) > 5.0 * goertzel(&samples, FREQ * 2.0),
            "expected the recorded {FREQ} Hz tone in the wav"
        );
    }
}
