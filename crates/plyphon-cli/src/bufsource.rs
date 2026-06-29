//! A filesystem [`BufferSource`] for the server's `/b_allocRead`/`/b_read`, and a small `block_on`.
//!
//! The server keeps buffer loads off the OSC-handling path: `apply` *queues* a load and
//! [`OscDispatcher::run_pending`](plyphon_osc::OscDispatcher::run_pending) services it. Natively a
//! filesystem read resolves on the first poll, so a trivial [`block_on`] (the same one the
//! `example-sampler` uses, built on the stable no-op waker - no `unsafe`) drives it.

use std::future::Future;
use std::io::{Seek, Write};
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use hound::WavWriter;
use plyphon_buffers::{
    BufFuture, BufferData, BufferSink, BufferSinkStream, BufferSource, DefSource, LoadError,
    ReadRegion, SaveError, StreamInfo,
};
use plyphon_osc::Host;

use crate::cli::SampleFormat;
use crate::wav;

/// The server's [`Host`]: bundles the filesystem-backed capabilities the dispatcher drives through
/// `run_pending` (sound-file loads, def-file loads; buffer saves as they land).
pub struct CliHost;

impl Host for CliHost {
    fn buffer_source(&self) -> Option<&dyn BufferSource> {
        Some(&FsSource)
    }

    fn buffer_sink(&self) -> Option<&dyn BufferSink> {
        Some(&FsSink)
    }

    fn def_source(&self) -> Option<&dyn DefSource> {
        Some(&FsDefs)
    }
}

/// A [`BufferSource`] that reads `/b_allocRead`/`/b_read` keys as WAV file paths.
pub struct FsSource;

impl BufferSource for FsSource {
    fn load<'a>(
        &'a self,
        key: &'a str,
        region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        let result = load_region(key, region);
        Box::pin(async move { result })
    }
}

/// A [`DefSource`] that reads `/d_load`/`/d_loadDir` keys as SCgf (`.scsyndef`) file paths.
pub struct FsDefs;

impl DefSource for FsDefs {
    fn read_def<'a>(&'a self, key: &'a str) -> BufFuture<'a, Result<Vec<u8>, LoadError>> {
        let result = std::fs::read(key).map_err(|err| LoadError::Io(err.to_string()));
        Box::pin(async move { result })
    }

    fn read_def_dir<'a>(&'a self, key: &'a str) -> BufFuture<'a, Result<Vec<Vec<u8>>, LoadError>> {
        let result = read_def_dir(key);
        Box::pin(async move { result })
    }
}

/// Read every `.scsyndef` file under `dir`, returning each one's bytes.
fn read_def_dir(dir: &str) -> Result<Vec<Vec<u8>>, LoadError> {
    let entries = std::fs::read_dir(dir).map_err(|err| LoadError::Io(err.to_string()))?;
    let mut blobs = Vec::new();
    for entry in entries {
        let path = entry.map_err(|err| LoadError::Io(err.to_string()))?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("scsyndef") {
            blobs.push(std::fs::read(&path).map_err(|err| LoadError::Io(err.to_string()))?);
        }
    }
    Ok(blobs)
}

/// A filesystem [`BufferSink`] for the server's `/b_write`: `key` is the output path; `open_write`
/// creates a 32-bit-float WAV there with the channel count and sample rate from `info`. (scsynth's
/// `/b_write` header/sample-format arguments are not honoured - the CLI always writes float WAV.)
pub struct FsSink;

impl BufferSink for FsSink {
    fn open_write<'a>(
        &'a self,
        key: &'a str,
        info: StreamInfo,
    ) -> BufFuture<'a, Result<Box<dyn BufferSinkStream>, SaveError>> {
        Box::pin(async move {
            let spec = wav::spec(SampleFormat::F32, info.num_channels, info.sample_rate);
            let writer = WavWriter::create(key, spec).map_err(|e| SaveError::Io(e.to_string()))?;
            Ok(Box::new(WavSink {
                writer: Some(writer),
                info,
            }) as Box<dyn BufferSinkStream>)
        })
    }
}

/// A [`BufferSinkStream`] backed by a hound WAV writer (32-bit float), generic over the underlying
/// sink so a test can target an in-memory cursor while the server targets a file.
struct WavSink<W: Write + Seek> {
    writer: Option<WavWriter<W>>,
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

/// Read `key` as a WAV file and return the requested `region` as interleaved `f32`.
fn load_region(key: &str, region: ReadRegion) -> Result<BufferData, LoadError> {
    let bytes = std::fs::read(key).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => LoadError::NotFound(key.to_string()),
        _ => LoadError::Io(e.to_string()),
    })?;
    let wav = wav::decode(&bytes).map_err(LoadError::Decode)?;
    let mut data = BufferData {
        samples: wav.samples,
        num_channels: wav.channels,
        sample_rate: wav.sample_rate,
    };
    apply_region(&mut data, region);
    Ok(data)
}

/// Trim `data` to `region` (scsynth's `/b_read` start-frame/frame-count offsets).
fn apply_region(data: &mut BufferData, region: ReadRegion) {
    let channels = data.num_channels.max(1);
    let total = data.samples.len() / channels;
    let start = (region.start_frame as usize).min(total);
    let count = region
        .num_frames
        .map_or(total - start, |n| (n as usize).min(total - start));
    if start == 0 && count == total {
        return;
    }
    data.samples = data.samples[start * channels..(start + count) * channels].to_vec();
}

/// Block the current thread until `future` resolves.
///
/// The server's only async work is filesystem buffer loads, whose futures are ready on the first
/// poll, so a spin on the stable no-op waker suffices (no runtime, no `unsafe`).
pub fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The server's `/b_write` sink: write a ramp through `FsSink` to a temp WAV, then decode it back
    /// and confirm it round-trips - exercising `FsSink`/`WavSink` and the float-WAV format end to end.
    #[test]
    fn fs_sink_round_trips_a_ramp_through_a_wav() {
        let path = std::env::temp_dir().join(format!("plyphon-bwrite-{}.wav", std::process::id()));
        let key = path.to_str().expect("temp path is valid utf-8");
        let info = StreamInfo {
            num_channels: 1,
            sample_rate: 48_000.0,
            total_frames: None,
        };
        let ramp: Vec<f32> = (0..200).map(|f| f as f32).collect();

        let mut sink = block_on(FsSink.open_write(key, info)).expect("open the wav");
        block_on(sink.write(&ramp)).expect("write the ramp");
        block_on(sink.close()).expect("finalize the wav");

        let bytes = std::fs::read(&path).expect("read back the wav");
        std::fs::remove_file(&path).ok();
        let wav = wav::decode(&bytes).expect("decode the wav");
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.samples, ramp, "the wav did not round-trip the ramp");
    }
}
