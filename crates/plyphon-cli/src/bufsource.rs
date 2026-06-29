//! A filesystem [`BufferSource`] for the server's `/b_allocRead`/`/b_read`, and a small `block_on`.
//!
//! The server keeps buffer loads off the OSC-handling path: `apply` *queues* a load and
//! [`OscDispatcher::run_pending`](plyphon_osc::OscDispatcher::run_pending) services it. Natively a
//! filesystem read resolves on the first poll, so a trivial [`block_on`] (the same one the
//! `example-sampler` uses, built on the stable no-op waker - no `unsafe`) drives it.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

use plyphon_buffers::{BufFuture, BufferData, BufferSource, DefSource, LoadError, ReadRegion};
use plyphon_osc::Host;

use crate::wav;

/// The server's [`Host`]: bundles the filesystem-backed capabilities the dispatcher drives through
/// `run_pending` (sound-file loads, def-file loads; buffer saves as they land).
pub struct CliHost;

impl Host for CliHost {
    fn buffer_source(&self) -> Option<&dyn BufferSource> {
        Some(&FsSource)
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
