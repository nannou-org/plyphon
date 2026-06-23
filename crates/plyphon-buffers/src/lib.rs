//! Async buffer-source traits: the I/O seam for loading sample data into the plyphon engine.
//!
//! The `plyphon` engine is deliberately I/O-free - it only ever installs a finished
//! [`Buffer`] (see [`plyphon::controller::Controller::buffer_set`]). *Where* the
//! samples come from - a sound file, a key-value store, the network, an embedded asset - is an
//! application concern. This crate is the shared contract for that: an application implements
//! [`BufferSource`] (and, for streaming, [`BufferStream`]) over whatever storage it likes, and the
//! decoded [`BufferData`] converts straight into a [`Buffer`] for installation.
//!
//! # Async, but not `Send`
//!
//! Loading is `async` because that is the only shape general enough to cover *every* backend: a
//! synchronous one (a filesystem read, an in-memory map) implements it by returning a ready future,
//! while a genuinely asynchronous one (IndexedDB, `fetch`, a network store) cannot be expressed
//! synchronously at all without preloading. The returned future is intentionally **not** `Send`, so
//! it also fits single-threaded `wasm32` executors; an application that wants multi-threaded loading
//! on native simply drives the future on a dedicated thread (mirroring scsynth's NRT thread).
//!
//! This crate defines no loaders itself - see `plyphon-example-sampler` for a reference
//! `BufferSource` (a small WAV decoder) implemented inline.

#![forbid(unsafe_code)]

use std::future::Future;
use std::pin::Pin;

use plyphon::Buffer;
use thiserror::Error;

/// A boxed future returned by a [`BufferSource`]/[`BufferStream`].
///
/// Boxed so the traits stay object-safe (usable as `dyn BufferSource`), and **not** `Send` so the
/// same trait works on single-threaded `wasm32` executors. Drive it on whatever executor suits the
/// platform: a `block_on` on a background thread natively, `spawn_local` on the web.
pub type BufFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Decoded interleaved sample data - the output of a [`BufferSource`] and the input to a [`Buffer`].
#[derive(Clone, Debug)]
pub struct BufferData {
    /// Interleaved samples (frame-major), `num_frames * num_channels` long.
    pub samples: Vec<f32>,
    /// Number of channels.
    pub num_channels: usize,
    /// The data's own sample rate in Hz.
    pub sample_rate: f64,
}

impl From<BufferData> for Buffer {
    fn from(data: BufferData) -> Buffer {
        Buffer::from_interleaved(data.samples, data.num_channels, data.sample_rate)
    }
}

/// The region of a sound resource to read, mirroring scsynth's `/b_allocRead` / `/b_read` offsets.
#[derive(Clone, Copy, Debug)]
pub struct ReadRegion {
    /// First frame to read.
    pub start_frame: u64,
    /// Number of frames to read, or `None` for "to the end".
    pub num_frames: Option<u64>,
}

impl ReadRegion {
    /// The whole resource.
    pub fn all() -> Self {
        ReadRegion {
            start_frame: 0,
            num_frames: None,
        }
    }
}

impl Default for ReadRegion {
    fn default() -> Self {
        ReadRegion::all()
    }
}

/// Metadata about an open [`BufferStream`].
#[derive(Clone, Copy, Debug)]
pub struct StreamInfo {
    /// Number of channels.
    pub num_channels: usize,
    /// The stream's sample rate in Hz.
    pub sample_rate: f64,
    /// Total length in frames, if known.
    pub total_frames: Option<u64>,
}

/// An error loading or streaming sample data. Variants carry a description string (data, not a
/// wrapped cause), so callers can match on the kind or display the message directly.
#[derive(Debug, Error)]
pub enum LoadError {
    /// No resource exists for the given key.
    #[error("resource not found: {0}")]
    NotFound(String),
    /// The bytes could not be decoded into samples.
    #[error("decode failed: {0}")]
    Decode(String),
    /// The underlying storage or transport failed.
    #[error("i/o error: {0}")]
    Io(String),
    /// The source does not support the requested operation (e.g. streaming).
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// A source of decoded sample data, implemented by the application over its chosen storage.
///
/// `key` identifies the resource (a path, a store key, a URL - whatever the implementation means by
/// it). [`load`](BufferSource::load) reads a whole region into memory (backing `/b_allocRead` and
/// `/b_read`); [`open`](BufferSource::open) starts a sequential stream (backing
/// `Buffer.cueSoundFile` + `DiskIn`) and defaults to [`LoadError::Unsupported`] for one-shot-only
/// sources.
pub trait BufferSource {
    /// Read `region` of `key` fully into memory.
    fn load<'a>(
        &'a self,
        key: &'a str,
        region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>>;

    /// Open `key` for sequential streaming. Defaults to unsupported.
    fn open<'a>(
        &'a self,
        _key: &'a str,
    ) -> BufFuture<'a, Result<Box<dyn BufferStream>, LoadError>> {
        Box::pin(async { Err(LoadError::Unsupported("streaming".to_string())) })
    }
}

/// A sequential, seekable stream of sample frames, for disk-streaming playback (`DiskIn`).
pub trait BufferStream {
    /// The stream's channel count, sample rate, and (if known) total length.
    fn info(&self) -> StreamInfo;

    /// Read the next frames into `out` (interleaved), returning the number of frames read (0 at the
    /// end of a non-looping stream).
    fn read<'a>(&'a mut self, out: &'a mut [f32]) -> BufFuture<'a, Result<usize, LoadError>>;

    /// Seek so the next [`read`](BufferStream::read) starts at `frame`.
    fn seek<'a>(&'a mut self, frame: u64) -> BufFuture<'a, Result<(), LoadError>>;
}
