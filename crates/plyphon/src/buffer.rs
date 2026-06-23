//! Sample buffers - plyphon's port of scsynth's `SndBuf`.
//!
//! A [`Buffer`] is pure in-memory sample data: interleaved `f32` samples plus a frame count, channel
//! count, and sample rate. The engine never reads or decodes sound files; a buffer is built off the
//! audio thread (allocating, possibly loading from storage - a host concern) and then installed into
//! the [`World`](crate::world::World)'s buffer table with [`Controller::buffer_set`] over the command
//! ring, exactly like a synth. The audio thread only ever reads a finished `Buffer`; UGens such as
//! `PlayBuf` reach it (read-only) through [`ProcessContext`](crate::ugen::ProcessContext).
//!
//! [`Controller::buffer_set`]: crate::controller::Controller::buffer_set

/// A bank of interleaved audio samples (scsynth's `SndBuf`).
///
/// Samples are stored frame-major: frame `f`'s channels occupy `data[f*ch .. (f+1)*ch]`.
#[derive(Clone, Debug)]
pub struct Buffer {
    /// `num_frames * num_channels` samples, interleaved (frame-major).
    data: Box<[f32]>,
    num_frames: usize,
    num_channels: usize,
    sample_rate: f64,
}

impl Buffer {
    /// A zeroed buffer of `num_frames` frames and `num_channels` channels (scsynth's `/b_alloc`).
    pub fn zeroed(num_frames: usize, num_channels: usize, sample_rate: f64) -> Self {
        Buffer {
            data: vec![0.0; num_frames * num_channels].into_boxed_slice(),
            num_frames,
            num_channels,
            sample_rate,
        }
    }

    /// A buffer wrapping already-interleaved samples (e.g. decoded from a sound file by the host).
    ///
    /// The frame count is `samples.len() / num_channels`; any trailing partial frame is dropped.
    pub fn from_interleaved(samples: Vec<f32>, num_channels: usize, sample_rate: f64) -> Self {
        let channels = num_channels.max(1);
        let num_frames = samples.len() / channels;
        let mut data = samples;
        data.truncate(num_frames * channels);
        Buffer {
            data: data.into_boxed_slice(),
            num_frames,
            num_channels: channels,
            sample_rate,
        }
    }

    /// Number of frames (samples per channel).
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// The buffer's own sample rate in Hz (so playback UGens can rate-correct against the engine's).
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// All samples, interleaved (frame-major).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Sample at `frame`, `channel`. Returns 0.0 for an out-of-range index, so RT readers never panic.
    pub fn sample(&self, frame: usize, channel: usize) -> f32 {
        if frame >= self.num_frames || channel >= self.num_channels {
            return 0.0;
        }
        self.data[frame * self.num_channels + channel]
    }

    /// Overwrite sample at `frame`, `channel` (scsynth's `/b_set`). No-op if out of range, RT-safe.
    pub fn set_sample(&mut self, frame: usize, channel: usize, value: f32) {
        if frame < self.num_frames && channel < self.num_channels {
            self.data[frame * self.num_channels + channel] = value;
        }
    }

    /// Zero every sample (scsynth's `/b_zero`).
    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }
}

/// One slot in the [`BufferTable`]. An enum (rather than `Option`) so a streaming variant can be
/// added later without changing the table API (see the `buffer` module docs).
enum BufferSlot {
    /// No buffer installed.
    Empty,
    /// An in-memory buffer.
    Loaded(Box<Buffer>),
}

/// The [`World`](crate::world::World)'s fixed-capacity table of buffers, indexed by buffer number.
///
/// Allocated once at construction. Installing a buffer is an O(1) swap that hands any previous buffer
/// back for off-audio-thread dropping; the audio thread only ever reads through [`BufferTable::get`].
pub struct BufferTable {
    slots: Vec<BufferSlot>,
}

impl BufferTable {
    /// A table of `capacity` empty slots (indices `0..capacity` are valid).
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(BufferSlot::Empty);
        }
        BufferTable { slots }
    }

    /// The buffer at `index`, or `None` if the slot is empty or out of range. RT-safe (no panic).
    pub fn get(&self, index: usize) -> Option<&Buffer> {
        match self.slots.get(index) {
            Some(BufferSlot::Loaded(buffer)) => Some(buffer),
            _ => None,
        }
    }

    /// Install `buffer` at `index`, returning whatever buffer must now be dropped off the audio
    /// thread: the one it replaced, or - if `index` is out of range - `buffer` itself (uninstalled).
    pub fn set(&mut self, index: usize, buffer: Box<Buffer>) -> Option<Box<Buffer>> {
        match self.slots.get_mut(index) {
            Some(slot) => match core::mem::replace(slot, BufferSlot::Loaded(buffer)) {
                BufferSlot::Loaded(old) => Some(old),
                BufferSlot::Empty => None,
            },
            None => Some(buffer),
        }
    }

    /// Empty `index`, returning the buffer it held (if any) for off-audio-thread dropping.
    pub fn free(&mut self, index: usize) -> Option<Box<Buffer>> {
        match self.slots.get_mut(index) {
            Some(slot) => match core::mem::replace(slot, BufferSlot::Empty) {
                BufferSlot::Loaded(old) => Some(old),
                BufferSlot::Empty => None,
            },
            None => None,
        }
    }
}
