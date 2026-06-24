//! Disk-streaming playback - the chunk-queue behind `DiskIn`, plyphon's safe-Rust answer to
//! scsynth's disk-streamed buffers.
//!
//! A streaming buffer is not a flat buffer but a *queue of audio chunks*. An off-RT feeder fills
//! fixed-capacity [`Chunk`]s (allocated once at cue time, then recycled - never on the audio thread)
//! and hands them to the RT side over a lock-free ring; `DiskIn` pops filled chunks, plays them, and
//! returns emptied chunks over a second ring. The audio thread never reads files, allocates, or
//! blocks; an empty queue (underrun) plays silence.
//!
//! This is the safe counterpart to scsynth's ring buffer whose halves RT and the NRT thread fill in
//! lock-step: instead of sharing one buffer's memory across threads, plyphon passes ownership of
//! whole chunks over SPSC rings, so no memory is shared mutably across threads.

use rtrb::{Consumer, Producer, PushError, RingBuffer};

/// A fixed-capacity block of interleaved samples passed between the feeder and the audio thread.
pub struct Chunk {
    /// Interleaved samples, `capacity * channels` long.
    data: Box<[f32]>,
    /// Valid frames currently in `data` (`<= capacity`).
    frames: usize,
    channels: usize,
}

impl Chunk {
    fn new(capacity: usize, channels: usize) -> Self {
        Chunk {
            data: vec![0.0; capacity * channels].into_boxed_slice(),
            frames: 0,
            channels,
        }
    }

    /// Capacity in frames.
    pub fn capacity(&self) -> usize {
        self.data.len() / self.channels
    }

    /// The full interleaved sample buffer, for a feeder to fill (`capacity * channels` long).
    pub fn samples_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Record how many frames the feeder filled (clamped to capacity).
    pub fn set_frames(&mut self, frames: usize) {
        self.frames = frames.min(self.capacity());
    }
}

/// The RT-side consumer of a stream: lives in the buffer table, read by `DiskIn`.
pub struct StreamPlayback {
    /// Filled chunks arriving from the feeder.
    chunks: Consumer<Chunk>,
    /// Emptied chunks returned to the feeder.
    recycle: Producer<Chunk>,
    current: Option<Chunk>,
    cursor: usize,
    channels: usize,
    sample_rate: f64,
}

impl StreamPlayback {
    /// The stream's channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The stream's sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Pull up to `frames` frames, calling `emit(frame, channel, sample)` for each sample of each
    /// produced frame, and return how many frames were produced (fewer than `frames` on underrun).
    /// Recycles each exhausted chunk back to the feeder. RT-safe (no allocation or blocking).
    pub fn read(
        &mut self,
        frames: usize,
        out_channels: usize,
        mut emit: impl FnMut(usize, usize, f32),
    ) -> usize {
        let mut produced = 0;
        while produced < frames {
            let exhausted = match &self.current {
                Some(chunk) => self.cursor >= chunk.frames,
                None => true,
            };
            if exhausted {
                if let Some(done) = self.current.take() {
                    let _ = self.recycle.push(done);
                }
                match self.chunks.pop() {
                    Ok(chunk) => {
                        self.current = Some(chunk);
                        self.cursor = 0;
                    }
                    Err(_) => break, // underrun: nothing queued
                }
            }
            let chunk = match &self.current {
                Some(chunk) if self.cursor < chunk.frames => chunk,
                _ => break, // a freshly popped empty chunk; treat as underrun
            };
            let channels = self.channels.min(out_channels);
            let base = self.cursor * self.channels;
            for c in 0..channels {
                emit(produced, c, chunk.data[base + c]);
            }
            self.cursor += 1;
            produced += 1;
        }
        produced
    }
}

/// The off-RT producer side of a stream, driven by a feeder to keep the queue full.
pub struct StreamProducer {
    /// Filled chunks sent to the RT side.
    chunks: Producer<Chunk>,
    /// Emptied chunks returned by the RT side.
    recycle: Consumer<Chunk>,
    /// Empty chunks not yet filled (starts holding the whole pool).
    spare: Vec<Chunk>,
    channels: usize,
}

impl StreamProducer {
    /// The stream's channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Take an empty chunk to fill, from the spare pool or one the RT side recycled. `None` when all
    /// chunks are in flight (the queue is full); try again after the RT side consumes some.
    pub fn take_empty(&mut self) -> Option<Chunk> {
        if let Some(chunk) = self.spare.pop() {
            return Some(chunk);
        }
        if let Ok(mut chunk) = self.recycle.pop() {
            chunk.frames = 0;
            return Some(chunk);
        }
        None
    }

    /// Return an unused empty chunk to the pool (e.g. the stream ended before filling it).
    pub fn return_empty(&mut self, chunk: Chunk) {
        self.spare.push(chunk);
    }

    /// Push a filled chunk to the RT side. Returns it back (as `Err`) if the queue is momentarily
    /// full - keep it and retry later.
    pub fn push(&mut self, chunk: Chunk) -> Result<(), Chunk> {
        self.chunks
            .push(chunk)
            .map_err(|PushError::Full(chunk)| chunk)
    }
}

/// Create a cued stream: the RT [`StreamPlayback`] (to install in the buffer table) and the off-RT
/// [`StreamProducer`] (handed to a feeder). Allocates `num_chunks` chunks of `chunk_frames` frames.
pub fn cue(
    channels: usize,
    sample_rate: f64,
    chunk_frames: usize,
    num_chunks: usize,
) -> (Box<StreamPlayback>, StreamProducer) {
    let channels = channels.max(1);
    let capacity = chunk_frames.max(1);
    let count = num_chunks.max(2);
    let (chunks_tx, chunks_rx) = RingBuffer::<Chunk>::new(count + 1);
    let (recycle_tx, recycle_rx) = RingBuffer::<Chunk>::new(count + 1);
    let spare = (0..count).map(|_| Chunk::new(capacity, channels)).collect();
    let playback = Box::new(StreamPlayback {
        chunks: chunks_rx,
        recycle: recycle_tx,
        current: None,
        cursor: 0,
        channels,
        sample_rate,
    });
    let producer = StreamProducer {
        chunks: chunks_tx,
        recycle: recycle_rx,
        spare,
        channels,
    };
    (playback, producer)
}
