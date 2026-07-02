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

use alloc::boxed::Box;
use alloc::vec::Vec;

use rtrb::{Consumer, Producer, PushError, RingBuffer};

use crate::interp::cubicinterp;

/// Frames retained for `VDiskIn`'s 4-point cubic resampling window (`[a, b, c, d]`, the read position
/// between `b` and `c`).
const RESAMPLE_WINDOW: usize = 4;

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

    /// Valid frames currently held.
    pub fn frames(&self) -> usize {
        self.frames
    }

    /// The filled prefix (`frames * channels` interleaved samples), for a drainer to write out.
    pub fn filled_samples(&self) -> &[f32] {
        &self.data[..self.frames * self.channels]
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
    /// `VDiskIn` resampling state: the fractional read position between the window's `b` and `c`
    /// frames.
    phase: f64,
    /// The 4-frame cubic-interpolation window (`RESAMPLE_WINDOW * channels`), only used by
    /// [`read_resampled`](StreamPlayback::read_resampled).
    window: Vec<f32>,
    /// Whether the resampling window has been primed with the first source frames.
    primed: bool,
}

/// Copy the next source frame into `out` (length `channels`), advancing the chunk cursor and recycling
/// each exhausted chunk to the feeder. Returns `false` on underrun (nothing queued). Field args (not
/// `&mut self`) so the caller can also borrow the disjoint window slice.
fn next_source_frame(
    current: &mut Option<Chunk>,
    cursor: &mut usize,
    chunks: &mut Consumer<Chunk>,
    recycle: &mut Producer<Chunk>,
    channels: usize,
    out: &mut [f32],
) -> bool {
    let exhausted = match current {
        Some(chunk) => *cursor >= chunk.frames,
        None => true,
    };
    if exhausted {
        if let Some(done) = current.take() {
            let _ = recycle.push(done);
        }
        match chunks.pop() {
            Ok(chunk) => {
                *current = Some(chunk);
                *cursor = 0;
            }
            Err(_) => return false,
        }
    }
    let chunk = match current {
        Some(chunk) if *cursor < chunk.frames => chunk,
        _ => return false,
    };
    let base = *cursor * channels;
    for (c, o) in out.iter_mut().take(channels).enumerate() {
        *o = chunk.data[base + c];
    }
    *cursor += 1;
    true
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

    /// Pull up to `frames` output frames, reading the stream at fractional `rate` (frames per output
    /// frame) with 4-point cubic interpolation, calling `emit(frame, channel, sample)` per produced
    /// sample; returns how many frames were produced (fewer on underrun). `VDiskIn`'s resampled read.
    ///
    /// The stream is forward-only (a chunk queue), so a 4-frame window `[a, b, c, d]` is retained and
    /// slid forward one source frame at a time as the phase crosses integer boundaries - the cubic
    /// analogue of [`read`](StreamPlayback::read)'s 1:1 drain. `rate` is clamped to `>= 0` (no reverse
    /// play - the stream cannot rewind). Looping is a host concern (a looping `BufferStream` feeds a
    /// queue that never ends); this drops to silence at end-of-stream like `read`.
    pub fn read_resampled(
        &mut self,
        frames: usize,
        out_channels: usize,
        rate: f64,
        mut emit: impl FnMut(usize, usize, f32),
    ) -> usize {
        let ch = self.channels;
        let rate = rate.max(0.0);
        if !self.primed {
            // `a` starts at silence (before the stream); `b`/`c`/`d` are the first three frames. If the
            // very first frame is not yet queued, stay unprimed and play silence until it arrives.
            self.window[..ch].fill(0.0);
            if !next_source_frame(
                &mut self.current,
                &mut self.cursor,
                &mut self.chunks,
                &mut self.recycle,
                ch,
                &mut self.window[ch..2 * ch],
            ) {
                return 0;
            }
            for slot in 2..RESAMPLE_WINDOW {
                let (head, tail) = self.window.split_at_mut(slot * ch);
                if !next_source_frame(
                    &mut self.current,
                    &mut self.cursor,
                    &mut self.chunks,
                    &mut self.recycle,
                    ch,
                    &mut tail[..ch],
                ) {
                    tail[..ch].fill(0.0);
                }
                let _ = head;
            }
            self.primed = true;
            self.phase = 0.0;
        }

        let outc = ch.min(out_channels);
        let mut produced = 0;
        while produced < frames {
            let frac = self.phase as f32;
            for c in 0..outc {
                let (a, b, cc, d) = (
                    self.window[c],
                    self.window[ch + c],
                    self.window[2 * ch + c],
                    self.window[3 * ch + c],
                );
                emit(produced, c, cubicinterp(frac, a, b, cc, d));
            }
            produced += 1;
            self.phase += rate;
            // Slide the window forward one source frame per integer the phase crossed.
            while self.phase >= 1.0 {
                self.phase -= 1.0;
                self.window.copy_within(ch..RESAMPLE_WINDOW * ch, 0);
                let last = (RESAMPLE_WINDOW - 1) * ch;
                let (head, tail) = self.window.split_at_mut(last);
                let _ = head;
                if !next_source_frame(
                    &mut self.current,
                    &mut self.cursor,
                    &mut self.chunks,
                    &mut self.recycle,
                    ch,
                    &mut tail[..ch],
                ) {
                    return produced; // underrun: the caller silences the rest
                }
            }
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
        phase: 0.0,
        window: vec![0.0; RESAMPLE_WINDOW * channels],
        primed: false,
    });
    let producer = StreamProducer {
        chunks: chunks_tx,
        recycle: recycle_rx,
        spare,
        channels,
    };
    (playback, producer)
}

/// The RT-side producer of a recording stream: lives in the buffer table, filled by `DiskOut`. The
/// exact inverse of [`StreamPlayback`] - the audio thread copies its block into a [`Chunk`] and hands
/// ownership of whole chunks to the off-RT [`StreamConsumer`] over a lock-free ring, so no buffer
/// memory is ever shared mutably across threads (scsynth instead shares a raw `float*`, which is only
/// race-free while the disk thread keeps up).
pub struct StreamRecording {
    /// Filled chunks sent off-RT.
    filled: Producer<Chunk>,
    /// Emptied chunks returning from the consumer.
    recycle: Consumer<Chunk>,
    current: Option<Chunk>,
    cursor: usize,
    channels: usize,
    sample_rate: f64,
}

impl StreamRecording {
    /// The stream's channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The stream's sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Write `frames` frames, calling `sample(frame, channel)` for each channel of each frame, and
    /// return how many frames were recorded (fewer on overrun). RT-safe (no allocation or blocking):
    /// an empty chunk is taken from the recycle ring; if none is available, or the filled ring is
    /// full, the surplus audio is dropped (a bounded overrun) - never blocking, allocating, or
    /// dropping a `Chunk` (a `Box` free on the audio thread is forbidden).
    pub fn write(
        &mut self,
        frames: usize,
        in_channels: usize,
        mut sample: impl FnMut(usize, usize) -> f32,
    ) -> usize {
        let channels = self.channels;
        let mut recorded = 0;
        while recorded < frames {
            // Ensure a chunk with room to write into.
            if self.current.is_none() {
                match self.recycle.pop() {
                    Ok(mut chunk) => {
                        chunk.frames = 0;
                        self.current = Some(chunk);
                        self.cursor = 0;
                    }
                    Err(_) => break, // overrun: no empty chunk; drop the rest of this block
                }
            }
            let cursor = self.cursor;
            let cap = {
                let chunk = self.current.as_mut().unwrap();
                let cap = chunk.data.len() / channels;
                let base = cursor * channels;
                for c in 0..channels {
                    chunk.data[base + c] = if c < in_channels {
                        sample(recorded, c)
                    } else {
                        0.0
                    };
                }
                cap
            };
            self.cursor += 1;
            recorded += 1;
            if self.cursor >= cap {
                let mut chunk = self.current.take().unwrap();
                chunk.frames = cap;
                if let Err(PushError::Full(chunk)) = self.filled.push(chunk) {
                    // The consumer is behind: keep the full chunk and overwrite it next round (its
                    // audio is lost - a bounded overrun). Never drop it (no `Box` free on the RT
                    // thread) and never block.
                    self.current = Some(chunk);
                    self.cursor = 0;
                }
            }
        }
        recorded
    }

    /// Whether the off-RT [`StreamConsumer`] has been dropped, so nothing will ever drain this
    /// recording again. A bounded copy-out polls this to abandon a recording whose host gave up (e.g.
    /// a sink that failed to open), instead of spinning forever on a recycle ring that never refills.
    pub fn is_abandoned(&self) -> bool {
        self.filled.is_abandoned()
    }

    /// Push the partially-filled current chunk to the consumer, so a recording that ends mid-chunk
    /// still delivers its final frames. Returns `true` when nothing remains to flush (no partial chunk,
    /// or it was pushed); `false` if the filled ring is momentarily full - keep the chunk and retry
    /// next block. RT-safe (no allocation, no blocking, never drops a `Chunk`).
    ///
    /// A continuous `DiskOut` never needs this (it drops the sub-chunk tail as a bounded overrun), but
    /// a *bounded* copy-out (`/b_write`) must flush to deliver every frame exactly.
    pub fn flush(&mut self) -> bool {
        if self.cursor == 0 {
            return true;
        }
        let Some(mut chunk) = self.current.take() else {
            return true;
        };
        chunk.frames = self.cursor;
        match self.filled.push(chunk) {
            Ok(()) => {
                self.cursor = 0;
                true
            }
            Err(PushError::Full(chunk)) => {
                self.current = Some(chunk);
                false
            }
        }
    }
}

/// The off-RT consumer side of a recording stream, drained by a sink. The inverse of
/// [`StreamProducer`].
pub struct StreamConsumer {
    /// Filled chunks arriving from the RT side.
    filled: Consumer<Chunk>,
    /// Emptied chunks returned to the RT side.
    recycle: Producer<Chunk>,
    channels: usize,
    sample_rate: f64,
}

impl StreamConsumer {
    /// The stream's channel count.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The stream's sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Pop the next filled chunk, if any (`None` when the recorder has not filled one yet).
    pub fn pop_filled(&mut self) -> Option<Chunk> {
        self.filled.pop().ok()
    }

    /// Return a drained chunk to the RT side as an empty (dropped if the ring is momentarily full).
    pub fn recycle(&mut self, mut chunk: Chunk) {
        chunk.frames = 0;
        let _ = self.recycle.push(chunk);
    }

    /// Whether the recording is complete: every filled chunk has been drained *and* the RT-side
    /// [`StreamRecording`] has been dropped, so no further chunk can ever arrive. A drainer polls this
    /// to know when to close its sink. (`is_abandoned` reports the dropped producer; `is_empty` that
    /// nothing remains queued.) A finite recording that is still running but momentarily drained is
    /// *not* finished - the producer is still alive.
    pub fn is_finished(&self) -> bool {
        self.filled.is_empty() && self.filled.is_abandoned()
    }
}

/// Create a recording stream: the RT [`StreamRecording`] (to install in the buffer table) and the
/// off-RT [`StreamConsumer`] (handed to a drainer). Allocates `num_chunks` chunks of `chunk_frames`
/// frames and pre-loads them onto the recycle ring so the recorder can fill them immediately.
pub fn cue_recording(
    channels: usize,
    sample_rate: f64,
    chunk_frames: usize,
    num_chunks: usize,
) -> (Box<StreamRecording>, StreamConsumer) {
    let channels = channels.max(1);
    let capacity = chunk_frames.max(1);
    let count = num_chunks.max(2);
    let (filled_tx, filled_rx) = RingBuffer::<Chunk>::new(count + 1);
    let (mut recycle_tx, recycle_rx) = RingBuffer::<Chunk>::new(count + 1);
    // The whole pool starts as empties on the recycle ring (RT-reachable), the inverse of `cue`
    // handing the spare pool to the off-RT feeder.
    for _ in 0..count {
        let _ = recycle_tx.push(Chunk::new(capacity, channels));
    }
    let recording = Box::new(StreamRecording {
        filled: filled_tx,
        recycle: recycle_rx,
        current: None,
        cursor: 0,
        channels,
        sample_rate,
    });
    let consumer = StreamConsumer {
        filled: filled_rx,
        recycle: recycle_tx,
        channels,
        sample_rate,
    };
    (recording, consumer)
}

#[cfg(test)]
mod tests {
    use super::{Chunk, cue_recording};

    /// Collect every filled chunk the consumer can pop into a flat interleaved `Vec`, recycling each.
    fn drain(consumer: &mut super::StreamConsumer) -> Vec<f32> {
        let mut out = Vec::new();
        while let Some(chunk) = consumer.pop_filled() {
            out.extend_from_slice(chunk.filled_samples());
            consumer.recycle(chunk);
        }
        out
    }

    #[test]
    fn recording_round_trips_a_ramp() {
        let (mut rec, mut consumer) = cue_recording(2, 48_000.0, 4, 3);
        // 8 frames of a known ramp: sample(frame, ch) = frame*10 + ch.
        let recorded = rec.write(8, 2, |frame, ch| (frame * 10 + ch) as f32);
        assert_eq!(recorded, 8);
        let got = drain(&mut consumer);
        let expected: Vec<f32> = (0..8)
            .flat_map(|frame| (0..2).map(move |ch| (frame * 10 + ch) as f32))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn recording_overruns_without_panic_and_recovers() {
        // Two chunks of 4 frames, never drained: the recorder fills both, then drops further audio.
        let (mut rec, mut consumer) = cue_recording(1, 48_000.0, 4, 2);
        assert_eq!(rec.write(4, 1, |_, _| 1.0), 4); // fills chunk 1, pushed
        assert_eq!(rec.write(4, 1, |_, _| 2.0), 4); // fills chunk 2, pushed
        // Both chunks are now in flight and none recycled: no empty available -> overrun, drop.
        assert_eq!(rec.write(4, 1, |_, _| 3.0), 0);
        // Drain one chunk back to the recorder; recording can proceed again (pool size is constant).
        let chunk = consumer.pop_filled().expect("a filled chunk");
        consumer.recycle(chunk);
        assert_eq!(rec.write(4, 1, |_, _| 4.0), 4);
    }

    #[test]
    fn flush_delivers_the_final_partial_chunk() {
        let (mut rec, mut consumer) = cue_recording(1, 48_000.0, 4, 3);
        // 6 frames into chunks of 4: one full chunk is pushed, a 2-frame partial is held back.
        assert_eq!(rec.write(6, 1, |frame, _| frame as f32), 6);
        assert_eq!(drain(&mut consumer), vec![0.0, 1.0, 2.0, 3.0]);
        // The partial only arrives after a flush - the difference a bounded copy-out depends on.
        assert!(rec.flush());
        assert_eq!(drain(&mut consumer), vec![4.0, 5.0]);
        // A second flush is a no-op (nothing pending).
        assert!(rec.flush());
        assert!(consumer.pop_filled().is_none());
    }

    #[test]
    fn is_finished_requires_drain_then_drop() {
        let (mut rec, mut consumer) = cue_recording(1, 48_000.0, 4, 2);
        assert!(!consumer.is_finished()); // producer alive, nothing queued
        rec.write(4, 1, |_, _| 1.0); // one full chunk pushed
        drop(rec); // RT side gone, but a chunk is still queued
        assert!(!consumer.is_finished()); // abandoned but not yet empty
        let chunk = consumer.pop_filled().expect("the queued chunk");
        consumer.recycle(chunk);
        assert!(consumer.is_finished()); // empty + abandoned
    }

    #[test]
    fn filled_samples_reflects_partial_frames() {
        let mut chunk = Chunk::new(8, 1);
        chunk.samples_mut()[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        chunk.set_frames(3);
        assert_eq!(chunk.filled_samples(), &[1.0, 2.0, 3.0]);
    }
}
