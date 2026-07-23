//! Sample buffers - plyphon's port of scsynth's `SndBuf`.
//!
//! A [`Buffer`] is pure in-memory sample data: interleaved `f32` samples plus a frame count, channel
//! count, and sample rate. The engine never reads or decodes sound files; a buffer is built off the
//! audio thread (allocating, possibly loading from storage - a host concern) and then installed into
//! the engine's buffer table with `Controller::buffer_set` over the command
//! ring, exactly like a synth. The audio thread only ever reads a finished `Buffer`; units such as
//! `PlayBuf` reach it (read-only) through the unit `io` free fns.
//!
//! A table slot can instead hold a disk-streaming endpoint (see [`crate::stream`]), read by `DiskIn`.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::stream::{StreamPlayback, StreamRecording};

/// How a buffer holding an FFT-chain frame interprets its packed bins: Cartesian (`Complex`, re/im
/// pairs) or polar (`Polar`, mag/phase pairs). scsynth's `SndBuf::coord`. Tracked so chained `PV_*`
/// units convert idempotently (a polar op only converts a complex frame once, and vice versa). It is
/// irrelevant for an ordinary sample buffer, which is left at the default [`Complex`](Self::Complex).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum SpectrumCoord {
    /// Cartesian bins: `[dc, nyq, re0, im0, re1, im1, ...]`. The form an `FFT` writes.
    #[default]
    Complex,
    /// Polar bins: `[dc, nyq, mag0, phase0, mag1, phase1, ...]`.
    Polar,
}

impl SpectrumCoord {
    /// Encode as a small integer tag, so storage that must be `Pod` (a graph-local buffer's header
    /// word) can carry the coordinate form.
    pub fn to_tag(self) -> u32 {
        match self {
            SpectrumCoord::Complex => 0,
            SpectrumCoord::Polar => 1,
        }
    }

    /// Decode a tag produced by [`SpectrumCoord::to_tag`] (any tag other than `1` reads as
    /// [`Complex`](Self::Complex), the zeroed-storage default).
    pub fn from_tag(tag: u32) -> SpectrumCoord {
        if tag == 1 {
            SpectrumCoord::Polar
        } else {
            SpectrumCoord::Complex
        }
    }
}

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
    /// The bin interpretation when this buffer carries an FFT-chain frame (default for sample data).
    coord: SpectrumCoord,
}

impl Buffer {
    /// A zeroed buffer of `num_frames` frames and `num_channels` channels (scsynth's `/b_alloc`).
    pub fn zeroed(num_frames: usize, num_channels: usize, sample_rate: f64) -> Self {
        Buffer {
            data: vec![0.0; num_frames * num_channels].into_boxed_slice(),
            num_frames,
            num_channels,
            sample_rate,
            coord: SpectrumCoord::Complex,
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
            coord: SpectrumCoord::Complex,
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

    /// The buffer's own sample rate in Hz (so playback units can rate-correct against the engine's).
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// All samples, interleaved (frame-major).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// All samples, interleaved, mutably - for a unit that rewrites a whole buffer in one pass from
    /// the audio thread (e.g. `FFT` writing a packed spectrum). RT-safe (no reallocation).
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// The bin interpretation when this buffer holds an FFT-chain frame (scsynth's `SndBuf::coord`).
    pub fn coord(&self) -> SpectrumCoord {
        self.coord
    }

    /// Record this buffer's bin interpretation (an `FFT` write or a `PV_*` conversion sets it).
    pub fn set_coord(&mut self, coord: SpectrumCoord) {
        self.coord = coord;
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

    /// Overwrite the sample at flat (interleaved) index `index`. scsynth's `/b_set`/`/b_setn` address
    /// the interleaved sample array directly, so `index` is `frame * num_channels + channel`. No-op if
    /// out of range, RT-safe.
    pub fn set_flat(&mut self, index: usize, value: f32) {
        if let Some(slot) = self.data.get_mut(index) {
            *slot = value;
        }
    }

    /// Overwrite the buffer's sample rate (scsynth's `/b_setSampleRate`). Metadata only - playback
    /// units rate-correct against it.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
    }

    /// Zero every sample (scsynth's `/b_zero`).
    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }

    /// Copy `count` samples within this buffer from flat `src_start` to flat `dst_start`, overlap-safe
    /// and clamped to the buffer (`/b_gen "copy"` with the same source and destination). RT-safe.
    pub fn copy_within(&mut self, dst_start: usize, src_start: usize, count: usize) {
        let len = self.data.len();
        let count = count
            .min(len.saturating_sub(src_start))
            .min(len.saturating_sub(dst_start));
        if count == 0 {
            return;
        }
        self.data
            .copy_within(src_start..src_start + count, dst_start);
    }

    /// Copy `count` samples from `src` (flat `src_start`) into this buffer at flat `dst_start`,
    /// clamped to both buffers (`/b_gen "copy"` across buffers). RT-safe.
    pub fn copy_from(&mut self, src: &Buffer, dst_start: usize, src_start: usize, count: usize) {
        let count = count
            .min(self.data.len().saturating_sub(dst_start))
            .min(src.data.len().saturating_sub(src_start));
        if count == 0 {
            return;
        }
        self.data[dst_start..dst_start + count]
            .copy_from_slice(&src.data[src_start..src_start + count]);
    }

    /// A read-only [`BufView`] of this buffer.
    pub fn view(&self) -> BufView<'_> {
        BufView {
            data: &self.data,
            num_frames: self.num_frames,
            num_channels: self.num_channels,
            sample_rate: self.sample_rate,
            coord: self.coord,
        }
    }

    /// A mutable [`BufViewMut`] of this buffer (splitting the sample and coord borrows so the view
    /// can hand both out).
    pub fn view_mut(&mut self) -> BufViewMut<'_> {
        BufViewMut {
            data: &mut self.data,
            num_frames: self.num_frames,
            num_channels: self.num_channels,
            sample_rate: self.sample_rate,
            coord: CoordSlot::Enum(&mut self.coord),
        }
    }
}

/// A read-only borrowed view of buffer-shaped sample storage - the uniform shape every buffer
/// consumer (`PlayBuf`, `BufRd`, the `BufInfo` units, ...) reads through, whether the samples live in
/// a table-owned [`Buffer`] or in a graph-local buffer span (a `LocalBuf`). `Copy`, so it passes by
/// value like the `&Buffer` it replaces; its accessors mirror [`Buffer`]'s read API.
#[derive(Copy, Clone)]
pub struct BufView<'a> {
    data: &'a [f32],
    num_frames: usize,
    num_channels: usize,
    sample_rate: f64,
    coord: SpectrumCoord,
}

impl<'a> BufView<'a> {
    /// View raw interleaved sample storage as a buffer of the given shape (how a graph-local buffer
    /// presents itself). `data` must hold `num_frames * num_channels` samples, frame-major.
    pub fn from_parts(
        data: &'a [f32],
        num_frames: usize,
        num_channels: usize,
        sample_rate: f64,
        coord: SpectrumCoord,
    ) -> Self {
        BufView {
            data,
            num_frames,
            num_channels,
            sample_rate,
            coord,
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

    /// The buffer's own sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// All samples, interleaved (frame-major). Returns the underlying storage's lifetime (not the
    /// view's), so the slice can outlive a temporary view.
    pub fn data(self) -> &'a [f32] {
        self.data
    }

    /// The bin interpretation when this buffer holds an FFT-chain frame.
    pub fn coord(&self) -> SpectrumCoord {
        self.coord
    }

    /// Sample at `frame`, `channel`. Returns 0.0 for an out-of-range index, so RT readers never panic.
    pub fn sample(&self, frame: usize, channel: usize) -> f32 {
        if frame >= self.num_frames || channel >= self.num_channels {
            return 0.0;
        }
        self.data[frame * self.num_channels + channel]
    }
}

/// Where a [`BufViewMut`]'s coordinate flag lives: the [`SpectrumCoord`] field of a table-owned
/// [`Buffer`], or the `u32` tag word of a graph-local buffer (which must be `Pod` storage).
enum CoordSlot<'a> {
    /// A table-owned buffer's coord field.
    Enum(&'a mut SpectrumCoord),
    /// A graph-local buffer's coord tag word (see [`SpectrumCoord::to_tag`]).
    Tag(&'a mut u32),
}

/// The mutable counterpart of [`BufView`] - what the writing buffer consumers (`BufWr`, `RecordBuf`,
/// `FFT` and the `PV_*` chain, ...) work through, whether the samples live in a table-owned
/// [`Buffer`] or in a graph-local buffer span. Its accessors mirror [`Buffer`]'s API.
pub struct BufViewMut<'a> {
    data: &'a mut [f32],
    num_frames: usize,
    num_channels: usize,
    sample_rate: f64,
    coord: CoordSlot<'a>,
}

impl<'a> BufViewMut<'a> {
    /// View raw interleaved sample storage, with its coordinate flag in a separate `u32` tag word
    /// (how a graph-local buffer presents itself). `data` must hold `num_frames * num_channels`
    /// samples, frame-major.
    pub fn from_tagged_parts(
        data: &'a mut [f32],
        num_frames: usize,
        num_channels: usize,
        sample_rate: f64,
        coord_tag: &'a mut u32,
    ) -> Self {
        BufViewMut {
            data,
            num_frames,
            num_channels,
            sample_rate,
            coord: CoordSlot::Tag(coord_tag),
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

    /// The buffer's own sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// All samples, interleaved (frame-major).
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// All samples, interleaved, mutably.
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }

    /// Consume the view, returning the samples at the underlying storage's lifetime - for a consumer
    /// (a buffer-backed delay) that resolves the view once and then works on the bare slice.
    pub fn into_data(self) -> &'a mut [f32] {
        self.data
    }

    /// The bin interpretation when this buffer holds an FFT-chain frame.
    pub fn coord(&self) -> SpectrumCoord {
        match &self.coord {
            CoordSlot::Enum(coord) => **coord,
            CoordSlot::Tag(tag) => SpectrumCoord::from_tag(**tag),
        }
    }

    /// Record this buffer's bin interpretation (an `FFT` write or a `PV_*` conversion sets it).
    pub fn set_coord(&mut self, coord: SpectrumCoord) {
        match &mut self.coord {
            CoordSlot::Enum(slot) => **slot = coord,
            CoordSlot::Tag(tag) => **tag = coord.to_tag(),
        }
    }

    /// Sample at `frame`, `channel`. Returns 0.0 for an out-of-range index, so RT readers never panic.
    pub fn sample(&self, frame: usize, channel: usize) -> f32 {
        if frame >= self.num_frames || channel >= self.num_channels {
            return 0.0;
        }
        self.data[frame * self.num_channels + channel]
    }

    /// Overwrite the sample at flat (interleaved) index `index` (`frame * num_channels + channel`).
    /// No-op if out of range, RT-safe.
    pub fn set_flat(&mut self, index: usize, value: f32) {
        if let Some(slot) = self.data.get_mut(index) {
            *slot = value;
        }
    }
}

/// Wrap or clamp a phase into `[0, hi]` - scsynth's `sc_loop`. Returns the resolved phase and whether
/// an end was hit with looping off (which a calc-rate buffer unit uses to mark itself done). When
/// looping, the phase wraps modulo `hi`; when not, it clamps to `hi` (high) or `0` (low). Shared by
/// the buffer units (`BufWr`, `Dbufrd`, `Dbufwr`); demand-rate callers ignore the returned flag.
pub fn sc_loop(mut x: f64, hi: f64, looping: bool) -> (f64, bool) {
    if x >= hi {
        if !looping {
            return (hi, true);
        }
        x -= hi;
        if x < hi {
            return (x, false);
        }
    } else if x < 0.0 {
        if !looping {
            return (0.0, true);
        }
        x += hi;
        if x >= 0.0 {
            return (x, false);
        }
    } else {
        return (x, false);
    }
    (x - hi * crate::math::floor(x / hi), false)
}

/// One slot in the [`BufferTable`]: empty, a flat in-memory buffer, a disk-streaming playback
/// endpoint, or a disk-streaming recording endpoint.
pub enum BufferSlot {
    /// No buffer installed.
    Empty,
    /// An in-memory buffer (read by `PlayBuf`).
    Loaded(Box<Buffer>),
    /// A streaming playback endpoint (read by `DiskIn`).
    Stream(Box<StreamPlayback>),
    /// A streaming recording endpoint (written by `DiskOut`).
    Recording(Box<StreamRecording>),
}

/// The engine's fixed-capacity table of buffers, indexed by buffer number.
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

    /// The number of buffer slots (scsynth's `mNumSndBufs`), i.e. the table's fixed capacity. This is
    /// what `NumBuffers` reports - the slot count, not how many are currently loaded.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// The flat buffer at `index`, or `None` if the slot is empty, a stream, or out of range.
    /// RT-safe (no panic).
    pub fn get(&self, index: usize) -> Option<&Buffer> {
        match self.slots.get(index) {
            Some(BufferSlot::Loaded(buffer)) => Some(buffer),
            _ => None,
        }
    }

    /// The flat buffer at `index`, mutably (for in-place sample writes - `/b_set`/`/b_setn`/
    /// `/b_fill`/`/b_setSampleRate`), or `None` if the slot is empty, a stream, or out of range.
    /// RT-safe (no panic).
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Buffer> {
        match self.slots.get_mut(index) {
            Some(BufferSlot::Loaded(buffer)) => Some(buffer),
            _ => None,
        }
    }

    /// Buffer `a` mutably and buffer `b` immutably, as disjoint borrows - for a two-buffer spectral op
    /// (e.g. `PV_MagMul` reading `b` while rewriting `a`). `None` unless `a != b` and both are loaded,
    /// in-range buffers. RT-safe (no panic); like [`copy_region`](Self::copy_region) it uses
    /// `split_at_mut` to hand out the two slots without `unsafe`.
    pub fn pair_mut(&mut self, a: usize, b: usize) -> Option<(&mut Buffer, &Buffer)> {
        if a == b {
            return None;
        }
        let hi = a.max(b);
        if hi >= self.slots.len() {
            return None;
        }
        let (left, right) = self.slots.split_at_mut(hi);
        let (a_slot, b_slot): (&mut BufferSlot, &BufferSlot) = if a < b {
            (&mut left[a], &right[0])
        } else {
            (&mut right[0], &left[b])
        };
        match (a_slot, b_slot) {
            (BufferSlot::Loaded(a_buf), BufferSlot::Loaded(b_buf)) => Some((a_buf, b_buf)),
            _ => None,
        }
    }

    /// Copy `count` samples from buffer `src` (flat `src_start`) into buffer `dst` (flat `dst_start`),
    /// overlap-safe when `dst == src` (`/b_gen "copy"`). No-op for empty/stream/out-of-range slots.
    /// RT-safe.
    pub fn copy_region(
        &mut self,
        dst: usize,
        dst_start: usize,
        src: usize,
        src_start: usize,
        count: usize,
    ) {
        if dst == src {
            if let Some(BufferSlot::Loaded(buffer)) = self.slots.get_mut(dst) {
                buffer.copy_within(dst_start, src_start, count);
            }
            return;
        }
        let hi = dst.max(src);
        if hi >= self.slots.len() {
            return;
        }
        // `split_at_mut` hands out the two slots as disjoint `&mut`s without `unsafe`.
        let (left, right) = self.slots.split_at_mut(hi);
        let (dst_slot, src_slot) = if dst < src {
            (&mut left[dst], &mut right[0])
        } else {
            (&mut right[0], &mut left[src])
        };
        if let (BufferSlot::Loaded(dst_buf), BufferSlot::Loaded(src_buf)) = (dst_slot, src_slot) {
            dst_buf.copy_from(src_buf, dst_start, src_start, count);
        }
    }

    /// The streaming endpoint at `index`, mutably (for `DiskIn` to pull chunks), or `None` if the
    /// slot is empty, a flat buffer, or out of range. RT-safe (no panic).
    pub fn stream_mut(&mut self, index: usize) -> Option<&mut StreamPlayback> {
        match self.slots.get_mut(index) {
            Some(BufferSlot::Stream(stream)) => Some(stream),
            _ => None,
        }
    }

    /// The recording endpoint at `index`, mutably (for `DiskOut` to push chunks), or `None` if the
    /// slot is empty, a flat buffer, a playback stream, or out of range. RT-safe (no panic).
    pub fn recording_mut(&mut self, index: usize) -> Option<&mut StreamRecording> {
        match self.slots.get_mut(index) {
            Some(BufferSlot::Recording(recording)) => Some(recording),
            _ => None,
        }
    }

    /// Install flat `buffer` at `index`, returning the slot it replaced (or `buffer` itself if
    /// `index` is out of range) so the caller can drop it off the audio thread.
    pub fn set(&mut self, index: usize, buffer: Box<Buffer>) -> Option<BufferSlot> {
        self.replace(index, BufferSlot::Loaded(buffer))
    }

    /// Install a streaming endpoint at `index` (scsynth's `Buffer.cueSoundFile`), returning the
    /// replaced slot for off-audio-thread dropping.
    pub fn cue(&mut self, index: usize, stream: Box<StreamPlayback>) -> Option<BufferSlot> {
        self.replace(index, BufferSlot::Stream(stream))
    }

    /// Install a recording endpoint at `index` (for `DiskOut`), returning the replaced slot for
    /// off-audio-thread dropping.
    pub fn cue_recording(
        &mut self,
        index: usize,
        recording: Box<StreamRecording>,
    ) -> Option<BufferSlot> {
        self.replace(index, BufferSlot::Recording(recording))
    }

    /// Empty `index`, returning the slot it held (if any) for off-audio-thread dropping.
    pub fn free(&mut self, index: usize) -> Option<BufferSlot> {
        self.replace(index, BufferSlot::Empty)
    }

    /// Swap `slot` into `index`, returning the displaced slot to be dropped off the audio thread
    /// (an `Empty` displacement is reported as `None`; an out-of-range index returns `slot` itself).
    fn replace(&mut self, index: usize, slot: BufferSlot) -> Option<BufferSlot> {
        match self.slots.get_mut(index) {
            Some(existing) => match core::mem::replace(existing, slot) {
                BufferSlot::Empty => None,
                old => Some(old),
            },
            None => Some(slot),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_flat_writes_by_interleaved_index() {
        let mut buf = Buffer::zeroed(3, 2, 48_000.0); // 3 frames x 2 channels = 6 samples
        buf.set_flat(0, 1.0); // frame 0, channel 0
        buf.set_flat(3, 2.0); // frame 1, channel 1
        assert_eq!(buf.sample(0, 0), 1.0);
        assert_eq!(buf.sample(1, 1), 2.0);
        assert_eq!(buf.data(), &[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        // Out of range is a silent no-op.
        buf.set_flat(6, 9.0);
        assert_eq!(buf.data().len(), 6);
    }

    #[test]
    fn set_sample_rate_overwrites_metadata() {
        let mut buf = Buffer::zeroed(4, 1, 48_000.0);
        buf.set_sample_rate(22_050.0);
        assert_eq!(buf.sample_rate(), 22_050.0);
    }

    #[test]
    fn get_mut_only_for_loaded_slots() {
        let mut table = BufferTable::new(2);
        assert!(table.get_mut(0).is_none()); // empty
        table.set(0, Box::new(Buffer::zeroed(2, 1, 48_000.0)));
        table.get_mut(0).expect("loaded slot").set_flat(1, 0.5);
        assert_eq!(table.get(0).expect("loaded").sample(1, 0), 0.5);
        assert!(table.get_mut(5).is_none()); // out of range
    }

    fn loaded(samples: &[f32]) -> Box<Buffer> {
        Box::new(Buffer::from_interleaved(samples.to_vec(), 1, 48_000.0))
    }

    #[test]
    fn copy_region_across_buffers() {
        let mut table = BufferTable::new(2);
        table.set(0, loaded(&[0.0; 4]));
        table.set(1, loaded(&[1.0, 2.0, 3.0, 4.0]));
        table.copy_region(0, 1, 1, 0, 2); // dst 0 @1 <- src 1 @0, 2 samples
        assert_eq!(table.get(0).unwrap().data(), &[0.0, 1.0, 2.0, 0.0]);
    }

    #[test]
    fn copy_region_within_one_buffer_is_overlap_safe() {
        let mut table = BufferTable::new(1);
        table.set(0, loaded(&[1.0, 2.0, 3.0, 4.0, 5.0]));
        table.copy_region(0, 1, 0, 0, 4); // shift [1..5) right by one, overlapping
        assert_eq!(table.get(0).unwrap().data(), &[1.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn copy_region_clamps_out_of_range() {
        let mut table = BufferTable::new(2);
        table.set(0, loaded(&[0.0, 0.0]));
        table.set(1, loaded(&[7.0, 8.0]));
        table.copy_region(0, 0, 1, 0, 99); // count clamps to 2
        assert_eq!(table.get(0).unwrap().data(), &[7.0, 8.0]);
        table.copy_region(0, 0, 5, 0, 1); // unknown source slot: no-op
        assert_eq!(table.get(0).unwrap().data(), &[7.0, 8.0]);
    }
}
