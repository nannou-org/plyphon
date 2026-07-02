//! Buses - the shared signal storage that `Out`/`In` units write to and read from.
//!
//! plyphon mirrors scsynth's two bus banks, both owned by [`Buses`] and lent to units (by `&mut`)
//! through the synth process loop:
//!
//! - an audio bus bank ([`AudioBus`]): `block_size` samples per channel, laid out as the hardware
//!   *output* channels, then the hardware *input* channels, then *private* channels for routing
//!   between synths. `In.ar`/`Out.ar` index this single channel space directly, exactly as in
//!   SuperCollider (`Out.ar(0, ..)` is the first output; `In.ar(numOutputs, ..)` reads the first
//!   hardware input).
//! - a control bus bank ([`ControlBus`]): one value per channel per control block, for `In.kr`,
//!   `Out.kr`, `/c_set`, and `/n_map` control-to-bus mapping.
//!
//! Audio channels track the `buf_counter` they were last written in, so several synths can sum into
//! one channel within a block (scsynth's `Out` "touched" accumulate-vs-copy behaviour); control
//! channels do the same for `Out.kr`.

use alloc::vec::Vec;
use core::ops::Range;

/// A bank of audio-rate bus channels.
///
/// Channels are stored flat (channel-major): channel `c` occupies `data[c*bs .. (c+1)*bs]`.
#[derive(Clone, Debug)]
pub struct AudioBus {
    /// `num_channels * block_size` samples, channel-major.
    data: Vec<f32>,
    /// Per-channel `buf_counter` of the most recent write.
    touched: Vec<u64>,
    num_channels: usize,
    block_size: usize,
}

impl AudioBus {
    /// Allocate `num_channels` channels of `block_size` samples, zeroed.
    pub fn new(num_channels: usize, block_size: usize) -> Self {
        AudioBus {
            data: vec![0.0; num_channels * block_size],
            touched: vec![0; num_channels],
            num_channels,
            block_size,
        }
    }

    /// Number of channels in the bus.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Samples per channel per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Read channel `ch`'s current block.
    pub fn channel(&self, ch: usize) -> &[f32] {
        let start = ch * self.block_size;
        &self.data[start..start + self.block_size]
    }

    /// Mutable access to channel `ch`'s current block (e.g. for the host to deposit input).
    pub fn channel_mut(&mut self, ch: usize) -> &mut [f32] {
        let start = ch * self.block_size;
        &mut self.data[start..start + self.block_size]
    }

    /// Write `src` into channel `ch` for block `buf_counter`.
    ///
    /// If the channel was already written this block it accumulates (sums); otherwise it overwrites
    /// and marks the channel touched. `src` shorter than a block leaves the remainder zeroed on a
    /// fresh write, mirroring scsynth's `Out` semantics.
    pub fn write_accumulate(&mut self, ch: usize, buf_counter: u64, src: &[f32]) {
        self.write_accumulate_decimated(ch, buf_counter, 0, src, 1);
    }

    /// Write `src` into channel `ch` at sample `offset` for block `buf_counter`, taking every
    /// `factor`-th sample of `src` - the reblock/resample boundary form. A graph running at a smaller
    /// block and/or oversampled rate writes each sub-block tick into its own slice of the
    /// World-block-wide channel, decimating its `factor`x-oversampled interior down to the World rate
    /// (scsynth's `Out_next_a_reblock`, which zeroes the channel on the first writer).
    ///
    /// The first writer of the channel this block clears the **whole** channel and marks it touched,
    /// so every later tick (and every other synth) then accumulates into its own slice over a clean
    /// zero (or a co-writer's signal). `offset == 0`, `factor == 1`, full-block `src` reduces to
    /// [`write_accumulate`](Self::write_accumulate).
    pub fn write_accumulate_decimated(
        &mut self,
        ch: usize,
        buf_counter: u64,
        offset: usize,
        src: &[f32],
        factor: usize,
    ) {
        if ch >= self.num_channels {
            return;
        }
        let factor = factor.max(1);
        let bs = self.block_size;
        // Read/clear `touched` before borrowing `data` (disjoint fields, sequenced).
        let first = self.touched[ch] != buf_counter;
        self.touched[ch] = buf_counter;
        let start = ch * bs;
        let dst = &mut self.data[start..start + bs];
        if first {
            dst.fill(0.0);
        }
        // World-rate output samples = the decimated source length, clamped into the channel slice.
        let count = (src.len() / factor).min(bs.saturating_sub(offset));
        if factor == 1 {
            // The common (non-oversampled) case: a contiguous zip the compiler can vectorize.
            for (d, &s) in dst[offset..offset + count].iter_mut().zip(src) {
                *d += s;
            }
        } else {
            for j in 0..count {
                dst[offset + j] += src[j * factor];
            }
        }
    }

    /// Overwrite channel `ch`'s samples `[offset, offset + src.len()/factor)` with `src` (decimated
    /// by `factor`) for block `buf_counter`, replacing whatever was there and marking the channel
    /// touched - scsynth's `ReplaceOut`. Unlike [`write_accumulate_decimated`](Self::write_accumulate_decimated)
    /// it neither sums nor clears the channel: each writer overwrites only its own slice. `offset == 0`,
    /// `factor == 1`, full-block `src` overwrites the whole channel.
    pub fn write_replace_decimated(
        &mut self,
        ch: usize,
        buf_counter: u64,
        offset: usize,
        src: &[f32],
        factor: usize,
    ) {
        if ch >= self.num_channels {
            return;
        }
        let factor = factor.max(1);
        let bs = self.block_size;
        self.touched[ch] = buf_counter;
        let start = ch * bs;
        let dst = &mut self.data[start..start + bs];
        let count = (src.len() / factor).min(bs.saturating_sub(offset));
        if factor == 1 {
            // The common (non-oversampled) case: a straight copy.
            dst[offset..offset + count].copy_from_slice(&src[..count]);
        } else {
            for j in 0..count {
                dst[offset + j] = src[j * factor];
            }
        }
    }

    /// Crossfade `src` (decimated by `factor`) into channel `ch`'s samples `[offset, offset + len)`
    /// for block `buf_counter`: `dst = dst*(1 - xfade) + src*xfade` - scsynth's `XOut`. The first
    /// writer of the channel this block clears the **whole** channel before its own slice (exactly
    /// as [`write_accumulate_decimated`](Self::write_accumulate_decimated) does), so a crossfade is
    /// always against this block's audio or silence - never a prior block's - including every later
    /// tick-slice under reblock. `offset == 0`, `factor == 1`, full-block `src` crossfades the whole
    /// channel.
    pub fn write_crossfade_decimated(
        &mut self,
        ch: usize,
        buf_counter: u64,
        offset: usize,
        src: &[f32],
        factor: usize,
        xfade: f32,
    ) {
        if ch >= self.num_channels {
            return;
        }
        let factor = factor.max(1);
        let bs = self.block_size;
        let first = self.touched[ch] != buf_counter;
        self.touched[ch] = buf_counter;
        let start = ch * bs;
        let dst = &mut self.data[start..start + bs];
        if first {
            dst.fill(0.0);
        }
        let count = (src.len() / factor).min(bs.saturating_sub(offset));
        if factor == 1 {
            // The common (non-oversampled) case: a contiguous zip the compiler can vectorize.
            for (d, &s) in dst[offset..offset + count].iter_mut().zip(src) {
                *d = *d * (1.0 - xfade) + s * xfade;
            }
        } else {
            for j in 0..count {
                dst[offset + j] = dst[offset + j] * (1.0 - xfade) + src[j * factor] * xfade;
            }
        }
    }

    /// Has channel `ch` been written during block `buf_counter`?
    pub fn is_touched(&self, ch: usize, buf_counter: u64) -> bool {
        self.touched[ch] == buf_counter
    }

    /// Mark every channel in `range` as written for block `buf_counter` without touching its
    /// samples (out-of-range channels are ignored). Used for the hardware input channels, whose
    /// samples the host deposits outside the `write_*` path.
    pub fn touch_range(&mut self, range: Range<usize>, buf_counter: u64) {
        for ch in range {
            if let Some(t) = self.touched.get_mut(ch) {
                *t = buf_counter;
            }
        }
    }

    /// Zero every channel in `range` that was *not* written during block `buf_counter`, so stale
    /// audio from an earlier block is not re-emitted on a channel nothing wrote to this block.
    pub fn silence_untouched_range(&mut self, range: Range<usize>, buf_counter: u64) {
        let bs = self.block_size;
        for ch in range {
            if ch < self.num_channels && self.touched[ch] != buf_counter {
                let start = ch * bs;
                for sample in &mut self.data[start..start + bs] {
                    *sample = 0.0;
                }
            }
        }
    }
}

/// A bank of control-rate bus channels: one value per channel per control block.
///
/// Like [`AudioBus`], channels track the block they were last written in, so multiple `Out.kr`
/// into one channel sum. `/c_set` overwrites a channel without marking it touched, matching
/// scsynth: a same-block `Out.kr` then overwrites it on its first (untouched) write.
#[derive(Clone, Debug)]
pub struct ControlBus {
    data: Vec<f32>,
    touched: Vec<u64>,
    num_channels: usize,
}

impl ControlBus {
    /// Allocate `num_channels` channels, zeroed.
    pub fn new(num_channels: usize) -> Self {
        ControlBus {
            data: vec![0.0; num_channels],
            touched: vec![0; num_channels],
            num_channels,
        }
    }

    /// Number of channels in the bus.
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Read channel `ch`'s current value (0.0 for an out-of-range channel).
    pub fn read(&self, ch: usize) -> f32 {
        self.data.get(ch).copied().unwrap_or(0.0)
    }

    /// Write `value` to channel `ch` for block `buf_counter`, summing if the channel was already
    /// written this block, overwriting otherwise (scsynth's `Out.kr` semantics).
    pub fn write_accumulate(&mut self, ch: usize, buf_counter: u64, value: f32) {
        if ch >= self.num_channels {
            return;
        }
        if self.touched[ch] == buf_counter {
            self.data[ch] += value;
        } else {
            self.data[ch] = value;
            self.touched[ch] = buf_counter;
        }
    }

    /// Overwrite channel `ch` with `value` for block `buf_counter`, marking it touched (scsynth's
    /// `ReplaceOut.kr`): replaces whatever an earlier same-block `Out.kr` wrote.
    pub fn write_replace(&mut self, ch: usize, buf_counter: u64, value: f32) {
        if ch < self.num_channels {
            self.data[ch] = value;
            self.touched[ch] = buf_counter;
        }
    }

    /// Crossfade `value` into channel `ch` for block `buf_counter`: `ch = ch*(1-xfade) + value*xfade`
    /// (scsynth's `XOut.kr`). The first writer of the channel this block treats the existing value as
    /// zero (so it lands `value*xfade`) and marks the channel touched.
    pub fn write_crossfade(&mut self, ch: usize, buf_counter: u64, value: f32, xfade: f32) {
        if ch >= self.num_channels {
            return;
        }
        let cur = if self.touched[ch] == buf_counter {
            self.data[ch]
        } else {
            0.0
        };
        self.data[ch] = cur * (1.0 - xfade) + value * xfade;
        self.touched[ch] = buf_counter;
    }

    /// Set channel `ch` to `value` (scsynth's `/c_set`): a persistent overwrite that does not mark
    /// the channel touched, so a same-block `Out.kr` still overwrites rather than sums onto it.
    pub fn set(&mut self, ch: usize, value: f32) {
        if let Some(slot) = self.data.get_mut(ch) {
            *slot = value;
        }
    }
}

/// The engine's bus banks: an [`AudioBus`] (output, then input, then private channels) and a
/// [`ControlBus`]. Owned by the engine and lent to units during processing.
#[derive(Clone, Debug)]
pub struct Buses {
    audio: AudioBus,
    control: ControlBus,
    output_channels: usize,
    input_channels: usize,
}

impl Buses {
    /// Allocate the bus banks. The audio bank holds `output_channels + input_channels +
    /// private_channels` channels in that order; the control bank holds `control_channels`.
    pub fn new(
        output_channels: usize,
        input_channels: usize,
        private_channels: usize,
        control_channels: usize,
        block_size: usize,
    ) -> Self {
        let audio_channels = output_channels + input_channels + private_channels;
        Buses {
            audio: AudioBus::new(audio_channels, block_size),
            control: ControlBus::new(control_channels),
            output_channels,
            input_channels,
        }
    }

    /// The audio bus bank.
    pub fn audio(&self) -> &AudioBus {
        &self.audio
    }

    /// The audio bus bank, mutably (for `Out.ar`).
    pub fn audio_mut(&mut self) -> &mut AudioBus {
        &mut self.audio
    }

    /// The control bus bank.
    pub fn control(&self) -> &ControlBus {
        &self.control
    }

    /// The control bus bank, mutably (for `Out.kr` and `/c_set`).
    pub fn control_mut(&mut self) -> &mut ControlBus {
        &mut self.control
    }

    /// Number of hardware output channels (the first audio bus channels).
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    /// Number of hardware input channels (the audio bus channels following the outputs).
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Zero any output channel not written this block, so silence is real silence rather than the
    /// previous block's audio repeated. Input and private channels persist (the host writes inputs;
    /// private channels are overwritten by their next `Out`).
    pub fn silence_untouched_outputs(&mut self, buf_counter: u64) {
        self.audio
            .silence_untouched_range(0..self.output_channels, buf_counter);
    }

    /// Mark the hardware input channels as written for block `buf_counter`, so `In.ar` reads them
    /// as live. The World calls this once per block whether or not the host supplied input: absent
    /// input is silence (zeros), which is still a valid read - matching scsynth, whose input buses
    /// are always valid for the block.
    pub fn touch_inputs(&mut self, buf_counter: u64) {
        let base = self.output_channels;
        self.audio
            .touch_range(base..base + self.input_channels, buf_counter);
    }

    /// Deposit one block of interleaved host input into the input bus region.
    ///
    /// `input_block` holds up to `block_size` frames of `in_channels`-wide interleaved samples; any
    /// frames or channels beyond what is supplied are zeroed. Channels past the engine's input count
    /// are ignored.
    pub fn write_input(&mut self, input_block: &[f32], in_channels: usize) {
        let bs = self.audio.block_size();
        let base = self.output_channels;
        let frames = input_block
            .len()
            .checked_div(in_channels)
            .unwrap_or(0)
            .min(bs);
        for c in 0..self.input_channels {
            let chan = self.audio.channel_mut(base + c);
            for (i, sample) in chan.iter_mut().enumerate() {
                *sample = if i < frames && c < in_channels {
                    input_block[i * in_channels + c]
                } else {
                    0.0
                };
            }
        }
    }
}
