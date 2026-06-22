//! Audio buses - the shared signal storage that `Out`/`In` UGens write to and read from.
//!
//! Channels are stored flat (channel-major): channel `c` occupies `data[c*bs .. (c+1)*bs]`. Each
//! channel tracks the `buf_counter` it was last written in, so several synths can sum into one
//! channel within a block (scsynth's `Out` "touched" accumulate-vs-copy behaviour).

/// A bank of audio-rate bus channels.
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

    /// Write `src` into channel `ch` for block `buf_counter`.
    ///
    /// If the channel was already written this block it accumulates (sums); otherwise it overwrites
    /// and marks the channel touched. `src` shorter than a block leaves the remainder zeroed on a
    /// fresh write, mirroring scsynth's `Out` semantics.
    pub fn write_accumulate(&mut self, ch: usize, buf_counter: u64, src: &[f32]) {
        let bs = self.block_size;
        let start = ch * bs;
        let dst = &mut self.data[start..start + bs];
        if self.touched[ch] == buf_counter {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d += s;
            }
        } else {
            let n = dst.len().min(src.len());
            dst[..n].copy_from_slice(&src[..n]);
            for d in dst[n..].iter_mut() {
                *d = 0.0;
            }
            self.touched[ch] = buf_counter;
        }
    }

    /// Has channel `ch` been written during block `buf_counter`?
    pub fn is_touched(&self, ch: usize, buf_counter: u64) -> bool {
        self.touched[ch] == buf_counter
    }
}
