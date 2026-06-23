//! `Io` - the handle UGens use to reach the World's shared signal and sample storage.
//!
//! A UGen has private wires for its own signals, but `In`/`Out`/`PlayBuf`/`DiskIn` also need the
//! World's shared storage: the audio and control [`buses`](crate::bus) and the
//! [`buffer table`](crate::buffer). Rather than hand a UGen raw `&mut` access to those (which would
//! let it resize a bus or swap a buffer - operations that belong to the `World`/`Controller`), the
//! synth process loop lends it an [`Io`]: a narrow, audited surface of exactly the per-channel and
//! per-buffer operations a UGen legitimately performs. `Io` also owns the block counter, so a UGen
//! cannot mis-stamp a bus write.
//!
//! This bundles what would otherwise be two `&mut` arguments into one and keeps the mutation surface
//! explicit - plyphon's safe-Rust answer to scsynth's "UGens mutate everything through `mWorld`".

use crate::buffer::{Buffer, BufferTable};
use crate::bus::Buses;
use crate::stream::StreamPlayback;

/// The borrowed, restricted view of the World's shared storage handed to [`Ugen::process`].
///
/// [`Ugen::process`]: crate::ugen::Ugen::process
pub struct Io<'a> {
    buses: &'a mut Buses,
    buffers: &'a mut BufferTable,
    buf_counter: u64,
}

impl<'a> Io<'a> {
    /// Lend the World's storage for one block. Called by the synth process loop.
    pub(crate) fn new(
        buses: &'a mut Buses,
        buffers: &'a mut BufferTable,
        buf_counter: u64,
    ) -> Self {
        Io {
            buses,
            buffers,
            buf_counter,
        }
    }

    /// Audio bus channel `ch` for this block (an empty slice if `ch` is out of range), for `In.ar`.
    pub fn audio_in(&self, ch: usize) -> &[f32] {
        let audio = self.buses.audio();
        if ch < audio.num_channels() {
            audio.channel(ch)
        } else {
            &[]
        }
    }

    /// Accumulate `src` into audio bus channel `ch` for this block (`Out.ar`). Out-of-range is a
    /// no-op.
    pub fn write_audio(&mut self, ch: usize, src: &[f32]) {
        self.buses
            .audio_mut()
            .write_accumulate(ch, self.buf_counter, src);
    }

    /// Control bus channel `ch`'s current value (0.0 if out of range), for `In.kr`.
    pub fn control_in(&self, ch: usize) -> f32 {
        self.buses.control().read(ch)
    }

    /// Accumulate `value` into control bus channel `ch` for this block (`Out.kr`). Out-of-range is a
    /// no-op.
    pub fn write_control(&mut self, ch: usize, value: f32) {
        self.buses
            .control_mut()
            .write_accumulate(ch, self.buf_counter, value);
    }

    /// The (flat, in-memory) buffer at `index`, if one is installed, for `PlayBuf`.
    pub fn buffer(&self, index: usize) -> Option<&Buffer> {
        self.buffers.get(index)
    }

    /// The streaming endpoint at `index`, mutably (to pull chunks), for `DiskIn`.
    pub fn stream_mut(&mut self, index: usize) -> Option<&mut StreamPlayback> {
        self.buffers.stream_mut(index)
    }
}
