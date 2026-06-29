//! `DiskOut` - streams audio out to a cued recording buffer, plyphon's port of scsynth's `DiskOut`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// `DiskOut.ar(bufnum, channelsArray)`: streams its input channels to the disk-recording buffer
/// `bufnum`, one buffer channel per input. It copies each block into the recording stream's chunk
/// queue (drained off the audio thread by a sink); a full queue drops audio (a bounded overrun), and
/// no recording cued at `bufnum` discards the input. Like scsynth it has one output: a running count
/// of frames written so far (for the client to track recording progress).
///
/// Inputs: `[bufnum, ch0, ch1, …]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DiskOut {
    /// Number of recorded channels (the inputs after `bufnum`).
    num_channels: u32,
    /// Total frames written so far (scsynth's `m_framewritten`), the value output each sample.
    frames_written: u32,
}

impl DiskOut {
    const BUFNUM: usize = 0;
    /// First recorded-channel input index.
    const FIRST_CHANNEL: usize = 1;
}

impl Unit for DiskOut {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`, so we can also take `&mut` recording.
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;
        let num_channels = self.num_channels as usize;
        let block = ctx.audio.block_size;

        if let Some(rec) = unit::recording_at_mut(ctx.buffers, bufnum) {
            rec.write(block, num_channels, |frame, ch| {
                sample_channel(&ins, Self::FIRST_CHANNEL + ch, frame)
            });
        }

        // One output: the running frame count (scsynth's `out[j] = framew++`), incremented per
        // sample regardless of overruns, persisted across blocks.
        let mut written = self.frames_written;
        for slot in ctx.outs.audio(0).iter_mut() {
            *slot = written as f32;
            written = written.wrapping_add(1);
        }
        self.frames_written = written;
        DoneAction::Nothing
    }
}

/// Sample channel input `i` at within-block index `k` (per sample at audio rate; the single value
/// broadcast otherwise). `0` if `i` is past the unit's inputs.
fn sample_channel(ins: &Inputs<'_>, i: usize, k: usize) -> f32 {
    if i >= ins.len() {
        0.0
    } else if ins.rate(i) == Rate::Audio {
        ins.audio(i)[k]
    } else {
        ins.control(i)
    }
}

/// Constructor for [`DiskOut`]. The channel count is the inputs after `bufnum`.
pub struct DiskOutCtor;

impl UnitDef for DiskOutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        if count <= DiskOut::FIRST_CHANNEL {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DiskOut {
            num_channels: (count - DiskOut::FIRST_CHANNEL) as u32,
            frames_written: 0,
        }))
    }
}
