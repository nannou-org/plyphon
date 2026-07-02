//! `VDiskIn` - plays a disk-streamed buffer at a variable rate, plyphon's port of scsynth's `VDiskIn`.
//!
//! Like [`DiskIn`](crate::unit::DiskIn), but reads the stream at a control `rate` (frames per output
//! sample) with 4-point cubic interpolation, so the streamed audio can be transposed on the fly. The
//! resampling lives in the transport ([`StreamPlayback::read_resampled`](plyphon_dsp::stream)); the
//! unit just forwards `rate`. `rate` is read once per block (block-constant, a documented
//! simplification of scsynth's per-sample slope).
//!
//! Looping is a host concern: `loop = 1` is served by cueing a looping `BufferStream` (a queue that
//! never ends), so the `loop` input is accepted but not acted on at the RT level. `sendID`
//! (`/done`-on-loop notifications) is accepted and ignored for now - it would need a new RT->host
//! reply path.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// `VDiskIn.ar(numChannels, bufnum, rate, loop, sendID)`: play the disk-streamed buffer `bufnum` at
/// `rate` (1 = native), one output per channel. Inputs `[bufnum, rate, loop, sendID]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VDiskIn {
    num_channels: u32,
}

impl VDiskIn {
    const BUFNUM: usize = 0;
    const RATE: usize = 1;
}

impl Unit for VDiskIn {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_channels = self.num_channels as usize;
        let bufnum = ctx.ins.control(Self::BUFNUM).max(0.0) as usize;
        let rate = ctx.ins.control(Self::RATE) as f64;
        let block = ctx.outs.audio(0).len();
        let produced = match unit::stream_at_mut(ctx.buffers, bufnum) {
            Some(stream) => stream.read_resampled(block, num_channels, rate, |frame, ch, value| {
                ctx.outs.audio(ch)[frame] = value;
            }),
            None => 0,
        };
        // An underrun (or no stream cued) plays silence for the rest of the block.
        for ch in 0..num_channels {
            ctx.outs.audio(ch)[produced..block].fill(0.0);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`VDiskIn`]. The output count (the stream's channel count) is fixed here.
pub struct VDiskInCtor;

impl UnitDef for VDiskInCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(VDiskIn {
            num_channels: ctx.num_outputs as u32,
        }))
    }
}
