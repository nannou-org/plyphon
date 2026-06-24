//! `DiskIn` - plays audio streamed from a cued buffer, plyphon's port of scsynth's `DiskIn`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// `DiskIn.ar(numChannels, bufnum)`: plays the disk-streamed buffer `bufnum`, one output per channel,
/// at the stream's native rate. It pulls pre-filled chunks from the stream's queue (filled off the
/// audio thread); an empty queue (an underrun, or no stream cued at `bufnum`) plays silence.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DiskIn {
    num_channels: u32,
}

impl DiskIn {
    const BUFNUM: usize = 0;
}

impl Unit for DiskIn {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_channels = self.num_channels as usize;
        let bufnum = ctx.ins.control(Self::BUFNUM).max(0.0) as usize;
        let block = ctx.outs.audio(0).len();
        let produced = match unit::stream_at_mut(ctx.buffers, bufnum) {
            Some(stream) => stream.read(block, num_channels, |frame, ch, value| {
                ctx.outs.audio(ch)[frame] = value;
            }),
            None => 0,
        };
        // An underrun (or the no-stream case) plays silence for the rest of the block.
        for ch in 0..num_channels {
            ctx.outs.audio(ch)[produced..block].fill(0.0);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DiskIn`]. The output count (the stream's channel count) is fixed here.
pub struct DiskInCtor;

impl UnitDef for DiskInCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(DiskIn {
            num_channels: ctx.num_outputs as u32,
        }))
    }
}
