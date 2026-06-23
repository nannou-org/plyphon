//! `DiskIn` - plays audio streamed from a cued buffer, plyphon's port of scsynth's `DiskIn`.

use crate::error::BuildError;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{self, DoneAction, ProcessCtx, Ugen};

/// `DiskIn.ar(numChannels, bufnum)`: plays the disk-streamed buffer `bufnum`, one output per channel,
/// at the stream's native rate. It pulls pre-filled chunks from the stream's queue (filled off the
/// audio thread); an empty queue (an underrun, or no stream cued at `bufnum`) plays silence.
pub struct DiskIn {
    num_channels: usize,
}

impl DiskIn {
    const BUFNUM: usize = 0;
}

impl Ugen for DiskIn {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bufnum = ctx.ins.control(Self::BUFNUM).max(0.0) as usize;
        let block = ctx.outs.audio(0).len();
        let produced = match ugen::stream_at_mut(ctx.buffers, bufnum) {
            Some(stream) => stream.read(block, self.num_channels, |frame, ch, value| {
                ctx.outs.audio(ch)[frame] = value;
            }),
            None => 0,
        };
        // An underrun (or the no-stream case) plays silence for the rest of the block.
        for ch in 0..self.num_channels {
            ctx.outs.audio(ch)[produced..block].fill(0.0);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DiskIn`]. The output count (the stream's channel count) is fixed here.
pub struct DiskInCtor;

impl UgenCtor for DiskInCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        Ok(Box::new(DiskIn {
            num_channels: ctx.num_outputs,
        }))
    }
}
