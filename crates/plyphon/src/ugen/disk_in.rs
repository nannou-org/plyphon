//! `DiskIn` - plays audio streamed from a cued buffer, plyphon's port of scsynth's `DiskIn`.

use crate::error::BuildError;
use crate::io::Io;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{DoneAction, Inputs, Outputs, ProcessContext, Ugen};

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
    fn process(
        &mut self,
        _ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        io: &mut Io,
    ) -> DoneAction {
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;
        let block = outs.audio(0).len();
        let produced = match io.stream_mut(bufnum) {
            Some(stream) => stream.read(block, self.num_channels, |frame, ch, value| {
                outs.audio(ch)[frame] = value;
            }),
            None => 0,
        };
        // An underrun (or the no-stream case) plays silence for the rest of the block.
        for ch in 0..self.num_channels {
            outs.audio(ch)[produced..block].fill(0.0);
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
