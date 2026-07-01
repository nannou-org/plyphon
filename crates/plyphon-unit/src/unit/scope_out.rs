//! `ScopeOut` - streams input channels to the control layer for live monitoring/analysis.
//!
//! plyphon's answer to scsynth's `ScopeOut`/`ScopeOut2`. scsynth's `ScopeOut2` streams every input
//! sample to a GUI through a *shared-memory* triple-buffer ring; plyphon has no shared memory, so this
//! writes into the same RT-safe chunked recording stream `DiskOut` uses - a bounded, lock-free SPSC
//! chunk ring the app drains off the audio thread (`Controller::cue_scope` returns the
//! `StreamConsumer`). Every input sample is streamed, in order, at whatever rate the SynthDef assigns;
//! a slow consumer causes a bounded overrun (surplus dropped), never a block or allocation on the
//! audio thread.
//!
//! Inputs: `[bufnum, ch0, ch1, …]` (scsynth's `ScopeOut` shape). Output-less, like scsynth's
//! `ScopeOut`. Several `ScopeOut` units may run in one graph, each on a distinct `bufnum`, to tap
//! several points at once - each is an independent cued recording stream.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::stream_channels_to_recording;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// `ScopeOut.ar(bufnum, channelsArray)`: streams its input channels to the scope stream cued at
/// `bufnum` (one stream channel per input), for the app to drain and display.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ScopeOut {
    /// Number of scoped channels (the inputs after `bufnum`).
    num_channels: u32,
}

impl ScopeOut {
    const BUFNUM: usize = 0;
    /// First scoped-channel input index.
    const FIRST_CHANNEL: usize = 1;
}

impl Unit for ScopeOut {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`, so we can also take `&mut` recording.
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;
        let num_channels = self.num_channels as usize;
        let block = ctx.audio.block_size;

        stream_channels_to_recording(
            &ins,
            ctx.buffers,
            block,
            bufnum,
            Self::FIRST_CHANNEL,
            num_channels,
        );
        DoneAction::Nothing
    }
}

/// Constructor for [`ScopeOut`]. The channel count is the inputs after `bufnum`; it has no outputs.
pub struct ScopeOutCtor;

impl UnitDef for ScopeOutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        if count <= ScopeOut::FIRST_CHANNEL {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(ScopeOut {
            num_channels: (count - ScopeOut::FIRST_CHANNEL) as u32,
        }))
    }
}
