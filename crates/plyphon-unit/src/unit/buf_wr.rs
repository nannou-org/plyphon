//! `BufWr` - writes input channels into a buffer at a phase index, plyphon's port of scsynth's `BufWr`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::sample_channel;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::buffer::sc_loop;

/// `BufWr.ar(inputArray, bufnum, phase, loop)`: writes its input channels into buffer `bufnum` at the
/// (truncated) frame index `phase` each sample - no interpolation, the inverse of `BufRd`. `loop`
/// wraps `phase` into the buffer; otherwise it clamps at the ends. It has no signal output (one
/// always-silent output, like scsynth).
///
/// `Pod` state for the rt-pool: just the channel count (the phase comes from the input each sample).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BufWr {
    /// Number of written channels (the inputs after the three fixed args).
    num_channels: u32,
}

impl BufWr {
    const BUFNUM: usize = 0;
    const PHASE: usize = 1;
    const LOOP: usize = 2;
    /// First written-channel input index.
    const FIRST_CHANNEL: usize = 3;
}

impl Unit for BufWr {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // One always-silent output (scsynth's `ClearUnitOutputs(unit, 1)`).
        ctx.outs.audio(0).fill(0.0);

        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`, so we can also take `&mut buffer`.
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;
        let looping = ins.control(Self::LOOP) != 0.0;
        let block = ctx.audio.block_size;
        let num_channels = self.num_channels as usize;

        // scsynth's `checkBuffer`: the written channels must fit the buffer (`<= bufChannels`).
        let mut buffer = match unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum) {
            Some(buffer) if buffer.num_frames() > 0 && num_channels <= buffer.num_channels() => {
                buffer
            }
            _ => return DoneAction::Nothing,
        };
        let stride = buffer.num_channels();
        // `loopMax`: wrapping bound (whole buffer when looping, last frame when not).
        let loop_max = (buffer.num_frames() - if looping { 0 } else { 1 }) as f64;

        let mut done = false;
        for k in 0..block {
            let (phase, hit) = sc_loop(
                sample_channel(&ins, Self::PHASE, k) as f64,
                loop_max,
                looping,
            );
            done |= hit;
            let base = phase as usize * stride;
            for ch in 0..num_channels {
                buffer.set_flat(base + ch, sample_channel(&ins, Self::FIRST_CHANNEL + ch, k));
            }
        }
        // Reaching an end with `loop` off marks the unit done (scsynth's `sc_loop` sets `mDone`).
        if done {
            ctx.done.mark_done();
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`BufWr`]. The channel count is the inputs after the three fixed args.
pub struct BufWrCtor;

impl UnitDef for BufWrCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        if count <= BufWr::FIRST_CHANNEL {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(BufWr {
            num_channels: (count - BufWr::FIRST_CHANNEL) as u32,
        }))
    }
}
