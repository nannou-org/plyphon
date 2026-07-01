//! `RecordBuf` - records input channels into a buffer, plyphon's port of scsynth's `RecordBuf`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::sample_channel;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::buffer::Buffer;

/// `RecordBuf.ar(inputArray, bufnum, offset, recLevel, preLevel, run, loop, trigger, doneAction)`:
/// records its input channels into buffer `bufnum` through a write head, mixing into what is already
/// there (`new*recLevel + old*preLevel`, an overdub). `run` moves the head forward (`>0`), backward
/// (`<0`), or holds it (`0`); a rising `trigger` rewinds it to `offset`; `loop` wraps the head, else
/// it stops at the end and fires `doneAction` once. It has no signal output (one always-silent
/// output, like scsynth).
///
/// `Pod` state for the rt-pool: the write head (in interleaved samples), the channel count, the two
/// carried level ramp endpoints, the previous trigger, and the latched done state - seven 4-byte
/// fields, so `repr(C)` packs them with no implicit padding.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RecordBuf {
    /// Write head, in interleaved samples (`frame * bufChannels`); signed, since reverse `run` walks
    /// it below zero before wrapping.
    write_pos: i32,
    /// Number of recorded channels (the inputs after the eight fixed args).
    num_channels: u32,
    /// Carried `recLevel` ramp endpoint (scsynth's `m_recLevel`), for `CALCSLOPE`.
    rec_level: f32,
    /// Carried `preLevel` ramp endpoint (scsynth's `m_preLevel`).
    pre_level: f32,
    /// Previous trigger sample, for rising-edge detection across blocks.
    prev_trig: f32,
    /// `0`/`1`: whether the non-loop end-of-buffer done action has already fired.
    done: u32,
    /// Latched [`DoneAction`] tag, read once from the `doneAction` input at init.
    done_action: u32,
}

impl RecordBuf {
    const BUFNUM: usize = 0;
    const OFFSET: usize = 1;
    const REC_LEVEL: usize = 2;
    const PRE_LEVEL: usize = 3;
    const RUN: usize = 4;
    const LOOP: usize = 5;
    const TRIG: usize = 6;
    const DONE: usize = 7;
    /// First recorded-channel input index; channels are the inputs from here on.
    const FIRST_CHANNEL: usize = 8;

    /// Overdub one frame's channels into the buffer at flat sample index `pos` (`pos + ch` per
    /// channel), sampling each channel input at within-block index `k`. Out-of-range slots are
    /// silently skipped (`set_flat`/`data().get` are bounds-checked), so a channel/buffer-size or
    /// offset mismatch never reads or writes out of bounds.
    #[allow(clippy::too_many_arguments)]
    fn write_frame(
        &self,
        buffer: &mut Buffer,
        pos: i32,
        ins: &Inputs<'_>,
        k: usize,
        rec: f32,
        pre: f32,
    ) {
        let base = pos as usize;
        for ch in 0..self.num_channels as usize {
            let input = sample_channel(ins, Self::FIRST_CHANNEL + ch, k);
            let idx = base + ch;
            let value = if pre != 0.0 {
                input * rec + buffer.data().get(idx).copied().unwrap_or(0.0) * pre
            } else {
                input * rec
            };
            buffer.set_flat(idx, value);
        }
    }
}

impl Unit for RecordBuf {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Latch the levels and the done action from the initial inputs (scsynth's `*_Ctor`), and seed
        // the head from `offset` (scaled by the input count, as scsynth's ctor does).
        self.rec_level = read_input(&ctx.ins, Self::REC_LEVEL, 1.0);
        self.pre_level = read_input(&ctx.ins, Self::PRE_LEVEL, 0.0);
        self.write_pos = read_input(&ctx.ins, Self::OFFSET, 0.0) as i32 * self.num_channels as i32;
        self.done_action = if ctx.ins.len() > Self::DONE {
            DoneAction::from_code(ctx.ins.control(Self::DONE)).to_tag()
        } else {
            DoneAction::Nothing.to_tag()
        };
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // One always-silent output (scsynth's `ClearUnitOutputs(unit, 1)`).
        ctx.outs.audio(0).fill(0.0);

        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`, so we can also take `&mut buffer`.
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;
        let rec_target = read_input(&ins, Self::REC_LEVEL, 1.0);
        let pre_target = read_input(&ins, Self::PRE_LEVEL, 0.0);
        let run = read_input(&ins, Self::RUN, 1.0);
        let looping = read_input(&ins, Self::LOOP, 0.0) != 0.0;
        let trig = read_input(&ins, Self::TRIG, 0.0);
        let offset = read_input(&ins, Self::OFFSET, 0.0) as i32;
        let block = ctx.audio.block_size;
        // `CALCSLOPE`: one ramp step per sample over the block (audio-rate slope factor = 1/block).
        let slope = ctx.audio.slope_factor as f32;

        let buffer = match crate::unit::buffer_at_mut(ctx.buffers, bufnum) {
            Some(buffer) if buffer.num_frames() > 0 => buffer,
            _ => {
                // No buffer: do nothing (scsynth clears outputs and returns).
                self.prev_trig = trig;
                return DoneAction::Nothing;
            }
        };
        let stride = buffer.num_channels() as i32;
        let buf_samples = (buffer.num_frames() * buffer.num_channels()) as i32;

        let rec_slope = (rec_target - self.rec_level) * slope;
        let pre_slope = (pre_target - self.pre_level) * slope;
        // scsynth resets the running levels to the previous block's endpoints, then ramps.
        let mut rec = self.rec_level;
        let mut pre = self.pre_level;
        let mut write_pos = self.write_pos;
        let rising = trig > 0.0 && self.prev_trig <= 0.0;

        if looping {
            if rising {
                self.done = 0;
                write_pos = offset * stride;
            }
            // Pre-wrap the head into range.
            if write_pos < 0 {
                write_pos = buf_samples - stride;
            } else if write_pos >= buf_samples {
                write_pos = 0;
            }
            if run > 0.0 {
                for k in 0..block {
                    self.write_frame(buffer, write_pos, &ins, k, rec, pre);
                    write_pos += stride;
                    if write_pos >= buf_samples {
                        write_pos = 0;
                    }
                    rec += rec_slope;
                    pre += pre_slope;
                }
            } else if run < 0.0 {
                for k in 0..block {
                    self.write_frame(buffer, write_pos, &ins, k, rec, pre);
                    write_pos -= stride;
                    if write_pos < 0 {
                        write_pos = buf_samples - stride;
                    }
                    rec += rec_slope;
                    pre += pre_slope;
                }
            }
        } else {
            if rising {
                self.done = 0;
                write_pos = offset * stride;
            }
            if run > 0.0 {
                let nsmps = (buf_samples - write_pos).clamp(0, block as i32 * stride);
                for k in 0..(nsmps / stride) as usize {
                    self.write_frame(buffer, write_pos, &ins, k, rec, pre);
                    write_pos += stride;
                    rec += rec_slope;
                    pre += pre_slope;
                }
            } else if run < 0.0 {
                let nsmps = write_pos.clamp(0, block as i32 * stride);
                for k in 0..(nsmps / stride) as usize {
                    self.write_frame(buffer, write_pos, &ins, k, rec, pre);
                    write_pos -= stride;
                    rec += rec_slope;
                    pre += pre_slope;
                }
            }
        }

        let mut action = DoneAction::Nothing;
        if !looping && write_pos >= buf_samples && self.done == 0 {
            self.done = 1;
            action = DoneAction::from_tag(self.done_action);
        }
        // Mark done (scsynth's `mDone`) so a watcher observes completion even with done action 0.
        if self.done != 0 {
            ctx.done.mark_done();
        }

        self.prev_trig = trig;
        self.write_pos = write_pos;
        self.rec_level = rec;
        self.pre_level = pre;
        action
    }
}

/// Read input `i` as a single value, or `default` if the unit was built with fewer inputs.
fn read_input(ins: &Inputs<'_>, i: usize, default: f32) -> f32 {
    if ins.len() > i {
        ins.control(i)
    } else {
        default
    }
}

/// Constructor for [`RecordBuf`]. The channel count is the inputs after the eight fixed args.
pub struct RecordBufCtor;

impl UnitDef for RecordBufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        if count <= RecordBuf::FIRST_CHANNEL {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(RecordBuf {
            write_pos: 0,
            num_channels: (count - RecordBuf::FIRST_CHANNEL) as u32,
            rec_level: 1.0,
            pre_level: 0.0,
            prev_trig: 0.0,
            done: 0,
            done_action: DoneAction::Nothing.to_tag(),
        }))
    }
}
