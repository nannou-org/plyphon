//! `PlayBuf` - plays a buffer back at a given rate, plyphon's port of scsynth's `PlayBuf`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, Inputs, Outputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `PlayBuf.ar(numChannels, bufnum, rate, trigger, startPos, loop, doneAction)`: reads consecutive
/// frames of buffer `bufnum`, one output per buffer channel, advancing the play head by `rate`
/// frames per sample with linear interpolation. As in scsynth, `rate` is in buffer frames per server
/// sample (multiply by a buffer-rate scale for natural-pitch playback of an off-rate buffer). On a
/// rising `trigger` the head jumps to `startPos`; at the end it either wraps (`loop != 0`) or holds
/// the last frame and fires `doneAction` once. Reading a missing buffer outputs silence.
///
/// `Pod` state for the rt-pool: `f64` head first, then the channel count, previous trigger, and two
/// `0`/`1` flags - four 4-byte fields, so `repr(C)` packs them with no implicit padding.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PlayBuf {
    /// Play head, in fractional buffer frames.
    phase: f64,
    num_channels: u32,
    /// Previous trigger value, for rising-edge detection.
    prev_trig: f32,
    /// `0`/`1`: whether the head has been seeded with `startPos` yet.
    started: u32,
    /// `0`/`1`: whether the end-of-buffer done action has already fired (non-looping).
    done: u32,
}

impl PlayBuf {
    const BUFNUM: usize = 0;
    const RATE: usize = 1;
    const TRIG: usize = 2;
    const START: usize = 3;
    const LOOP: usize = 4;
    const DONE: usize = 5;

    fn silence(&self, outs: &mut Outputs<'_>) {
        for ch in 0..self.num_channels as usize {
            outs.audio(ch).fill(0.0);
        }
    }
}

impl Unit for PlayBuf {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bufnum = ctx.ins.control(Self::BUFNUM).max(0.0) as usize;
        let buffer = match unit::buffer_at(ctx.buffers, bufnum) {
            Some(buffer) if buffer.num_frames() > 0 => buffer,
            _ => {
                self.silence(&mut ctx.outs);
                return DoneAction::Nothing;
            }
        };
        let frames = buffer.num_frames();
        let last = (frames - 1) as f64;

        let rate = read_input(&ctx.ins, Self::RATE, 1.0) as f64;
        let looping = read_input(&ctx.ins, Self::LOOP, 0.0) != 0.0;
        let start_pos = read_input(&ctx.ins, Self::START, 0.0) as f64;
        let trig = read_input(&ctx.ins, Self::TRIG, 0.0);
        let done_action = if ctx.ins.len() > Self::DONE {
            DoneAction::from_code(ctx.ins.control(Self::DONE))
        } else {
            DoneAction::Nothing
        };

        if self.started == 0 {
            self.phase = start_pos.clamp(0.0, last);
            self.started = 1;
        }
        // A rising trigger restarts playback from `startPos`.
        if self.prev_trig <= 0.0 && trig > 0.0 {
            self.phase = start_pos.clamp(0.0, last);
            self.done = 0;
        }
        self.prev_trig = trig;

        let mut action = DoneAction::Nothing;
        let block = ctx.outs.audio(0).len();
        for i in 0..block {
            let floor = math::floor(self.phase);
            let frac = (self.phase - floor) as f32;
            let i0 = floor as usize;
            let i1 = if looping {
                (i0 + 1) % frames
            } else {
                (i0 + 1).min(frames - 1)
            };
            for ch in 0..self.num_channels as usize {
                let a = buffer.sample(i0, ch);
                let b = buffer.sample(i1, ch);
                ctx.outs.audio(ch)[i] = a + (b - a) * frac;
            }

            self.phase += rate;
            if looping {
                self.phase = math::rem_euclid(self.phase, frames as f64);
            } else if self.phase > last {
                self.phase = last;
                if self.done == 0 {
                    self.done = 1;
                    action = action.max(done_action);
                }
            } else if self.phase < 0.0 {
                self.phase = 0.0;
                if self.done == 0 {
                    self.done = 1;
                    action = action.max(done_action);
                }
            }
        }
        // Reaching the end marks the unit done (scsynth's `mDone`), so a watcher can observe it even
        // when the done action is 0. Re-armed by a rising trigger clearing `self.done`.
        if self.done != 0 {
            ctx.done.mark_done();
        }
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

/// Constructor for [`PlayBuf`]. The output count (the buffer's channel count) is fixed here.
pub struct PlayBufCtor;

impl UnitDef for PlayBufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(PlayBuf {
            phase: 0.0,
            num_channels: ctx.num_outputs as u32,
            prev_trig: 0.0,
            started: 0,
            done: 0,
        }))
    }
}
