//! `PlayBuf` - plays a buffer back at a given rate, plyphon's port of scsynth's `PlayBuf`.

use crate::bus::Buses;
use crate::error::BuildError;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{DoneAction, Inputs, Outputs, ProcessContext, Ugen};

/// `PlayBuf.ar(numChannels, bufnum, rate, trigger, startPos, loop, doneAction)`: reads consecutive
/// frames of buffer `bufnum`, one output per buffer channel, advancing the play head by `rate`
/// frames per sample with linear interpolation. As in scsynth, `rate` is in buffer frames per server
/// sample (multiply by a buffer-rate scale for natural-pitch playback of an off-rate buffer). On a
/// rising `trigger` the head jumps to `startPos`; at the end it either wraps (`loop != 0`) or holds
/// the last frame and fires `doneAction` once. Reading a missing buffer outputs silence.
pub struct PlayBuf {
    num_channels: usize,
    /// Play head, in fractional buffer frames.
    phase: f64,
    /// Previous trigger value, for rising-edge detection.
    prev_trig: f32,
    /// Whether the head has been seeded with `startPos` yet.
    started: bool,
    /// Whether the end-of-buffer done action has already fired (non-looping).
    done: bool,
}

impl PlayBuf {
    const BUFNUM: usize = 0;
    const RATE: usize = 1;
    const TRIG: usize = 2;
    const START: usize = 3;
    const LOOP: usize = 4;
    const DONE: usize = 5;

    fn silence(&self, outs: &mut Outputs<'_>) {
        for ch in 0..self.num_channels {
            outs.audio(ch).fill(0.0);
        }
    }
}

impl Ugen for PlayBuf {
    fn process(
        &mut self,
        ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        _buses: &mut Buses,
    ) -> DoneAction {
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;
        let buffer = match ctx.buffers.get(bufnum) {
            Some(buffer) if buffer.num_frames() > 0 => buffer,
            _ => {
                self.silence(outs);
                return DoneAction::Nothing;
            }
        };
        let frames = buffer.num_frames();
        let last = (frames - 1) as f64;

        let rate = read_input(&ins, Self::RATE, 1.0) as f64;
        let looping = read_input(&ins, Self::LOOP, 0.0) != 0.0;
        let start_pos = read_input(&ins, Self::START, 0.0) as f64;
        let trig = read_input(&ins, Self::TRIG, 0.0);
        let done_action = if ins.len() > Self::DONE {
            DoneAction::from_code(ins.control(Self::DONE))
        } else {
            DoneAction::Nothing
        };

        if !self.started {
            self.phase = start_pos.clamp(0.0, last);
            self.started = true;
        }
        // A rising trigger restarts playback from `startPos`.
        if self.prev_trig <= 0.0 && trig > 0.0 {
            self.phase = start_pos.clamp(0.0, last);
            self.done = false;
        }
        self.prev_trig = trig;

        let mut action = DoneAction::Nothing;
        let block = outs.audio(0).len();
        for i in 0..block {
            let floor = self.phase.floor();
            let frac = (self.phase - floor) as f32;
            let i0 = floor as usize;
            let i1 = if looping {
                (i0 + 1) % frames
            } else {
                (i0 + 1).min(frames - 1)
            };
            for ch in 0..self.num_channels {
                let a = buffer.sample(i0, ch);
                let b = buffer.sample(i1, ch);
                outs.audio(ch)[i] = a + (b - a) * frac;
            }

            self.phase += rate;
            if looping {
                self.phase = self.phase.rem_euclid(frames as f64);
            } else if self.phase > last {
                self.phase = last;
                if !self.done {
                    self.done = true;
                    action = action.max(done_action);
                }
            } else if self.phase < 0.0 {
                self.phase = 0.0;
                if !self.done {
                    self.done = true;
                    action = action.max(done_action);
                }
            }
        }
        action
    }
}

/// Read input `i` as a single value, or `default` if the UGen was built with fewer inputs.
fn read_input(ins: &Inputs<'_>, i: usize, default: f32) -> f32 {
    if ins.len() > i {
        ins.control(i)
    } else {
        default
    }
}

/// Constructor for [`PlayBuf`]. The output count (the buffer's channel count) is fixed here.
pub struct PlayBufCtor;

impl UgenCtor for PlayBufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        Ok(Box::new(PlayBuf {
            num_channels: ctx.num_outputs,
            phase: 0.0,
            prev_trig: 0.0,
            started: false,
            done: false,
        }))
    }
}
