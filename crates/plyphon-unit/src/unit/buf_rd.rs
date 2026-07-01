//! `BufRd` - reads a buffer at an arbitrary phase, plyphon's port of scsynth's `BufRd`
//! (`DelayUGens.cpp`). The read counterpart to [`BufWr`](crate::unit::buf_wr).
//!
//! Unlike [`PlayBuf`](crate::unit::play_buf), whose head advances internally at a `rate`, `BufRd` reads
//! wherever its `phase` input points (typically a `Phasor` or an LFO), so the same buffer can be
//! scrubbed, granulated or resynthesised. One output per requested channel; `interpolation` (a build
//! constant) selects none/linear/cubic, and `loop` chooses wrap vs clamp at the buffer ends.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::buffer_at;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Outputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::buffer::{Buffer, sc_loop};
use plyphon_dsp::interp::{cubicinterp, lininterp};
use plyphon_dsp::rate::Rate;

const BUFNUM: usize = 0;
const PHASE: usize = 1;
const LOOP: usize = 2;
const INTERP: usize = 3;

/// Read channel `ch` of `buf` (of `frames` frames) at integer frame `iphase` + `frac`, interpolating
/// per `interp` (`4` cubic, `2` linear, else none). Neighbour frames wrap (`looping`) or clamp.
fn read_frame(
    buf: &Buffer,
    iphase: i64,
    frac: f32,
    ch: usize,
    interp: u32,
    looping: bool,
    frames: usize,
) -> f32 {
    let at = |k: i64| {
        let f = if looping {
            k.rem_euclid(frames as i64)
        } else {
            k.clamp(0, frames as i64 - 1)
        };
        buf.sample(f as usize, ch)
    };
    match interp {
        4 => cubicinterp(
            frac,
            at(iphase - 1),
            at(iphase),
            at(iphase + 1),
            at(iphase + 2),
        ),
        2 => lininterp(frac, at(iphase), at(iphase + 1)),
        _ => at(iphase),
    }
}

/// `BufRd.ar(numChannels, bufnum, phase, loop, interpolation)`: read buffer `bufnum` at the given
/// `phase` (in frames), one output per channel.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BufRd {
    num_channels: u32,
    interp: u32,
}

impl BufRd {
    fn silence(&self, outs: &mut Outputs<'_>) {
        for ch in 0..self.num_channels as usize {
            outs.audio(ch).fill(0.0);
        }
    }
}

impl Unit for BufRd {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let bufnum = ins.control(BUFNUM).max(0.0) as usize;
        let looping = ins.control(LOOP) != 0.0;
        let phase_audio = (ins.rate(PHASE) == Rate::Audio).then(|| ins.audio(PHASE));
        let phase_ctrl = ins.control(PHASE);
        let interp = self.interp;
        let num_out = self.num_channels as usize;

        let buffer = match buffer_at(ctx.buffers, bufnum) {
            Some(b) if b.num_frames() > 0 => b,
            _ => {
                self.silence(&mut ctx.outs);
                return DoneAction::Nothing;
            }
        };
        let frames = buffer.num_frames();
        let bufchans = buffer.num_channels();
        let loop_max = if looping {
            frames as f64
        } else {
            (frames - 1) as f64
        };

        let block = ctx.outs.audio(0).len();
        for i in 0..block {
            let raw = phase_audio.map_or(phase_ctrl, |s| s[i]) as f64;
            let (phase, _) = sc_loop(raw, loop_max, looping);
            let iphase = phase as i64;
            let frac = (phase - iphase as f64) as f32;
            for ch in 0..num_out {
                let v = if ch < bufchans {
                    read_frame(buffer, iphase, frac, ch, interp, looping, frames)
                } else {
                    0.0
                };
                ctx.outs.audio(ch)[i] = v;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`BufRd`].
pub struct BufRdCtor;

impl UnitDef for BufRdCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        let interp = match ctx.const_input(INTERP).map(|v| v as u32) {
            Some(1) => 1,
            Some(4) => 4,
            _ => 2,
        };
        Ok(unit_spec(BufRd {
            num_channels: ctx.num_outputs.max(1) as u32,
            interp,
        }))
    }
}
