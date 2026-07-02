//! The shared writable delay tap - plyphon's ports of scsynth's `DelTapWr`/`DelTapRd`
//! (`DelayUGens.cpp`).
//!
//! Unlike the [delay-line family](crate::unit::delay), which owns its line, these split a single
//! delay line held in a `/b_alloc`'d **mono** buffer into a writer and one or more readers:
//!
//! - [`DelTapWr`] writes its input into the buffer at a monotonically wrapping write head, and outputs
//!   that head position each sample. The position is an integer disguised as a float - carried through
//!   the audio wire by [`f32::from_bits`]/[`f32::to_bits`], exactly as scsynth reinterprets the wire.
//! - [`DelTapRd`] reads the writer's head off that wire (its first sample), then reads the buffer
//!   `delTime` seconds behind the head, wrapping into the buffer and interpolating (none/linear/cubic).
//!
//! Several `DelTapRd`s can tap one `DelTapWr` at different delays - a multi-tap delay from one line.
//! `DelTapWr` zeroes the buffer on its first block so a reader never taps stale content.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, buffer_at_mut, unit_spec};
use plyphon_dsp::interp::{cubicinterp, lininterp};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// Read a mono delay buffer `buf` at fractional position `phase` (wrapped into the buffer) with
/// `interp` (`4` cubic, `2` linear, else none) - scsynth's `DelTapRd` interpolation variants. Every
/// neighbour index wraps modulo the buffer length, so a tap near the wrap point reads across it.
fn tap_read(buf: &[f32], phase: f64, interp: u32) -> f32 {
    let n = buf.len();
    let p = math::rem_euclid(phase, n as f64);
    let i = (p as usize) % n;
    let frac = (p - i as f64) as f32;
    match interp {
        4 => {
            let i0 = (i + n - 1) % n;
            let i1 = (i + 1) % n;
            let i2 = (i + 2) % n;
            cubicinterp(frac, buf[i0], buf[i], buf[i1], buf[i2])
        }
        2 => lininterp(frac, buf[i], buf[(i + 1) % n]),
        _ => buf[i],
    }
}

/// `DelTapWr.ar(buffer, in)`: write `in` into the mono buffer at `bufnum` at a wrapping write head, and
/// output the head position each sample (an integer carried through the float wire via its bits) for a
/// [`DelTapRd`] to read.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DelTapWr {
    /// The write head (`m_phase`), wrapping modulo the buffer length.
    phase: u32,
    /// `1` until the first block has zeroed the buffer (scsynth zeroes it in its ctor).
    first: u32,
}

impl DelTapWr {
    const BUFNUM: usize = 0;
    const IN: usize = 1;
}

impl Unit for DelTapWr {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let in_audio = (ins.rate(Self::IN) == Rate::Audio).then(|| ins.audio(Self::IN));
        let in_ctrl = ins.control(Self::IN);
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;

        let out = ctx.outs.audio(0);
        let buf = match buffer_at_mut(ctx.buffers, bufnum) {
            Some(b) if b.num_channels() == 1 && !b.data().is_empty() => b.data_mut(),
            _ => {
                out.fill(0.0);
                return DoneAction::Nothing;
            }
        };
        let bufsamples = buf.len();

        // The buffer is the delay line; zero it once so a reader never taps prior content.
        if self.first != 0 {
            buf.fill(0.0);
            self.first = 0;
            self.phase = 0;
        }

        let mut phase = self.phase as usize;
        for (j, o) in out.iter_mut().enumerate() {
            let x = in_audio.map_or(in_ctrl, |s| s[j]);
            buf[phase] = x;
            *o = f32::from_bits(phase as u32);
            phase += 1;
            if phase == bufsamples {
                phase = 0;
            }
        }
        self.phase = phase as u32;
        DoneAction::Nothing
    }
}

/// Constructor for [`DelTapWr`].
pub struct DelTapWrCtor;

impl UnitDef for DelTapWrCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DelTapWr { phase: 0, first: 1 }))
    }
}

/// `DelTapRd.ar(buffer, phase, delTime, interp)`: read the mono buffer at `bufnum` `delTime` seconds
/// behind the write head that `phase` (from a [`DelTapWr`]) reports, interpolating per `interp`
/// (`1` none / `2` linear / `4` cubic).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DelTapRd {
    /// The delay in samples last block (`m_delTime`), slope-interpolated when it changes.
    del_time: f32,
    /// Interpolation order (`1`/`2`/`4`).
    interp: u32,
    /// `1` if `delTime` is audio-rate (per-sample), else control/scalar (slope on change).
    delay_audio: u32,
    _pad: u32,
}

impl DelTapRd {
    const BUFNUM: usize = 0;
    const PHASE: usize = 1;
    const DELTIME: usize = 2;

    /// The write head reported on the `phase` wire this block (its first sample), as the integer it
    /// encodes. A control/constant wire broadcasts one value.
    fn head(ins: &crate::unit::Inputs<'_>) -> u32 {
        let f = if ins.rate(Self::PHASE) == Rate::Audio {
            ins.audio(Self::PHASE).first().copied().unwrap_or(0.0)
        } else {
            ins.control(Self::PHASE)
        };
        f.to_bits()
    }
}

impl Unit for DelTapRd {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.del_time = ctx.ins.control(Self::DELTIME) * ctx.own.sample_rate as f32;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.own.sample_rate as f32;
        let interp = self.interp;
        let ins = ctx.ins;
        let mut phase_in = Self::head(&ins);
        let dt_audio = (self.delay_audio != 0).then(|| ins.audio(Self::DELTIME));
        let dt_ctrl = ins.control(Self::DELTIME);

        let out = ctx.outs.audio(0);
        let buf = match buffer_at_mut(ctx.buffers, ins.control(Self::BUFNUM).max(0.0) as usize) {
            Some(b) if b.num_channels() == 1 && !b.data().is_empty() => b.data(),
            _ => {
                out.fill(0.0);
                return DoneAction::Nothing;
            }
        };
        let n = out.len();

        match dt_audio {
            Some(dt) => {
                for (j, o) in out.iter_mut().enumerate() {
                    let del = dt[j] * sr;
                    *o = tap_read(buf, phase_in as f64 - del as f64, interp);
                    phase_in = phase_in.wrapping_add(1);
                }
            }
            None if dt_ctrl * sr == self.del_time => {
                let del = self.del_time as f64;
                for o in out.iter_mut() {
                    *o = tap_read(buf, phase_in as f64 - del, interp);
                    phase_in = phase_in.wrapping_add(1);
                }
            }
            None => {
                let next = dt_ctrl * sr;
                let mut del = self.del_time;
                let slope = (next - del) / n as f32;
                for o in out.iter_mut() {
                    del += slope;
                    *o = tap_read(buf, phase_in as f64 - del as f64, interp);
                    phase_in = phase_in.wrapping_add(1);
                }
                self.del_time = next;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DelTapRd`].
pub struct DelTapRdCtor;

impl UnitDef for DelTapRdCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        // `interp` is a build-time constant selecting the read variant (scsynth picks the calc func by
        // it at ctor); default to no interpolation for a non-constant or unknown value.
        let interp = match ctx.const_input(3).map(|v| v as u32) {
            Some(2) => 2,
            Some(4) => 4,
            _ => 1,
        };
        let delay_audio = matches!(ctx.input_rates[DelTapRd::DELTIME], Rate::Audio) as u32;
        Ok(unit_spec(DelTapRd {
            del_time: 0.0,
            interp,
            delay_audio,
            _pad: 0,
        }))
    }
}
