//! Additive/modal resonator banks - plyphon's ports of scsynth's `Klang` and `Klank`
//! (`OscUGens.cpp`).
//!
//! Both sum a bank of second-order resonators whose coefficients come from a trailing spec array
//! (`[freq, amp, phase|ringtime]` triples). `Klang` is *additive*: each partial is a self-oscillating
//! `2·cos(w)` resonator (the `FSinOsc` recurrence) seeded to the partial's amplitude and phase, so the
//! bank is a fixed sum of sines. `Klank` is *modal*: each partial is a decaying `Ringz`-style
//! resonator driven by a shared excitation input, so an impulse/noise burst rings the whole bank.
//!
//! The per-partial coefficients + running state live in the unit's [auxiliary memory](crate::unit::Aux)
//! (sized from the spec length at build). They are computed once, on the first block, from the live
//! spec inputs (scsynth computes them in the ctor); the aux is not zeroed at instantiation, so a
//! `warmed` flag guards that one-time setup, after which the block loop only advances the state.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::sample_channel;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::math;

/// `ln(0.001)`, scsynth's `log001`, for the modal decay coefficient (-60 dB over the ring time).
const LOG001: f32 = -6.907_755_4;

/// Compute one additive-bank sample: advance every `2cos(w)` resonator and sum. State is `[y1, y2,
/// b1]` per partial.
fn klang_tick(buf: &mut [f32], n: usize) -> f32 {
    let mut acc = 0.0;
    for p in 0..n {
        let base = p * 3;
        let (y1, y2, b1) = (buf[base], buf[base + 1], buf[base + 2]);
        let y0 = b1 * y1 - y2;
        acc += y0;
        buf[base + 1] = y1;
        buf[base] = y0;
    }
    acc
}

/// Seed the `Klang` bank's coefficients/state from the spec inputs (scsynth's `Klang_SetCoefs`).
fn klang_set_coefs(ins: &Inputs<'_>, buf: &mut [f32], n: usize, rps: f32) {
    let freqscale = ins.control(0) * rps;
    let freqoffset = ins.control(1) * rps;
    for i in 0..n {
        let j = 2 + i * 3;
        let w = ins.control(j) * freqscale + freqoffset;
        let level = ins.control(j + 1);
        let phase = ins.control(j + 2);
        let base = i * 3;
        if phase != 0.0 {
            buf[base] = level * math::sin(phase - w);
            buf[base + 1] = level * math::sin(phase - w - w);
        } else {
            buf[base] = level * -math::sin(w);
            buf[base + 1] = level * -math::sin(w + w);
        }
        buf[base + 2] = 2.0 * math::cos(w);
    }
}

/// `Klang.ar(freqscale, freqoffset, [freq, amp, phase]...)`: a fixed additive bank of sine partials.
/// Inputs `0`/`1` scale/offset every partial frequency; each following triple is one partial.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Klang {
    num_partials: u32,
    /// `0` until the first block seeds the coefficients into aux, then `1`.
    warmed: u32,
    audio: u32,
}

impl Unit for Klang {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let n = self.num_partials as usize;
        let ins = ctx.ins;
        let rps = if audio {
            ctx.audio.radians_per_sample
        } else {
            ctx.control.radians_per_sample
        } as f32;
        let warm = self.warmed == 0;
        self.warmed = 1;

        let buf = ctx.aux.f32_mut();
        if buf.len() < n * 3 {
            if audio {
                ctx.outs.audio(0).fill(0.0);
            } else {
                *ctx.outs.control(0) = 0.0;
            }
            return DoneAction::Nothing;
        }
        if warm {
            klang_set_coefs(&ins, buf, n, rps);
        }
        if audio {
            for o in ctx.outs.audio(0).iter_mut() {
                *o = klang_tick(buf, n);
            }
        } else {
            *ctx.outs.control(0) = klang_tick(buf, n);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Klang`]. Partial count is `(inputs - 2) / 3`.
pub struct KlangCtor;

impl UnitDef for KlangCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        if count < 5 || !(count - 2).is_multiple_of(3) {
            return Err(BuildError::WrongInputCount);
        }
        let n = (count - 2) / 3;
        let aux_bytes = n * 3 * core::mem::size_of::<f32>();
        Ok(unit_spec_aux(
            Klang {
                num_partials: n as u32,
                warmed: 0,
                audio: (ctx.rate == plyphon_dsp::rate::Rate::Audio) as u32,
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}

/// Compute one modal-bank sample: drive every 2-pole resonator with `x` and sum the scaled outputs.
/// State is `[y1, y2, b1, b2, a0]` per partial.
fn klank_tick(buf: &mut [f32], n: usize, x: f32) -> f32 {
    let mut acc = 0.0;
    for p in 0..n {
        let base = p * 5;
        let (y1, y2, b1, b2, a0) = (
            buf[base],
            buf[base + 1],
            buf[base + 2],
            buf[base + 3],
            buf[base + 4],
        );
        let y0 = x + b1 * y1 + b2 * y2;
        acc += a0 * y0;
        buf[base + 1] = y1;
        buf[base] = y0;
    }
    acc
}

/// Seed the `Klank` bank's coefficients from the spec inputs (scsynth's `Klank_SetCoefs`).
fn klank_set_coefs(ins: &Inputs<'_>, buf: &mut [f32], n: usize, rps: f32, sr: f32) {
    let freqscale = ins.control(1) * rps;
    let freqoffset = ins.control(2) * rps;
    let decayscale = ins.control(3);
    for i in 0..n {
        let j = 4 + i * 3;
        let w = ins.control(j) * freqscale + freqoffset;
        let level = ins.control(j + 1);
        let time = ins.control(j + 2) * decayscale;
        let r = if time == 0.0 {
            0.0
        } else {
            math::exp(LOG001 / (time * sr))
        };
        let two_r = 2.0 * r;
        let r2 = r * r;
        let cost = (two_r * math::cos(w)) / (1.0 + r2);
        let base = i * 5;
        buf[base] = 0.0;
        buf[base + 1] = 0.0;
        buf[base + 2] = two_r * cost;
        buf[base + 3] = -r2;
        buf[base + 4] = level * 0.25;
    }
}

/// `Klank.ar(input, freqscale, freqoffset, decayscale, [freq, amp, ringtime]...)`: a bank of decaying
/// resonators, all driven by `input`. Inputs `1`/`2` scale/offset every partial frequency, `3` scales
/// every ring time; each following triple is one resonant mode.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Klank {
    num_partials: u32,
    /// `0` until the first block seeds the coefficients into aux, then `1`.
    warmed: u32,
    audio: u32,
}

impl Klank {
    const INPUT: usize = 0;
}

impl Unit for Klank {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let n = self.num_partials as usize;
        let ins = ctx.ins;
        let (rps, sr) = if audio {
            (ctx.audio.radians_per_sample, ctx.audio.sample_rate)
        } else {
            (ctx.control.radians_per_sample, ctx.control.sample_rate)
        };
        let (rps, sr) = (rps as f32, sr as f32);
        let warm = self.warmed == 0;
        self.warmed = 1;

        let buf = ctx.aux.f32_mut();
        if buf.len() < n * 5 {
            if audio {
                ctx.outs.audio(0).fill(0.0);
            } else {
                *ctx.outs.control(0) = 0.0;
            }
            return DoneAction::Nothing;
        }
        if warm {
            klank_set_coefs(&ins, buf, n, rps, sr);
        }
        if audio {
            for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
                *o = klank_tick(buf, n, sample_channel(&ins, Self::INPUT, i));
            }
        } else {
            *ctx.outs.control(0) = klank_tick(buf, n, sample_channel(&ins, Self::INPUT, 0));
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Klank`]. Partial count is `(inputs - 4) / 3`.
pub struct KlankCtor;

impl UnitDef for KlankCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        if count < 7 || !(count - 4).is_multiple_of(3) {
            return Err(BuildError::WrongInputCount);
        }
        let n = (count - 4) / 3;
        let aux_bytes = n * 5 * core::mem::size_of::<f32>();
        Ok(unit_spec_aux(
            Klank {
                num_partials: n as u32,
                warmed: 0,
                audio: (ctx.rate == plyphon_dsp::rate::Rate::Audio) as u32,
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}
