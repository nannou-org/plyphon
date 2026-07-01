//! Wavetable oscillators - plyphon's ports of scsynth's `Osc`/`OscN`/`COsc`/`VOsc`/`VOsc3`
//! (`OscUGens.cpp`).
//!
//! These read a user buffer as a single-cycle wavetable, sweeping it with a normalised phase
//! accumulator (in cycles, kept in `[0, 1)`) exactly as [`SinOsc`](crate::unit::SinOsc) sweeps the
//! shared sine table. `Osc`/`COsc`/`VOsc`/`VOsc3` interpolate a buffer in scsynth's `(a, b)` wavetable
//! format (fill it with `/b_gen … wavetable`); `OscN` truncates a plain-sample buffer (nearest-lower),
//! so it needs no special format.
//!
//! As with `SinOsc`, the frequency input is read at audio or control rate (chosen at build) while the
//! phase-offset input is control-rate; scsynth's audio-rate-phase variants are not ported.

use core::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::buffer::{Buffer, BufferTable};
use plyphon_dsp::interp::lininterp;
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::wavetable::lookup_wavetable;

/// Freq calc-variant tags, chosen from the frequency input's rate at build time (scsynth picks one of
/// `Osc_next_{i,a}{k,a}`; we branch on the freq rate once per block). Stored as a `u32` so the state
/// is [`Pod`] and lives in the rt-pool.
mod calc {
    /// Frequency is constant or control-rate (one value per block).
    pub const FREQ_CONTROL: u32 = 0;
    /// Frequency is audio-rate (one value per sample).
    pub const FREQ_AUDIO: u32 = 1;
}

/// Wrap a phase in cycles into `[0, 1)`.
#[inline]
fn wrap_unit(x: f32) -> f32 {
    x - math::floor(x)
}

/// The freq calc variant for a unit whose frequency is input `freq_input`.
fn freq_calc(ctx: &BuildContext<'_>, freq_input: usize) -> u32 {
    match ctx.input_rates.get(freq_input).copied() {
        Some(Rate::Audio) => calc::FREQ_AUDIO,
        _ => calc::FREQ_CONTROL,
    }
}

/// The `(a, b)` wavetable sample slice of `buffer`, if it is a valid wavetable (scsynth's
/// `verify_wavetable`): the frame count must be a power of two within scsynth's ceiling, so its
/// `frames / 2` logical samples index cleanly. `None` (→ silence) otherwise, matching scsynth zeroing
/// its output on a bad table.
fn wavetable_data(buffer: &Buffer) -> Option<&[f32]> {
    let frames = buffer.num_frames();
    ((2..=131_072).contains(&frames) && frames.is_power_of_two()).then(|| buffer.data())
}

/// Read `table` at normalised `phase` in cycles by truncating to the nearest-lower sample (no
/// interpolation) - `OscN`'s harder-edged lookup. Only the fractional part of `phase` is used.
#[inline]
fn lookup_trunc(table: &[f32], phase: f32) -> f32 {
    let n = table.len();
    if n == 0 {
        return 0.0;
    }
    let frac_phase = phase - math::floor(phase); // wrap into [0, 1)
    let i = ((frac_phase * n as f32) as usize).min(n - 1);
    table[i]
}

/// `Osc.ar(bufnum, freq, phase)`: a wavetable oscillator reading buffer `bufnum` (in scsynth's `(a, b)`
/// wavetable format - fill it with `/b_gen … wavetable`) with linear interpolation. `phase` is a phase
/// offset in radians. A missing or non-power-of-two buffer outputs silence.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Osc {
    /// Normalised phase accumulator in cycles, kept in `[0, 1)`.
    phase: f32,
    /// Which freq calc variant (see [`calc`]), chosen from the freq input rate at build time.
    calc: u32,
}

impl Osc {
    const BUFNUM: usize = 0;
    const FREQ: usize = 1;
    const PHASE: usize = 2;
}

impl Unit for Osc {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bufnum = ctx.ins.control(Self::BUFNUM).max(0.0) as usize;
        let phase_offset = ctx.ins.control(Self::PHASE) / TAU;
        let sample_dur = ctx.audio.sample_dur as f32;
        let wt = match unit::buffer_at(ctx.buffers, bufnum).and_then(wavetable_data) {
            Some(wt) => wt,
            None => {
                ctx.outs.audio(0).fill(0.0);
                return DoneAction::Nothing;
            }
        };
        match self.calc {
            calc::FREQ_AUDIO => {
                let freq = ctx.ins.audio(Self::FREQ);
                for (o, &f) in ctx.outs.audio(0).iter_mut().zip(freq) {
                    *o = lookup_wavetable(wt, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + f * sample_dur);
                }
            }
            _ => {
                let inc = ctx.ins.control(Self::FREQ) * sample_dur;
                for o in ctx.outs.audio(0).iter_mut() {
                    *o = lookup_wavetable(wt, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + inc);
                }
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Osc`].
pub struct OscCtor;

impl UnitDef for OscCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Osc {
            phase: 0.0,
            calc: freq_calc(ctx, Osc::FREQ),
        }))
    }
}

/// `OscN.ar(bufnum, freq, phase)`: a non-interpolating wavetable oscillator - it truncates to the
/// nearest-lower sample of a plain (non-wavetable-format) buffer `bufnum`, giving a harder, more
/// aliased tone than [`Osc`]. `phase` is a phase offset in radians. A missing/empty buffer outputs
/// silence. Unlike `Osc` the table need not be a power of two (the phase is a float, not a masked
/// fixed-point index).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OscN {
    /// Normalised phase accumulator in cycles, kept in `[0, 1)`.
    phase: f32,
    /// Which freq calc variant (see [`calc`]), chosen from the freq input rate at build time.
    calc: u32,
}

impl OscN {
    const BUFNUM: usize = 0;
    const FREQ: usize = 1;
    const PHASE: usize = 2;
}

impl Unit for OscN {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bufnum = ctx.ins.control(Self::BUFNUM).max(0.0) as usize;
        let phase_offset = ctx.ins.control(Self::PHASE) / TAU;
        let sample_dur = ctx.audio.sample_dur as f32;
        let table = match unit::buffer_at(ctx.buffers, bufnum) {
            Some(buffer) if buffer.num_frames() > 0 => buffer.data(),
            _ => {
                ctx.outs.audio(0).fill(0.0);
                return DoneAction::Nothing;
            }
        };
        match self.calc {
            calc::FREQ_AUDIO => {
                let freq = ctx.ins.audio(Self::FREQ);
                for (o, &f) in ctx.outs.audio(0).iter_mut().zip(freq) {
                    *o = lookup_trunc(table, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + f * sample_dur);
                }
            }
            _ => {
                let inc = ctx.ins.control(Self::FREQ) * sample_dur;
                for o in ctx.outs.audio(0).iter_mut() {
                    *o = lookup_trunc(table, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + inc);
                }
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`OscN`].
pub struct OscNCtor;

impl UnitDef for OscNCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(OscN {
            phase: 0.0,
            calc: freq_calc(ctx, OscN::FREQ),
        }))
    }
}

/// `COsc.ar(bufnum, freq, beats)`: a chorusing wavetable oscillator - two [`Osc`]-style readers of the
/// same buffer detuned by `beats` Hz (one at `freq + beats/2`, one at `freq - beats/2`) summed, so the
/// two slowly-drifting copies beat against each other. `freq` and `beats` are control-rate (scsynth's
/// single `COsc_next`). A missing or non-power-of-two buffer outputs silence.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct COsc {
    /// The two detuned phase accumulators, each in cycles kept in `[0, 1)`.
    phase1: f32,
    phase2: f32,
}

impl COsc {
    const BUFNUM: usize = 0;
    const FREQ: usize = 1;
    const BEATS: usize = 2;
}

impl Unit for COsc {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bufnum = ctx.ins.control(Self::BUFNUM).max(0.0) as usize;
        let freq = ctx.ins.control(Self::FREQ);
        let beats = ctx.ins.control(Self::BEATS) * 0.5; // half the beat spread each side
        let sample_dur = ctx.audio.sample_dur as f32;
        let wt = match unit::buffer_at(ctx.buffers, bufnum).and_then(wavetable_data) {
            Some(wt) => wt,
            None => {
                ctx.outs.audio(0).fill(0.0);
                return DoneAction::Nothing;
            }
        };
        let inc1 = (freq + beats) * sample_dur;
        let inc2 = (freq - beats) * sample_dur;
        for o in ctx.outs.audio(0).iter_mut() {
            *o = lookup_wavetable(wt, self.phase1) + lookup_wavetable(wt, self.phase2);
            self.phase1 = wrap_unit(self.phase1 + inc1);
            self.phase2 = wrap_unit(self.phase2 + inc2);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`COsc`].
pub struct COscCtor;

impl UnitDef for COscCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(COsc {
            phase1: 0.0,
            phase2: 0.0,
        }))
    }
}

/// The crossfaded read of two consecutive wavetables at `bufindex`/`bufindex + 1` in the buffer table,
/// each read at `phase`, blended by `level` (the fractional buffer position). `0.0` unless both slots
/// hold valid, equal-size wavetables - scsynth's `VOsc` silences a missing/mismatched bank member.
fn crossfade_read(buffers: &BufferTable, bufindex: usize, phase: f32, level: f32) -> f32 {
    let table0 = unit::buffer_at(buffers, bufindex).and_then(wavetable_data);
    let table1 = unit::buffer_at(buffers, bufindex + 1).and_then(wavetable_data);
    match (table0, table1) {
        (Some(t0), Some(t1)) if t0.len() == t1.len() => lininterp(
            level,
            lookup_wavetable(t0, phase),
            lookup_wavetable(t1, phase),
        ),
        _ => 0.0,
    }
}

/// `VOsc.ar(bufpos, freq, phase)`: a wavetable-*crossfade* oscillator. `bufpos` selects a position in a
/// bank of consecutively-numbered wavetable buffers; the output crossfades between buffer
/// `floor(bufpos)` and the next by the fractional part, so sweeping `bufpos` morphs smoothly through
/// the bank (all members must be equal-size `(a, b)` wavetables). `bufpos` is interpolated across the
/// block from its previous value, so a moving `bufpos` sweeps sample-accurately. `freq` is audio or
/// control rate, `phase` a control-rate offset in radians.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VOsc {
    /// Phase accumulator in cycles, kept in `[0, 1)`.
    phase: f32,
    /// The previous block's `bufpos`, so this block can ramp from it.
    bufpos: f32,
    /// Which freq calc variant (see [`calc`]).
    calc: u32,
    /// `0` until the first block seeds `bufpos`, then `1`.
    warmed: u32,
}

impl VOsc {
    const BUFPOS: usize = 0;
    const FREQ: usize = 1;
    const PHASE: usize = 2;
}

impl Unit for VOsc {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let next_bufpos = ctx.ins.control(Self::BUFPOS);
        let phase_offset = ctx.ins.control(Self::PHASE) / TAU;
        let sample_dur = ctx.audio.sample_dur as f32;
        let audio_freq = self.calc == calc::FREQ_AUDIO;
        let freq_slice = if audio_freq {
            ctx.ins.audio(Self::FREQ)
        } else {
            &[]
        };
        let freq_ctl = if audio_freq {
            0.0
        } else {
            ctx.ins.control(Self::FREQ)
        };

        let prev = if self.warmed == 0 {
            self.warmed = 1;
            next_bufpos
        } else {
            self.bufpos
        };
        self.bufpos = next_bufpos;
        let bufdiff = next_bufpos - prev;

        let block = ctx.outs.audio(0).len();
        for i in 0..block {
            let cur = prev + bufdiff * (i as f32 / block as f32);
            let base = math::floor(cur);
            let level = cur - base;
            let bufindex = base.max(0.0) as usize;
            ctx.outs.audio(0)[i] =
                crossfade_read(ctx.buffers, bufindex, self.phase + phase_offset, level);
            // Audio rate: `freq_slice` has one value per sample; control rate: it is empty and `.get`
            // falls back to the per-block `freq_ctl`.
            let f = freq_slice.get(i).copied().unwrap_or(freq_ctl);
            self.phase = wrap_unit(self.phase + f * sample_dur);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`VOsc`].
pub struct VOscCtor;

impl UnitDef for VOscCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(VOsc {
            phase: 0.0,
            bufpos: 0.0,
            calc: freq_calc(ctx, VOsc::FREQ),
            warmed: 0,
        }))
    }
}

/// `VOsc3.ar(bufpos, freq1, freq2, freq3)`: three [`VOsc`]-style oscillators (control-rate `freq1..3`)
/// summed and read from the same crossfaded buffer-bank position - a three-voice detuned wavetable
/// oscillator. Like `VOsc`, `bufpos` crossfades a bank of equal-size wavetables and is ramped across
/// the block. No phase input.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VOsc3 {
    /// The three phase accumulators, each in cycles kept in `[0, 1)`.
    phase1: f32,
    phase2: f32,
    phase3: f32,
    /// The previous block's `bufpos`, so this block can ramp from it.
    bufpos: f32,
    /// `0` until the first block seeds `bufpos`, then `1`.
    warmed: u32,
}

impl VOsc3 {
    const BUFPOS: usize = 0;
    const FREQ1: usize = 1;
    const FREQ2: usize = 2;
    const FREQ3: usize = 3;
}

impl Unit for VOsc3 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let next_bufpos = ctx.ins.control(Self::BUFPOS);
        let sample_dur = ctx.audio.sample_dur as f32;
        let inc1 = ctx.ins.control(Self::FREQ1) * sample_dur;
        let inc2 = ctx.ins.control(Self::FREQ2) * sample_dur;
        let inc3 = ctx.ins.control(Self::FREQ3) * sample_dur;

        let prev = if self.warmed == 0 {
            self.warmed = 1;
            next_bufpos
        } else {
            self.bufpos
        };
        self.bufpos = next_bufpos;
        let bufdiff = next_bufpos - prev;

        let block = ctx.outs.audio(0).len();
        for i in 0..block {
            let cur = prev + bufdiff * (i as f32 / block as f32);
            let base = math::floor(cur);
            let level = cur - base;
            let bufindex = base.max(0.0) as usize;
            let voices = crossfade_read(ctx.buffers, bufindex, self.phase1, level)
                + crossfade_read(ctx.buffers, bufindex, self.phase2, level)
                + crossfade_read(ctx.buffers, bufindex, self.phase3, level);
            ctx.outs.audio(0)[i] = voices;
            self.phase1 = wrap_unit(self.phase1 + inc1);
            self.phase2 = wrap_unit(self.phase2 + inc2);
            self.phase3 = wrap_unit(self.phase3 + inc3);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`VOsc3`].
pub struct VOsc3Ctor;

impl UnitDef for VOsc3Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(VOsc3 {
            phase1: 0.0,
            phase2: 0.0,
            phase3: 0.0,
            bufpos: 0.0,
            warmed: 0,
        }))
    }
}
