//! Granular-synthesis UGens - plyphon's ports of scsynth's `GrainSin`/`GrainFM`/`GrainIn`/`GrainBuf`
//! and (in future batches) `Warp1` (`GrainUGens.cpp`).
//!
//! Each is a bank of short overlapping *grains* spawned on a rising trigger. A grain is a windowed
//! tone (a sine for `GrainSin`, an FM pair for `GrainFM`, a live input for `GrainIn`, a buffer read for
//! `GrainBuf`) that fades in and out over `dur` seconds and is panned across the output channels. The
//! grains live in a fixed in-struct array (no allocation); a finished grain is removed by swapping the
//! last active grain into its slot. The default window is scsynth's inline `sin²` (Hann) recurrence; an
//! `envbufnum >= 0` reads a user buffer as the window instead.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::{buffer_at, sample_channel};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, Outputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::math;
use plyphon_dsp::ops;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::wavetable::lookup_cycle;

/// Maximum simultaneous grains per unit (scsynth's fixed `kMaxGrains`, also the plyphon cap on the
/// `maxGrains` input).
const MAX_GRAINS: usize = 64;

/// Read the window buffer `win` at fractional position `pos` with linear interpolation (0 out of range).
fn sample_window(win: Option<&[f32]>, pos: f64) -> f32 {
    match win {
        Some(w) if w.len() >= 2 => {
            let i = pos as usize;
            let a = w.get(i).copied().unwrap_or(0.0);
            let b = w.get(i + 1).copied().unwrap_or(0.0);
            a + (b - a) * (pos - i as f64) as f32
        }
        _ => 0.0,
    }
}

/// The current grain-window amplitude, advancing the window state. For the default window (`win_type <
/// 0`) this is the `sin²` recurrence `y0 = b1*y1 - y2; amp = y1²`; otherwise it interpolates the custom
/// window buffer `win`.
fn window_amp(
    win_type: f32,
    b1: f64,
    y1: &mut f64,
    y2: &mut f64,
    win_pos: &mut f64,
    win_inc: f64,
    win: Option<&[f32]>,
) -> f32 {
    if win_type < 0.0 {
        let amp = (*y1 * *y1) as f32;
        let y0 = b1 * *y1 - *y2;
        *y2 = *y1;
        *y1 = y0;
        amp
    } else {
        let amp = sample_window(win, *win_pos);
        *win_pos += win_inc;
        amp
    }
}

/// A grain's start channel and equal-power pan gains for `pan` across `num_out` output channels
/// (scsynth's `CALC_GRAIN_PAN`).
fn grain_pan(pan: f32, num_out: usize) -> (usize, f32, f32) {
    use core::f32::consts::FRAC_PI_2;
    if num_out > 2 {
        let pan = ops::wrap(pan * 0.5, 0.0, 1.0);
        let cpan = num_out as f32 * pan + 0.5;
        let ipan = math::floor(cpan);
        let angle = (cpan - ipan) * FRAC_PI_2;
        let mut chan = ipan as usize;
        if chan >= num_out {
            chan -= num_out;
        }
        (chan, math::cos(angle), math::sin(angle))
    } else if num_out == 2 {
        let angle = (pan * 0.5 + 0.5).clamp(0.0, 1.0) * FRAC_PI_2;
        (0, math::cos(angle), math::sin(angle))
    } else {
        (0, 1.0, 0.0)
    }
}

/// The grain length in samples from a `dur` (seconds), floored to at least 4 (scsynth's minimum).
fn grain_counter(dur: f32, sample_rate: f64) -> i32 {
    (math::floor(dur as f64 * sample_rate) as i32).max(4)
}

/// Seed a grain's window state from its length and window type (`win_type` = `envbufnum`, `< 0` for the
/// default `sin²` window). Returns `(b1, y1, y2, win_pos, win_inc)`.
fn init_window(win_type: f32, counter: i32, win: Option<&[f32]>) -> (f64, f64, f64, f64, f64) {
    if win_type < 0.0 {
        let w = core::f64::consts::PI / counter as f64;
        (2.0 * math::cos(w), math::sin(w), 0.0, 0.0, 0.0)
    } else {
        let win_samples = win.map(<[f32]>::len).unwrap_or(0);
        (0.0, 0.0, 0.0, 0.0, win_samples as f64 / counter as f64)
    }
}

/// Accumulate `value * pan` into output channels `chan` and (for >1 output) `chan + 1` at sample `j`.
fn pan_out(
    outs: &mut Outputs<'_>,
    chan: usize,
    j: usize,
    value: f32,
    pan1: f32,
    pan2: f32,
    num_out: usize,
) {
    outs.audio(chan)[j] += value * pan1;
    if num_out > 1 {
        let chan2 = if chan + 1 >= num_out { 0 } else { chan + 1 };
        outs.audio(chan2)[j] += value * pan2;
    }
}

/// The env (window) buffer for `envbufnum` (`< 0` -> the default window, `None`).
fn env_buffer(buffers: &BufferTable, envbufnum: f32) -> Option<&[f32]> {
    if envbufnum >= 0.0 {
        buffer_at(buffers, envbufnum as usize).map(|b| b.data())
    } else {
        None
    }
}

/// One grain of [`GrainSin`]: a windowed sine.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GrainSinG {
    b1: f64,
    y1: f64,
    y2: f64,
    win_pos: f64,
    win_inc: f64,
    phase: f64,
    inc: f64,
    counter: i32,
    chan: i32,
    pan1: f32,
    pan2: f32,
    win_type: f32,
    _pad: u32,
}

impl GrainSinG {
    /// Render this grain from output sample `off` to the end of the block (or the end of the grain),
    /// accumulating into `outs`; returns `true` once the grain is finished.
    fn render(
        &mut self,
        outs: &mut Outputs<'_>,
        block: usize,
        off: usize,
        table: &[f32],
        win: Option<&[f32]>,
        num_out: usize,
    ) -> bool {
        let n = (block - off).min(self.counter.max(0) as usize);
        let chan = self.chan as usize;
        for j in off..off + n {
            let osc = lookup_cycle(table, self.phase as f32);
            let amp = window_amp(
                self.win_type,
                self.b1,
                &mut self.y1,
                &mut self.y2,
                &mut self.win_pos,
                self.win_inc,
                win,
            );
            pan_out(outs, chan, j, amp * osc, self.pan1, self.pan2, num_out);
            let p = self.phase + self.inc;
            self.phase = p - math::floor(p);
            self.counter -= 1;
        }
        self.counter <= 0
    }
}

/// `GrainSin.ar(numChannels, trigger, dur, freq, pan, envbufnum, maxGrains)`: granular synthesis with
/// windowed sine grains. A rising `trigger` spawns a grain of length `dur` at frequency `freq`, panned
/// by `pan` across the `numChannels` outputs. `envbufnum` selects a window buffer (`-1` for the default
/// `sin²` window); `maxGrains` caps simultaneous grains (at most 64 here).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GrainSin {
    grains: [GrainSinG; MAX_GRAINS],
    num_active: u32,
    num_channels: u32,
    max_grains: u32,
    prev_trig: f32,
    trig_audio: u32,
    _pad: u32,
}

impl GrainSin {
    const TRIG: usize = 0;
    const DUR: usize = 1;
    const FREQ: usize = 2;
    const PAN: usize = 3;
    const ENVBUF: usize = 4;

    /// Spawn a grain at output offset `off`, reading its parameters at that sample, and render it for
    /// the rest of the block.
    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        ins: &Inputs<'_>,
        outs: &mut Outputs<'_>,
        buffers: &BufferTable,
        block: usize,
        off: usize,
        table: &[f32],
        sample_rate: f64,
    ) {
        if self.num_active as usize >= self.max_grains as usize {
            return; // too many grains
        }
        let num_out = self.num_channels as usize;
        let counter = grain_counter(sample_channel(ins, Self::DUR, off), sample_rate);
        let freq = sample_channel(ins, Self::FREQ, off);
        let pan = sample_channel(ins, Self::PAN, off);
        let win_type = ins.control(Self::ENVBUF);
        let win = env_buffer(buffers, win_type);
        let (b1, y1, y2, win_pos, win_inc) = init_window(win_type, counter, win);
        let (chan, pan1, pan2) = grain_pan(pan, num_out);
        let mut grain = GrainSinG {
            b1,
            y1,
            y2,
            win_pos,
            win_inc,
            phase: 0.0,
            inc: freq as f64 / sample_rate,
            counter,
            chan: chan as i32,
            pan1,
            pan2,
            win_type,
            _pad: 0,
        };
        let finished = grain.render(outs, block, off, table, win, num_out);
        if !finished {
            self.grains[self.num_active as usize] = grain;
            self.num_active += 1;
        }
    }
}

impl Unit for GrainSin {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_out = self.num_channels as usize;
        let block = ctx.outs.audio(0).len();
        for ch in 0..num_out {
            ctx.outs.audio(ch).fill(0.0);
        }
        let sample_rate = ctx.audio.sample_rate;
        let table = ctx.wavetables.sine();
        let ins = ctx.ins;
        let win = env_buffer(ctx.buffers, ins.control(Self::ENVBUF));

        // Advance every active grain from the block start; remove finished grains by swapping the last
        // active grain into the vacated slot.
        let mut k = 0;
        while k < self.num_active as usize {
            let finished = self.grains[k].render(&mut ctx.outs, block, 0, table, win, num_out);
            if finished {
                self.num_active -= 1;
                self.grains[k] = self.grains[self.num_active as usize];
            } else {
                k += 1;
            }
        }

        // Scan the trigger for rising edges and spawn a grain at each.
        if self.trig_audio != 0 {
            let mut prev = self.prev_trig;
            for i in 0..block {
                let t = ins.audio(Self::TRIG)[i];
                if prev <= 0.0 && t > 0.0 {
                    self.spawn(
                        &ins,
                        &mut ctx.outs,
                        ctx.buffers,
                        block,
                        i,
                        table,
                        sample_rate,
                    );
                }
                prev = t;
            }
            self.prev_trig = prev;
        } else {
            let t = ins.control(Self::TRIG);
            if self.prev_trig <= 0.0 && t > 0.0 {
                self.spawn(
                    &ins,
                    &mut ctx.outs,
                    ctx.buffers,
                    block,
                    0,
                    table,
                    sample_rate,
                );
            }
            self.prev_trig = t;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`GrainSin`].
pub struct GrainSinCtor;

impl UnitDef for GrainSinCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 6 {
            return Err(BuildError::WrongInputCount);
        }
        let max_grains = ctx
            .const_input(5)
            .map(|m| (m as usize).clamp(1, MAX_GRAINS))
            .unwrap_or(MAX_GRAINS) as u32;
        Ok(unit_spec(GrainSin {
            grains: [GrainSinG::zeroed(); MAX_GRAINS],
            num_active: 0,
            num_channels: ctx.num_outputs.max(1) as u32,
            max_grains,
            prev_trig: 0.0,
            trig_audio: (ctx.input_rates.first() == Some(&Rate::Audio)) as u32,
            _pad: 0,
        }))
    }
}
