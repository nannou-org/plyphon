//! Granular-synthesis UGens - plyphon's ports of scsynth's `GrainSin`/`GrainFM`/`GrainIn`/`GrainBuf`,
//! `TGrains` and `Warp1` (`GrainUGens.cpp`, `DelayUGens.cpp`).
//!
//! Most are a bank of short overlapping *grains* spawned on a rising trigger. A grain is a windowed
//! tone (a sine for `GrainSin`, an FM pair for `GrainFM`, a live input for `GrainIn`, a buffer read for
//! `GrainBuf`/`TGrains`) that fades in and out over `dur` seconds and is panned across the output
//! channels. The grains live in a fixed in-struct array (no allocation); a finished grain is removed by
//! swapping the last active grain into its slot. The default window is scsynth's inline `sin²` (Hann)
//! recurrence; an `envbufnum >= 0` reads a user buffer as the window instead.
//!
//! `Warp1` is the exception: a self-triggering granular time-stretcher with no trigger input. Each
//! output channel runs its own independent grain cloud (a decorrelated read of the same buffer), so its
//! per-channel grain banks live in [auxiliary memory](crate::unit::Aux) sized from the channel count,
//! and its window sizes are randomised by a per-unit [`Rng`].

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::{buffer_at, sample_channel};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    BuiltUnit, DoneAction, Inputs, Outputs, ProcessCtx, Unit, unit_spec, unit_spec_aux,
};
use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::interp::{cubicinterp, lininterp};
use plyphon_dsp::math;
use plyphon_dsp::ops;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::rng::Rng;
use plyphon_dsp::wavetable::lookup_cycle;

/// scsynth's fixed cap on `Warp1`'s output channels (`WarpWinGrain mGrains[16][kMaxGrains]`); the
/// per-channel grain counters live in fixed arrays of this length in the unit's `Pod` state.
const MAX_WARP_CHANNELS: usize = 16;

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

/// The default `sin²` (Hann) window amplitude, advancing the recurrence `y0 = b1*y1 - y2; amp = y1²`.
fn default_window_amp(b1: f64, y1: &mut f64, y2: &mut f64) -> f32 {
    let amp = (*y1 * *y1) as f32;
    let y0 = b1 * *y1 - *y2;
    *y2 = *y1;
    *y1 = y0;
    amp
}

/// The current grain-window amplitude, advancing the window state. For the default window (`win_type <
/// 0`) this is the `sin²` recurrence; otherwise it interpolates the custom window buffer `win`.
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
        default_window_amp(b1, y1, y2)
    } else {
        let amp = sample_window(win, *win_pos);
        *win_pos += win_inc;
        amp
    }
}

/// Read mono buffer `buf` at fractional frame `phase` (wrapped into the buffer) with `interp` (`>= 4`
/// cubic, `>= 2` linear, else none) - the grain buffer read.
fn read_buffer(buf: &[f32], phase: f64, interp: i32) -> f32 {
    let n = buf.len();
    if n == 0 {
        return 0.0;
    }
    let p = math::rem_euclid(phase, n as f64);
    let i = (p as usize) % n;
    let frac = (p - i as f64) as f32;
    if interp >= 4 {
        let i0 = (i + n - 1) % n;
        let i2 = (i + 1) % n;
        let i3 = (i + 2) % n;
        cubicinterp(frac, buf[i0], buf[i], buf[i2], buf[i3])
    } else if interp >= 2 {
        lininterp(frac, buf[i], buf[(i + 1) % n])
    } else {
        buf[i]
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

/// Bind the pan target channels out of `outs` once per grain render: the grain's primary channel
/// and (for multichannel output) the wrapped neighbour it pans against - so the per-sample
/// accumulate ([`pan_out`]) indexes pre-bound slices instead of re-slicing the scratch twice per
/// sample.
fn pan_channels<'a>(
    outs: &'a mut Outputs<'_>,
    chan: usize,
    num_out: usize,
) -> (&'a mut [f32], Option<&'a mut [f32]>) {
    if num_out > 1 {
        let chan2 = if chan + 1 >= num_out { 0 } else { chan + 1 };
        match outs.audio_pair(chan, chan2) {
            Some((c1, c2)) => (c1, Some(c2)),
            // Unreachable (`chan2 != chan` whenever `num_out > 1`, and both are in scratch
            // range); degrade to silence rather than panic - `pan_out` writes through `get_mut`.
            None => (&mut [], None),
        }
    } else {
        (outs.audio(chan), None)
    }
}

/// Accumulate `value * pan` into the grain's pre-bound pan channels at sample `j`.
#[inline]
fn pan_out(
    ch1: &mut [f32],
    ch2: &mut Option<&mut [f32]>,
    j: usize,
    value: f32,
    pan1: f32,
    pan2: f32,
) {
    if let Some(o) = ch1.get_mut(j) {
        *o += value * pan1;
    }
    if let Some(o) = ch2.as_mut().and_then(|c2| c2.get_mut(j)) {
        *o += value * pan2;
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

/// Call `spawn(off)` at each rising edge of the trigger over the block, and return the new previous
/// value. An audio-rate trigger is scanned per sample; a control-rate one fires at most once (offset 0).
fn scan_triggers(
    trig_audio: bool,
    prev: f32,
    ctrl: f32,
    audio: &[f32],
    mut spawn: impl FnMut(usize),
) -> f32 {
    if trig_audio {
        let mut p = prev;
        for (i, &t) in audio.iter().enumerate() {
            if p <= 0.0 && t > 0.0 {
                spawn(i);
            }
            p = t;
        }
        p
    } else {
        if prev <= 0.0 && ctrl > 0.0 {
            spawn(0);
        }
        ctrl
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
        let (ch1, mut ch2) = pan_channels(outs, chan, num_out);
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
            pan_out(ch1, &mut ch2, j, amp * osc, self.pan1, self.pan2);
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
        block: usize,
        off: usize,
        table: &[f32],
        win: Option<&[f32]>,
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
        let sample_rate = ctx.own.sample_rate;
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
        let trig_audio = self.trig_audio != 0;
        let audio = if trig_audio {
            ins.audio(Self::TRIG)
        } else {
            &[]
        };
        self.prev_trig = scan_triggers(
            trig_audio,
            self.prev_trig,
            ins.control(Self::TRIG),
            audio,
            |off| {
                self.spawn(&ins, &mut ctx.outs, block, off, table, win, sample_rate);
            },
        );
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
        Ok(unit_spec(GrainSin {
            grains: [GrainSinG::zeroed(); MAX_GRAINS],
            num_active: 0,
            num_channels: ctx.num_outputs.max(1) as u32,
            max_grains: max_grains(ctx, 5),
            prev_trig: 0.0,
            trig_audio: trig_is_audio(ctx),
            _pad: 0,
        }))
    }
}

/// The `maxGrains` input (index `i`) clamped to `1..=MAX_GRAINS`, or the cap if it is not constant.
fn max_grains(ctx: &BuildContext<'_>, i: usize) -> u32 {
    ctx.const_input(i)
        .map(|m| (m as usize).clamp(1, MAX_GRAINS))
        .unwrap_or(MAX_GRAINS) as u32
}

/// Whether the trigger input (index 0) is audio-rate.
fn trig_is_audio(ctx: &BuildContext<'_>) -> u32 {
    (ctx.input_rates.first() == Some(&Rate::Audio)) as u32
}

/// One grain of [`GrainFM`]: a windowed FM carrier/modulator pair.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GrainFMG {
    b1: f64,
    y1: f64,
    y2: f64,
    win_pos: f64,
    win_inc: f64,
    cphase: f64,
    mphase: f64,
    minc: f64,
    counter: i32,
    chan: i32,
    carbase: f32,
    deviation: f32,
    pan1: f32,
    pan2: f32,
    win_type: f32,
    _pad: u32,
}

impl GrainFMG {
    #[allow(clippy::too_many_arguments)]
    fn render(
        &mut self,
        outs: &mut Outputs<'_>,
        block: usize,
        off: usize,
        table: &[f32],
        win: Option<&[f32]>,
        num_out: usize,
        sample_dur: f64,
    ) -> bool {
        let n = (block - off).min(self.counter.max(0) as usize);
        let chan = self.chan as usize;
        let (ch1, mut ch2) = pan_channels(outs, chan, num_out);
        for j in off..off + n {
            let thismod = lookup_cycle(table, self.mphase as f32) as f64 * self.deviation as f64;
            let carrier = lookup_cycle(table, self.cphase as f32);
            let amp = window_amp(
                self.win_type,
                self.b1,
                &mut self.y1,
                &mut self.y2,
                &mut self.win_pos,
                self.win_inc,
                win,
            );
            pan_out(ch1, &mut ch2, j, amp * carrier, self.pan1, self.pan2);
            let cp = self.cphase + (self.carbase as f64 + thismod) * sample_dur;
            self.cphase = cp - math::floor(cp);
            let mp = self.mphase + self.minc;
            self.mphase = mp - math::floor(mp);
            self.counter -= 1;
        }
        self.counter <= 0
    }
}

/// `GrainFM.ar(numChannels, trigger, dur, carfreq, modfreq, index, pan, envbufnum, maxGrains)`: FM
/// granular synthesis - each grain is a sine carrier at `carfreq` frequency-modulated by a sine at
/// `modfreq` with modulation `index` (peak deviation `index * modfreq` Hz), windowed and panned like
/// [`GrainSin`].
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GrainFM {
    grains: [GrainFMG; MAX_GRAINS],
    num_active: u32,
    num_channels: u32,
    max_grains: u32,
    prev_trig: f32,
    trig_audio: u32,
    _pad: u32,
}

impl GrainFM {
    const TRIG: usize = 0;
    const DUR: usize = 1;
    const CARFREQ: usize = 2;
    const MODFREQ: usize = 3;
    const INDEX: usize = 4;
    const PAN: usize = 5;
    const ENVBUF: usize = 6;

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        ins: &Inputs<'_>,
        outs: &mut Outputs<'_>,
        block: usize,
        off: usize,
        table: &[f32],
        win: Option<&[f32]>,
        sample_rate: f64,
    ) {
        if self.num_active as usize >= self.max_grains as usize {
            return;
        }
        let num_out = self.num_channels as usize;
        let counter = grain_counter(sample_channel(ins, Self::DUR, off), sample_rate);
        let carfreq = sample_channel(ins, Self::CARFREQ, off);
        let modfreq = sample_channel(ins, Self::MODFREQ, off);
        let index = sample_channel(ins, Self::INDEX, off);
        let pan = sample_channel(ins, Self::PAN, off);
        let win_type = ins.control(Self::ENVBUF);
        let (b1, y1, y2, win_pos, win_inc) = init_window(win_type, counter, win);
        let (chan, pan1, pan2) = grain_pan(pan, num_out);
        let sample_dur = 1.0 / sample_rate;
        let mut grain = GrainFMG {
            b1,
            y1,
            y2,
            win_pos,
            win_inc,
            cphase: 0.0,
            mphase: 0.0,
            minc: modfreq as f64 * sample_dur,
            counter,
            chan: chan as i32,
            carbase: carfreq,
            deviation: index * modfreq,
            pan1,
            pan2,
            win_type,
            _pad: 0,
        };
        if !grain.render(outs, block, off, table, win, num_out, sample_dur) {
            self.grains[self.num_active as usize] = grain;
            self.num_active += 1;
        }
    }
}

impl Unit for GrainFM {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_out = self.num_channels as usize;
        let block = ctx.outs.audio(0).len();
        for ch in 0..num_out {
            ctx.outs.audio(ch).fill(0.0);
        }
        let sample_rate = ctx.own.sample_rate;
        let sample_dur = ctx.own.sample_dur;
        let table = ctx.wavetables.sine();
        let ins = ctx.ins;
        let win = env_buffer(ctx.buffers, ins.control(Self::ENVBUF));

        let mut k = 0;
        while k < self.num_active as usize {
            if self.grains[k].render(&mut ctx.outs, block, 0, table, win, num_out, sample_dur) {
                self.num_active -= 1;
                self.grains[k] = self.grains[self.num_active as usize];
            } else {
                k += 1;
            }
        }

        let trig_audio = self.trig_audio != 0;
        let audio = if trig_audio {
            ins.audio(Self::TRIG)
        } else {
            &[]
        };
        self.prev_trig = scan_triggers(
            trig_audio,
            self.prev_trig,
            ins.control(Self::TRIG),
            audio,
            |off| {
                self.spawn(&ins, &mut ctx.outs, block, off, table, win, sample_rate);
            },
        );
        DoneAction::Nothing
    }
}

/// Constructor for [`GrainFM`].
pub struct GrainFMCtor;

impl UnitDef for GrainFMCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 8 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(GrainFM {
            grains: [GrainFMG::zeroed(); MAX_GRAINS],
            num_active: 0,
            num_channels: ctx.num_outputs.max(1) as u32,
            max_grains: max_grains(ctx, 7),
            prev_trig: 0.0,
            trig_audio: trig_is_audio(ctx),
            _pad: 0,
        }))
    }
}

/// One grain of [`GrainIn`]: a window applied to the live input.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GrainInG {
    b1: f64,
    y1: f64,
    y2: f64,
    win_pos: f64,
    win_inc: f64,
    counter: i32,
    chan: i32,
    pan1: f32,
    pan2: f32,
    win_type: f32,
    _pad: u32,
}

impl GrainInG {
    fn render(
        &mut self,
        outs: &mut Outputs<'_>,
        block: usize,
        off: usize,
        win: Option<&[f32]>,
        num_out: usize,
        input: &[f32],
    ) -> bool {
        let n = (block - off).min(self.counter.max(0) as usize);
        let chan = self.chan as usize;
        let (ch1, mut ch2) = pan_channels(outs, chan, num_out);
        for j in off..off + n {
            let x = input.get(j).copied().unwrap_or(0.0);
            let amp = window_amp(
                self.win_type,
                self.b1,
                &mut self.y1,
                &mut self.y2,
                &mut self.win_pos,
                self.win_inc,
                win,
            );
            pan_out(ch1, &mut ch2, j, amp * x, self.pan1, self.pan2);
            self.counter -= 1;
        }
        self.counter <= 0
    }
}

/// `GrainIn.ar(numChannels, trigger, dur, in, pan, envbufnum, maxGrains)`: grains that window the live
/// input signal `in` - a rising trigger starts a `dur`-long window over `in`, panned across the
/// outputs. Overlapping grains sum windowed copies of the same input.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GrainIn {
    grains: [GrainInG; MAX_GRAINS],
    num_active: u32,
    num_channels: u32,
    max_grains: u32,
    prev_trig: f32,
    trig_audio: u32,
    _pad: u32,
}

impl GrainIn {
    const TRIG: usize = 0;
    const DUR: usize = 1;
    const IN: usize = 2;
    const PAN: usize = 3;
    const ENVBUF: usize = 4;

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        ins: &Inputs<'_>,
        outs: &mut Outputs<'_>,
        block: usize,
        off: usize,
        win: Option<&[f32]>,
        input: &[f32],
        sample_rate: f64,
    ) {
        if self.num_active as usize >= self.max_grains as usize {
            return;
        }
        let num_out = self.num_channels as usize;
        let counter = grain_counter(sample_channel(ins, Self::DUR, off), sample_rate);
        let pan = sample_channel(ins, Self::PAN, off);
        let win_type = ins.control(Self::ENVBUF);
        let (b1, y1, y2, win_pos, win_inc) = init_window(win_type, counter, win);
        let (chan, pan1, pan2) = grain_pan(pan, num_out);
        let mut grain = GrainInG {
            b1,
            y1,
            y2,
            win_pos,
            win_inc,
            counter,
            chan: chan as i32,
            pan1,
            pan2,
            win_type,
            _pad: 0,
        };
        if !grain.render(outs, block, off, win, num_out, input) {
            self.grains[self.num_active as usize] = grain;
            self.num_active += 1;
        }
    }
}

impl Unit for GrainIn {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_out = self.num_channels as usize;
        let block = ctx.outs.audio(0).len();
        for ch in 0..num_out {
            ctx.outs.audio(ch).fill(0.0);
        }
        let sample_rate = ctx.own.sample_rate;
        let ins = ctx.ins;
        let win = env_buffer(ctx.buffers, ins.control(Self::ENVBUF));
        let input = ins.audio(Self::IN);

        let mut k = 0;
        while k < self.num_active as usize {
            if self.grains[k].render(&mut ctx.outs, block, 0, win, num_out, input) {
                self.num_active -= 1;
                self.grains[k] = self.grains[self.num_active as usize];
            } else {
                k += 1;
            }
        }

        let trig_audio = self.trig_audio != 0;
        let audio = if trig_audio {
            ins.audio(Self::TRIG)
        } else {
            &[]
        };
        self.prev_trig = scan_triggers(
            trig_audio,
            self.prev_trig,
            ins.control(Self::TRIG),
            audio,
            |off| {
                self.spawn(&ins, &mut ctx.outs, block, off, win, input, sample_rate);
            },
        );
        DoneAction::Nothing
    }
}

/// Constructor for [`GrainIn`].
pub struct GrainInCtor;

impl UnitDef for GrainInCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 6 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(GrainIn {
            grains: [GrainInG::zeroed(); MAX_GRAINS],
            num_active: 0,
            num_channels: ctx.num_outputs.max(1) as u32,
            max_grains: max_grains(ctx, 5),
            prev_trig: 0.0,
            trig_audio: trig_is_audio(ctx),
            _pad: 0,
        }))
    }
}

/// The `(num_frames, sample_rate)` of the buffer at `bufnum`, or `(0, sample_rate)` if absent.
fn buffer_info(buffers: &BufferTable, bufnum: i32, sample_rate: f64) -> (usize, f64) {
    match buffer_at(buffers, bufnum.max(0) as usize) {
        Some(b) if bufnum >= 0 => (b.num_frames(), b.sample_rate()),
        _ => (0, sample_rate),
    }
}

/// One grain of [`GrainBuf`]: a windowed buffer read.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GrainBufG {
    b1: f64,
    y1: f64,
    y2: f64,
    win_pos: f64,
    win_inc: f64,
    phase: f64,
    rate: f64,
    counter: i32,
    chan: i32,
    bufnum: i32,
    interp: i32,
    pan1: f32,
    pan2: f32,
    win_type: f32,
    _pad: u32,
}

impl GrainBufG {
    fn render(
        &mut self,
        outs: &mut Outputs<'_>,
        block: usize,
        off: usize,
        buffers: &BufferTable,
        win: Option<&[f32]>,
        num_out: usize,
    ) -> bool {
        let n = (block - off).min(self.counter.max(0) as usize);
        let chan = self.chan as usize;
        let (ch1, mut ch2) = pan_channels(outs, chan, num_out);
        let snd = buffer_at(buffers, self.bufnum.max(0) as usize).map(|b| b.data());
        for j in off..off + n {
            let sample = snd.map_or(0.0, |b| read_buffer(b, self.phase, self.interp));
            let amp = window_amp(
                self.win_type,
                self.b1,
                &mut self.y1,
                &mut self.y2,
                &mut self.win_pos,
                self.win_inc,
                win,
            );
            pan_out(ch1, &mut ch2, j, amp * sample, self.pan1, self.pan2);
            self.phase += self.rate;
            self.counter -= 1;
        }
        self.counter <= 0
    }
}

/// `GrainBuf.ar(numChannels, trigger, dur, sndbuf, rate, pos, interp, pan, envbufnum, maxGrains)`:
/// granular playback from a mono buffer. Each grain reads `sndbuf` from normalised start `pos` at
/// `rate` (pitch), interpolated per `interp` (1 none / 2 linear / 4 cubic), windowed and panned.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GrainBuf {
    grains: [GrainBufG; MAX_GRAINS],
    num_active: u32,
    num_channels: u32,
    max_grains: u32,
    prev_trig: f32,
    trig_audio: u32,
    _pad: u32,
}

impl GrainBuf {
    const TRIG: usize = 0;
    const DUR: usize = 1;
    const SNDBUF: usize = 2;
    const RATE: usize = 3;
    const POS: usize = 4;
    const INTERP: usize = 5;
    const PAN: usize = 6;
    const ENVBUF: usize = 7;

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        ins: &Inputs<'_>,
        outs: &mut Outputs<'_>,
        buffers: &BufferTable,
        block: usize,
        off: usize,
        win: Option<&[f32]>,
        sample_rate: f64,
    ) {
        if self.num_active as usize >= self.max_grains as usize {
            return;
        }
        let num_out = self.num_channels as usize;
        let counter = grain_counter(sample_channel(ins, Self::DUR, off), sample_rate);
        let bufnum = sample_channel(ins, Self::SNDBUF, off) as i32;
        let rate_in = sample_channel(ins, Self::RATE, off);
        let pos = sample_channel(ins, Self::POS, off);
        let interp = sample_channel(ins, Self::INTERP, off) as i32;
        let pan = sample_channel(ins, Self::PAN, off);
        let win_type = ins.control(Self::ENVBUF);
        let (buf_frames, buf_sr) = buffer_info(buffers, bufnum, sample_rate);
        let (b1, y1, y2, win_pos, win_inc) = init_window(win_type, counter, win);
        let (chan, pan1, pan2) = grain_pan(pan, num_out);
        let mut grain = GrainBufG {
            b1,
            y1,
            y2,
            win_pos,
            win_inc,
            phase: pos as f64 * buf_frames as f64,
            rate: rate_in as f64 * buf_sr / sample_rate,
            counter,
            chan: chan as i32,
            bufnum,
            interp,
            pan1,
            pan2,
            win_type,
            _pad: 0,
        };
        if !grain.render(outs, block, off, buffers, win, num_out) {
            self.grains[self.num_active as usize] = grain;
            self.num_active += 1;
        }
    }
}

impl Unit for GrainBuf {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_out = self.num_channels as usize;
        let block = ctx.outs.audio(0).len();
        for ch in 0..num_out {
            ctx.outs.audio(ch).fill(0.0);
        }
        let sample_rate = ctx.own.sample_rate;
        let ins = ctx.ins;
        let win = env_buffer(ctx.buffers, ins.control(Self::ENVBUF));

        let mut k = 0;
        while k < self.num_active as usize {
            if self.grains[k].render(&mut ctx.outs, block, 0, ctx.buffers, win, num_out) {
                self.num_active -= 1;
                self.grains[k] = self.grains[self.num_active as usize];
            } else {
                k += 1;
            }
        }

        let trig_audio = self.trig_audio != 0;
        let audio = if trig_audio {
            ins.audio(Self::TRIG)
        } else {
            &[]
        };
        self.prev_trig = scan_triggers(
            trig_audio,
            self.prev_trig,
            ins.control(Self::TRIG),
            audio,
            |off| {
                self.spawn(
                    &ins,
                    &mut ctx.outs,
                    ctx.buffers,
                    block,
                    off,
                    win,
                    sample_rate,
                );
            },
        );
        DoneAction::Nothing
    }
}

/// Constructor for [`GrainBuf`].
pub struct GrainBufCtor;

impl UnitDef for GrainBufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 9 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(GrainBuf {
            grains: [GrainBufG::zeroed(); MAX_GRAINS],
            num_active: 0,
            num_channels: ctx.num_outputs.max(1) as u32,
            max_grains: max_grains(ctx, 8),
            prev_trig: 0.0,
            trig_audio: trig_is_audio(ctx),
            _pad: 0,
        }))
    }
}

/// One grain of [`TGrains`]: a windowed buffer read whose gains already fold in the grain amplitude.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TGrainG {
    phase: f64,
    rate: f64,
    b1: f64,
    y1: f64,
    y2: f64,
    pan1: f32,
    pan2: f32,
    counter: i32,
    bufnum: i32,
    chan: i32,
    interp: i32,
}

impl TGrainG {
    fn render(
        &mut self,
        outs: &mut Outputs<'_>,
        block: usize,
        off: usize,
        buffers: &BufferTable,
        num_out: usize,
    ) -> bool {
        let n = (block - off).min(self.counter.max(0) as usize);
        let chan = self.chan as usize;
        let (ch1, mut ch2) = pan_channels(outs, chan, num_out);
        let snd = buffer_at(buffers, self.bufnum.max(0) as usize).map(|b| b.data());
        for j in off..off + n {
            let sample = snd.map_or(0.0, |b| read_buffer(b, self.phase, self.interp));
            let amp = default_window_amp(self.b1, &mut self.y1, &mut self.y2);
            pan_out(ch1, &mut ch2, j, amp * sample, self.pan1, self.pan2);
            self.phase += self.rate;
            self.counter -= 1;
        }
        self.counter <= 0
    }
}

/// `TGrains.ar(numChannels, trigger, bufnum, rate, centerPos, dur, pan, amp, interp)`: triggered buffer
/// grains centred on `centerPos` (seconds). Like [`GrainBuf`] but always using the default `sin²`
/// window, with an explicit `amp` folded into the pan gains and the grain centred (not started) on the
/// position.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TGrains {
    grains: [TGrainG; MAX_GRAINS],
    num_active: u32,
    num_channels: u32,
    prev_trig: f32,
    trig_audio: u32,
}

impl TGrains {
    const TRIG: usize = 0;
    const BUFNUM: usize = 1;
    const RATE: usize = 2;
    const CENTER: usize = 3;
    const DUR: usize = 4;
    const PAN: usize = 5;
    const AMP: usize = 6;
    const INTERP: usize = 7;

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        ins: &Inputs<'_>,
        outs: &mut Outputs<'_>,
        buffers: &BufferTable,
        block: usize,
        off: usize,
        sample_rate: f64,
    ) {
        if self.num_active as usize >= MAX_GRAINS {
            return;
        }
        let num_out = self.num_channels as usize;
        let counter = grain_counter(sample_channel(ins, Self::DUR, off), sample_rate);
        let bufnum = sample_channel(ins, Self::BUFNUM, off) as i32;
        let rate_in = sample_channel(ins, Self::RATE, off);
        let center = sample_channel(ins, Self::CENTER, off);
        let pan = sample_channel(ins, Self::PAN, off);
        let amp = sample_channel(ins, Self::AMP, off);
        let interp = sample_channel(ins, Self::INTERP, off) as i32;
        let (_buf_frames, buf_sr) = buffer_info(buffers, bufnum, sample_rate);
        let rate = rate_in as f64 * buf_sr / sample_rate;
        // The grain is centred on `centerPos` seconds, so it starts half a grain before it.
        let phase = center as f64 * buf_sr - 0.5 * counter as f64 * rate;
        let (chan, cos, sin) = grain_pan(pan, num_out);
        let w = core::f64::consts::PI / counter as f64;
        let mut grain = TGrainG {
            phase,
            rate,
            b1: 2.0 * math::cos(w),
            y1: math::sin(w),
            y2: 0.0,
            pan1: amp * cos,
            pan2: amp * sin,
            counter,
            bufnum,
            chan: chan as i32,
            interp,
        };
        if !grain.render(outs, block, off, buffers, num_out) {
            self.grains[self.num_active as usize] = grain;
            self.num_active += 1;
        }
    }
}

impl Unit for TGrains {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_out = self.num_channels as usize;
        let block = ctx.outs.audio(0).len();
        for ch in 0..num_out {
            ctx.outs.audio(ch).fill(0.0);
        }
        let sample_rate = ctx.own.sample_rate;
        let ins = ctx.ins;

        let mut k = 0;
        while k < self.num_active as usize {
            if self.grains[k].render(&mut ctx.outs, block, 0, ctx.buffers, num_out) {
                self.num_active -= 1;
                self.grains[k] = self.grains[self.num_active as usize];
            } else {
                k += 1;
            }
        }

        let trig_audio = self.trig_audio != 0;
        let audio = if trig_audio {
            ins.audio(Self::TRIG)
        } else {
            &[]
        };
        self.prev_trig = scan_triggers(
            trig_audio,
            self.prev_trig,
            ins.control(Self::TRIG),
            audio,
            |off| {
                self.spawn(&ins, &mut ctx.outs, ctx.buffers, block, off, sample_rate);
            },
        );
        DoneAction::Nothing
    }
}

/// Constructor for [`TGrains`].
pub struct TGrainsCtor;

impl UnitDef for TGrainsCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 8 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(TGrains {
            grains: [TGrainG::zeroed(); MAX_GRAINS],
            num_active: 0,
            num_channels: ctx.num_outputs.max(1) as u32,
            prev_trig: 0.0,
            trig_audio: trig_is_audio(ctx),
        }))
    }
}

/// One grain of [`Warp1`]: a windowed buffer read written straight to its channel's output (no pan -
/// each `Warp1` channel is its own grain cloud).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WarpG {
    phase: f64,
    rate: f64,
    win_pos: f64,
    win_inc: f64,
    b1: f64,
    y1: f64,
    y2: f64,
    counter: i32,
    interp: i32,
    win_type: f32,
    _pad: u32,
}

impl WarpG {
    /// Render this grain from output sample `off`, accumulating into its channel's `out`; returns
    /// `true` once the grain is finished. The window buffer is re-resolved from `win_type` each block
    /// (scsynth's `GET_GRAIN_AMP_PARAMS`).
    fn render(
        &mut self,
        out: &mut [f32],
        block: usize,
        off: usize,
        snd: &[f32],
        buffers: &BufferTable,
    ) -> bool {
        let n = (block - off).min(self.counter.max(0) as usize);
        let win = env_buffer(buffers, self.win_type);
        for o in out.iter_mut().take(off + n).skip(off) {
            let sample = read_buffer(snd, self.phase, self.interp);
            let amp = window_amp(
                self.win_type,
                self.b1,
                &mut self.y1,
                &mut self.y2,
                &mut self.win_pos,
                self.win_inc,
                win,
            );
            *o += amp * sample;
            self.phase += self.rate;
            self.counter -= 1;
        }
        self.counter <= 0
    }
}

/// `Warp1.ar(numChannels, bufnum, pointer, freqScale, windowSize, envbufnum, overlaps, windowRandRatio,
/// interp)`: a granular time-stretcher/pitch-shifter. It reads mono `bufnum` from normalised position
/// `pointer` in overlapping windowed grains at pitch `freqScale`, self-triggering every
/// `windowSize/overlaps` seconds (no trigger input). Each output channel runs an independent grain
/// cloud - the same buffer read with independently randomised (`windowRandRatio`) window sizes - so the
/// channels decorrelate into a spread.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Warp1 {
    rng: Rng,
    num_channels: u32,
    /// Active grain count per channel (indices `0..num_channels`).
    num_active: [u32; MAX_WARP_CHANNELS],
    /// Samples until the next grain spawns per channel; seeded to `1` so a grain starts at once.
    next_grain: [i32; MAX_WARP_CHANNELS],
}

impl Warp1 {
    const BUFNUM: usize = 0;
    const POINTER: usize = 1;
    const FREQ_SCALE: usize = 2;
    const WINDOW_SIZE: usize = 3;
    const ENVBUF: usize = 4;
    const OVERLAPS: usize = 5;
    const WINDOW_RAND: usize = 6;
    const INTERP: usize = 7;
}

impl Unit for Warp1 {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_out = self.num_channels as usize;
        let block = ctx.outs.audio(0).len();
        for ch in 0..num_out {
            ctx.outs.audio(ch).fill(0.0);
        }
        let sample_rate = ctx.own.sample_rate;
        let sample_dur = ctx.own.sample_dur;
        let ins = ctx.ins;

        // The sound buffer is resolved once from `bufnum` (scsynth reads it per block, not per grain).
        let bufnum = ins.control(Self::BUFNUM) as i32;
        let (snd, buf_frames, buf_sr) = match buffer_at(ctx.buffers, bufnum.max(0) as usize) {
            Some(b) if bufnum >= 0 => (b.data(), b.num_frames(), b.sample_rate()),
            _ => return DoneAction::Nothing,
        };
        let buf_rate_scale = buf_sr * sample_dur;

        let grains = ctx.aux.cast_mut::<WarpG>();
        for n in 0..num_out {
            let base = n * MAX_GRAINS;
            let bank = &mut grains[base..base + MAX_GRAINS];

            // Render the grains already active on this channel.
            let mut i = 0;
            while i < self.num_active[n] as usize {
                let done = {
                    let out = ctx.outs.audio(n);
                    bank[i].render(out, block, 0, snd, ctx.buffers)
                };
                if done {
                    self.num_active[n] -= 1;
                    bank[i] = bank[self.num_active[n] as usize];
                } else {
                    i += 1;
                }
            }

            // Scan the block for self-triggered spawns.
            let mut next_grain = self.next_grain[n];
            for smp in 0..block {
                next_grain -= 1;
                if next_grain != 0 {
                    continue;
                }
                if self.num_active[n] as usize + 1 >= MAX_GRAINS {
                    break;
                }
                let overlaps = sample_channel(&ins, Self::OVERLAPS, smp);
                let win_rand = sample_channel(&ins, Self::WINDOW_RAND, smp);
                let win_randamt = self.rng.next_bipolar() as f64 * win_rand as f64;
                let raw = sample_channel(&ins, Self::WINDOW_SIZE, smp) as f64 * sample_rate;
                let counter = math::floor(raw + raw * win_randamt).max(4.0) as i32;
                next_grain = (counter as f32 / overlaps) as i32;

                let win_type = sample_channel(&ins, Self::ENVBUF, smp);
                let win = env_buffer(ctx.buffers, win_type);
                let (b1, y1, y2, win_pos, win_inc) = init_window(win_type, counter, win);
                let idx = base + self.num_active[n] as usize;
                grains[idx] = WarpG {
                    phase: sample_channel(&ins, Self::POINTER, smp) as f64 * buf_frames as f64,
                    rate: sample_channel(&ins, Self::FREQ_SCALE, smp) as f64 * buf_rate_scale,
                    win_pos,
                    win_inc,
                    b1,
                    y1,
                    y2,
                    counter,
                    interp: sample_channel(&ins, Self::INTERP, smp) as i32,
                    win_type,
                    _pad: 0,
                };
                self.num_active[n] += 1;
                let done = {
                    let out = ctx.outs.audio(n);
                    grains[idx].render(out, block, smp, snd, ctx.buffers)
                };
                if done {
                    self.num_active[n] -= 1;
                    grains[idx] = grains[base + self.num_active[n] as usize];
                }
            }
            self.next_grain[n] = next_grain;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Warp1`].
pub struct Warp1Ctor;

impl UnitDef for Warp1Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 8 {
            return Err(BuildError::WrongInputCount);
        }
        let num_channels = ctx.num_outputs.max(1);
        if num_channels > MAX_WARP_CHANNELS {
            return Err(BuildError::TooManyOutputs {
                needed: num_channels,
                limit: MAX_WARP_CHANNELS,
            });
        }
        let aux_bytes = num_channels * MAX_GRAINS * core::mem::size_of::<WarpG>();
        Ok(unit_spec_aux(
            Warp1 {
                rng: Rng::new(0),
                num_channels: num_channels as u32,
                num_active: [0; MAX_WARP_CHANNELS],
                next_grain: [1; MAX_WARP_CHANNELS],
            },
            aux_bytes,
            core::mem::align_of::<WarpG>(),
        ))
    }
}
