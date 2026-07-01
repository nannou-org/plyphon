//! The delay-line family - plyphon's ports of scsynth's `DelayN/L/C`, `CombN/L/C`, `AllpassN/L/C` and
//! their buffer-backed twins `BufDelayN/L/C`, `BufCombN/L/C`, `BufAllpassN/L/C` (`DelayUGens.cpp`).
//!
//! The plain `Delay*`/`Comb*`/`Allpass*` use per-instance [auxiliary memory](crate::unit::Aux): the
//! delay line is sized at build time from the scalar `maxdelaytime` and lives in the synth's pool block
//! (the safe stand-in for scsynth's `RTAlloc`'d `float* m_dlybuf`). The `Buf*` twins instead use a
//! `/b_alloc`'d buffer (addressed by `bufnum`, resolved each block via [`buffer_at_mut`]) as the line -
//! so the line is shared, resizable and outlives the synth. A buffer of `N` samples is used only up to
//! its largest power-of-two prefix `2^floor(log2 N)` (scsynth's `BUFMASK`), so the same power-of-two
//! circular addressing applies. All variants share one read kernel; the family splits only three ways:
//!
//! - **interpolation** of the fractional tap - `N` none, `L` linear, `C` cubic (`Interp`);
//! - **feedback** - a plain delay ([`Delay`]) writes the input, while a comb/allpass
//!   ([`FeedbackDelay`]) recirculates the delayed value with a coefficient derived from `decaytime`;
//! - **allpass vs comb** - the allpass additionally subtracts the feed-forward path.
//!
//! The aux arena is *not* zeroed at instantiation (it may recycle a freed synth's dirty memory), so
//! while the line is still filling (`numoutput < len`) reads use scsynth's cold-start (`_z`) guard:
//! any tap before the start of writing reads `0` rather than stale memory.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, buffer_at, buffer_at_mut, unit_spec,
    unit_spec_aux,
};
use plyphon_dsp::interp::{cubicinterp, lininterp};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// `ln(0.001)`, scsynth's `log001`, the decay a comb/allpass reaches over its `decaytime` (-60 dB).
const LOG001: f32 = -6.907_755_4;

/// Calc-variant tags, chosen from the `delaytime` input's rate at build time (scsynth selects a
/// `_next` vs `_next_a` calc func by `INRATE(2)`). Stored as a `u32` so the state stays [`Pod`].
mod calc {
    /// `delaytime` is constant or control-rate: one value per block, slope-interpolated when it
    /// changes (scsynth's `CALCSLOPE`).
    pub const DELAY_CONTROL: u32 = 0;
    /// `delaytime` is audio-rate: recomputed every sample, no slope.
    pub const DELAY_AUDIO: u32 = 1;
}

/// How a fractional delay tap is interpolated. Stored as a `u32` tag so the state stays [`Pod`].
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Interp {
    /// No interpolation - truncate to the nearest integer sample (`DelayN`/`CombN`/`AllpassN`).
    None,
    /// Linear interpolation between the two adjacent samples (`DelayL`/`CombL`/`AllpassL`).
    Lin,
    /// 4-point cubic interpolation (`DelayC`/`CombC`/`AllpassC`).
    Cubic,
}

impl Interp {
    fn to_tag(self) -> u32 {
        match self {
            Interp::None => 0,
            Interp::Lin => 1,
            Interp::Cubic => 2,
        }
    }

    fn from_tag(tag: u32) -> Interp {
        match tag {
            1 => Interp::Lin,
            2 => Interp::Cubic,
            _ => Interp::None,
        }
    }

    /// The minimum delay in samples (scsynth's `minDelaySamples`). Cubic *feedback* delays reserve two
    /// samples of headroom for the interpolator; every other case reserves one (note plain `DelayC`
    /// keeps the `1`-sample minimum, matching scsynth's `InterpolationUnit` inheritance).
    pub(crate) fn min_delay(self, feedback: bool) -> f32 {
        if feedback && self == Interp::Cubic {
            2.0
        } else {
            1.0
        }
    }
}

/// scsynth's `DelayUnit_AllocDelayLine` length in `f32`s: `NEXTPOWEROFTWO(ceil(maxdelay*SR + 1) +
/// BUFLENGTH)`. The `+1` lets a read sit one sample behind a write at the same phase; the `+block`
/// headroom keeps the write head and any delayed read from colliding within a block; the power-of-two
/// length makes circular addressing a single mask.
pub(crate) fn line_len(max_delay: f32, sr: f64, block: usize) -> u32 {
    let base = math::ceil(max_delay.max(0.0) as f64 * sr + 1.0) as i64;
    let len = (base + block as i64).max(1) as u64;
    len.next_power_of_two() as u32
}

/// Clamp a delay in samples to `[min, max]` (scsynth's `CalcDelay`/`sc_clip`). NaN-safe: the
/// `max`/`min` order maps a NaN to `min` rather than propagating it onto the read index.
#[inline]
pub(crate) fn clamp_delay(samples: f32, min: f32, max: f32) -> f32 {
    samples.max(min).min(max)
}

/// scsynth's `sc_CalcFeedback`: the recirculation coefficient giving a -60 dB (factor 0.001) decay
/// over `decaytime` seconds for a loop of length `delaytime`; negative `decaytime` flips the sign.
#[inline]
pub(crate) fn calc_feedback(delaytime: f32, decaytime: f32) -> f32 {
    if delaytime == 0.0 || decaytime == 0.0 {
        return 0.0;
    }
    let absret = math::exp(LOG001 * delaytime / decaytime.abs());
    absret.copysign(decaytime)
}

/// Read the delayed value `idsamp + frac` samples behind the write head, interpolated per `interp`.
///
/// When `warm` (the line has filled) the taps wrap freely with `mask`. While cold, `iwrphase < len`
/// so the read phase is compared as a signed index and any tap before the start of writing
/// contributes `0` - scsynth's `_z` checked helpers, reproduced so recycled dirty aux never leaks.
#[inline]
pub(crate) fn read_delayed(
    buf: &[f32],
    iwrphase: u32,
    mask: u32,
    idsamp: i64,
    frac: f32,
    interp: Interp,
    warm: bool,
) -> f32 {
    if warm {
        let tap = |back: i64| buf[((iwrphase as i64 - back) as u32 & mask) as usize];
        match interp {
            Interp::None => tap(idsamp),
            Interp::Lin => lininterp(frac, tap(idsamp), tap(idsamp + 1)),
            Interp::Cubic => cubicinterp(
                frac,
                tap(idsamp - 1),
                tap(idsamp),
                tap(idsamp + 1),
                tap(idsamp + 2),
            ),
        }
    } else {
        let rd = iwrphase as i64 - idsamp;
        let at = |ph: i64| buf[(ph as u32 & mask) as usize];
        match interp {
            Interp::None => {
                if rd < 0 {
                    0.0
                } else {
                    at(rd)
                }
            }
            Interp::Lin => {
                if rd < 0 {
                    0.0
                } else if rd - 1 < 0 {
                    let d1 = at(rd);
                    d1 - frac * d1
                } else {
                    lininterp(frac, at(rd), at(rd - 1))
                }
            }
            Interp::Cubic => {
                if rd + 1 < 0 {
                    0.0
                } else {
                    let d0 = at(rd + 1);
                    let (d1, d2, d3) = if rd < 0 {
                        (0.0, 0.0, 0.0)
                    } else if rd - 1 < 0 {
                        (at(rd), 0.0, 0.0)
                    } else if rd - 2 < 0 {
                        (at(rd), at(rd - 1), 0.0)
                    } else {
                        (at(rd), at(rd - 1), at(rd - 2))
                    };
                    cubicinterp(frac, d0, d1, d2, d3)
                }
            }
        }
    }
}

/// One plain-delay sample: write `x` at the head, read the (interpolated) delayed tap, advance the
/// head. Writing first lets a cubic tap read one sample ahead of the main tap at the shortest delay.
#[inline]
#[allow(clippy::too_many_arguments)]
fn delay_tick(
    buf: &mut [f32],
    iwrphase: &mut u32,
    mask: u32,
    idsamp: i64,
    frac: f32,
    interp: Interp,
    x: f32,
    warm: bool,
) -> f32 {
    buf[(*iwrphase & mask) as usize] = x;
    let y = read_delayed(buf, *iwrphase, mask, idsamp, frac, interp, warm);
    *iwrphase = iwrphase.wrapping_add(1);
    y
}

/// One feedback (comb/allpass) sample: read the delayed value first, then write the recirculated
/// input. A comb writes `x + feedbk * value` and outputs `value`; an allpass writes `x + feedbk *
/// value` and outputs `value - feedbk * written`, cancelling the feed-forward path.
#[inline]
#[allow(clippy::too_many_arguments)]
fn feedback_tick(
    buf: &mut [f32],
    iwrphase: &mut u32,
    mask: u32,
    idsamp: i64,
    frac: f32,
    interp: Interp,
    x: f32,
    feedbk: f32,
    allpass: bool,
    warm: bool,
) -> f32 {
    let value = read_delayed(buf, *iwrphase, mask, idsamp, frac, interp, warm);
    let out = if allpass {
        let dwr = x + feedbk * value;
        buf[(*iwrphase & mask) as usize] = dwr;
        value - feedbk * dwr
    } else {
        buf[(*iwrphase & mask) as usize] = x + feedbk * value;
        value
    };
    *iwrphase = iwrphase.wrapping_add(1);
    out
}

const IN: usize = 0;
const MAXDELAY: usize = 1;
const DELAY: usize = 2;
const DECAY: usize = 3;

/// `DelayN/L/C.ar(in, maxdelaytime, delaytime)`: a delay line with no feedback, tapped with no,
/// linear, or cubic interpolation. The line is the [`ProcessCtx::aux`] slice, sized to `len` `f32`s at
/// build time. Field names mirror scsynth's `DelayUnit` (minus the `float* m_dlybuf` pointer).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Delay {
    /// Current delay in samples (`m_dsamp`), possibly fractional and mid-slope.
    dsamp: f32,
    /// The `delaytime` seen last block (`m_delaytime`), to detect a change and slope toward the new
    /// value. Only used for control-rate `delaytime`.
    delaytime: f32,
    /// Delay-line length in samples (`m_idelaylen`), a power of two.
    len: u32,
    /// `len - 1`, the wrap mask for the power-of-two circular buffer (`m_mask`).
    mask: u32,
    /// Monotonic write phase (`m_iwrphase`); only masked at use, so it may wrap freely once warm.
    iwrphase: u32,
    /// Samples written so far, saturating at `len` (`m_numoutput`); while `< len` the cold-start guard
    /// applies.
    numoutput: u32,
    /// Which calc variant (see [`calc`]), chosen from the `delaytime` rate at build time.
    calc: u32,
    /// Interpolation tag (see [`Interp`]).
    interp: u32,
}

impl Unit for Delay {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Seed `dsamp`/`delaytime` from the initial `delaytime` so the first block uses the steady
        // path (no ramp-from-zero), mirroring scsynth's `DelayUnit_Reset`.
        let dt = ctx.ins.control(DELAY);
        let min = Interp::from_tag(self.interp).min_delay(false);
        self.delaytime = dt;
        self.dsamp = clamp_delay(dt * ctx.audio.sample_rate as f32, min, self.len as f32);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let max = self.len as f32;
        let interp = Interp::from_tag(self.interp);
        let min = interp.min_delay(false);
        let mask = self.mask;
        let mut iwrphase = self.iwrphase;
        let warm = self.numoutput >= self.len;

        let in_audio = (ctx.ins.rate(IN) == Rate::Audio).then(|| ctx.ins.audio(IN));
        let in_ctrl = ctx.ins.control(IN);
        let dt_audio = (self.calc == calc::DELAY_AUDIO).then(|| ctx.ins.audio(DELAY));
        let dt_ctrl = ctx.ins.control(DELAY);

        let out = ctx.outs.audio(0);
        let buf = ctx.aux.f32_mut();
        if buf.is_empty() {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        let n = out.len();
        let input = |i: usize| in_audio.map_or(in_ctrl, |s| s[i]);

        match dt_audio {
            Some(dt) => {
                for (i, o) in out.iter_mut().enumerate() {
                    let dsamp = clamp_delay(dt[i] * sr, min, max);
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    *o = delay_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        warm,
                    );
                }
            }
            None if dt_ctrl == self.delaytime => {
                let dsamp = self.dsamp;
                let idsamp = dsamp as i64;
                let frac = dsamp - idsamp as f32;
                for (i, o) in out.iter_mut().enumerate() {
                    *o = delay_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        warm,
                    );
                }
            }
            None => {
                let next = clamp_delay(dt_ctrl * sr, min, max);
                let mut dsamp = self.dsamp;
                let slope = (next - dsamp) / n as f32;
                for (i, o) in out.iter_mut().enumerate() {
                    dsamp += slope;
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    *o = delay_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        warm,
                    );
                }
                self.dsamp = dsamp;
                self.delaytime = dt_ctrl;
            }
        }

        self.iwrphase = iwrphase;
        if !warm {
            self.numoutput = self.numoutput.saturating_add(n as u32).min(self.len);
        }
        DoneAction::Nothing
    }
}

/// `CombN/L/C` and `AllpassN/L/C.ar(in, maxdelaytime, delaytime, decaytime)`: a delay line that
/// recirculates its output. `decaytime` sets the feedback coefficient (`sc_CalcFeedback`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FeedbackDelay {
    /// Current delay in samples (`m_dsamp`), possibly fractional and mid-slope.
    dsamp: f32,
    /// The `delaytime` seen last block (`m_delaytime`).
    delaytime: f32,
    /// The `decaytime` seen last block (`m_decaytime`).
    decaytime: f32,
    /// The recirculation coefficient last block (`m_feedbk`).
    feedbk: f32,
    /// Delay-line length in samples (`m_idelaylen`), a power of two.
    len: u32,
    /// `len - 1`, the wrap mask (`m_mask`).
    mask: u32,
    /// Monotonic write phase (`m_iwrphase`).
    iwrphase: u32,
    /// Samples written so far, saturating at `len` (`m_numoutput`).
    numoutput: u32,
    /// Which calc variant (see [`calc`]).
    calc: u32,
    /// Interpolation tag (see [`Interp`]).
    interp: u32,
    /// `1` for an allpass, `0` for a comb.
    allpass: u32,
}

impl Unit for FeedbackDelay {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let dt = ctx.ins.control(DELAY);
        let decay = ctx.ins.control(DECAY);
        let min = Interp::from_tag(self.interp).min_delay(true);
        self.delaytime = dt;
        self.decaytime = decay;
        self.dsamp = clamp_delay(dt * ctx.audio.sample_rate as f32, min, self.len as f32);
        self.feedbk = calc_feedback(dt, decay);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let max = self.len as f32;
        let interp = Interp::from_tag(self.interp);
        let min = interp.min_delay(true);
        let allpass = self.allpass != 0;
        let mask = self.mask;
        let mut iwrphase = self.iwrphase;
        let warm = self.numoutput >= self.len;

        let in_audio = (ctx.ins.rate(IN) == Rate::Audio).then(|| ctx.ins.audio(IN));
        let in_ctrl = ctx.ins.control(IN);
        let dt_audio = (self.calc == calc::DELAY_AUDIO).then(|| ctx.ins.audio(DELAY));
        let dt_ctrl = ctx.ins.control(DELAY);
        // `decaytime` is always read at scalar/control rate, even when `delaytime` is audio-rate.
        let decay = ctx.ins.control(DECAY);

        let out = ctx.outs.audio(0);
        let buf = ctx.aux.f32_mut();
        if buf.is_empty() {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        let n = out.len();
        let input = |i: usize| in_audio.map_or(in_ctrl, |s| s[i]);

        match dt_audio {
            Some(dt) => {
                // Audio-rate delaytime: recompute the tap and coefficient every sample, no slope.
                for (i, o) in out.iter_mut().enumerate() {
                    let del = dt[i];
                    let dsamp = clamp_delay(del * sr, min, max);
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    let feedbk = calc_feedback(del, decay);
                    *o = feedback_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        feedbk,
                        allpass,
                        warm,
                    );
                }
            }
            None if dt_ctrl == self.delaytime && decay == self.decaytime => {
                let dsamp = self.dsamp;
                let idsamp = dsamp as i64;
                let frac = dsamp - idsamp as f32;
                let feedbk = self.feedbk;
                for (i, o) in out.iter_mut().enumerate() {
                    *o = feedback_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        feedbk,
                        allpass,
                        warm,
                    );
                }
            }
            None => {
                // Changed control-rate delaytime and/or decaytime: slope both across the block.
                let next_dsamp = clamp_delay(dt_ctrl * sr, min, max);
                let next_feedbk = calc_feedback(dt_ctrl, decay);
                let mut dsamp = self.dsamp;
                let mut feedbk = self.feedbk;
                let dsamp_slope = (next_dsamp - dsamp) / n as f32;
                let feedbk_slope = (next_feedbk - feedbk) / n as f32;
                for (i, o) in out.iter_mut().enumerate() {
                    dsamp += dsamp_slope;
                    feedbk += feedbk_slope;
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    *o = feedback_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        feedbk,
                        allpass,
                        warm,
                    );
                }
                self.dsamp = dsamp;
                self.feedbk = feedbk;
                self.delaytime = dt_ctrl;
                self.decaytime = decay;
            }
        }

        self.iwrphase = iwrphase;
        if !warm {
            self.numoutput = self.numoutput.saturating_add(n as u32).min(self.len);
        }
        DoneAction::Nothing
    }
}

/// Build a delay line, validating inputs and sizing the aux buffer from the constant `maxdelaytime`.
/// Returns `(len, mask, calc, aux_bytes)`.
fn build_line(
    ctx: &BuildContext<'_>,
    min_inputs: usize,
) -> Result<(u32, u32, u32, usize), BuildError> {
    if ctx.input_rates.len() < min_inputs {
        return Err(BuildError::WrongInputCount);
    }
    // `maxdelaytime` sizes the line, so it must be a compile-time constant (scsynth reads it once at
    // ctor and never again).
    let max_delay = ctx
        .const_input(MAXDELAY)
        .ok_or(BuildError::AuxRequiresConstant { input: MAXDELAY })?;
    let len = line_len(max_delay, ctx.audio.sample_rate, ctx.audio.block_size);
    let calc = match ctx.input_rates[DELAY] {
        Rate::Audio => calc::DELAY_AUDIO,
        _ => calc::DELAY_CONTROL,
    };
    let aux_bytes = len as usize * core::mem::size_of::<f32>();
    Ok((len, len - 1, calc, aux_bytes))
}

/// Constructor for [`Delay`] (`DelayN`/`DelayL`/`DelayC`), parameterized by [`Interp`].
pub struct DelayCtor(pub Interp);

impl UnitDef for DelayCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let (len, mask, calc, aux_bytes) = build_line(ctx, 3)?;
        Ok(unit_spec_aux(
            Delay {
                dsamp: 0.0,
                delaytime: 0.0,
                len,
                mask,
                iwrphase: 0,
                numoutput: 0,
                calc,
                interp: self.0.to_tag(),
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}

/// Constructor for [`FeedbackDelay`] (`CombN/L/C`, `AllpassN/L/C`), parameterized by [`Interp`] and
/// whether it is an allpass (else a comb).
pub struct FeedbackDelayCtor {
    pub interp: Interp,
    pub allpass: bool,
}

impl UnitDef for FeedbackDelayCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let (len, mask, calc, aux_bytes) = build_line(ctx, 4)?;
        Ok(unit_spec_aux(
            FeedbackDelay {
                dsamp: 0.0,
                delaytime: 0.0,
                decaytime: 0.0,
                feedbk: 0.0,
                len,
                mask,
                iwrphase: 0,
                numoutput: 0,
                calc,
                interp: self.interp.to_tag(),
                allpass: self.allpass as u32,
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}

// Buffer-backed delays (`BufDelay*`, `BufComb*`, `BufAllpass*`): the delay line is a `/b_alloc`'d
// buffer rather than aux memory. Inputs are `[bufnum, in, delaytime(, decaytime)]` - `in` is at index
// 1, and there is no `maxdelaytime` (the buffer sizes the line).
const BUF_BUFNUM: usize = 0;
const BUF_IN: usize = 1;
const BUF_DELAY: usize = 2;
const BUF_DECAY: usize = 3;

/// The power-of-two prefix length and wrap mask a buffer-backed delay uses for a buffer of
/// `buf_samples` samples (scsynth's `BUFMASK` = `PREVIOUSPOWEROFTWO(bufSamples) - 1`): only the largest
/// `2^floor(log2 bufSamples)` samples are addressed as the circular line.
fn buf_line(buf_samples: usize) -> (usize, u32) {
    if buf_samples == 0 {
        return (0, 0);
    }
    let len = 1usize << (usize::BITS - 1 - buf_samples.leading_zeros());
    (len, (len - 1) as u32)
}

/// `BufDelayN/L/C.ar(bufnum, in, delaytime)`: a no-feedback delay line living in the buffer at
/// `bufnum`. Otherwise identical to [`Delay`], but the line is resolved each block from the buffer
/// table (so it may be shared or resized) and its length is the buffer's power-of-two prefix.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BufDelay {
    /// Current delay in samples (`m_dsamp`), possibly fractional and mid-slope.
    dsamp: f32,
    /// The `delaytime` seen last block (`m_delaytime`).
    delaytime: f32,
    /// Monotonic write phase (`m_iwrphase`).
    iwrphase: u32,
    /// Samples written so far, saturating at the line length (`m_numoutput`); while below it the
    /// cold-start guard applies so a not-yet-written tap reads `0` rather than the buffer's prior
    /// content.
    numoutput: u32,
    /// Which calc variant (see [`calc`]), chosen from the `delaytime` rate at build time.
    calc: u32,
    /// Interpolation tag (see [`Interp`]).
    interp: u32,
}

impl Unit for BufDelay {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let dt = ctx.ins.control(BUF_DELAY);
        let min = Interp::from_tag(self.interp).min_delay(false);
        let bufnum = ctx.ins.control(BUF_BUFNUM).max(0.0) as usize;
        let max = buffer_at(ctx.buffers, bufnum).map_or(0, |b| buf_line(b.data().len()).1 as usize);
        self.delaytime = dt;
        self.dsamp = clamp_delay(dt * ctx.audio.sample_rate as f32, min, max as f32);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let interp = Interp::from_tag(self.interp);
        let min = interp.min_delay(false);
        let mut iwrphase = self.iwrphase;

        let ins = ctx.ins;
        let in_audio = (ins.rate(BUF_IN) == Rate::Audio).then(|| ins.audio(BUF_IN));
        let in_ctrl = ins.control(BUF_IN);
        let dt_audio = (self.calc == calc::DELAY_AUDIO).then(|| ins.audio(BUF_DELAY));
        let dt_ctrl = ins.control(BUF_DELAY);
        let bufnum = ins.control(BUF_BUFNUM).max(0.0) as usize;

        let out = ctx.outs.audio(0);
        let buf = match buffer_at_mut(ctx.buffers, bufnum) {
            Some(b) => b.data_mut(),
            None => {
                out.fill(0.0);
                return DoneAction::Nothing;
            }
        };
        let (len, mask) = buf_line(buf.len());
        if len == 0 {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        let max = mask as f32;
        let warm = self.numoutput as usize >= len;
        let n = out.len();
        let input = |i: usize| in_audio.map_or(in_ctrl, |s| s[i]);

        match dt_audio {
            Some(dt) => {
                for (i, o) in out.iter_mut().enumerate() {
                    let dsamp = clamp_delay(dt[i] * sr, min, max);
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    *o = delay_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        warm,
                    );
                }
            }
            None if dt_ctrl == self.delaytime => {
                let dsamp = self.dsamp;
                let idsamp = dsamp as i64;
                let frac = dsamp - idsamp as f32;
                for (i, o) in out.iter_mut().enumerate() {
                    *o = delay_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        warm,
                    );
                }
            }
            None => {
                let next = clamp_delay(dt_ctrl * sr, min, max);
                let mut dsamp = self.dsamp;
                let slope = (next - dsamp) / n as f32;
                for (i, o) in out.iter_mut().enumerate() {
                    dsamp += slope;
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    *o = delay_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        warm,
                    );
                }
                self.dsamp = dsamp;
                self.delaytime = dt_ctrl;
            }
        }

        self.iwrphase = iwrphase;
        if !warm {
            self.numoutput = (self.numoutput.saturating_add(n as u32)).min(len as u32);
        }
        DoneAction::Nothing
    }
}

/// `BufCombN/L/C` and `BufAllpassN/L/C.ar(bufnum, in, delaytime, decaytime)`: the buffer-backed twin of
/// [`FeedbackDelay`] - a comb/allpass whose line is the buffer at `bufnum`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BufFeedbackDelay {
    dsamp: f32,
    delaytime: f32,
    decaytime: f32,
    feedbk: f32,
    iwrphase: u32,
    numoutput: u32,
    calc: u32,
    interp: u32,
    allpass: u32,
    _pad: u32,
}

impl Unit for BufFeedbackDelay {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let dt = ctx.ins.control(BUF_DELAY);
        let decay = ctx.ins.control(BUF_DECAY);
        let min = Interp::from_tag(self.interp).min_delay(true);
        let bufnum = ctx.ins.control(BUF_BUFNUM).max(0.0) as usize;
        let max = buffer_at(ctx.buffers, bufnum).map_or(0, |b| buf_line(b.data().len()).1 as usize);
        self.delaytime = dt;
        self.decaytime = decay;
        self.dsamp = clamp_delay(dt * ctx.audio.sample_rate as f32, min, max as f32);
        self.feedbk = calc_feedback(dt, decay);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let interp = Interp::from_tag(self.interp);
        let min = interp.min_delay(true);
        let allpass = self.allpass != 0;
        let mut iwrphase = self.iwrphase;

        let ins = ctx.ins;
        let in_audio = (ins.rate(BUF_IN) == Rate::Audio).then(|| ins.audio(BUF_IN));
        let in_ctrl = ins.control(BUF_IN);
        let dt_audio = (self.calc == calc::DELAY_AUDIO).then(|| ins.audio(BUF_DELAY));
        let dt_ctrl = ins.control(BUF_DELAY);
        let decay = ins.control(BUF_DECAY);
        let bufnum = ins.control(BUF_BUFNUM).max(0.0) as usize;

        let out = ctx.outs.audio(0);
        let buf = match buffer_at_mut(ctx.buffers, bufnum) {
            Some(b) => b.data_mut(),
            None => {
                out.fill(0.0);
                return DoneAction::Nothing;
            }
        };
        let (len, mask) = buf_line(buf.len());
        if len == 0 {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        let max = mask as f32;
        let warm = self.numoutput as usize >= len;
        let n = out.len();
        let input = |i: usize| in_audio.map_or(in_ctrl, |s| s[i]);

        match dt_audio {
            Some(dt) => {
                for (i, o) in out.iter_mut().enumerate() {
                    let del = dt[i];
                    let dsamp = clamp_delay(del * sr, min, max);
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    let feedbk = calc_feedback(del, decay);
                    *o = feedback_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        feedbk,
                        allpass,
                        warm,
                    );
                }
            }
            None if dt_ctrl == self.delaytime && decay == self.decaytime => {
                let dsamp = self.dsamp;
                let idsamp = dsamp as i64;
                let frac = dsamp - idsamp as f32;
                let feedbk = self.feedbk;
                for (i, o) in out.iter_mut().enumerate() {
                    *o = feedback_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        feedbk,
                        allpass,
                        warm,
                    );
                }
            }
            None => {
                let next_dsamp = clamp_delay(dt_ctrl * sr, min, max);
                let next_feedbk = calc_feedback(dt_ctrl, decay);
                let mut dsamp = self.dsamp;
                let mut feedbk = self.feedbk;
                let dsamp_slope = (next_dsamp - dsamp) / n as f32;
                let feedbk_slope = (next_feedbk - feedbk) / n as f32;
                for (i, o) in out.iter_mut().enumerate() {
                    dsamp += dsamp_slope;
                    feedbk += feedbk_slope;
                    let idsamp = dsamp as i64;
                    let frac = dsamp - idsamp as f32;
                    *o = feedback_tick(
                        buf,
                        &mut iwrphase,
                        mask,
                        idsamp,
                        frac,
                        interp,
                        input(i),
                        feedbk,
                        allpass,
                        warm,
                    );
                }
                self.dsamp = dsamp;
                self.feedbk = feedbk;
                self.delaytime = dt_ctrl;
                self.decaytime = decay;
            }
        }

        self.iwrphase = iwrphase;
        if !warm {
            self.numoutput = (self.numoutput.saturating_add(n as u32)).min(len as u32);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`BufDelay`] (`BufDelayN`/`BufDelayL`/`BufDelayC`), parameterized by [`Interp`].
pub struct BufDelayCtor(pub Interp);

impl UnitDef for BufDelayCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        let calc = match ctx.input_rates[BUF_DELAY] {
            Rate::Audio => calc::DELAY_AUDIO,
            _ => calc::DELAY_CONTROL,
        };
        Ok(unit_spec(BufDelay {
            dsamp: 0.0,
            delaytime: 0.0,
            iwrphase: 0,
            numoutput: 0,
            calc,
            interp: self.0.to_tag(),
        }))
    }
}

/// Constructor for [`BufFeedbackDelay`] (`BufCombN/L/C`, `BufAllpassN/L/C`), parameterized by
/// [`Interp`] and whether it is an allpass (else a comb).
pub struct BufFeedbackDelayCtor {
    pub interp: Interp,
    pub allpass: bool,
}

impl UnitDef for BufFeedbackDelayCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        let calc = match ctx.input_rates[BUF_DELAY] {
            Rate::Audio => calc::DELAY_AUDIO,
            _ => calc::DELAY_CONTROL,
        };
        Ok(unit_spec(BufFeedbackDelay {
            dsamp: 0.0,
            delaytime: 0.0,
            decaytime: 0.0,
            feedbk: 0.0,
            iwrphase: 0,
            numoutput: 0,
            calc,
            interp: self.interp.to_tag(),
            allpass: self.allpass as u32,
            _pad: 0,
        }))
    }
}
