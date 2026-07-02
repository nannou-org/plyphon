//! Dynamics processors - plyphon's ports of scsynth's `Compander`, `DetectSilence`, `Limiter` and
//! `Normalizer` (`FilterUGens.cpp`).
//!
//! `Compander` is a general compressor/expander/gate driven by a side-chain control signal;
//! `DetectSilence` fires a done action (and outputs `1`) once its input has stayed below a threshold
//! for a given time. `Limiter` and `Normalizer` are look-ahead peak processors: they delay the signal
//! by `dur` seconds (a triple-buffer in aux memory) so a gain glide can be set up before each peak
//! arrives - `Limiter` only attenuates (capping the peak *at* `level`), `Normalizer` always rescales
//! (driving the peak *to* `level`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec, unit_spec_aux};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// `ln(0.1) == -ln(10)` - the `-20 dB`-over-`time` target scsynth uses for `Compander`'s
/// attack/release coefficients.
const LOG1: f32 = -core::f32::consts::LN_10;

/// The attack/release coefficient decaying to `-20 dB` over `time` seconds (0 for an immediate
/// response), matching scsynth's `time == 0 ? 0 : exp(log1 / (time * SR))`.
fn comp_coef(time: f32, sample_rate: f32) -> f32 {
    if time == 0.0 {
        0.0
    } else {
        math::exp(LOG1 / (time * sample_rate))
    }
}

/// `Compander.ar(in, control, thresh, slopeBelow, slopeAbove, clampTime, relaxTime)`: applies a gain
/// to `in` derived from the amplitude of `control`, giving compression/expansion above/below
/// `thresh`. `clampTime`/`relaxTime` set the attack/release of the amplitude follower.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Compander {
    clamp: f32,
    clampcoef: f32,
    relax: f32,
    relaxcoef: f32,
    gain: f32,
    prevmaxval: f32,
}

impl Unit for Compander {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.own.sample_rate as f32;
        let thresh = ctx.ins.control(2);
        let slope_below = ctx.ins.control(3);
        let slope_above = ctx.ins.control(4);
        let clamp = ctx.ins.control(5);
        let relax = ctx.ins.control(6);
        if clamp != self.clamp {
            self.clampcoef = comp_coef(clamp, sr);
            self.clamp = clamp;
        }
        if relax != self.relax {
            self.relaxcoef = comp_coef(relax, sr);
            self.relax = relax;
        }
        let (clampcoef, relaxcoef) = (self.clampcoef, self.relaxcoef);

        // Follow the amplitude of the side-chain control signal (fast attack, slow release).
        let mut prevmaxval = self.prevmaxval;
        for &c in ctx.ins.audio(1) {
            let val = c.abs();
            let coef = if val < prevmaxval {
                relaxcoef
            } else {
                clampcoef
            };
            prevmaxval = val + (prevmaxval - val) * coef;
        }
        self.prevmaxval = prevmaxval;

        // The target gain: `(maxval/thresh)^(slope - 1)`, with the below-thresh path zapped so tiny
        // values do not blow the gain up.
        let next_gain = if prevmaxval < thresh {
            if slope_below == 1.0 {
                1.0
            } else {
                let g = math::powf(prevmaxval / thresh, slope_below - 1.0);
                let a = g.abs();
                if a < 1e-15 {
                    0.0
                } else if a > 1e15 {
                    1.0
                } else {
                    g
                }
            }
        } else if slope_above == 1.0 {
            1.0
        } else {
            math::powf(prevmaxval / thresh, slope_above - 1.0)
        };

        // Apply the gain, ramping from the previous block's gain to `next_gain` to avoid zipper noise.
        let mut gain = self.gain;
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        let gain_slope = (next_gain - gain) / out.len().max(1) as f32;
        for (o, &x) in out.iter_mut().zip(input) {
            *o = x * gain;
            gain += gain_slope;
        }
        self.gain = gain;
        DoneAction::Nothing
    }
}

/// Constructor for [`Compander`].
pub struct CompanderCtor;

impl UnitDef for CompanderCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 7 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Compander {
            clamp: f32::NAN, // force coefficient computation on the first block
            clampcoef: 0.0,
            relax: f32::NAN,
            relaxcoef: 0.0,
            gain: 1.0,
            prevmaxval: 0.0,
        }))
    }
}

/// `DetectSilence.ar(in, amp, time, doneAction)`: outputs `0` while `in` is active; once `|in|` has
/// stayed at or below `amp` for `time` seconds (after first exceeding it), outputs `1` and fires the
/// `doneAction` once.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DetectSilence {
    counter: i32,
    fired: u32,
    audio: u32,
}

impl DetectSilence {
    const IN: usize = 0;
    const THRESH: usize = 1;
    const TIME: usize = 2;
    const DONE: usize = 3;
}

impl Unit for DetectSilence {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let thresh = ctx.ins.control(Self::THRESH);
        let end_counter = (ctx.own.sample_rate as f32 * ctx.ins.control(Self::TIME)) as i32;
        let input = sig(&ctx.ins, Self::IN);
        let mut counter = self.counter;
        let mut completed = false;
        drive(ctx, audio_out, |i| {
            let val = input.at(i).abs();
            if val > thresh {
                counter = 0;
                0.0
            } else if counter >= 0 {
                counter += 1;
                if counter >= end_counter {
                    completed = true;
                    1.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        });
        self.counter = counter;
        if completed {
            ctx.done.mark_done();
            if self.fired == 0 {
                self.fired = 1;
                let action = if ctx.ins.len() > Self::DONE {
                    DoneAction::from_code(ctx.ins.control(Self::DONE))
                } else {
                    DoneAction::Nothing
                };
                return action;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DetectSilence`].
pub struct DetectSilenceCtor;

impl UnitDef for DetectSilenceCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DetectSilence {
            counter: -1, // wait for the first above-threshold sample before counting silence
            fired: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// Whether a [`LookAhead`] normalizes (drives the peak *to* `level`) or limits (caps it *at* `level`).
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum LookAheadMode {
    /// `Normalizer`: target gain is always `level / peak`.
    Normalizer,
    /// `Limiter`: target gain is `min(1, level / peak)` - only attenuates.
    Limiter,
}

impl LookAheadMode {
    fn to_tag(self) -> u32 {
        match self {
            LookAheadMode::Normalizer => 0,
            LookAheadMode::Limiter => 1,
        }
    }
}

/// The look-ahead target gain for peak `maxval` and target `amp`, per mode (scsynth's `next_level`).
fn look_ahead_gain(mode: u32, maxval: f32, amp: f32) -> f32 {
    if mode == LookAheadMode::Limiter.to_tag() {
        if maxval > amp { amp / maxval } else { 1.0 }
    } else if maxval <= 0.00001 {
        // Near-silence guard, so the boost gain stays finite.
        100_000.0 * amp
    } else {
        amp / maxval
    }
}

/// `Limiter.ar(in, level, dur)` / `Normalizer.ar(in, level, dur)`: a look-ahead peak processor. The
/// signal is delayed by `dur` seconds through a rotating triple-buffer while the peak over the last two
/// `dur`-length regions sets a gain that ramps in before the peak reaches the output, so limiting is
/// click-free. `dur` is fixed at build (it sizes the buffer); `level` is read per block. Outputs
/// silence for the first `2*dur` seconds (the look-ahead latency).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LookAhead {
    /// Current gain, ramped by `slope` each sample.
    level: f32,
    slope: f32,
    /// Running peak of the region being written, and of the previous region.
    curmaxval: f32,
    prevmaxval: f32,
    /// `1 / bufsize` - ramps a gain change over exactly one region.
    slopefactor: f32,
    /// Write/read position within the current region.
    pos: i32,
    /// Region rotations so far; real output begins at `2`.
    flips: i32,
    /// Samples per region (`ceil(dur * sampleRate)`).
    bufsize: u32,
    /// Sample offsets of the write / pending / read regions into the aux buffer.
    in_off: u32,
    mid_off: u32,
    out_off: u32,
    /// `0` = `Normalizer`, `1` = `Limiter`.
    mode: u32,
}

impl LookAhead {
    const IN: usize = 0;
    const LEVEL: usize = 1;
    const DUR: usize = 2;
}

impl Unit for LookAhead {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let amp = ctx.ins.control(Self::LEVEL);
        let n = self.bufsize as usize;
        let mode = self.mode;
        let slopefactor = self.slopefactor;
        let input = ctx.ins.audio(Self::IN);
        let block = input.len();
        let buf = ctx.aux.f32_mut();
        if n == 0 || buf.len() < 3 * n {
            ctx.outs.audio(0).fill(0.0);
            return DoneAction::Nothing;
        }
        let out = ctx.outs.audio(0);

        let (mut pos, mut flips) = (self.pos as usize, self.flips);
        let (mut level, mut slope) = (self.level, self.slope);
        let (mut curmaxval, mut prevmaxval) = (self.curmaxval, self.prevmaxval);
        let (mut in_off, mut mid_off, mut out_off) = (
            self.in_off as usize,
            self.mid_off as usize,
            self.out_off as usize,
        );

        let mut i = 0;
        while i < block {
            let nsmps = (block - i).min(n - pos);
            let active = flips >= 2;
            for _ in 0..nsmps {
                let x = input[i];
                buf[in_off + pos] = x;
                out[i] = if active {
                    level * buf[out_off + pos]
                } else {
                    0.0
                };
                level += slope;
                let a = x.abs();
                if a > curmaxval {
                    curmaxval = a;
                }
                pos += 1;
                i += 1;
            }
            if pos >= n {
                pos = 0;
                let maxval2 = prevmaxval.max(curmaxval);
                prevmaxval = curmaxval;
                curmaxval = 0.0;
                let next_level = look_ahead_gain(mode, maxval2, amp);
                slope = (next_level - level) * slopefactor;
                // Rotate the regions: read <- pending, pending <- just-written, write <- old read.
                let temp = out_off;
                out_off = mid_off;
                mid_off = in_off;
                in_off = temp;
                flips += 1;
            }
        }

        self.pos = pos as i32;
        self.flips = flips;
        self.level = level;
        self.slope = slope;
        self.curmaxval = curmaxval;
        self.prevmaxval = prevmaxval;
        self.in_off = in_off as u32;
        self.mid_off = mid_off as u32;
        self.out_off = out_off as u32;
        DoneAction::Nothing
    }
}

/// Constructor for [`LookAhead`] (`Limiter`/`Normalizer`), parameterized by [`LookAheadMode`].
pub struct LookAheadCtor(pub LookAheadMode);

impl UnitDef for LookAheadCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        // `dur` sizes the look-ahead buffer, so it must be a compile-time constant (as in scsynth).
        let dur = ctx
            .const_input(LookAhead::DUR)
            .ok_or(BuildError::AuxRequiresConstant {
                input: LookAhead::DUR,
            })?;
        let n = (math::ceil(dur as f64 * ctx.audio.sample_rate) as usize).max(1);
        let aux_bytes = 3 * n * core::mem::size_of::<f32>();
        Ok(unit_spec_aux(
            LookAhead {
                level: 1.0,
                slope: 0.0,
                curmaxval: 0.0,
                prevmaxval: 0.0,
                slopefactor: 1.0 / n as f32,
                pos: 0,
                flips: 0,
                bufsize: n as u32,
                in_off: 0,
                mid_off: n as u32,
                out_off: 2 * n as u32,
                mode: self.0.to_tag(),
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}
