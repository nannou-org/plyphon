//! Counting and timing units - plyphon's ports of scsynth's `PulseCount`, `PulseDivider`, `Stepper`,
//! `ZeroCrossing`, `Timer`, `Sweep` and `Phasor` (`TriggerUGens.cpp`).
//!
//! Like the [trigger](crate::unit::trigger) units, these detect rising edges (`> 0` after `<= 0`) and
//! read each input at its declared rate via the shared `Sig` helper. `Sweep`/`Phasor` accumulate a
//! phase in `f64` for precision. Coefficient-like inputs (division amount, min/max, rate, bounds) are
//! read once per block.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// `PulseCount.ar/kr(trig, reset)`: counts rising `trig` edges; a rising `reset` zeroes the count.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PulseCount {
    prev_trig: f32,
    prev_reset: f32,
    level: f32,
    audio: u32,
}

impl PulseCount {
    fn step(&mut self, cur_trig: f32, cur_reset: f32) -> f32 {
        if self.prev_reset <= 0.0 && cur_reset > 0.0 {
            self.level = 0.0;
        } else if self.prev_trig <= 0.0 && cur_trig > 0.0 {
            self.level += 1.0;
        }
        self.prev_trig = cur_trig;
        self.prev_reset = cur_reset;
        self.level
    }
}

impl Unit for PulseCount {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let reset = sig(&ctx.ins, 1);
        drive(ctx, audio_out, |i| self.step(trig.at(i), reset.at(i)));
        DoneAction::Nothing
    }
}

/// `PulseDivider.ar/kr(trig, div, start)`: emits a `1` on every `div`-th rising `trig`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PulseDivider {
    prev_trig: f32,
    counter: i32,
    audio: u32,
}

impl PulseDivider {
    fn step(&mut self, cur: f32, div: i32) -> f32 {
        let mut z = 0.0;
        if self.prev_trig <= 0.0 && cur > 0.0 {
            self.counter += 1;
            if self.counter >= div {
                self.counter = 0;
                z = 1.0;
            }
        }
        self.prev_trig = cur;
        z
    }
}

impl Unit for PulseDivider {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let div = ctx.ins.control(1) as i32;
        drive(ctx, audio_out, |i| self.step(trig.at(i), div));
        DoneAction::Nothing
    }
}

/// `Stepper.ar/kr(trig, reset, min, max, step, resetval)`: a counter stepping by `step` and wrapping
/// within `[min, max]` on each rising `trig`; a rising `reset` jumps to `resetval` (wrapped).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Stepper {
    prev_trig: f32,
    prev_reset: f32,
    level: f32,
    audio: u32,
}

/// Integer wrap into `[lo, hi]` (scsynth's `sc_wrap(int, int, int)`).
fn wrap_i(x: i32, lo: i32, hi: i32) -> i32 {
    let n = hi - lo + 1;
    if n <= 0 {
        lo
    } else {
        (x - lo).rem_euclid(n) + lo
    }
}

impl Stepper {
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        cur_trig: f32,
        cur_reset: f32,
        min: i32,
        max: i32,
        stp: i32,
        resetval: i32,
    ) -> f32 {
        if self.prev_reset <= 0.0 && cur_reset > 0.0 {
            self.level = wrap_i(resetval, min, max) as f32;
        } else if self.prev_trig <= 0.0 && cur_trig > 0.0 {
            self.level = wrap_i(self.level as i32 + stp, min, max) as f32;
        }
        self.prev_trig = cur_trig;
        self.prev_reset = cur_reset;
        self.level
    }
}

impl Unit for Stepper {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let reset = sig(&ctx.ins, 1);
        let min = ctx.ins.control(2) as i32;
        let max = ctx.ins.control(3) as i32;
        let stp = ctx.ins.control(4) as i32;
        let resetval = ctx.ins.control(5) as i32;
        drive(ctx, audio_out, |i| {
            self.step(trig.at(i), reset.at(i), min, max, stp, resetval)
        });
        DoneAction::Nothing
    }
}

/// `ZeroCrossing.ar(in)`: estimates `in`'s fundamental frequency (Hz) from its zero-crossing period.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ZeroCrossing {
    prev_in: f32,
    prev_frac: f32,
    level: f32,
    counter: i32,
    audio: u32,
}

impl ZeroCrossing {
    fn step(&mut self, cur: f32, sr: f32) -> f32 {
        self.counter += 1;
        if self.counter > 4 && self.prev_in <= 0.0 && cur > 0.0 {
            let frac = -self.prev_in / (cur - self.prev_in);
            self.level = sr / (frac + self.counter as f32 - self.prev_frac);
            self.prev_frac = frac;
            self.counter = 0;
        }
        self.prev_in = cur;
        self.level
    }
}

impl Unit for ZeroCrossing {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let sr = ctx.audio.sample_rate as f32;
        drive(ctx, audio_out, |i| self.step(input.at(i), sr));
        DoneAction::Nothing
    }
}

/// `Timer.ar/kr(trig)`: outputs the time in seconds between successive rising `trig` edges.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Timer {
    prev_in: f32,
    prev_frac: f32,
    level: f32,
    counter: i32,
    audio: u32,
}

impl Timer {
    fn step(&mut self, cur: f32, sample_dur: f32) -> f32 {
        self.counter += 1;
        if self.prev_in <= 0.0 && cur > 0.0 {
            let frac = -self.prev_in / (cur - self.prev_in);
            self.level = sample_dur * (frac + self.counter as f32 - self.prev_frac);
            self.prev_frac = frac;
            self.counter = 0;
        }
        self.prev_in = cur;
        self.level
    }
}

impl Unit for Timer {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let sample_dur = 1.0 / ctx.audio.sample_rate as f32;
        drive(ctx, audio_out, |i| self.step(trig.at(i), sample_dur));
        DoneAction::Nothing
    }
}

/// `Sweep.ar/kr(trig, rate)`: a linear ramp climbing at `rate` per second, restarted by a rising
/// `trig`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Sweep {
    level: f64,
    prev_in: f32,
    audio: u32,
}

impl Sweep {
    fn step(&mut self, cur: f32, rate: f32, sample_dur: f32) -> f32 {
        let step = rate as f64 * sample_dur as f64;
        if self.prev_in <= 0.0 && cur > 0.0 {
            let frac = (-self.prev_in / (cur - self.prev_in)) as f64;
            self.level = frac * step;
        }
        let out = self.level;
        self.level += step;
        self.prev_in = cur;
        out as f32
    }
}

impl Unit for Sweep {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let rate = sig(&ctx.ins, 1);
        let sample_dur = 1.0 / ctx.audio.sample_rate as f32;
        drive(ctx, audio_out, |i| {
            self.step(trig.at(i), rate.at(i), sample_dur)
        });
        DoneAction::Nothing
    }
}

/// `Phasor.ar/kr(trig, rate, start, end, resetPos)`: a ramp advancing by `rate` per sample and
/// wrapping within `[start, end)`, jumping to `resetPos` on a rising `trig`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Phasor {
    level: f64,
    prev_in: f32,
    audio: u32,
}

/// Wrap `x` into `[lo, hi)` in `f64` (scsynth's `sc_wrap` on the phasor accumulator).
fn wrap64(mut x: f64, lo: f64, hi: f64) -> f64 {
    let range = hi - lo;
    if range <= 0.0 {
        return lo;
    }
    if x >= hi {
        x -= range;
        if x < hi {
            return x;
        }
    } else if x < lo {
        x += range;
        if x >= lo {
            return x;
        }
    } else {
        return x;
    }
    x - range * math::floor((x - lo) / range)
}

impl Phasor {
    fn step(&mut self, cur: f32, rate: f32, start: f64, end: f64, reset_pos: f64) -> f32 {
        if self.prev_in <= 0.0 && cur > 0.0 {
            let frac = 1.0 - self.prev_in / (cur - self.prev_in);
            self.level = reset_pos + frac as f64 * rate as f64;
        }
        let out = self.level;
        self.level += rate as f64;
        self.level = wrap64(self.level, start, end);
        self.prev_in = cur;
        out as f32
    }
}

impl Unit for Phasor {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let rate = sig(&ctx.ins, 1);
        let start = ctx.ins.control(2) as f64;
        let end = ctx.ins.control(3) as f64;
        let reset_pos = ctx.ins.control(4) as f64;
        drive(ctx, audio_out, |i| {
            self.step(trig.at(i), rate.at(i), start, end, reset_pos)
        });
        DoneAction::Nothing
    }
}

/// Build a timing unit whose numeric state is zero-initialised, with the given minimum input count.
macro_rules! timing_ctor {
    ($ctor:ident, $unit:ident, $min_inputs:expr, { $($field:ident: $init:expr),* $(,)? }) => {
        #[doc = concat!("Constructor for [`", stringify!($unit), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.len() < $min_inputs {
                    return Err(BuildError::WrongInputCount);
                }
                Ok(unit_spec($unit {
                    $($field: $init,)*
                    audio: (ctx.rate == Rate::Audio) as u32,
                }))
            }
        }
    };
}

timing_ctor!(PulseCountCtor, PulseCount, 2, { prev_trig: 0.0, prev_reset: 0.0, level: 0.0 });
timing_ctor!(PulseDividerCtor, PulseDivider, 2, { prev_trig: 0.0, counter: 0 });
timing_ctor!(StepperCtor, Stepper, 6, { prev_trig: 0.0, prev_reset: 0.0, level: 0.0 });
timing_ctor!(ZeroCrossingCtor, ZeroCrossing, 1, { prev_in: 0.0, prev_frac: 0.0, level: 0.0, counter: 0 });
timing_ctor!(TimerCtor, Timer, 1, { prev_in: 0.0, prev_frac: 0.0, level: 0.0, counter: -1 });
timing_ctor!(SweepCtor, Sweep, 2, { level: 0.0, prev_in: 0.0 });
timing_ctor!(PhasorCtor, Phasor, 5, { level: 0.0, prev_in: 0.0 });
