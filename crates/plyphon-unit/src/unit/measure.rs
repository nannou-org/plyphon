//! Signal-measurement units - plyphon's ports of scsynth's `Peak`, `RunningMin`, `RunningMax`,
//! `PeakFollower`, `MostChange`, `LeastChange` and `LastValue` (`TriggerUGens.cpp`).
//!
//! Each tracks a running statistic of its input(s) and outputs it at the unit's own rate. `Peak`,
//! `RunningMin` and `RunningMax` hold the running |max|/min/max and reset on a trigger; `PeakFollower`
//! is an amplitude envelope follower (instant attack, exponential release); `MostChange`/`LeastChange`
//! pick whichever of two inputs moved the most/least; `LastValue` samples-and-holds until its input
//! changes by more than a threshold. Inputs are read at their declared rate via the shared `sig`
//! helper, and state depending on the first input is seeded in [`Unit::init`].

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// `Peak.ar/kr(in, trig)`: the running peak of `|in|`; a rising edge of `trig` resets it to the
/// current `|in|` so it starts tracking a fresh peak.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Peak {
    level: f32,
    prev_trig: f32,
    audio: u32,
}

impl Unit for Peak {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.level = ctx.ins.control(0).abs();
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let trig = sig(&ctx.ins, 1);
        drive(ctx, audio, |i| {
            let inlevel = input.at(i).abs();
            self.level = inlevel.max(self.level);
            let out = self.level;
            let cur = trig.at(i);
            if self.prev_trig <= 0.0 && cur > 0.0 {
                self.level = inlevel;
            }
            self.prev_trig = cur;
            out
        });
        DoneAction::Nothing
    }
}

/// `RunningMin.ar/kr(in, trig)`: the running minimum of `in`; a rising edge of `trig` resets it.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RunningMin {
    level: f32,
    prev_trig: f32,
    audio: u32,
}

impl Unit for RunningMin {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.level = ctx.ins.control(0);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let trig = sig(&ctx.ins, 1);
        drive(ctx, audio, |i| {
            let inlevel = input.at(i);
            self.level = inlevel.min(self.level);
            let out = self.level;
            let cur = trig.at(i);
            if self.prev_trig <= 0.0 && cur > 0.0 {
                self.level = inlevel;
            }
            self.prev_trig = cur;
            out
        });
        DoneAction::Nothing
    }
}

/// `RunningMax.ar/kr(in, trig)`: the running maximum of `in`; a rising edge of `trig` resets it.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RunningMax {
    level: f32,
    prev_trig: f32,
    audio: u32,
}

impl Unit for RunningMax {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.level = ctx.ins.control(0);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let trig = sig(&ctx.ins, 1);
        drive(ctx, audio, |i| {
            let inlevel = input.at(i);
            self.level = inlevel.max(self.level);
            let out = self.level;
            let cur = trig.at(i);
            if self.prev_trig <= 0.0 && cur > 0.0 {
                self.level = inlevel;
            }
            self.prev_trig = cur;
            out
        });
        DoneAction::Nothing
    }
}

/// `PeakFollower.ar/kr(in, decay)`: an amplitude envelope follower - the level jumps up to `|in|`
/// instantly and decays toward it by `decay` each sample otherwise (`decay` in `[0, 1)`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PeakFollower {
    level: f32,
    audio: u32,
}

impl Unit for PeakFollower {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.level = ctx.ins.control(0).abs();
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let decay = ctx.ins.control(1);
        drive(ctx, audio, |i| {
            let inlevel = input.at(i).abs();
            if inlevel >= self.level {
                self.level = inlevel;
            } else {
                self.level = inlevel + decay * (self.level - inlevel);
            }
            self.level
        });
        DoneAction::Nothing
    }
}

/// `MostChange.ar/kr(a, b)`: passes through whichever of `a`/`b` changed more since the last sample
/// (ties keep the previous winner).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MostChange {
    prev_a: f32,
    prev_b: f32,
    /// `0` if `a` was output last, `1` if `b` - the tie-break.
    recent: u32,
    audio: u32,
}

impl Unit for MostChange {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.prev_a = ctx.ins.control(0);
        self.prev_b = ctx.ins.control(1);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let a = sig(&ctx.ins, 0);
        let b = sig(&ctx.ins, 1);
        drive(ctx, audio, |i| {
            let xa = a.at(i);
            let xb = b.at(i);
            let diff = (xa - self.prev_a).abs() - (xb - self.prev_b).abs();
            let out = if diff > 0.0 {
                self.recent = 0;
                xa
            } else if diff < 0.0 {
                self.recent = 1;
                xb
            } else if self.recent != 0 {
                xb
            } else {
                xa
            };
            self.prev_a = xa;
            self.prev_b = xb;
            out
        });
        DoneAction::Nothing
    }
}

/// `LeastChange.ar/kr(a, b)`: passes through whichever of `a`/`b` changed *less* since the last
/// sample (ties keep the previous winner).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LeastChange {
    prev_a: f32,
    prev_b: f32,
    recent: u32,
    audio: u32,
}

impl Unit for LeastChange {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.prev_a = ctx.ins.control(0);
        self.prev_b = ctx.ins.control(1);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let a = sig(&ctx.ins, 0);
        let b = sig(&ctx.ins, 1);
        drive(ctx, audio, |i| {
            let xa = a.at(i);
            let xb = b.at(i);
            let diff = (xa - self.prev_a).abs() - (xb - self.prev_b).abs();
            let out = if diff < 0.0 {
                self.recent = 0;
                xa
            } else if diff > 0.0 {
                self.recent = 1;
                xb
            } else if self.recent != 0 {
                xb
            } else {
                xa
            };
            self.prev_a = xa;
            self.prev_b = xb;
            out
        });
        DoneAction::Nothing
    }
}

/// `LastValue.ar/kr(in, diff)`: samples and holds `in`, stepping to a new held value only once the
/// input has moved at least `diff` from the last accepted value (a hysteresis quantiser).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LastValue {
    /// The held (output) value.
    prev: f32,
    /// The last accepted input value the threshold is measured from.
    curr: f32,
    audio: u32,
}

impl Unit for LastValue {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let v = ctx.ins.control(0);
        self.prev = v;
        self.curr = v;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let delta = ctx.ins.control(1);
        drive(ctx, audio, |i| {
            let inval = input.at(i);
            if (inval - self.curr).abs() >= delta {
                self.prev = self.curr;
                self.curr = inval;
            }
            self.prev
        });
        DoneAction::Nothing
    }
}

/// Build a measurement unit whose state is zero-initialised except for the output-rate flag (the rest
/// is seeded in [`Unit::init`] from the first input). Requires `min_inputs` inputs.
macro_rules! measure_ctor {
    ($ctor:ident, $unit:ident, $min_inputs:expr) => {
        #[doc = concat!("Constructor for [`", stringify!($unit), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.len() < $min_inputs {
                    return Err(BuildError::WrongInputCount);
                }
                let mut unit: $unit = Zeroable::zeroed();
                unit.audio = (ctx.rate == Rate::Audio) as u32;
                Ok(unit_spec(unit))
            }
        }
    };
}

measure_ctor!(PeakCtor, Peak, 2);
measure_ctor!(RunningMinCtor, RunningMin, 2);
measure_ctor!(RunningMaxCtor, RunningMax, 2);
measure_ctor!(PeakFollowerCtor, PeakFollower, 2);
measure_ctor!(MostChangeCtor, MostChange, 2);
measure_ctor!(LeastChangeCtor, LeastChange, 2);
measure_ctor!(LastValueCtor, LastValue, 2);
