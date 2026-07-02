//! Trigger, sample-and-hold and flip-flop units - plyphon's ports of scsynth's `Trig`, `Trig1`,
//! `TDelay`, `Latch`, `Gate`, `ToggleFF`, `SetResetFF` and `Schmidt` (`TriggerUGens.cpp`).
//!
//! A "trigger" is a rising edge: a sample that is `> 0` where the previous was `<= 0`. Each unit
//! carries the previous input sample(s) across block boundaries so edges are detected at the block
//! seam too. The signal and trigger inputs are read at whatever rate the SynthDef assigns (per-sample
//! at audio rate, one value at control rate) via a small `Sig` helper, and the output is produced at
//! the unit's own rate.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// An input read either per-sample (audio rate) or as a single held value (control rate).
pub(crate) enum Sig<'a> {
    Audio(&'a [f32]),
    Control(f32),
}

impl Sig<'_> {
    /// The input's value at frame `i` (the held value at every frame when control-rate).
    pub(crate) fn at(&self, i: usize) -> f32 {
        match *self {
            Sig::Audio(s) => s[i],
            Sig::Control(v) => v,
        }
    }
}

/// Resolve input `i` to a [`Sig`] at its declared rate. The returned slice is tied to the World's
/// data lifetime, so it never conflicts with the mutable borrow of the output.
pub(crate) fn sig<'a>(ins: &Inputs<'a>, i: usize) -> Sig<'a> {
    if ins.rate(i) == Rate::Audio {
        Sig::Audio(ins.audio(i))
    } else {
        Sig::Control(ins.control(i))
    }
}

/// Drive the output at the unit's rate, computing each frame with `f`.
pub(crate) fn drive(ctx: &mut ProcessCtx<'_>, audio_out: bool, mut f: impl FnMut(usize) -> f32) {
    if audio_out {
        for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
            *o = f(i);
        }
    } else {
        *ctx.outs.control(0) = f(0);
    }
}

/// The trigger duration `dur` seconds as a sample count, at least 1 (scsynth's `dur*sr + .5`).
fn dur_samples(dur: f32, sr: f32) -> i32 {
    ((dur * sr + 0.5) as i32).max(1)
}

/// `Trig1.ar/kr(trig, dur)`: outputs `1` for `dur` seconds after each rising trigger, else `0`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Trig1 {
    prev: f32,
    counter: i32,
    audio: u32,
}

impl Trig1 {
    fn step(&mut self, cur: f32, dur: f32, sr: f32) -> f32 {
        let out = if self.counter > 0 {
            self.counter -= 1;
            if self.counter != 0 { 1.0 } else { 0.0 }
        } else if cur > 0.0 && self.prev <= 0.0 {
            self.counter = dur_samples(dur, sr);
            1.0
        } else {
            0.0
        };
        self.prev = cur;
        out
    }
}

impl Unit for Trig1 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let dur = ctx.ins.control(1);
        let sr = ctx.own.sample_rate as f32;
        drive(ctx, audio_out, |i| self.step(trig.at(i), dur, sr));
        DoneAction::Nothing
    }
}

/// `Trig.ar/kr(trig, dur)`: like [`Trig1`] but holds the trigger's value (not `1`) for `dur` seconds.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Trig {
    prev: f32,
    level: f32,
    counter: i32,
    audio: u32,
}

impl Trig {
    fn step(&mut self, cur: f32, dur: f32, sr: f32) -> f32 {
        let out = if self.counter > 0 {
            self.counter -= 1;
            if self.counter != 0 { self.level } else { 0.0 }
        } else if cur > 0.0 && self.prev <= 0.0 {
            self.counter = dur_samples(dur, sr);
            self.level = cur;
            self.level
        } else {
            0.0
        };
        self.prev = cur;
        out
    }
}

impl Unit for Trig {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let dur = ctx.ins.control(1);
        let sr = ctx.own.sample_rate as f32;
        drive(ctx, audio_out, |i| self.step(trig.at(i), dur, sr));
        DoneAction::Nothing
    }
}

/// `TDelay.ar/kr(trig, dur)`: delays each rising trigger by `dur` seconds, emitting a one-sample `1`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TDelay {
    prev: f32,
    counter: i32,
    audio: u32,
}

impl TDelay {
    fn step(&mut self, cur: f32, dur: f32, sr: f32) -> f32 {
        let out = if self.counter > 1 {
            self.counter -= 1;
            0.0
        } else if self.counter <= 0 {
            if cur > 0.0 && self.prev <= 0.0 {
                self.counter = dur_samples(dur, sr);
            }
            0.0
        } else {
            self.counter = 0;
            1.0
        };
        self.prev = cur;
        out
    }
}

impl Unit for TDelay {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let dur = ctx.ins.control(1);
        let sr = ctx.own.sample_rate as f32;
        drive(ctx, audio_out, |i| self.step(trig.at(i), dur, sr));
        DoneAction::Nothing
    }
}

/// `ToggleFF.ar/kr(trig)`: a toggle flip-flop - flips between `0` and `1` on each rising trigger.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ToggleFF {
    prev: f32,
    level: f32,
    audio: u32,
}

impl ToggleFF {
    fn step(&mut self, cur: f32) -> f32 {
        if cur > 0.0 && self.prev <= 0.0 {
            self.level = 1.0 - self.level;
        }
        self.prev = cur;
        self.level
    }
}

impl Unit for ToggleFF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        drive(ctx, audio_out, |i| self.step(trig.at(i)));
        DoneAction::Nothing
    }
}

/// `SetResetFF.ar/kr(trig, reset)`: a set-reset flip-flop - a rising `trig` sets the output to `1`, a
/// rising `reset` sets it to `0` (reset wins when both fire).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SetResetFF {
    prev_trig: f32,
    prev_reset: f32,
    level: f32,
    audio: u32,
}

impl SetResetFF {
    fn step(&mut self, cur_trig: f32, cur_reset: f32) -> f32 {
        if self.prev_reset <= 0.0 && cur_reset > 0.0 {
            self.level = 0.0;
        } else if self.prev_trig <= 0.0 && cur_trig > 0.0 {
            self.level = 1.0;
        }
        self.prev_trig = cur_trig;
        self.prev_reset = cur_reset;
        self.level
    }
}

impl Unit for SetResetFF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let trig = sig(&ctx.ins, 0);
        let reset = sig(&ctx.ins, 1);
        drive(ctx, audio_out, |i| self.step(trig.at(i), reset.at(i)));
        DoneAction::Nothing
    }
}

/// `Latch.ar/kr(in, trig)`: sample-and-hold - samples `in` on each rising `trig`, holding it between.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Latch {
    prev_trig: f32,
    level: f32,
    audio: u32,
}

impl Latch {
    fn step(&mut self, cur_in: f32, cur_trig: f32) -> f32 {
        if self.prev_trig <= 0.0 && cur_trig > 0.0 {
            self.level = cur_in;
        }
        self.prev_trig = cur_trig;
        self.level
    }
}

impl Unit for Latch {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let trig = sig(&ctx.ins, 1);
        drive(ctx, audio_out, |i| self.step(input.at(i), trig.at(i)));
        DoneAction::Nothing
    }
}

/// `Gate.ar/kr(in, trig)`: passes `in` while `trig > 0`, otherwise holds the last passed value.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Gate {
    level: f32,
    audio: u32,
}

impl Gate {
    fn step(&mut self, cur_in: f32, cur_trig: f32) -> f32 {
        if cur_trig > 0.0 {
            self.level = cur_in;
        }
        self.level
    }
}

impl Unit for Gate {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let trig = sig(&ctx.ins, 1);
        drive(ctx, audio_out, |i| self.step(input.at(i), trig.at(i)));
        DoneAction::Nothing
    }
}

/// `Schmidt.ar/kr(in, lo, hi)`: a Schmitt trigger - output goes to `1` when `in` rises above `hi` and
/// back to `0` when it falls below `lo` (hysteresis).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Schmidt {
    level: f32,
    audio: u32,
}

impl Schmidt {
    fn step(&mut self, cur_in: f32, lo: f32, hi: f32) -> f32 {
        if self.level == 1.0 {
            if cur_in < lo {
                self.level = 0.0;
            }
        } else if cur_in > hi {
            self.level = 1.0;
        }
        self.level
    }
}

impl Unit for Schmidt {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let lo = ctx.ins.control(1);
        let hi = ctx.ins.control(2);
        drive(ctx, audio_out, |i| self.step(input.at(i), lo, hi));
        DoneAction::Nothing
    }
}

/// Build a trigger unit whose state is zero-initialised except for the output-rate flag.
macro_rules! trigger_ctor {
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

trigger_ctor!(Trig1Ctor, Trig1, 2, { prev: 0.0, counter: 0 });
trigger_ctor!(TrigCtor, Trig, 2, { prev: 0.0, level: 0.0, counter: 0 });
trigger_ctor!(TDelayCtor, TDelay, 2, { prev: 0.0, counter: 0 });
trigger_ctor!(ToggleFFCtor, ToggleFF, 1, { prev: 0.0, level: 0.0 });
trigger_ctor!(SetResetFFCtor, SetResetFF, 2, { prev_trig: 0.0, prev_reset: 0.0, level: 0.0 });
trigger_ctor!(LatchCtor, Latch, 2, { prev_trig: 0.0, level: 0.0 });
trigger_ctor!(GateCtor, Gate, 2, { level: 0.0 });
trigger_ctor!(SchmidtCtor, Schmidt, 3, { level: 0.0 });
