//! Low-frequency and dynamic noise generators - plyphon's ports of scsynth's `LFNoise0/1/2` and
//! `LFClipNoise` (`NoiseUGens.cpp`) and `LFDNoise0/1/3` and `LFDClipNoise` (`DynNoiseUGens.cpp`).
//!
//! All produce a new random value at an average `freq`, differing in how they bridge between values:
//! `*Noise0`/`*ClipNoise` hold a step, `*Noise1` ramps linearly, `*Noise2`/`*DNoise3` interpolate
//! smoothly (quadratic/cubic). The `LF*` units count whole samples between values (so transitions are
//! quantised to the sample rate, and `freq` is read once per block); the dynamic `LFD*` units run a
//! floating phase decremented by `freq * sampleDur` (so `freq` may be modulated at audio rate and
//! transitions land off-grid). Each draws from the synth's random stream ([`ProcessCtx::rgen`]),
//! working on a local copy through a block as scsynth's `RGET`/`RPUT` do.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::noise::coin;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::interp::cubicinterp;
use plyphon_dsp::ops::wrap;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::rng::Rng;

/// The whole-sample period `sr / max(freq, 0.001)` a counter-based `LF*` unit holds a value for, at
/// least `floor` samples (scsynth clamps `LFNoise2` to 2, the others to 1).
fn period(rate_sr: f32, freq: f32, floor: i32) -> i32 {
    ((rate_sr / freq.max(0.001)) as i32).max(floor)
}

/// Run one block of a counter-based `LF*` unit: `freq` read once, `step` per output frame with a
/// local copy of the synth's random stream, written back afterwards.
fn run_counter(
    ctx: &mut ProcessCtx<'_>,
    audio: bool,
    mut step: impl FnMut(&mut Rng, f32, f32) -> f32,
) {
    let freq = ctx.ins.control(0);
    let sr = ctx.own.sample_rate as f32;
    let mut rgen = *ctx.rgen;
    drive(ctx, audio, |_| step(&mut rgen, freq, sr));
    *ctx.rgen = rgen;
}

/// Run one block of a dynamic `LFD*` unit: `freq` read per frame, `step` per output frame with a
/// local copy of the synth's random stream, written back afterwards.
fn run_phase(
    ctx: &mut ProcessCtx<'_>,
    audio: bool,
    mut step: impl FnMut(&mut Rng, f32, f32) -> f32,
) {
    let freq = sig(&ctx.ins, 0);
    let smpdur = ctx.own.sample_dur as f32;
    let mut rgen = *ctx.rgen;
    drive(ctx, audio, |i| step(&mut rgen, freq.at(i), smpdur));
    *ctx.rgen = rgen;
}

/// `LFNoise0.ar/kr(freq)`: a step of random values in `[-1, 1)`, a new value every `sr / freq`
/// samples.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFNoise0 {
    level: f32,
    counter: i32,
    audio: u32,
}

impl LFNoise0 {
    fn step(&mut self, rgen: &mut Rng, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            self.counter = period(rate_sr, freq, 1);
            self.level = rgen.next_bipolar();
        }
        self.counter -= 1;
        self.level
    }
}

impl Unit for LFNoise0 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_counter(ctx, self.audio != 0, |rgen, freq, sr| {
            self.step(rgen, freq, sr)
        });
        DoneAction::Nothing
    }
}

/// `LFClipNoise.ar/kr(freq)`: like [`LFNoise0`] but each held value is `+1` or `-1` (a random
/// square).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFClipNoise {
    level: f32,
    counter: i32,
    audio: u32,
}

impl LFClipNoise {
    fn step(&mut self, rgen: &mut Rng, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            self.counter = period(rate_sr, freq, 1);
            self.level = coin(rgen);
        }
        self.counter -= 1;
        self.level
    }
}

impl Unit for LFClipNoise {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_counter(ctx, self.audio != 0, |rgen, freq, sr| {
            self.step(rgen, freq, sr)
        });
        DoneAction::Nothing
    }
}

/// `LFNoise1.ar/kr(freq)`: random values in `[-1, 1)` joined by straight-line ramps (a new target
/// every `sr / freq` samples).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFNoise1 {
    level: f32,
    slope: f32,
    counter: i32,
    audio: u32,
}

impl LFNoise1 {
    fn step(&mut self, rgen: &mut Rng, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            self.counter = period(rate_sr, freq, 1);
            let next = rgen.next_bipolar();
            self.slope = (next - self.level) / self.counter as f32;
        }
        let out = self.level;
        self.level += self.slope;
        self.counter -= 1;
        out
    }
}

impl Unit for LFNoise1 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's constructor draws the starting level, then runs the calc.
        self.level = ctx.rgen.next_bipolar();
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_counter(ctx, self.audio != 0, |rgen, freq, sr| {
            self.step(rgen, freq, sr)
        });
        DoneAction::Nothing
    }
}

/// `LFNoise2.ar/kr(freq)`: random values joined by quadratic curves for a smoother contour than
/// [`LFNoise1`] (a new target every `sr / freq` samples, clamped to at least two).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFNoise2 {
    level: f32,
    slope: f32,
    curve: f32,
    next_value: f32,
    next_midpt: f32,
    counter: i32,
    audio: u32,
}

impl LFNoise2 {
    fn step(&mut self, rgen: &mut Rng, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            let value = self.next_value;
            self.next_value = rgen.next_bipolar();
            self.level = self.next_midpt;
            self.next_midpt = (self.next_value + value) * 0.5;
            self.counter = period(rate_sr, freq, 2);
            let seg = self.counter as f32;
            self.curve =
                2.0 * (self.next_midpt - self.level - seg * self.slope) / (seg * seg + seg);
        }
        let out = self.level;
        self.slope += self.curve;
        self.level += self.slope;
        self.counter -= 1;
        out
    }
}

impl Unit for LFNoise2 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's constructor draws the first target and its midpoint, then runs the calc.
        self.next_value = ctx.rgen.next_bipolar();
        self.next_midpt = self.next_value * 0.5;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_counter(ctx, self.audio != 0, |rgen, freq, sr| {
            self.step(rgen, freq, sr)
        });
        DoneAction::Nothing
    }
}

/// `LFDNoise0.ar/kr(freq)`: the dynamic (off-grid) counterpart of [`LFNoise0`] - a random step held
/// until a floating phase, decremented by `freq * sampleDur`, wraps past zero. `freq` may be audio
/// rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDNoise0 {
    phase: f32,
    level: f32,
    audio: u32,
}

impl LFDNoise0 {
    fn step(&mut self, rgen: &mut Rng, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.level = rgen.next_bipolar();
        }
        self.level
    }
}

impl Unit for LFDNoise0 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_phase(ctx, self.audio != 0, |rgen, freq, smpdur| {
            self.step(rgen, freq, smpdur)
        });
        DoneAction::Nothing
    }
}

/// `LFDClipNoise.ar/kr(freq)`: the dynamic counterpart of [`LFClipNoise`] - a random `+1`/`-1` square
/// switched when the floating phase wraps past zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDClipNoise {
    phase: f32,
    level: f32,
    audio: u32,
}

impl LFDClipNoise {
    fn step(&mut self, rgen: &mut Rng, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.level = coin(rgen);
        }
        self.level
    }
}

impl Unit for LFDClipNoise {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_phase(ctx, self.audio != 0, |rgen, freq, smpdur| {
            self.step(rgen, freq, smpdur)
        });
        DoneAction::Nothing
    }
}

/// `LFDNoise1.ar/kr(freq)`: the dynamic counterpart of [`LFNoise1`] - a straight-line ramp between
/// random values as the floating phase falls from one toward zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDNoise1 {
    phase: f32,
    prev_level: f32,
    next_level: f32,
    audio: u32,
}

impl LFDNoise1 {
    fn step(&mut self, rgen: &mut Rng, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.prev_level = self.next_level;
            self.next_level = rgen.next_bipolar();
        }
        self.next_level + self.phase * (self.prev_level - self.next_level)
    }
}

impl Unit for LFDNoise1 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's constructor draws the first target level, then runs the calc.
        self.next_level = ctx.rgen.next_bipolar();
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_phase(ctx, self.audio != 0, |rgen, freq, smpdur| {
            self.step(rgen, freq, smpdur)
        });
        DoneAction::Nothing
    }
}

/// `LFDNoise3.ar/kr(freq)`: the dynamic counterpart of [`LFNoise2`] - a cubic curve through the last
/// four random values (each scaled by `0.8` to cap the interpolation overshoot at 1) as the floating
/// phase falls toward zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDNoise3 {
    phase: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    audio: u32,
}

/// A fresh random value scaled by `0.8` (scsynth caps the cubic overshoot at 1 this way).
fn draw3(rgen: &mut Rng) -> f32 {
    rgen.next_bipolar() * 0.8
}

impl LFDNoise3 {
    fn step(&mut self, rgen: &mut Rng, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.a = self.b;
            self.b = self.c;
            self.c = self.d;
            self.d = draw3(rgen);
        }
        cubicinterp(1.0 - self.phase, self.a, self.b, self.c, self.d)
    }
}

impl Unit for LFDNoise3 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's constructor draws the four starting levels, then runs the calc.
        self.a = draw3(ctx.rgen);
        self.b = draw3(ctx.rgen);
        self.c = draw3(ctx.rgen);
        self.d = draw3(ctx.rgen);
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        run_phase(ctx, self.audio != 0, |rgen, freq, smpdur| {
            self.step(rgen, freq, smpdur)
        });
        DoneAction::Nothing
    }
}

/// Build a low-frequency/dynamic noise unit at its zeroed starting state with its output-rate flag
/// (its constructor, [`Unit::init`], draws any starting values). Requires the `freq` input.
macro_rules! lf_noise_ctor {
    ($ctor:ident, $unit:ident) => {
        #[doc = concat!("Constructor for [`", stringify!($unit), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.is_empty() {
                    return Err(BuildError::WrongInputCount);
                }
                let mut unit: $unit = Zeroable::zeroed();
                unit.audio = (ctx.rate == Rate::Audio) as u32;
                Ok(unit_spec(unit))
            }
        }
    };
}

lf_noise_ctor!(LFNoise0Ctor, LFNoise0);
lf_noise_ctor!(LFNoise1Ctor, LFNoise1);
lf_noise_ctor!(LFNoise2Ctor, LFNoise2);
lf_noise_ctor!(LFClipNoiseCtor, LFClipNoise);
lf_noise_ctor!(LFDNoise0Ctor, LFDNoise0);
lf_noise_ctor!(LFDNoise1Ctor, LFDNoise1);
lf_noise_ctor!(LFDNoise3Ctor, LFDNoise3);
lf_noise_ctor!(LFDClipNoiseCtor, LFDClipNoise);
