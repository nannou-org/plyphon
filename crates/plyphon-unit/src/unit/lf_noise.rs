//! Low-frequency and dynamic noise generators - plyphon's ports of scsynth's `LFNoise0/1/2` and
//! `LFClipNoise` (`NoiseUGens.cpp`) and `LFDNoise0/1/3` and `LFDClipNoise` (`DynNoiseUGens.cpp`).
//!
//! All produce a new random value at an average `freq`, differing in how they bridge between values:
//! `*Noise0`/`*ClipNoise` hold a step, `*Noise1` ramps linearly, `*Noise2`/`*DNoise3` interpolate
//! smoothly (quadratic/cubic). The `LF*` units count whole samples between values (so transitions are
//! quantised to the sample rate, and `freq` is read once per block); the dynamic `LFD*` units run a
//! floating phase decremented by `freq * sampleDur` (so `freq` may be modulated at audio rate and
//! transitions land off-grid). Each embeds a per-unit [`Rng`] and reseeds it in [`Unit::reseed`].

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

/// The unit's own sample rate (audio rate for `.ar`, control rate for `.kr`), matching scsynth's
/// `unit->mRate->mSampleRate`.
fn rate_sr(ctx: &ProcessCtx<'_>, audio: bool) -> f32 {
    if audio {
        ctx.audio.sample_rate as f32
    } else {
        ctx.control.sample_rate as f32
    }
}

/// The unit's own sample duration, matching scsynth's `SAMPLEDUR` (`unit->mRate->mSampleDur`).
fn sample_dur(ctx: &ProcessCtx<'_>, audio: bool) -> f32 {
    if audio {
        ctx.audio.sample_dur as f32
    } else {
        ctx.control.sample_dur as f32
    }
}

/// `LFNoise0.ar/kr(freq)`: a step of random values in `[-1, 1)`, a new value every `sr / freq`
/// samples.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFNoise0 {
    rng: Rng,
    level: f32,
    counter: i32,
    audio: u32,
}

impl LFNoise0 {
    fn step(&mut self, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            self.counter = period(rate_sr, freq, 1);
            self.level = self.rng.next_bipolar();
        }
        self.counter -= 1;
        self.level
    }
}

impl Unit for LFNoise0 {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = ctx.ins.control(0);
        let sr = rate_sr(ctx, audio);
        drive(ctx, audio, |_| self.step(freq, sr));
        DoneAction::Nothing
    }
}

/// `LFClipNoise.ar/kr(freq)`: like [`LFNoise0`] but each held value is `+1` or `-1` (a random
/// square).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFClipNoise {
    rng: Rng,
    level: f32,
    counter: i32,
    audio: u32,
}

impl LFClipNoise {
    fn step(&mut self, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            self.counter = period(rate_sr, freq, 1);
            self.level = coin(&mut self.rng);
        }
        self.counter -= 1;
        self.level
    }
}

impl Unit for LFClipNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = ctx.ins.control(0);
        let sr = rate_sr(ctx, audio);
        drive(ctx, audio, |_| self.step(freq, sr));
        DoneAction::Nothing
    }
}

/// `LFNoise1.ar/kr(freq)`: random values in `[-1, 1)` joined by straight-line ramps (a new target
/// every `sr / freq` samples).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFNoise1 {
    rng: Rng,
    level: f32,
    slope: f32,
    counter: i32,
    audio: u32,
}

impl LFNoise1 {
    fn step(&mut self, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            self.counter = period(rate_sr, freq, 1);
            let next = self.rng.next_bipolar();
            self.slope = (next - self.level) / self.counter as f32;
        }
        let out = self.level;
        self.level += self.slope;
        self.counter -= 1;
        out
    }
}

impl Unit for LFNoise1 {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.level = self.rng.next_bipolar();
        self.slope = 0.0;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = ctx.ins.control(0);
        let sr = rate_sr(ctx, audio);
        drive(ctx, audio, |_| self.step(freq, sr));
        DoneAction::Nothing
    }
}

/// `LFNoise2.ar/kr(freq)`: random values joined by quadratic curves for a smoother contour than
/// [`LFNoise1`] (a new target every `sr / freq` samples, clamped to at least two).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFNoise2 {
    rng: Rng,
    level: f32,
    slope: f32,
    curve: f32,
    next_value: f32,
    next_midpt: f32,
    counter: i32,
    audio: u32,
}

impl LFNoise2 {
    fn step(&mut self, freq: f32, rate_sr: f32) -> f32 {
        if self.counter <= 0 {
            let value = self.next_value;
            self.next_value = self.rng.next_bipolar();
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
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.next_value = self.rng.next_bipolar();
        self.next_midpt = self.next_value * 0.5;
        self.level = 0.0;
        self.slope = 0.0;
        self.curve = 0.0;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = ctx.ins.control(0);
        let sr = rate_sr(ctx, audio);
        drive(ctx, audio, |_| self.step(freq, sr));
        DoneAction::Nothing
    }
}

/// `LFDNoise0.ar/kr(freq)`: the dynamic (off-grid) counterpart of [`LFNoise0`] - a random step held
/// until a floating phase, decremented by `freq * sampleDur`, wraps past zero. `freq` may be audio
/// rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDNoise0 {
    rng: Rng,
    phase: f32,
    level: f32,
    audio: u32,
}

impl LFDNoise0 {
    fn step(&mut self, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.level = self.rng.next_bipolar();
        }
        self.level
    }
}

impl Unit for LFDNoise0 {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = sig(&ctx.ins, 0);
        let smpdur = sample_dur(ctx, audio);
        drive(ctx, audio, |i| self.step(freq.at(i), smpdur));
        DoneAction::Nothing
    }
}

/// `LFDClipNoise.ar/kr(freq)`: the dynamic counterpart of [`LFClipNoise`] - a random `+1`/`-1` square
/// switched when the floating phase wraps past zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDClipNoise {
    rng: Rng,
    phase: f32,
    level: f32,
    audio: u32,
}

impl LFDClipNoise {
    fn step(&mut self, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.level = coin(&mut self.rng);
        }
        self.level
    }
}

impl Unit for LFDClipNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = sig(&ctx.ins, 0);
        let smpdur = sample_dur(ctx, audio);
        drive(ctx, audio, |i| self.step(freq.at(i), smpdur));
        DoneAction::Nothing
    }
}

/// `LFDNoise1.ar/kr(freq)`: the dynamic counterpart of [`LFNoise1`] - a straight-line ramp between
/// random values as the floating phase falls from one toward zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDNoise1 {
    rng: Rng,
    phase: f32,
    prev_level: f32,
    next_level: f32,
    audio: u32,
}

impl LFDNoise1 {
    fn step(&mut self, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.prev_level = self.next_level;
            self.next_level = self.rng.next_bipolar();
        }
        self.next_level + self.phase * (self.prev_level - self.next_level)
    }
}

impl Unit for LFDNoise1 {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.prev_level = 0.0;
        self.next_level = self.rng.next_bipolar();
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = sig(&ctx.ins, 0);
        let smpdur = sample_dur(ctx, audio);
        drive(ctx, audio, |i| self.step(freq.at(i), smpdur));
        DoneAction::Nothing
    }
}

/// `LFDNoise3.ar/kr(freq)`: the dynamic counterpart of [`LFNoise2`] - a cubic curve through the last
/// four random values (each scaled by `0.8` to cap the interpolation overshoot at 1) as the floating
/// phase falls toward zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFDNoise3 {
    rng: Rng,
    phase: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    audio: u32,
}

impl LFDNoise3 {
    /// A fresh random value scaled by `0.8` (scsynth caps the cubic overshoot at 1 this way).
    fn draw(&mut self) -> f32 {
        self.rng.next_bipolar() * 0.8
    }

    fn step(&mut self, freq: f32, smpdur: f32) -> f32 {
        self.phase -= freq * smpdur;
        if self.phase < 0.0 {
            self.phase = wrap(self.phase, 0.0, 1.0);
            self.a = self.b;
            self.b = self.c;
            self.c = self.d;
            self.d = self.draw();
        }
        cubicinterp(1.0 - self.phase, self.a, self.b, self.c, self.d)
    }
}

impl Unit for LFDNoise3 {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.a = self.draw();
        self.b = self.draw();
        self.c = self.draw();
        self.d = self.draw();
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let freq = sig(&ctx.ins, 0);
        let smpdur = sample_dur(ctx, audio);
        drive(ctx, audio, |i| self.step(freq.at(i), smpdur));
        DoneAction::Nothing
    }
}

/// Build a low-frequency/dynamic noise unit: seed its [`Rng`] from `ctx.seed`, set the output-rate
/// flag, and run the type's own `reseed` to prime any derived starting state (matching scsynth's
/// `Ctor`). Requires the `freq` input.
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
                unit.reseed(ctx.seed);
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
