//! The noise generators - plyphon's ports of scsynth's `WhiteNoise`, `ClipNoise`, `GrayNoise`,
//! `PinkNoise`, `BrownNoise`, `Dust` and `Dust2`, plus the chaotic/deterministic `Crackle`, `Logistic`,
//! `Hasher` and `MantissaMask` (`NoiseUGens.cpp`).
//!
//! The random generators draw from the synth's random stream ([`ProcessCtx::rgen`]), scsynth's
//! `mParent->mRGen`, so they share it with every other random unit on the stream and a `RandSeed`
//! restarts them too. A block loop works on a local copy of the stream and writes it back, as
//! scsynth's `RGET`/`RPUT` do. The coefficient-free ones output at whichever rate the SynthDef
//! assigns. The float bit-tricks in scsynth's `SC_RGen.h` (`frand`/`frand2`/`frand8`/`fcoin`, and
//! PinkNoise's mantissa packing) are reproduced with safe [`f32::from_bits`]. `Crackle`/`Logistic` are
//! deterministic chaotic maps (seeded from the stream / an `init` input); `Hasher`/`MantissaMask` are
//! pure deterministic bit manglers of their input.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;
use plyphon_dsp::rng::{Rng, hash};

/// GrayNoise's output scale, `1 / 2^31` (scsynth's `4.65661287308e-10`).
const GRAY_SCALE: f32 = 1.0 / 2_147_483_648.0;

/// One of `-1.0`/`+1.0` chosen at random (scsynth's `fcoin`).
pub(crate) fn coin(rng: &mut Rng) -> f32 {
    if rng.next_u32() & 0x8000_0000 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// A small step uniformly in `[-0.125, 0.125)` (scsynth's `frand8`), used by BrownNoise.
fn frand8(rng: &mut Rng) -> f32 {
    rng.next_unipolar() * 0.25 - 0.125
}

/// Fill the output at the unit's rate (a full audio block, or one control value) from `next`,
/// drawing from a local copy of the synth's random stream that is written back afterwards.
fn generate(ctx: &mut ProcessCtx<'_>, audio: bool, mut next: impl FnMut(&mut Rng) -> f32) {
    let mut rgen = *ctx.rgen;
    if audio {
        for o in ctx.outs.audio(0).iter_mut() {
            *o = next(&mut rgen);
        }
    } else {
        *ctx.outs.control(0) = next(&mut rgen);
    }
    *ctx.rgen = rgen;
}

/// Map input 0 through `f` to the output at the unit's rate (a control input broadcasts).
fn transform(ctx: &mut ProcessCtx<'_>, audio: bool, f: impl Fn(f32) -> f32) {
    let ins = ctx.ins;
    if audio {
        let in_audio = (ins.rate(0) == Rate::Audio).then(|| ins.audio(0));
        let in_ctrl = ins.control(0);
        for (k, o) in ctx.outs.audio(0).iter_mut().enumerate() {
            *o = f(in_audio.map_or(in_ctrl, |s| s[k]));
        }
    } else {
        *ctx.outs.control(0) = f(ins.control(0));
    }
}

/// `WhiteNoise.ar/kr`: samples drawn uniformly from `[-1, 1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WhiteNoise {
    /// `0`/`1`: audio-rate (a full block) vs control-rate (one value).
    audio: u32,
}

impl Unit for WhiteNoise {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        generate(ctx, self.audio != 0, Rng::next_bipolar);
        DoneAction::Nothing
    }
}

/// Constructor for [`WhiteNoise`].
pub struct WhiteNoiseCtor;

impl UnitDef for WhiteNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(WhiteNoise {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `ClipNoise.ar/kr`: noise clipped to `+1`/`-1` (a random square).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ClipNoise {
    audio: u32,
}

impl Unit for ClipNoise {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        generate(ctx, self.audio != 0, coin);
        DoneAction::Nothing
    }
}

/// Constructor for [`ClipNoise`].
pub struct ClipNoiseCtor;

impl UnitDef for ClipNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(ClipNoise {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `GrayNoise.ar/kr`: noise from randomly flipping bits of an accumulator (a flatter spectrum than
/// white noise).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GrayNoise {
    counter: u32,
    audio: u32,
}

impl Unit for GrayNoise {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let counter = &mut self.counter;
        generate(ctx, audio, |rgen| {
            *counter ^= 1u32 << (rgen.next_u32() & 31);
            (*counter as i32 as f32) * GRAY_SCALE
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`GrayNoise`].
pub struct GrayNoiseCtor;

impl UnitDef for GrayNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(GrayNoise {
            counter: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `PinkNoise.ar/kr`: 1/f (pink) noise via the Voss-McCartney dice algorithm (16 octave bands).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PinkNoise {
    dice: [u32; 16],
    total: u32,
    audio: u32,
}

impl Unit for PinkNoise {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's constructor rolls the 16 dice and their running total, then runs the calc.
        let mut total = 0u32;
        for d in &mut self.dice {
            let newrand = ctx.rgen.next_u32() >> 13;
            total = total.wrapping_add(newrand);
            *d = newrand;
        }
        self.total = total;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let dice = &mut self.dice;
        let total = &mut self.total;
        generate(ctx, audio, |rgen| {
            let counter = rgen.next_u32();
            let newrand = counter >> 13;
            let k = (counter.trailing_zeros() & 15) as usize;
            let prevrand = dice[k];
            dice[k] = newrand;
            *total = total.wrapping_add(newrand.wrapping_sub(prevrand));
            let newrand2 = rgen.next_u32() >> 13;
            // scsynth packs the accumulator into a float's mantissa (exponent 0x40000000 -> [2, 4))
            // and subtracts 3 to land in [-1, 1).
            f32::from_bits(total.wrapping_add(newrand2) | 0x4000_0000) - 3.0
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PinkNoise`].
pub struct PinkNoiseCtor;

impl UnitDef for PinkNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(PinkNoise {
            dice: [0; 16],
            total: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `BrownNoise.ar/kr`: Brownian (1/f^2) noise - a random walk reflected at `±1`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BrownNoise {
    level: f32,
    audio: u32,
}

impl Unit for BrownNoise {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's constructor draws the starting level and writes it, without running the calc.
        self.level = ctx.rgen.next_bipolar();
        *ctx.outs.control(0) = self.level;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let level = &mut self.level;
        generate(ctx, audio, |rgen| {
            *level += frand8(rgen);
            if *level > 1.0 {
                *level = 2.0 - *level;
            } else if *level < -1.0 {
                *level = -2.0 - *level;
            }
            *level
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`BrownNoise`].
pub struct BrownNoiseCtor;

impl UnitDef for BrownNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(BrownNoise {
            level: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Dust.ar/kr(density)`: random unipolar impulses (`[0, 1]`) at an average `density` per second.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dust {
    audio: u32,
}

impl Unit for Dust {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let density = ctx.ins.control(0);
        let sample_dur = ctx.own.sample_dur as f32;
        let thresh = density * sample_dur;
        let scale = if thresh > 0.0 { 1.0 / thresh } else { 0.0 };
        generate(ctx, audio, |rgen| {
            let z = rgen.next_unipolar();
            if z < thresh { z * scale } else { 0.0 }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`Dust`].
pub struct DustCtor;

impl UnitDef for DustCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Dust {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Dust2.ar/kr(density)`: random bipolar impulses (`[-1, 1]`) at an average `density` per second.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dust2 {
    audio: u32,
}

impl Unit for Dust2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let density = ctx.ins.control(0);
        let sample_dur = ctx.own.sample_dur as f32;
        let thresh = density * sample_dur;
        let scale = if thresh > 0.0 { 2.0 / thresh } else { 0.0 };
        generate(ctx, audio, |rgen| {
            let z = rgen.next_unipolar();
            if z < thresh { z * scale - 1.0 } else { 0.0 }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`Dust2`].
pub struct Dust2Ctor;

impl UnitDef for Dust2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Dust2 {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Crackle.ar(chaosParam)`: a chaotic noise from the map `y0 = |y1*param - y2 - 0.05|` (scsynth's
/// `Crackle`). Deterministic once seeded; `y1` starts from a double-precision draw (scsynth's `drand`)
/// so instances decorrelate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Crackle {
    y1: f32,
    y2: f32,
    audio: u32,
    _pad: u32,
}

impl Unit for Crackle {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's constructor seeds `y1` from a double-precision draw and runs the calc.
        self.y1 = ctx.rgen.next_unipolar_f64() as f32;
        self.y2 = 0.0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let param = ctx.ins.control(0);
        let audio = self.audio != 0;
        let mut y1 = self.y1;
        let mut y2 = self.y2;
        generate(ctx, audio, |_| {
            let y0 = (y1 * param - y2 - 0.05).abs();
            y2 = y1;
            y1 = y0;
            y0
        });
        self.y1 = y1;
        self.y2 = y2;
        DoneAction::Nothing
    }
}

/// Constructor for [`Crackle`].
pub struct CrackleCtor;

impl UnitDef for CrackleCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Crackle {
            y1: 0.0,
            y2: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
            _pad: 0,
        }))
    }
}

/// `Logistic.ar(chaosParam, freq, init)`: the logistic map `y = param*y*(1-y)`, iterated at `freq`
/// (held between iterations) - a route into chaos as `param` approaches 4. `y` starts from `init`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Logistic {
    y1: f64,
    counter: i32,
    audio: u32,
}

impl Unit for Logistic {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.y1 = ctx.ins.control(2) as f64;
        // scsynth's constructor iterates the map once with `Logistic_next_1`, which leaves the
        // sample counter at zero, so the first block iterates again at its first sample.
        let param = ctx.ins.control(0) as f64;
        self.y1 = param * self.y1 * (1.0 - self.y1);
        *ctx.outs.control(0) = self.y1 as f32;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let param = ctx.ins.control(0) as f64;
        let freq = ctx.ins.control(1);
        let sr = ctx.own.sample_rate as f32;
        let audio = self.audio != 0;
        let mut y1 = self.y1;
        let mut counter = self.counter;
        // Iterate the map when the sample counter expires (every `sr/freq` samples).
        let step = |c: &mut i32, y: &mut f64| {
            if *c <= 0 {
                *c = ((sr / freq.max(0.001)) as i32).max(1);
                *y = param * *y * (1.0 - *y);
            }
            *c -= 1;
            *y as f32
        };
        if audio {
            for o in ctx.outs.audio(0).iter_mut() {
                *o = step(&mut counter, &mut y1);
            }
        } else {
            *ctx.outs.control(0) = step(&mut counter, &mut y1);
        }
        self.y1 = y1;
        self.counter = counter;
        DoneAction::Nothing
    }
}

/// Constructor for [`Logistic`].
pub struct LogisticCtor;

impl UnitDef for LogisticCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Logistic {
            y1: 0.0,
            counter: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Hasher.ar/kr(in)`: a deterministic pseudo-random value in `[-1, 1)` per input sample (scsynth's
/// integer `Hash` of the input's bits, packed into a float). The same input always gives the same
/// output, so it "freezes" noise to a signal.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Hasher {
    audio: u32,
}

impl Unit for Hasher {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let map = |x: f32| {
            let bits = 0x4000_0000u32 | ((hash(x.to_bits() as i32) as u32) >> 9);
            f32::from_bits(bits) - 3.0
        };
        transform(ctx, self.audio != 0, map);
        DoneAction::Nothing
    }
}

/// Constructor for [`Hasher`].
pub struct HasherCtor;

impl UnitDef for HasherCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Hasher {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `MantissaMask.ar/kr(in, bits)`: keep only the top `bits` mantissa bits of the input, masking the
/// rest to zero - a cheap bit-crush/distortion.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MantissaMask {
    audio: u32,
}

impl Unit for MantissaMask {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bits = (ctx.ins.control(1) as i32).clamp(0, 23);
        let mask = (-1i32) << (23 - bits);
        let map = |x: f32| f32::from_bits((x.to_bits() as i32 & mask) as u32);
        transform(ctx, self.audio != 0, map);
        DoneAction::Nothing
    }
}

/// Constructor for [`MantissaMask`].
pub struct MantissaMaskCtor;

impl UnitDef for MantissaMaskCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(MantissaMask {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
