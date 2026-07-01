//! The noise generators - plyphon's ports of scsynth's `WhiteNoise`, `ClipNoise`, `GrayNoise`,
//! `PinkNoise`, `BrownNoise`, `Dust` and `Dust2` (`NoiseUGens.cpp`).
//!
//! Each embeds a per-unit [`Rng`] (scsynth's Taus88 `RGen`) in its `Pod` state and reseeds it in
//! [`Unit::reseed`]; the coefficient-free generators output at whichever rate the SynthDef assigns.
//! The float bit-tricks in scsynth's `SC_RGen.h` (`frand`/`frand2`/`frand8`/`fcoin`, and PinkNoise's
//! mantissa packing) are reproduced with safe [`f32::from_bits`].

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;
use plyphon_dsp::rng::Rng;

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

/// Fill the output at the unit's rate (a full audio block, or one control value) from `next`.
fn generate(ctx: &mut ProcessCtx<'_>, audio: bool, mut next: impl FnMut() -> f32) {
    if audio {
        for o in ctx.outs.audio(0).iter_mut() {
            *o = next();
        }
    } else {
        *ctx.outs.control(0) = next();
    }
}

/// `WhiteNoise.ar/kr`: samples drawn uniformly from `[-1, 1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WhiteNoise {
    rng: Rng,
    /// `0`/`1`: audio-rate (a full block) vs control-rate (one value).
    audio: u32,
}

impl Unit for WhiteNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.audio != 0 {
            for o in ctx.outs.audio(0).iter_mut() {
                *o = self.rng.next_bipolar();
            }
        } else {
            *ctx.outs.control(0) = self.rng.next_bipolar();
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`WhiteNoise`].
pub struct WhiteNoiseCtor;

impl UnitDef for WhiteNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(WhiteNoise {
            rng: Rng::new(ctx.seed),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `ClipNoise.ar/kr`: noise clipped to `+1`/`-1` (a random square).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ClipNoise {
    rng: Rng,
    audio: u32,
}

impl Unit for ClipNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let rng = &mut self.rng;
        generate(ctx, audio, || coin(rng));
        DoneAction::Nothing
    }
}

/// Constructor for [`ClipNoise`].
pub struct ClipNoiseCtor;

impl UnitDef for ClipNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(ClipNoise {
            rng: Rng::new(ctx.seed),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `GrayNoise.ar/kr`: noise from randomly flipping bits of an accumulator (a flatter spectrum than
/// white noise).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GrayNoise {
    rng: Rng,
    counter: u32,
    audio: u32,
}

impl Unit for GrayNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let counter = &mut self.counter;
        let rng = &mut self.rng;
        generate(ctx, audio, || {
            *counter ^= 1u32 << (rng.next_u32() & 31);
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
            rng: Rng::new(ctx.seed),
            counter: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `PinkNoise.ar/kr`: 1/f (pink) noise via the Voss-McCartney dice algorithm (16 octave bands).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PinkNoise {
    rng: Rng,
    dice: [u32; 16],
    total: u32,
    audio: u32,
}

/// Seed PinkNoise's 16 dice and their running total (scsynth's `PinkNoise_Ctor`).
fn pink_init(rng: &mut Rng) -> ([u32; 16], u32) {
    let mut dice = [0u32; 16];
    let mut total = 0u32;
    for d in &mut dice {
        let newrand = rng.next_u32() >> 13;
        total = total.wrapping_add(newrand);
        *d = newrand;
    }
    (dice, total)
}

impl Unit for PinkNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        let (dice, total) = pink_init(&mut self.rng);
        self.dice = dice;
        self.total = total;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let dice = &mut self.dice;
        let total = &mut self.total;
        let rng = &mut self.rng;
        generate(ctx, audio, || {
            let counter = rng.next_u32();
            let newrand = counter >> 13;
            let k = (counter.trailing_zeros() & 15) as usize;
            let prevrand = dice[k];
            dice[k] = newrand;
            *total = total.wrapping_add(newrand.wrapping_sub(prevrand));
            let newrand2 = rng.next_u32() >> 13;
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
        let mut rng = Rng::new(ctx.seed);
        let (dice, total) = pink_init(&mut rng);
        Ok(unit_spec(PinkNoise {
            rng,
            dice,
            total,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `BrownNoise.ar/kr`: Brownian (1/f^2) noise - a random walk reflected at `±1`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BrownNoise {
    rng: Rng,
    level: f32,
    audio: u32,
}

impl Unit for BrownNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
        self.level = self.rng.next_bipolar();
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let level = &mut self.level;
        let rng = &mut self.rng;
        generate(ctx, audio, || {
            *level += frand8(rng);
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
        let mut rng = Rng::new(ctx.seed);
        let level = rng.next_bipolar();
        Ok(unit_spec(BrownNoise {
            rng,
            level,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Dust.ar/kr(density)`: random unipolar impulses (`[0, 1]`) at an average `density` per second.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dust {
    rng: Rng,
    audio: u32,
}

impl Unit for Dust {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let density = ctx.ins.control(0);
        let sample_dur = 1.0 / ctx.audio.sample_rate as f32;
        let thresh = density * sample_dur;
        let scale = if thresh > 0.0 { 1.0 / thresh } else { 0.0 };
        let rng = &mut self.rng;
        generate(ctx, audio, || {
            let z = rng.next_unipolar();
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
            rng: Rng::new(ctx.seed),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Dust2.ar/kr(density)`: random bipolar impulses (`[-1, 1]`) at an average `density` per second.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dust2 {
    rng: Rng,
    audio: u32,
}

impl Unit for Dust2 {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let density = ctx.ins.control(0);
        let sample_dur = 1.0 / ctx.audio.sample_rate as f32;
        let thresh = density * sample_dur;
        let scale = if thresh > 0.0 { 2.0 / thresh } else { 0.0 };
        let rng = &mut self.rng;
        generate(ctx, audio, || {
            let z = rng.next_unipolar();
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
            rng: Rng::new(ctx.seed),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
