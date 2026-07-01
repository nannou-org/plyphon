//! Diagnostic guard units - plyphon's ports of scsynth's `CheckBadValues` and `Sanitize`
//! (`TestUGens.cpp`).
//!
//! Both classify each sample with IEEE-754 categories: [`CheckBadValues`] reports the category as a
//! code (`0` ok, `1` NaN, `2` infinite, `3` subnormal), while [`Sanitize`] passes the signal through
//! but swaps any bad sample for a replacement value. scsynth's `post` diagnostics (printing a line
//! per bad value from the RT thread) are omitted: plyphon does no printing on the audio thread, so
//! the `id`/`post` inputs are accepted for signature compatibility and otherwise ignored.

use core::num::FpCategory;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// scsynth's classification code for a sample: `1` NaN, `2` infinite, `3` subnormal, else `0`.
fn classify(x: f32) -> f32 {
    match x.classify() {
        FpCategory::Nan => 1.0,
        FpCategory::Infinite => 2.0,
        FpCategory::Subnormal => 3.0,
        FpCategory::Normal | FpCategory::Zero => 0.0,
    }
}

/// `CheckBadValues.ar/kr(in, id, post)`: report each sample's IEEE-754 category as a code - `0` for a
/// normal/zero value, `1` NaN, `2` infinite, `3` subnormal. The output is at the unit's own rate and
/// the input is read at whatever rate the SynthDef assigns it.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CheckBadValues {
    /// Non-zero when the unit runs at audio rate.
    audio: u32,
}

impl Unit for CheckBadValues {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        drive(ctx, audio_out, |i| classify(input.at(i)));
        DoneAction::Nothing
    }
}

/// Constructor for [`CheckBadValues`].
pub struct CheckBadValuesCtor;

impl UnitDef for CheckBadValuesCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(CheckBadValues {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Sanitize.ar/kr(in, replace)`: pass `in` through unchanged, but replace any NaN, infinite or
/// subnormal sample with `replace` (a signal in its own right, read at its declared rate). Guards a
/// chain against a blown-up value poisoning everything downstream.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Sanitize {
    /// Non-zero when the unit runs at audio rate.
    audio: u32,
}

impl Unit for Sanitize {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio_out = self.audio != 0;
        let input = sig(&ctx.ins, 0);
        let replace = sig(&ctx.ins, 1);
        drive(ctx, audio_out, |i| {
            let x = input.at(i);
            match x.classify() {
                FpCategory::Nan | FpCategory::Infinite | FpCategory::Subnormal => replace.at(i),
                FpCategory::Normal | FpCategory::Zero => x,
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`Sanitize`].
pub struct SanitizeCtor;

impl UnitDef for SanitizeCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Sanitize {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
