//! sc3-plugins phase-vocoder operators with deterministic Plyphon buffer ownership.
//!
//! `PV_MagSmooth` is based on work by Dan Stowell, Copyright 2006-2010, licensed under
//! GPL-2.0-or-later. `PV_Morph` is based on work by Bhob Rainey and SuperCollider contributors,
//! licensed under GPL-2.0-or-later. Both use Plyphon's fixed auxiliary storage and leave invalid
//! frames byte-for-byte unchanged.

use core::f32::consts::{PI, TAU};

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec, unit_spec_aux};
use plyphon_dsp::buffer::BufView;
use plyphon_dsp::fft::is_supported_size;
use plyphon_dsp::rate::Rate;

/// The largest supported FFT frame. A smoothed spectrum retains one value per ordinary bin plus
/// DC and Nyquist, so its auxiliary memory is fixed at compile time.
const MAX_FFT_SIZE: usize = 16_384;

/// Number of retained magnitude values needed by the largest supported FFT.
const MAX_MAGNITUDES: usize = MAX_FFT_SIZE / 2 + 1;

/// Number of ordinary complex/polar bins in the largest supported packed spectrum.
const MAX_ORDINARY_BINS: usize = MAX_FFT_SIZE / 2 - 1;

/// Resolve an exact non-negative integer buffer token without accepting infinity or a fractional
/// value. The caller invokes [`pv::pv_frame`] first so output-token normalization remains shared.
fn token_index(token: f32) -> Option<usize> {
    (token.is_finite() && token >= 0.0 && token.fract() == 0.0).then_some(token as usize)
}

/// Whether a buffer is a supported mono packed spectrum whose DC, Nyquist, and decoded polar bins
/// are all finite. Complex bins are decoded read-only so a rejected frame cannot change its
/// coordinate tag or samples.
fn valid_spectrum(buffer: BufView<'_>) -> bool {
    let frames = buffer.num_frames();
    let data = buffer.data();
    if buffer.num_channels() != 1
        || data.len() != frames
        || !is_supported_size(frames)
        || data.len() < 2
        || !data[0].is_finite()
        || !data[1].is_finite()
    {
        return false;
    }

    let bins = pv::bins(data);
    bins.len() == (frames - 2) / 2
        && bins.iter().all(|&bin| {
            if !bin.x.is_finite() || !bin.y.is_finite() {
                return false;
            }
            let polar = pv::bin_as_polar_apx(buffer.coord(), bin);
            polar.x.is_finite() && polar.y.is_finite()
        })
}

/// Compute one retained convex smoothing value, rejecting a non-finite rounded result.
fn smooth(previous: f32, current: f32, factor: f32) -> Option<f32> {
    let value = previous * factor + current * (1.0 - factor);
    value.is_finite().then_some(value)
}

/// Validate the shared ABI of a control-rate PV operator.
fn validate_pv_abi(ctx: &BuildContext<'_>, num_inputs: usize) -> Result<(), BuildError> {
    if ctx.input_rates.len() != num_inputs {
        return Err(BuildError::WrongInputCount);
    }
    if ctx.num_outputs != 1 {
        return Err(BuildError::WrongOutputCount {
            expected: 1,
            actual: ctx.num_outputs,
        });
    }
    if ctx.special_index != 0 {
        return Err(BuildError::UnsupportedOp(ctx.special_index));
    }
    if ctx.rate != Rate::Control
        || ctx
            .input_rates
            .iter()
            .any(|rate| !matches!(rate, Rate::Scalar | Rate::Control))
    {
        return Err(BuildError::UnsupportedUnitRate);
    }
    Ok(())
}

/// Wrap one phase once into `[-π, π]`.
fn wrap_phase_once(phase: f32) -> f32 {
    if phase > PI {
        phase - TAU
    } else if phase < -PI {
        phase + TAU
    } else {
        phase
    }
}

/// `PV_Freeze(buffer, freeze)`: retain spectral magnitudes while advancing phase by the last
/// observed per-bin phase difference.
///
/// Each FFT size has three audible warm-up stages. The first establishes the size, the second
/// stores magnitudes and phases, and the third establishes phase differences. Frozen frames then
/// reuse the retained magnitudes, DC, and Nyquist while their phases continue coherently.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvFreeze {
    frame_size: u32,
    stage: u32,
    last_freeze: f32,
    stored_dc: f32,
    stored_nyquist: f32,
}

impl Unit for PvFreeze {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let token = ctx.ins.control(0);
        let freeze_input = ctx.ins.control(1);
        let frame_index = pv::pv_frame(ctx);
        if token == f32::INFINITY {
            *ctx.outs.control(0) = -1.0;
            return DoneAction::Nothing;
        }
        let Some(frame_index) = frame_index else {
            return DoneAction::Nothing;
        };
        let Some(checked_index) = token_index(token).filter(|&index| index == frame_index) else {
            return DoneAction::Nothing;
        };
        let Some(view) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, checked_index) else {
            return DoneAction::Nothing;
        };
        if !valid_spectrum(view) {
            return DoneAction::Nothing;
        }

        let frame_size = view.num_frames();
        let freeze = if freeze_input.is_finite() {
            freeze_input
        } else {
            self.last_freeze
        };
        let size_changed = self.frame_size as usize != frame_size;
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, checked_index)
        else {
            return DoneAction::Nothing;
        };
        let Some(spectrum) = pv::to_polar_apx(&mut buffer) else {
            return DoneAction::Nothing;
        };
        self.last_freeze = freeze;

        let bins = spectrum.bins.len();
        let memory = ctx.aux.f32_mut();
        let (magnitudes, rest) = memory.split_at_mut(MAX_ORDINARY_BINS);
        let (previous_phases, phase_differences) = rest.split_at_mut(MAX_ORDINARY_BINS);
        let magnitudes = &mut magnitudes[..bins];
        let previous_phases = &mut previous_phases[..bins];
        let phase_differences = &mut phase_differences[..bins];

        if size_changed || self.stage == 0 {
            self.frame_size = frame_size as u32;
            self.stage = 1;
            return DoneAction::Nothing;
        }

        if self.stage == 1 {
            for ((magnitude, previous_phase), bin) in magnitudes
                .iter_mut()
                .zip(previous_phases.iter_mut())
                .zip(spectrum.bins.iter())
            {
                *magnitude = bin.x;
                *previous_phase = bin.y;
            }
            self.stored_dc = *spectrum.dc;
            self.stored_nyquist = *spectrum.nyq;
            self.stage = 2;
            return DoneAction::Nothing;
        }

        if self.stage == 2 {
            for ((magnitude, previous_phase), (difference, bin)) in magnitudes
                .iter_mut()
                .zip(previous_phases.iter_mut())
                .zip(phase_differences.iter_mut().zip(spectrum.bins.iter_mut()))
            {
                *difference = bin.y - *previous_phase;
                *previous_phase = bin.y;
                if freeze > 0.0 {
                    bin.x = *magnitude;
                } else {
                    *magnitude = bin.x;
                }
            }
            if freeze > 0.0 {
                *spectrum.dc = self.stored_dc;
                *spectrum.nyq = self.stored_nyquist;
            } else {
                self.stored_dc = *spectrum.dc;
                self.stored_nyquist = *spectrum.nyq;
            }
            self.stage = 3;
            return DoneAction::Nothing;
        }

        if freeze > 0.0 {
            for ((magnitude, previous_phase), (difference, bin)) in magnitudes
                .iter()
                .zip(previous_phases.iter_mut())
                .zip(phase_differences.iter().zip(spectrum.bins.iter_mut()))
            {
                bin.x = *magnitude;
                bin.y = wrap_phase_once(*previous_phase + *difference);
                *previous_phase = bin.y;
            }
            *spectrum.dc = self.stored_dc;
            *spectrum.nyq = self.stored_nyquist;
        } else {
            for ((magnitude, previous_phase), (difference, bin)) in magnitudes
                .iter_mut()
                .zip(previous_phases.iter_mut())
                .zip(phase_differences.iter_mut().zip(spectrum.bins.iter()))
            {
                *magnitude = bin.x;
                *difference = bin.y - *previous_phase;
                *previous_phase = bin.y;
            }
            self.stored_dc = *spectrum.dc;
            self.stored_nyquist = *spectrum.nyq;
        }

        DoneAction::Nothing
    }
}

/// Constructor for [`PvFreeze`].
pub struct PvFreezeCtor;

impl UnitDef for PvFreezeCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_pv_abi(ctx, 2)?;
        Ok(unit_spec_aux(
            PvFreeze {
                frame_size: 0,
                stage: 0,
                last_freeze: 0.0,
                stored_dc: 0.0,
                stored_nyquist: 0.0,
            },
            3 * MAX_ORDINARY_BINS * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

/// `PV_MagSmooth(buffer, factor)`: smooth magnitudes, DC, and Nyquist between ready frames while
/// retaining each bin's incoming phase.
///
/// The first valid frame at each FFT size initializes fixed auxiliary memory and is otherwise
/// unchanged. `factor` is remembered per voice and clamped to `[0, 1]`; a non-finite value reuses
/// the last finite factor, initially `0.1`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagSmooth {
    frame_size: u32,
    initialized: u32,
    last_factor: f32,
}

impl Unit for PvMagSmooth {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let token = ctx.ins.control(0);
        let factor_in = ctx.ins.control(1);
        let frame_index = pv::pv_frame(ctx);
        if token == f32::INFINITY {
            *ctx.outs.control(0) = -1.0;
            return DoneAction::Nothing;
        }
        let Some(frame_index) = frame_index else {
            return DoneAction::Nothing;
        };
        let Some(checked_index) = token_index(token).filter(|&index| index == frame_index) else {
            return DoneAction::Nothing;
        };
        let Some(view) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, checked_index) else {
            return DoneAction::Nothing;
        };
        if !valid_spectrum(view) {
            return DoneAction::Nothing;
        }

        let frame_size = view.num_frames();
        let factor = if factor_in.is_finite() {
            factor_in.clamp(0.0, 1.0)
        } else {
            self.last_factor
        };
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, checked_index)
        else {
            return DoneAction::Nothing;
        };
        let Some(spectrum) = pv::to_polar_apx(&mut buffer) else {
            return DoneAction::Nothing;
        };

        let numbins = spectrum.bins.len();
        let memory_len = numbins + 2;
        let memory = &mut ctx.aux.f32_mut()[..memory_len];
        self.last_factor = factor;

        if self.initialized == 0 || self.frame_size as usize != frame_size {
            for (slot, bin) in memory[..numbins].iter_mut().zip(spectrum.bins.iter()) {
                *slot = bin.x;
            }
            memory[numbins] = *spectrum.dc;
            memory[numbins + 1] = *spectrum.nyq;
            self.frame_size = frame_size as u32;
            self.initialized = 1;
            return DoneAction::Nothing;
        }

        for (slot, bin) in memory[..numbins].iter_mut().zip(spectrum.bins.iter_mut()) {
            if let Some(value) = smooth(*slot, bin.x, factor) {
                *slot = value;
                bin.x = value;
            }
        }
        if let Some(value) = smooth(memory[numbins], *spectrum.dc, factor) {
            memory[numbins] = value;
            *spectrum.dc = value;
        }
        if let Some(value) = smooth(memory[numbins + 1], *spectrum.nyq, factor) {
            memory[numbins + 1] = value;
            *spectrum.nyq = value;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagSmooth`].
pub struct PvMagSmoothCtor;

impl UnitDef for PvMagSmoothCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_pv_abi(ctx, 2)?;
        Ok(unit_spec_aux(
            PvMagSmooth {
                frame_size: 0,
                initialized: 0,
                last_factor: 0.1,
            },
            MAX_MAGNITUDES * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

/// Interpolate one polar bin atomically, retaining `a` if either rounded component is non-finite.
fn morph_bin(a: pv::Bin, b: pv::Bin, morph: f32) -> pv::Bin {
    let one_minus = 1.0 - morph;
    let out = pv::Bin {
        x: one_minus * a.x + morph * b.x,
        y: one_minus * a.y + morph * b.y,
    };
    if out.x.is_finite() && out.y.is_finite() {
        out
    } else {
        a
    }
}

/// `PV_Morph(bufferA, bufferB, morph)`: interpolate A's polar bins toward B while copying B's DC
/// and Nyquist into A.
///
/// A is mutated and passed downstream. B is decoded read-only in its current coordinate form, so
/// parallel consumers observe its original samples and coordinate tag. Aliased, differently sized,
/// malformed, or non-finite frames are deterministic no-ops.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMorph {
    last_morph: f32,
}

impl Unit for PvMorph {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let token_a = ctx.ins.control(0);
        let token_b = ctx.ins.control(1);
        let morph_in = ctx.ins.control(2);
        let frame_a = pv::pv_frame(ctx);
        if token_a == f32::INFINITY {
            *ctx.outs.control(0) = -1.0;
            return DoneAction::Nothing;
        }
        let Some(frame_a) = frame_a else {
            return DoneAction::Nothing;
        };
        let Some(index_a) = token_index(token_a).filter(|&index| index == frame_a) else {
            return DoneAction::Nothing;
        };
        let Some(index_b) = token_index(token_b) else {
            return DoneAction::Nothing;
        };
        if index_a == index_b {
            return DoneAction::Nothing;
        }

        let Some(view_a) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, index_a) else {
            return DoneAction::Nothing;
        };
        let Some(view_b) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, index_b) else {
            return DoneAction::Nothing;
        };
        if view_a.num_frames() != view_b.num_frames()
            || !valid_spectrum(view_a)
            || !valid_spectrum(view_b)
        {
            return DoneAction::Nothing;
        }

        let morph = if morph_in.is_finite() {
            morph_in.clamp(0.0, 1.0)
        } else {
            self.last_morph
        };
        let Some((mut buffer_a, buffer_b)) =
            unit::buffer_pair_mut(ctx.buffers, &mut ctx.local_bufs, index_a, index_b)
        else {
            return DoneAction::Nothing;
        };

        let coord_b = buffer_b.coord();
        let data_b = buffer_b.data();
        let bins_b = pv::bins(data_b);
        let Some(spectrum_a) = pv::to_polar_apx(&mut buffer_a) else {
            return DoneAction::Nothing;
        };
        self.last_morph = morph;

        *spectrum_a.dc = data_b[0];
        *spectrum_a.nyq = data_b[1];
        for (bin_a, &raw_b) in spectrum_a.bins.iter_mut().zip(bins_b) {
            *bin_a = morph_bin(*bin_a, pv::bin_as_polar_apx(coord_b, raw_b), morph);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMorph`].
pub struct PvMorphCtor;

impl UnitDef for PvMorphCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_pv_abi(ctx, 3)?;
        Ok(unit_spec(PvMorph { last_morph: 0.0 }))
    }
}
