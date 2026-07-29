//! sc3-plugins phase-vocoder operators.
//!
//! `PV_MagSmooth` is based on work by Dan Stowell, Copyright 2006-2010, licensed under
//! GPL-2.0-or-later. `PV_Morph` is based on work by Bhob Rainey and SuperCollider contributors,
//! licensed under GPL-2.0-or-later. Both use Plyphon's fixed auxiliary storage while preserving
//! sc3-plugins' ready-frame arithmetic and buffer mutation.

use core::f32::consts::{PI, TAU};

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    self, BuiltUnit, DoneAction, LocalBufs, ProcessCtx, Unit, pv, unit_spec, unit_spec_aux,
};
use plyphon_dsp::buffer::{BufView, BufferTable};
use plyphon_dsp::fft::is_supported_size;
use plyphon_dsp::rate::Rate;

/// The largest supported FFT frame. A smoothed spectrum retains one value per ordinary bin plus
/// DC and Nyquist, so its auxiliary memory is fixed at compile time.
const MAX_FFT_SIZE: usize = 16_384;

/// Number of retained magnitude values needed by the largest supported FFT.
const MAX_MAGNITUDES: usize = MAX_FFT_SIZE / 2 + 1;

/// Number of ordinary complex/polar bins in the largest supported packed spectrum.
const MAX_ORDINARY_BINS: usize = MAX_FFT_SIZE / 2 - 1;

/// Whether a buffer has a supported mono packed-spectrum shape.
///
/// scsynth assumes this shape after resolving an FFT chain. Plyphon checks it because its buffer
/// table can also contain arbitrary host-provided buffers.
fn valid_spectrum_shape(buffer: BufView<'_>) -> bool {
    valid_spectrum_parts(buffer.num_frames(), buffer.num_channels(), buffer.data())
}

/// Validate packed-spectrum dimensions without depending on a buffer view's mutability.
fn valid_spectrum_parts(frames: usize, channels: usize, data: &[f32]) -> bool {
    if channels != 1 || data.len() != frames || !is_supported_size(frames) || data.len() < 2 {
        return false;
    }

    pv::bins(data).len() == (frames - 2) / 2
}

/// Resolve a finite PV chain token, including the source fallback for an unknown local buffer.
fn pv_buffer_index(buffers: &BufferTable, local_bufs: &LocalBufs<'_>, token: f32) -> Option<usize> {
    if token < 0.0 || !token.is_finite() {
        return None;
    }
    let index = token as usize;
    let local_end = buffers.capacity().saturating_add(local_bufs.len());
    if index >= local_end && buffers.capacity() != 0 {
        Some(0)
    } else {
        Some(index)
    }
}

/// Apply sc3-plugins' raw smoothing arithmetic.
fn smooth(previous: f32, current: f32, factor: f32) -> f32 {
    previous * factor + current * (1.0 - factor)
}

/// Validate the shared ABI of a control-rate PV operator.
///
/// Chain tokens remain scalar/control values. The final modulation inlet also accepts an audio
/// wire, which [`Inputs::control`](crate::unit::Inputs::control) samples with scsynth's `IN0`
/// convention.
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
    let chain_inputs = &ctx.input_rates[..num_inputs - 1];
    let modulation = ctx.input_rates[num_inputs - 1];
    if ctx.rate != Rate::Control
        || chain_inputs
            .iter()
            .any(|rate| !matches!(rate, Rate::Scalar | Rate::Control | Rate::Audio))
        || !matches!(modulation, Rate::Scalar | Rate::Control | Rate::Audio)
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
/// The first FFT size has three audible warm-up stages. The first establishes the size, the second
/// stores magnitudes and phases, and the third establishes phase differences. Frozen frames then
/// reuse the retained magnitudes, DC, and Nyquist while their phases continue coherently. A later
/// size change is outside the live-chain host contract and is rejected without mutating history.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PvFreeze {
    frame_size: u32,
    stage: u32,
    stored_dc: f32,
    stored_nyquist: f32,
}

impl Unit for PvFreeze {
    /// Publishes the raw input chain token before the first calculation.
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        *ctx.outs.control(0) = ctx.ins.control(0);
    }

    /// Updates one ready spectrum while retaining the staged freeze history.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freeze_input = ctx.ins.control(1);
        let frame_index = pv::pv_frame(ctx);
        if frame_index.is_none() {
            return DoneAction::Nothing;
        }
        let Some(checked_index) = pv_buffer_index(ctx.buffers, &ctx.local_bufs, ctx.ins.control(0))
        else {
            return DoneAction::Nothing;
        };
        let Some(view) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, checked_index) else {
            return DoneAction::Nothing;
        };
        if !valid_spectrum_shape(view) {
            return DoneAction::Nothing;
        }

        let frame_size = view.num_frames();
        if self.frame_size != 0 && self.frame_size as usize != frame_size {
            return DoneAction::Nothing;
        }
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, checked_index)
        else {
            return DoneAction::Nothing;
        };
        let Some(spectrum) = pv::to_polar_apx(&mut buffer) else {
            return DoneAction::Nothing;
        };
        let freeze = freeze_input;

        let bins = spectrum.bins.len();
        let memory = ctx.aux.f32_mut();
        let (magnitudes, rest) = memory.split_at_mut(MAX_ORDINARY_BINS);
        let (previous_phases, phase_differences) = rest.split_at_mut(MAX_ORDINARY_BINS);
        let magnitudes = &mut magnitudes[..bins];
        let previous_phases = &mut previous_phases[..bins];
        let phase_differences = &mut phase_differences[..bins];

        if self.stage == 0 {
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
pub(super) struct PvFreezeCtor;

impl UnitDef for PvFreezeCtor {
    /// Validates the PV ABI and allocates fixed per-bin history.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_pv_abi(ctx, 2)?;
        Ok(unit_spec_aux(
            PvFreeze {
                frame_size: 0,
                stage: 0,
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
/// The first valid frame initializes fixed auxiliary memory before running the same smoothing
/// expression as every later frame. `factor` is applied directly, including values outside
/// `[0, 1]`, matching the source.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PvMagSmooth {
    frame_size: u32,
    initialized: u32,
}

impl Unit for PvMagSmooth {
    /// Publishes the raw input chain token before the first calculation.
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        *ctx.outs.control(0) = ctx.ins.control(0);
    }

    /// Smooths one ready spectrum against the retained magnitude history.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let frame_index = pv::pv_frame(ctx);
        if frame_index.is_none() {
            return DoneAction::Nothing;
        }
        let Some(checked_index) = pv_buffer_index(ctx.buffers, &ctx.local_bufs, ctx.ins.control(0))
        else {
            return DoneAction::Nothing;
        };
        let Some(view) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, checked_index) else {
            return DoneAction::Nothing;
        };
        if !valid_spectrum_shape(view) {
            return DoneAction::Nothing;
        }

        let frame_size = view.num_frames();
        if self.initialized != 0 && frame_size > self.frame_size as usize {
            return DoneAction::Nothing;
        }
        let factor = ctx.ins.control(1);
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

        if self.initialized == 0 {
            for (slot, bin) in memory[..numbins].iter_mut().zip(spectrum.bins.iter()) {
                *slot = bin.x;
            }
            memory[numbins] = *spectrum.dc;
            memory[numbins + 1] = *spectrum.nyq;
            self.frame_size = frame_size as u32;
            self.initialized = 1;
        }

        for (slot, bin) in memory[..numbins].iter_mut().zip(spectrum.bins.iter_mut()) {
            *slot = smooth(*slot, bin.x, factor);
            bin.x = *slot;
        }
        memory[numbins] = smooth(memory[numbins], *spectrum.dc, factor);
        *spectrum.dc = memory[numbins];
        memory[numbins + 1] = smooth(memory[numbins + 1], *spectrum.nyq, factor);
        *spectrum.nyq = memory[numbins + 1];
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagSmooth`].
pub(super) struct PvMagSmoothCtor;

impl UnitDef for PvMagSmoothCtor {
    /// Validates the PV ABI and allocates fixed per-bin magnitude history.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_pv_abi(ctx, 2)?;
        Ok(unit_spec_aux(
            PvMagSmooth {
                frame_size: 0,
                initialized: 0,
            },
            MAX_MAGNITUDES * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

/// Interpolate one polar bin with sc3-plugins' raw arithmetic.
fn morph_bin(a: pv::Bin, b: pv::Bin, morph: f32) -> pv::Bin {
    let one_minus = 1.0 - morph;
    pv::Bin {
        x: one_minus * a.x + morph * b.x,
        y: one_minus * a.y + morph * b.y,
    }
}

/// `PV_Morph(bufferA, bufferB, morph)`: interpolate A's polar bins toward B while copying B's DC
/// and Nyquist into A.
///
/// A is mutated and passed downstream. Both buffers are converted to polar form in place, matching
/// `ToPolarApx` in the source. Differently sized or malformed buffers are left unchanged.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PvMorph;

impl Unit for PvMorph {
    /// Publishes the raw A-chain token before the first calculation.
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        *ctx.outs.control(0) = ctx.ins.control(0);
    }

    /// Morphs one ready A spectrum toward B after converting both buffers to polar form.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let token_a = ctx.ins.control(0);
        let token_b = ctx.ins.control(1);
        if token_a < 0.0 || token_b < 0.0 || token_a.is_nan() || token_b.is_nan() {
            *ctx.outs.control(0) = -1.0;
            return DoneAction::Nothing;
        }
        *ctx.outs.control(0) = token_a;
        let Some(index_a) = pv_buffer_index(ctx.buffers, &ctx.local_bufs, token_a) else {
            return DoneAction::Nothing;
        };
        let Some(index_b) = pv_buffer_index(ctx.buffers, &ctx.local_bufs, token_b) else {
            return DoneAction::Nothing;
        };
        if index_a == index_b {
            let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, index_a)
            else {
                return DoneAction::Nothing;
            };
            if valid_spectrum_parts(buffer.num_frames(), buffer.num_channels(), buffer.data()) {
                let morph = ctx.ins.control(2);
                if let Some(spectrum) = pv::to_polar_apx(&mut buffer) {
                    for bin in spectrum.bins {
                        *bin = morph_bin(*bin, *bin, morph);
                    }
                }
            }
            return DoneAction::Nothing;
        }

        let Some(view_a) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, index_a) else {
            return DoneAction::Nothing;
        };
        let Some(view_b) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, index_b) else {
            return DoneAction::Nothing;
        };
        if view_a.num_frames() != view_b.num_frames()
            || !valid_spectrum_shape(view_a)
            || !valid_spectrum_shape(view_b)
        {
            return DoneAction::Nothing;
        }

        let morph = ctx.ins.control(2);
        let Some(mut buffer_a) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, index_a)
        else {
            return DoneAction::Nothing;
        };
        if pv::to_polar_apx(&mut buffer_a).is_none() {
            return DoneAction::Nothing;
        }
        drop(buffer_a);

        let Some(mut buffer_b) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, index_b)
        else {
            return DoneAction::Nothing;
        };
        if pv::to_polar_apx(&mut buffer_b).is_none() {
            return DoneAction::Nothing;
        }
        drop(buffer_b);

        let Some((mut buffer_a, buffer_b)) =
            unit::buffer_pair_mut(ctx.buffers, &mut ctx.local_bufs, index_a, index_b)
        else {
            return DoneAction::Nothing;
        };

        let Some(spectrum_a) = pv::to_polar_apx(&mut buffer_a) else {
            return DoneAction::Nothing;
        };
        let data_b = buffer_b.data();
        let bins_b = pv::bins(data_b);

        *spectrum_a.dc = data_b[0];
        *spectrum_a.nyq = data_b[1];
        for (bin_a, &bin_b) in spectrum_a.bins.iter_mut().zip(bins_b) {
            *bin_a = morph_bin(*bin_a, bin_b, morph);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMorph`].
pub(super) struct PvMorphCtor;

impl UnitDef for PvMorphCtor {
    /// Validates the two-buffer PV ABI and constructs a morph voice.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_pv_abi(ctx, 3)?;
        Ok(unit_spec(PvMorph))
    }
}
