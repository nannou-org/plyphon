//! `SpecCentroid`, `SpecFlatness` and `SpecPcile` - plyphon's ports of scsynth's spectral statistics
//! (`ML_SpecStats.cpp`). Compiled only with the `fft` feature.
//!
//! Each reads the FFT chain (input 0) and, on a frame, reduces the chain buffer's spectrum to one
//! value that it outputs and holds between frames. Unlike a `PV_*` unit it does not pass the chain
//! on. The frame preamble (`analysis_frame`) is shared with `Onsets`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec, unit_spec_pool};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// The analysis units' frame preamble - scsynth's `FFTAnalyser_GET_BUF` (`ML_SpecStats.cpp:66`) and
/// `Onsets_GET_BUF` (`Onsets.cpp:79`). Between frames (input 0 `< 0`) it writes the held `outval`
/// and returns `None`. On a frame it writes the chain value itself, which the unit overwrites with
/// its result once it has one, and returns the chain buffer's number.
pub(crate) fn analysis_frame(ctx: &mut ProcessCtx<'_>, outval: f32) -> Option<usize> {
    let fbufnum = ctx.ins.control(0);
    if fbufnum < 0.0 {
        *ctx.outs.control(0) = outval;
        return None;
    }
    *ctx.outs.control(0) = fbufnum;
    Some(fbufnum as u32 as usize)
}

/// The machine-listening units run at control rate, the only rate their language classes offer.
/// Each writes a single value per block, so an audio-rate instance would leave the rest of its
/// output block undefined; plyphon rejects one.
pub(crate) fn check_rate(ctx: &BuildContext<'_>, min_inputs: usize) -> Result<(), BuildError> {
    if ctx.rate == Rate::Audio {
        return Err(BuildError::UnsupportedRate(ctx.rate));
    }
    if ctx.input_rates.len() < min_inputs {
        return Err(BuildError::WrongInputCount);
    }
    Ok(())
}

/// `SpecCentroid.kr(buffer)`: the spectral centroid of each frame, in Hz - the magnitude-weighted
/// mean frequency of the bins and the Nyquist term (the DC term carries no weight). A silent frame
/// yields `0`.
///
/// The chain buffer is converted to polar form in place with `ToPolarApx`, and the bin-to-Hz factor
/// (`sample rate / buffer size`) is latched from the first frame.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SpecCentroid {
    /// The last centroid, held between frames (scsynth's `outval`).
    outval: f32,
    /// Hz per bin, latched from the first frame; `0` until then (scsynth's `m_bintofreq`).
    bintofreq: f32,
}

impl Unit for SpecCentroid {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `SpecCentroid_Ctor` (`ML_SpecStats.cpp:286`) zeroes its output and runs no calc.
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `SpecCentroid_next` (`ML_SpecStats.cpp:293`).
        let Some(bufnum) = analysis_frame(ctx, self.outval) else {
            return DoneAction::Nothing;
        };
        let sample_rate = ctx.audio.sample_rate;
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum) else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        let samples = buffer.data().len();
        // `ToPolarApx(buf)` (`ML_SpecStats.cpp:296`).
        let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex()) else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        // `GET_BINTOFREQ`: `FULLRATE / buf->samples`, computed in double and stored as a float.
        if self.bintofreq == 0.0 {
            self.bintofreq = (sample_rate / samples as f64) as f32;
        }
        let numbins = spectrum.bins.len();
        // Each magnitude is weighted by its bin number as a float product, then summed in double.
        let nyq = spectrum.nyq.abs();
        let mut num = f64::from(nyq * (numbins + 1) as f32);
        let mut denom = f64::from(nyq);
        for (i, bin) in spectrum.bins.iter().enumerate() {
            let mag = bin.x.abs();
            num += f64::from(mag * (i + 1) as f32);
            denom += f64::from(mag);
        }
        self.outval = if denom == 0.0 {
            0.0
        } else {
            (f64::from(self.bintofreq) * num / denom) as f32
        };
        *ctx.outs.control(0) = self.outval;
        DoneAction::Nothing
    }
}

/// Constructor for [`SpecCentroid`].
pub struct SpecCentroidCtor;

impl UnitDef for SpecCentroidCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        check_rate(ctx, 1)?;
        Ok(unit_spec(SpecCentroid::zeroed()))
    }
}

/// `SpecFlatness.kr(buffer)`: the spectral flatness of each frame - the geometric mean of the
/// magnitudes over their arithmetic mean, `1` for white noise and near `0` for a pure tone. The DC
/// and Nyquist terms count as magnitudes; a zero bin is left out of both means (its logarithm would
/// be `-inf`) but still counts towards `n`. An all-zero frame yields `0.8`, scsynth's empirical
/// value for very quiet white noise.
///
/// The chain buffer is converted to Cartesian form in place with `ToComplexApx`, and `1 / n` is
/// latched from the first frame.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SpecFlatness {
    /// `1 / (numbins + 2)`, latched from the first frame; `0` until then (scsynth's `m_oneovern`).
    oneovern: f64,
    /// The last flatness, held between frames (scsynth's `outval`).
    outval: f32,
    _pad: u32,
}

impl Unit for SpecFlatness {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `SpecFlatness_Ctor` (`ML_SpecStats.cpp:154`) zeroes its output and runs no calc.
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `SpecFlatness_next` (`ML_SpecStats.cpp:160`).
        let Some(bufnum) = analysis_frame(ctx, self.outval) else {
            return DoneAction::Nothing;
        };
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum) else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        // `ToComplexApx(buf)` (`ML_SpecStats.cpp:163`).
        let Some(spectrum) = pv::to_complex(&mut buffer, ctx.fft.complex()) else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        if self.oneovern == 0.0 {
            self.oneovern = 1.0 / (spectrum.bins.len() + 2) as f64;
        }
        // The logarithms are single precision (`std::log` of a float); the sums are double.
        let dc = spectrum.dc.abs();
        let nyq = spectrum.nyq.abs();
        let mut geommean = f64::from(math::ln(dc) + math::ln(nyq));
        let mut mean = f64::from(dc + nyq);
        for bin in spectrum.bins.iter() {
            let amp = math::sqrt(bin.x * bin.x + bin.y * bin.y);
            if amp != 0.0 {
                geommean += f64::from(math::ln(amp));
                mean += f64::from(amp);
            }
        }
        geommean = math::exp(geommean * self.oneovern);
        mean *= self.oneovern;
        self.outval = if mean == 0.0 {
            0.8
        } else {
            (geommean / mean) as f32
        };
        *ctx.outs.control(0) = self.outval;
        DoneAction::Nothing
    }
}

/// Constructor for [`SpecFlatness`].
pub struct SpecFlatnessCtor;

impl UnitDef for SpecFlatnessCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        check_rate(ctx, 1)?;
        Ok(unit_spec(SpecFlatness::zeroed()))
    }
}

/// `SpecPcile.kr(buffer, fraction = 0.5, interpolate = 0, binout = 0)`: the frequency below which
/// `fraction` of each frame's cumulative magnitude lies - the spectral rolloff (`0.5` is the
/// median).
///
/// The magnitudes of the DC term and the bins are summed cumulatively, the Nyquist term added to the
/// total, and the first bin whose running sum reaches `fraction` of that total is reported: as a
/// frequency by default, or as its bin number when `binout > 0`. `interpolate > 0` refines the
/// position linearly between that bin and the one before. When no bin reaches the target (a
/// negative `fraction`, say) the result is `0`. `interpolate` and `binout` are read when the synth
/// starts; `fraction` every frame.
///
/// The chain buffer is converted to Cartesian form in place with `ToComplexApx`. The running sums
/// live in a table of one `f32` per bin that the unit allocates from the engine's pool on the first
/// frame, which also fixes the bin count: a later frame with a different bin count is ignored,
/// outputting the chain value itself for that block.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SpecPcile {
    /// The last result, held between frames (scsynth's `outval`).
    outval: f32,
    /// Hz per unit of `bin + 1`: `sample rate * 0.5 / (numbins + 2)`, from the first frame (scsynth's
    /// `m_halfnyq_over_numbinsp2`).
    halfnyq_over_numbinsp2: f32,
    /// The first frame's bin count (scsynth's `m_numbins`).
    numbins: u32,
    /// `1` once the running-sum table is allocated (scsynth's non-null `m_tempbuf`).
    allocated: u32,
    /// `1` when `interpolate > 0` at synth start.
    interpolate: u32,
    /// `1` when `binout > 0` at synth start.
    binout: u32,
}

impl SpecPcile {
    const FRACTION: usize = 1;
    const INTERPOLATE: usize = 2;
    const BINOUT: usize = 3;
}

impl Unit for SpecPcile {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `SpecPcile_Ctor` (`ML_SpecStats.cpp:196`): latch the two flags, zero the output, no calc.
        self.interpolate = u32::from(ctx.ins.control(Self::INTERPOLATE) > 0.0);
        self.binout = u32::from(ctx.ins.control(Self::BINOUT) > 0.0);
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `SpecPcile_next` (`ML_SpecStats.cpp:206`).
        let Some(bufnum) = analysis_frame(ctx, self.outval) else {
            return DoneAction::Nothing;
        };
        let Some(samples) =
            unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum).map(|b| b.data().len())
        else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        let numbins = samples.saturating_sub(2) / 2;
        if self.allocated == 0 {
            // A failed allocation silences the unit for good, as `ClearUnitOnMemFailed` does.
            if !ctx.aux.alloc(numbins * core::mem::size_of::<f32>()) {
                return DoneAction::Nothing;
            }
            self.allocated = 1;
            self.numbins = numbins as u32;
            self.halfnyq_over_numbinsp2 = ctx.audio.sample_rate as f32 * 0.5 / (numbins + 2) as f32;
        } else if numbins != self.numbins as usize {
            // The chain value the preamble wrote stays on the output.
            return DoneAction::Nothing;
        }

        let fraction = ctx.ins.control(Self::FRACTION);
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum) else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        // `ToComplexApx(buf)` (`ML_SpecStats.cpp:227`).
        let Some(spectrum) = pv::to_complex(&mut buffer, ctx.fft.complex()) else {
            *ctx.outs.control(0) = self.outval;
            return DoneAction::Nothing;
        };
        let q = &mut ctx.aux.f32_mut()[..numbins];
        let mut cumul = spectrum.dc.abs();
        for (q, bin) in q.iter_mut().zip(spectrum.bins.iter()) {
            cumul += math::sqrt(bin.x * bin.x + bin.y * bin.y);
            *q = cumul;
        }
        cumul += spectrum.nyq.abs();
        let target = cumul * fraction;

        let interpolate = self.interpolate != 0;
        let mut bestposition = 0.0f32;
        for i in 0..numbins {
            // The reference's `!(q[i] < target)`, which a NaN target also passes.
            if q[i].partial_cmp(&target) != Some(core::cmp::Ordering::Less) {
                // The bin number refines forwards from `i`, the frequency backwards from `i + 1`.
                bestposition = if self.binout != 0 {
                    if interpolate && i != 0 {
                        i as f32 + (q[i] - target) / (q[i] - q[i - 1])
                    } else {
                        i as f32
                    }
                } else {
                    let binpos = if interpolate && i != 0 {
                        i as f32 + 1.0 - (q[i] - target) / (q[i] - q[i - 1])
                    } else {
                        i as f32 + 1.0
                    };
                    binpos * self.halfnyq_over_numbinsp2
                };
                break;
            }
        }
        self.outval = bestposition;
        *ctx.outs.control(0) = self.outval;
        DoneAction::Nothing
    }
}

/// Constructor for [`SpecPcile`]: the unit allocates its running-sum table on the first frame.
pub struct SpecPcileCtor;

impl UnitDef for SpecPcileCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        check_rate(ctx, SpecPcile::BINOUT + 1)?;
        Ok(unit_spec_pool(SpecPcile::zeroed()))
    }
}
