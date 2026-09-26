//! Single-buffer spectral (`PV_*`) operators - plyphon's ports of scsynth's `PV_MagAbove`,
//! `PV_MagBelow`, `PV_MagClip`, `PV_LocalMax`, `PV_PhaseShift90`, `PV_PhaseShift270`, `PV_BrickWall`,
//! `PV_Conj`, `PV_Diffuser`, `PV_BinShift`, `PV_MagSmear` and `PV_RectComb` (`PV_UGens.cpp`).
//!
//! Each edits the FFT-chain buffer in place each frame, using the shared [`pv`] plumbing:
//! [`pv::pv_frame`] for the frame preamble, [`pv::to_polar`]/[`pv::to_complex`] for the coordinate
//! form the op needs, or [`pv::spectrum`] for coordinate-independent bin edits. Compiled only with
//! the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec, unit_spec_pool};
use core::f64::consts::TAU;
use plyphon_dsp::math;

/// Which magnitude-threshold operation a [`PvMagThresh`] applies.
#[derive(Copy, Clone)]
pub enum MagKind {
    /// `PV_MagAbove` - pass bins whose magnitude is `>= thresh`, zero the rest.
    Above,
    /// `PV_MagBelow` - pass bins whose magnitude is `<= thresh`, zero the rest.
    Below,
    /// `PV_MagClip` - clip every bin's magnitude to at most `thresh`.
    Clip,
}

impl MagKind {
    fn to_tag(self) -> u32 {
        match self {
            MagKind::Above => 0,
            MagKind::Below => 1,
            MagKind::Clip => 2,
        }
    }

    /// Apply to a signed real term (`dc`/`nyq`), which is thresholded by its absolute value.
    fn real(tag: u32, val: f32, thresh: f32) -> f32 {
        match tag {
            1 => {
                if val.abs() > thresh {
                    0.0
                } else {
                    val
                }
            }
            2 => {
                if val.abs() > thresh {
                    if val < 0.0 { -thresh } else { thresh }
                } else {
                    val
                }
            }
            _ => {
                if val.abs() < thresh {
                    0.0
                } else {
                    val
                }
            }
        }
    }

    /// Apply to a (non-negative) bin magnitude.
    fn mag(tag: u32, mag: f32, thresh: f32) -> f32 {
        match tag {
            1 => {
                if mag > thresh {
                    0.0
                } else {
                    mag
                }
            }
            2 => mag.min(thresh),
            _ => {
                if mag < thresh {
                    0.0
                } else {
                    mag
                }
            }
        }
    }
}

/// `PV_MagAbove`/`PV_MagBelow`/`PV_MagClip(buffer, thresh)`: a magnitude gate/limiter, selected by
/// [`MagKind`].
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagThresh {
    kind: u32,
}

impl Unit for PvMagThresh {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let kind = self.kind;
        let thresh = ctx.ins.control(1);
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex())
        {
            *spectrum.dc = MagKind::real(kind, *spectrum.dc, thresh);
            *spectrum.nyq = MagKind::real(kind, *spectrum.nyq, thresh);
            for bin in spectrum.bins.iter_mut() {
                bin.x = MagKind::mag(kind, bin.x, thresh);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagThresh`], parameterised by [`MagKind`].
pub struct PvMagThreshCtor(pub MagKind);

impl UnitDef for PvMagThreshCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvMagThresh {
            kind: self.0.to_tag(),
        }))
    }
}

/// `PV_LocalMax(buffer, thresh)`: zero every bin that is not a local magnitude maximum (greater than
/// both neighbours and `thresh`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvLocalMax {
    _pad: u32,
}

impl Unit for PvLocalMax {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let thresh = ctx.ins.control(1);
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex())
        {
            let n = spectrum.bins.len();
            if n >= 2 {
                let dc = spectrum.dc.abs();
                let nyq = spectrum.nyq.abs();
                // DC is compared only with the bin above it.
                if dc < thresh || dc < spectrum.bins[0].x {
                    *spectrum.dc = 0.0;
                }
                // Bin 0 against DC and bin 1.
                let b0 = spectrum.bins[0].x;
                if b0 < thresh || b0 < dc || b0 < spectrum.bins[1].x {
                    spectrum.bins[0].x = 0.0;
                }
                // The middle bins against their two neighbours.
                for i in 1..n - 1 {
                    let mag = spectrum.bins[i].x;
                    if mag < thresh || mag < spectrum.bins[i - 1].x || mag < spectrum.bins[i + 1].x
                    {
                        spectrum.bins[i].x = 0.0;
                    }
                }
                // The last bin against the one below and the Nyquist.
                let last = spectrum.bins[n - 1].x;
                if last < thresh || last < nyq || last < spectrum.bins[n - 2].x {
                    spectrum.bins[n - 1].x = 0.0;
                }
                // Nyquist against the penultimate bin.
                if nyq < thresh || nyq < last {
                    *spectrum.nyq = 0.0;
                }
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvLocalMax`].
pub struct PvLocalMaxCtor;

impl UnitDef for PvLocalMaxCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvLocalMax { _pad: 0 }))
    }
}

/// `PV_PhaseShift90`/`PV_PhaseShift270(buffer)`: rotate every bin's phase by a quarter turn (one of
/// the two signs).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvPhaseQuarter {
    /// `0` = +90 degrees, `1` = -90 (270) degrees.
    negate: u32,
}

impl Unit for PvPhaseQuarter {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let negate = self.negate != 0;
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_complex(&mut buffer, ctx.fft.complex())
        {
            for bin in spectrum.bins.iter_mut() {
                let (re, im) = (bin.x, bin.y);
                if negate {
                    // 270 degrees: (re, im) -> (im, -re).
                    bin.x = im;
                    bin.y = -re;
                } else {
                    // 90 degrees: (re, im) -> (-im, re).
                    bin.x = -im;
                    bin.y = re;
                }
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvPhaseQuarter`]; `negate` picks 270 (`true`) vs 90 degrees.
pub struct PvPhaseQuarterCtor(pub bool);

impl UnitDef for PvPhaseQuarterCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvPhaseQuarter {
            negate: self.0 as u32,
        }))
    }
}

/// `PV_BrickWall(buffer, wipe)`: a brick-wall low/high pass - `wipe` in `(0, 1]` zeroes the lowest
/// `wipe` fraction of bins (high-pass); `[-1, 0)` zeroes the highest `|wipe|` fraction (low-pass).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvBrickWall {
    _pad: u32,
}

impl Unit for PvBrickWall {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let wipe_frac = ctx.ins.control(1);
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::spectrum(&mut buffer)
        {
            let numbins = spectrum.bins.len() as i32;
            let wipe = (wipe_frac * numbins as f32) as i32;
            if wipe > 0 {
                let wipe = wipe.min(numbins);
                *spectrum.dc = 0.0;
                for bin in &mut spectrum.bins[..wipe as usize] {
                    *bin = zero();
                }
                if wipe == numbins {
                    *spectrum.nyq = 0.0;
                }
            } else if wipe < 0 {
                let wipe = wipe.max(-numbins);
                if wipe == -numbins {
                    *spectrum.dc = 0.0;
                }
                for bin in &mut spectrum.bins[(numbins + wipe) as usize..] {
                    *bin = zero();
                }
                *spectrum.nyq = 0.0;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvBrickWall`].
pub struct PvBrickWallCtor;

impl UnitDef for PvBrickWallCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvBrickWall { _pad: 0 }))
    }
}

/// `PV_Conj(buffer)`: the complex conjugate of every bin (negate the imaginary part).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvConj {
    _pad: u32,
}

impl Unit for PvConj {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_complex(&mut buffer, ctx.fft.complex())
        {
            for bin in spectrum.bins.iter_mut() {
                bin.y = -bin.y;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvConj`].
pub struct PvConjCtor;

impl UnitDef for PvConjCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvConj { _pad: 0 }))
    }
}

/// A zeroed bin.
fn zero() -> pv::Bin {
    pv::Bin { x: 0.0, y: 0.0 }
}

/// `PV_Diffuser(buffer, trig)`: add a fixed, random phase offset per bin, re-randomising the
/// offsets on each rising `trig`. Smears transients over time (each bin's phase is decorrelated)
/// while leaving magnitudes untouched, so a steady tone is unchanged but an impulse is diffused.
///
/// `trig` doubles as the shifted-bin fraction: each frame offsets only the first
/// `clip(trig * numbins, 0, numbins)` bins (scsynth's `PV_Diffuser_next`), so `0` leaves every
/// phase untouched, `0.5` diffuses the lower half of the spectrum and `>= 1` the whole frame.
///
/// The offsets are drawn from the synth's random stream and held in `aux`, one `f32` per bin.
/// Like scsynth's `PV_Diffuser_next`, the unit allocates that table from the engine's pool on the
/// first frame, when the chain buffer's size gives the bin count.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvDiffuser {
    /// Number of bins the offset table covers, from the first frame's chain buffer.
    numbins: u32,
    /// Previous-block `trig` value, for rising-edge detection across blocks.
    prev_trig: f32,
    /// `1` once a rising `trig` has been seen since the last frame applied the offsets.
    retrigger: u32,
    /// `1` once the offset table is allocated (scsynth's non-null `m_shift`).
    allocated: u32,
}

impl Unit for PvDiffuser {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // Sample the trigger every block (frames are intermittent), latching a rising edge to
        // re-randomise on the next ready frame - scsynth's `m_prevtrig`/`m_triggered`.
        let trig = ctx.ins.control(1);
        if self.prev_trig <= 0.0 && trig > 0.0 {
            self.retrigger = 1;
        }
        self.prev_trig = trig;

        let Some(bufnum) = pv::pv_frame(ctx) else {
            return DoneAction::Nothing;
        };
        let Some(frames) =
            unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum).map(|b| b.num_frames())
        else {
            return DoneAction::Nothing;
        };
        let numbins = frames.saturating_sub(2) / 2;
        // The first frame fixes the bin count: allocate the table and randomise it. After that a
        // frame of a different size passes through untouched (scsynth's `numbins != m_numbins`
        // bail), and a latched trigger re-randomises.
        let choose = if self.allocated == 0 {
            if !ctx.aux.alloc(numbins * core::mem::size_of::<f32>()) {
                return DoneAction::Nothing;
            }
            self.allocated = 1;
            self.numbins = numbins as u32;
            true
        } else if numbins != self.numbins as usize {
            return DoneAction::Nothing;
        } else {
            core::mem::take(&mut self.retrigger) != 0
        };

        let shifts = &mut ctx.aux.f32_mut()[..numbins];
        if choose {
            for shift in shifts.iter_mut() {
                *shift = (ctx.rgen.next_unipolar() as f64 * TAU) as f32;
            }
        }
        // The trigger level also scales how many bins are offset - scsynth's
        // `n = sc_clip((int)(trig * numbins), 0, numbins)` - so a zero trig converts the frame to
        // polar but shifts nothing.
        let n = ((trig * numbins as f32) as i32).clamp(0, numbins as i32) as usize;
        if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex())
        {
            for (bin, &shift) in spectrum.bins.iter_mut().zip(shifts.iter()).take(n) {
                bin.y += shift;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvDiffuser`]: the unit allocates its offset table on the first frame.
pub struct PvDiffuserCtor;

impl UnitDef for PvDiffuserCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvDiffuser::zeroed())
        })
    }
}

/// scsynth's `MAKE_TEMP_BUF`: on the first frame, allocate a scratch as large as the chain buffer
/// (`buf->samples` floats) from the engine's pool and latch the frame's bin count; after that, a
/// frame with a different bin count passes through untouched. Returns the bin count to process, or
/// `None` to skip the frame - no such buffer, a changed size, or a failed allocation (after which the
/// engine silences the unit; scsynth leaves the scratch null and writes through it).
fn make_temp_buf(
    ctx: &mut ProcessCtx<'_>,
    bufnum: usize,
    numbins: &mut u32,
    allocated: &mut u32,
) -> Option<usize> {
    let data = unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum)?.data();
    let (frame_bins, bytes) = (
        data.len().saturating_sub(2) / 2,
        core::mem::size_of_val(data),
    );
    if *allocated == 0 {
        if !ctx.aux.alloc(bytes) {
            return None;
        }
        *allocated = 1;
        *numbins = frame_bins as u32;
    } else if frame_bins != *numbins as usize {
        return None;
    }
    Some(frame_bins)
}

/// `PV_BinShift(buffer, stretch, shift, interp)`: move every bin to a new position, stretching the
/// spectrum by `stretch` and offsetting it by `shift` bins - a frequency shift (linear, so harmonic
/// ratios change) when `stretch` is `1`, a spectral scaling when it is not.
///
/// Bin `i` lands at `shift + i * stretch`. `interp > 0` spreads it linearly across the two bins that
/// straddle that position; otherwise it goes wholly into the nearest one. Several sources can map
/// onto the same destination, so the destination spectrum is built from zero in the scratch and
/// copied back once every source has been read. The DC and Nyquist terms are not bins and pass
/// through unchanged. A destination outside the spectrum is dropped, as is a bin whose position is
/// not finite (undefined behaviour in the reference).
///
/// The op reads and writes Cartesian bins and leaves the frame in Cartesian form. Its scratch is
/// allocated on the first frame, as large as the chain buffer (scsynth's `MAKE_TEMP_BUF`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvBinShift {
    /// Bin count of the first frame.
    numbins: u32,
    /// `1` once the scratch is allocated (scsynth's non-null `m_tempbuf`).
    allocated: u32,
}

impl Unit for PvBinShift {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let Some(bufnum) = pv::pv_frame(ctx) else {
            return DoneAction::Nothing;
        };
        let Some(numbins) = make_temp_buf(ctx, bufnum, &mut self.numbins, &mut self.allocated)
        else {
            return DoneAction::Nothing;
        };
        let stretch = ctx.ins.control(1);
        let shift = ctx.ins.control(2);
        let interp = ctx.ins.control(3);
        let dest = &mut ctx.aux.f32_mut()[..2 * numbins];
        dest.fill(0.0);

        if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_complex(&mut buffer, ctx.fft.complex())
        {
            let mut fpos = shift;
            for bin in spectrum.bins.iter() {
                if fpos.is_finite() {
                    if interp > 0.0 {
                        let floor = math::floor(fpos);
                        let beta = fpos - floor;
                        let pos = floor as i64;
                        accumulate(dest, pos, 1.0 - beta, *bin);
                        accumulate(dest, pos + 1, beta, *bin);
                    } else {
                        // The reference's `(int32)(fpos + 0.5)`: the `0.5` is a `double`, so the
                        // sum rounds at double precision before truncating.
                        accumulate(dest, (f64::from(fpos) + 0.5) as i64, 1.0, *bin);
                    }
                }
                fpos += stretch;
            }
            for (bin, pair) in spectrum.bins.iter_mut().zip(dest.chunks_exact(2)) {
                bin.x = pair[0];
                bin.y = pair[1];
            }
        }
        DoneAction::Nothing
    }
}

/// Add `weight * bin` into destination bin `pos` of `dest` (two `f32` per bin), dropping it if `pos`
/// is not a bin.
fn accumulate(dest: &mut [f32], pos: i64, weight: f32, bin: pv::Bin) {
    let Some(pair) = usize::try_from(pos)
        .ok()
        .and_then(|pos| dest.get_mut(2 * pos..2 * pos + 2))
    else {
        return;
    };
    pair[0] += weight * bin.x;
    pair[1] += weight * bin.y;
}

/// Constructor for [`PvBinShift`]: the unit allocates its scratch on the first frame.
pub struct PvBinShiftCtor;

impl UnitDef for PvBinShiftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvBinShift::zeroed())
        })
    }
}

/// `PV_MagSmear(buffer, bins)`: replace each bin's magnitude with the average of the `2 * bins + 1`
/// magnitudes centred on it, blurring the spectrum along the frequency axis. Phases, and the DC and
/// Nyquist terms, are untouched.
///
/// The window is truncated at the spectrum's edges but the divisor is not, so the outermost bins are
/// attenuated - the reference's behaviour. `bins` is truncated to an integer and clamped to
/// `[0, numbins - 1]`, and the cost is `numbins * bins`.
///
/// The op reads and writes polar bins and leaves the frame in polar form. The smeared magnitudes go
/// to a scratch first, so every bin averages the frame's original magnitudes; the scratch is
/// allocated on the first frame, as large as the chain buffer (scsynth's `MAKE_TEMP_BUF`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagSmear {
    /// Bin count of the first frame.
    numbins: u32,
    /// `1` once the scratch is allocated (scsynth's non-null `m_tempbuf`).
    allocated: u32,
}

impl Unit for PvMagSmear {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let Some(bufnum) = pv::pv_frame(ctx) else {
            return DoneAction::Nothing;
        };
        let Some(numbins) = make_temp_buf(ctx, bufnum, &mut self.numbins, &mut self.allocated)
        else {
            return DoneAction::Nothing;
        };
        let width_in = ctx.ins.control(1);
        let smeared = &mut ctx.aux.f32_mut()[..numbins];

        if numbins > 0
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex())
        {
            let last = numbins as i32 - 1;
            let width = (width_in as i32).clamp(0, last);
            let scale = 1.0 / (2 * width + 1) as f32;
            for (j, out) in smeared.iter_mut().enumerate() {
                let lo = (j as i32 - width).max(0) as usize;
                let hi = (j as i32 + width).min(last) as usize;
                let sum: f32 = spectrum.bins[lo..=hi].iter().map(|bin| bin.x).sum();
                *out = sum * scale;
            }
            for (bin, &mag) in spectrum.bins.iter_mut().zip(smeared.iter()) {
                bin.x = mag;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagSmear`]: the unit allocates its scratch on the first frame.
pub struct PvMagSmearCtor;

impl UnitDef for PvMagSmearCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvMagSmear::zeroed())
        })
    }
}

/// `PV_RectComb(buffer, numTeeth, phase, width)`: zero every slot outside the teeth of a rectangular
/// comb laid across the spectrum, keeping `width` of each `1 / numTeeth` of the frame.
///
/// A running phase walks the spectrum, advancing by `numTeeth / (numbins + 1)` per slot from the DC
/// term through every bin to the Nyquist term; a slot survives when the phase is at most `width`.
/// The phase is wrapped by a single addition or subtraction per step, as the reference does, so a
/// `numTeeth` beyond the frame's slot count sweeps the comb off the spectrum instead of aliasing.
///
/// The op zeroes whole slots, which needs neither magnitudes nor real parts, so it edits the packed
/// frame without converting it and leaves its coordinate form as it found it.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvRectComb {
    /// The unit is stateless; the state block must still be a non-zero-sized `Pod`.
    _pad: u32,
}

impl Unit for PvRectComb {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_teeth = ctx.ins.control(1);
        let start_phase = ctx.ins.control(2);
        let width = ctx.ins.control(3);
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::spectrum(&mut buffer)
        {
            let step = num_teeth / (spectrum.bins.len() + 1) as f32;
            let mut phase = start_phase;
            if phase > width {
                *spectrum.dc = 0.0;
            }
            phase = wrap_phase(phase + step);
            for bin in spectrum.bins.iter_mut() {
                if phase > width {
                    *bin = zero();
                }
                phase = wrap_phase(phase + step);
            }
            if phase > width {
                *spectrum.nyq = 0.0;
            }
        }
        DoneAction::Nothing
    }
}

/// Bring a comb phase back toward `[0, 1)` with one addition or subtraction, as the reference does.
fn wrap_phase(phase: f32) -> f32 {
    if phase >= 1.0 {
        phase - 1.0
    } else if phase < 0.0 {
        phase + 1.0
    } else {
        phase
    }
}

/// Constructor for [`PvRectComb`].
pub struct PvRectCombCtor;

impl UnitDef for PvRectCombCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvRectComb { _pad: 0 }))
    }
}
