//! Single-buffer spectral (`PV_*`) operators - plyphon's ports of scsynth's `PV_MagAbove`,
//! `PV_MagBelow`, `PV_MagClip`, `PV_LocalMax`, `PV_PhaseShift90`, `PV_PhaseShift270`, `PV_BrickWall`
//! and `PV_Conj` (`PV_UGens.cpp`).
//!
//! Each edits the FFT-chain buffer in place each frame, using the shared [`pv`] plumbing:
//! [`pv::pv_frame`] for the frame preamble, [`pv::to_polar`]/[`pv::to_complex`] for the coordinate
//! form the op needs, or [`pv::spectrum`] for coordinate-independent bin edits. Compiled only with
//! the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::fft::{DEFAULT_MAX_FFT, resolve_fftsize};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec, unit_spec_aux};
use core::f32::consts::TAU;

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
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let kind = self.kind;
        let thresh = ctx.ins.control(1);
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer)
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
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let thresh = ctx.ins.control(1);
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer)
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
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let negate = self.negate != 0;
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_complex(&mut buffer)
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
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_complex(&mut buffer)
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

/// The most bins a deferred-size [`PvDiffuser`] reserves phase state for: one per bin of the largest
/// supported FFT (`[dc, nyq, bins...]` packs `(N - 2) / 2` bins).
const MAX_DIFFUSER_BINS: usize = (DEFAULT_MAX_FFT - 2) / 2;

/// `PV_Diffuser(buffer, trig)`: add a fixed, random phase offset to every bin, re-randomising the
/// offsets on each rising `trig`. Smears transients over time (each bin's phase is decorrelated)
/// while leaving magnitudes untouched, so a steady tone is unchanged but an impulse is diffused.
///
/// The offsets are drawn from the synth's shared random stream, held in `aux` (one `f32` per bin),
/// and reserved for the largest supported FFT since the chain buffer - hence the bin count - is not
/// known until the first frame. scsynth's `PV_Diffuser`, which lazily allocates the same table.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvDiffuser {
    /// Number of bins the offset table currently covers; `0` until the first frame resolves it.
    numbins: u32,
    /// Previous-block `trig` value, for rising-edge detection across blocks.
    prev_trig: f32,
    /// `1` once a rising `trig` has been seen since the last frame applied the offsets.
    retrigger: u32,
}

impl Unit for PvDiffuser {
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
        // The bin count is fixed by the chain buffer's size; resolve it once, capped at the aux
        // reservation. A newly-installed (or resized) buffer re-randomises from scratch.
        if self.numbins == 0 {
            match resolve_fftsize(ctx.buffers, &ctx.local_bufs, bufnum) {
                Some(n) => {
                    self.numbins = ((n.saturating_sub(2)) / 2).min(MAX_DIFFUSER_BINS) as u32;
                    self.retrigger = 1;
                }
                None => return DoneAction::Nothing,
            }
        }
        let numbins = self.numbins as usize;

        let shifts = &mut ctx.aux.f32_mut()[..numbins];
        if self.retrigger != 0 {
            for shift in shifts.iter_mut() {
                *shift = ctx.rgen.next_unipolar() * TAU;
            }
            self.retrigger = 0;
        }
        if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer)
        {
            for (bin, &shift) in spectrum.bins.iter_mut().zip(shifts.iter()) {
                bin.y += shift;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvDiffuser`]: reserves phase state for the largest supported FFT, since the
/// chain buffer's size is not known until the first frame.
pub struct PvDiffuserCtor;

impl UnitDef for PvDiffuserCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_aux(
            PvDiffuser {
                numbins: 0,
                prev_trig: 0.0,
                retrigger: 0,
            },
            MAX_DIFFUSER_BINS * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}
