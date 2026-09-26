//! More spectral (`PV_*`) operators - plyphon's ports of scsynth's `PV_MagFreeze`, `PV_MagShift`,
//! `PV_PhaseShift`, `PV_MagDiv`, `PV_BinWipe` and `PV_RectComb2` (`PV_UGens.cpp`) and
//! `PV_ConformalMap` (`PV_ThirdParty.cpp`).
//!
//! Each edits the FFT-chain buffer in place each frame. It converts the buffer with
//! [`pv::to_polar`] or [`pv::to_complex`] (scsynth's `ToPolarApx`/`ToComplexApx`) exactly where
//! scsynth's unit does, and leaves it unconverted where scsynth reads the raw packed frame. The
//! two-buffer units go through the shared `PV_GET_BUF2` preamble, [`pv::pv_pair`]. Compiled only
//! with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::pv_ops::{make_temp_buf, wrap_phase};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec, unit_spec_pool};
use plyphon_dsp::buffer::BufViewMut;
use plyphon_dsp::complex::ComplexTables;

/// `PV_MagFreeze(buffer, freeze = 0)`: while `freeze > 0`, hold every bin's magnitude (and the DC
/// and Nyquist terms) at the values of the last frame that arrived unfrozen; phases keep moving.
///
/// The held magnitudes live in `aux`, one `f32` per bin, allocated on the first frame from the
/// engine's pool (scsynth's `PV_MagFreeze_next` `RTAlloc`); that frame always stores, never holds,
/// so the table is filled before it is read. A later frame with a different bin count passes
/// through untouched. The op leaves the frame in polar form.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagFreeze {
    /// Bin count of the first frame (scsynth's `m_numbins`).
    numbins: u32,
    /// `1` once the magnitude table is allocated (scsynth's non-null `m_mags`).
    allocated: u32,
    /// The held DC term (scsynth's `m_dc`).
    dc: f32,
    /// The held Nyquist term (scsynth's `m_nyq`).
    nyq: f32,
}

impl Unit for PvMagFreeze {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let Some(bufnum) = pv::pv_frame(ctx) else {
            return DoneAction::Nothing;
        };
        let Some(samples) =
            unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum).map(|b| b.data().len())
        else {
            return DoneAction::Nothing;
        };
        let numbins = samples.saturating_sub(2) / 2;
        let mut freeze = ctx.ins.control(1);
        if self.allocated == 0 {
            if !ctx.aux.alloc(numbins * core::mem::size_of::<f32>()) {
                return DoneAction::Nothing;
            }
            self.allocated = 1;
            self.numbins = numbins as u32;
            // The first frame stores its magnitudes before any frame may read them.
            freeze = 0.0;
        } else if numbins != self.numbins as usize {
            return DoneAction::Nothing;
        }

        let mags = &mut ctx.aux.f32_mut()[..numbins];
        // `ToPolarApx(buf)` after the size check (PV_UGens.cpp:1120).
        if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex())
        {
            if freeze > 0.0 {
                for (bin, &mag) in spectrum.bins.iter_mut().zip(mags.iter()) {
                    bin.x = mag;
                }
                *spectrum.dc = self.dc;
                *spectrum.nyq = self.nyq;
            } else {
                for (mag, bin) in mags.iter_mut().zip(spectrum.bins.iter()) {
                    *mag = bin.x;
                }
                self.dc = *spectrum.dc;
                self.nyq = *spectrum.nyq;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagFreeze`]: the unit allocates its magnitude table on the first frame.
pub struct PvMagFreezeCtor;

impl UnitDef for PvMagFreezeCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvMagFreeze::zeroed())
        })
    }
}

/// `PV_MagShift(buffer, stretch = 1, shift = 0)`: move every bin's magnitude to a new bin, leaving
/// each bin's phase where it is - the magnitude-only counterpart of `PV_BinShift`.
///
/// Bin `i`'s magnitude lands in bin `(int)(shift + i * stretch + 0.5)`; magnitudes landing in the
/// same bin add, a bin nothing lands in gets magnitude zero, and a destination outside the spectrum
/// is dropped, as is a bin whose position is not finite (undefined behaviour in the reference). The
/// DC and Nyquist terms pass through. The op leaves the frame in polar form; its scratch is allocated
/// on the first frame, as large as the chain buffer (scsynth's `MAKE_TEMP_BUF`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagShift {
    /// Bin count of the first frame.
    numbins: u32,
    /// `1` once the scratch is allocated (scsynth's non-null `m_tempbuf`).
    allocated: u32,
}

impl Unit for PvMagShift {
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
        let dest = &mut ctx.aux.f32_mut()[..2 * numbins];

        // `ToPolarApx(buf)` after `MAKE_TEMP_BUF` (PV_UGens.cpp:429).
        if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex())
        {
            // The destination starts with zero magnitudes and the source's own phases.
            for (pair, bin) in dest.chunks_exact_mut(2).zip(spectrum.bins.iter()) {
                pair[0] = 0.0;
                pair[1] = bin.y;
            }
            let mut fpos = shift;
            for bin in spectrum.bins.iter() {
                if fpos.is_finite() {
                    // `(int32)(fpos + 0.5)`: the `0.5` is a `double`, so the sum rounds at double
                    // precision before truncating.
                    let pos = (f64::from(fpos) + 0.5) as i64;
                    if let Some(pair) = usize::try_from(pos)
                        .ok()
                        .filter(|&pos| pos < numbins)
                        .map(|pos| &mut dest[2 * pos..2 * pos + 2])
                    {
                        pair[0] += bin.x;
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

/// Constructor for [`PvMagShift`]: the unit allocates its scratch on the first frame.
pub struct PvMagShiftCtor;

impl UnitDef for PvMagShiftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvMagShift::zeroed())
        })
    }
}

/// scsynth's `TWOPI` in `PV_UGens.cpp`, the `float` literal `6.28318530717952646f`, which rounds to
/// the same `f32` as `TAU`.
const TWOPI: f32 = core::f32::consts::TAU;

/// `PV_PhaseShift(buffer, shift, integrate = 0)`: add `shift` radians to every bin's phase.
///
/// With `integrate > 0` (after truncation to an integer) the offset accumulates: each frame adds
/// `shift` plus the running total, and the total is then wrapped by `fmod` to one turn, so a steady
/// `shift` spins the phases frame by frame. The DC and Nyquist terms are untouched. The op leaves the
/// frame in polar form.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvPhaseShift {
    /// The running phase offset (scsynth's `m_phase_integral`).
    phase_integral: f32,
}

impl Unit for PvPhaseShift {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let shift = ctx.ins.control(1);
        let integrate = ctx.ins.control(2) as i32;
        // `PV_GET_BUF`, then `ToPolarApx(buf)` (PV_UGens.cpp:490-492).
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer, ctx.fft.complex())
        {
            let mut ashift = shift;
            if integrate > 0 {
                ashift += self.phase_integral;
                // `fmod(ashift, TWOPI)`: the remainder is exact, so its precision does not matter.
                self.phase_integral = ashift % TWOPI;
            }
            for bin in spectrum.bins.iter_mut() {
                bin.y += ashift;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvPhaseShift`].
pub struct PvPhaseShiftCtor;

impl UnitDef for PvPhaseShiftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvPhaseShift::zeroed()))
    }
}

/// `sc_max(a, b)`: `a` when it is greater, else `b` (so a NaN `a` yields `b`).
fn sc_max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `PV_MagDiv(bufferA, bufferB, zeroed = 0.0001)`: divide `A`'s magnitudes by `B`'s, keeping `A`'s
/// phases. Each divisor is floored at `zeroed`, so a quiet or negative `B` term divides by `zeroed`
/// instead. The DC and Nyquist terms divide the same way.
///
/// As in scsynth, both frames are converted to polar form in place (`ToPolarApx`), so `B` is left
/// polar too, and when both inputs name one buffer each term divides by itself.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagDiv {
    /// The unit is stateless; the state block must still be a non-zero-sized `Pod`.
    _pad: u32,
}

impl Unit for PvMagDiv {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let zeroed = ctx.ins.control(2);
        // `PV_GET_BUF2`, then `ToPolarApx(buf1)` and `ToPolarApx(buf2)` (PV_UGens.cpp:647-650).
        pv::pv_pair(ctx, pv::to_polar, |p, q| {
            *p.dc /= sc_max(q.dc(*p.dc), zeroed);
            *p.nyq /= sc_max(q.nyq(*p.nyq), zeroed);
            for (i, bin) in p.bins.iter_mut().enumerate() {
                bin.x /= sc_max(q.bin(i, *bin).x, zeroed);
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagDiv`].
pub struct PvMagDivCtor;

impl UnitDef for PvMagDivCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvMagDiv { _pad: 0 }))
    }
}
/// A [`pv::Convert`] that converts nothing, for the two-buffer units that copy bins as stored:
/// `PV_BinWipe` and `PV_RectComb2` read the raw packed frames, calling neither `ToPolarApx` nor
/// `ToComplexApx`.
fn as_stored<'b>(buf: &'b mut BufViewMut<'_>, _tables: &ComplexTables) -> Option<pv::Spectrum<'b>> {
    pv::spectrum(buf)
}

/// `PV_BinWipe(bufferA, bufferB, wipe = 0)`: replace part of `A`'s spectrum with `B`'s - `wipe` in
/// `(0, 1]` takes the lowest `wipe` fraction of bins from `B`, `[-1, 0)` the highest `|wipe|`
/// fraction.
///
/// A positive wipe always takes `B`'s DC term, and its Nyquist term too once it covers every bin; a
/// negative wipe mirrors that. The bins are copied as stored, without converting either frame, so
/// each frame keeps the coordinate form it arrived in (scsynth's `PV_BinWipe_next`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvBinWipe {
    /// The unit is stateless; the state block must still be a non-zero-sized `Pod`.
    _pad: u32,
}

impl Unit for PvBinWipe {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let wipe_frac = ctx.ins.control(2);
        // `PV_GET_BUF2`, then the raw frames (PV_UGens.cpp:598-601).
        pv::pv_pair(ctx, as_stored, |p, q| {
            let numbins = p.bins.len() as i32;
            let wipe = (wipe_frac * numbins as f32) as i32;
            if wipe > 0 {
                let wipe = wipe.min(numbins);
                *p.dc = q.dc(*p.dc);
                for i in 0..wipe as usize {
                    p.bins[i] = q.bin(i, p.bins[i]);
                }
                if wipe == numbins {
                    *p.nyq = q.nyq(*p.nyq);
                }
            } else if wipe < 0 {
                let wipe = wipe.max(-numbins);
                if wipe == -numbins {
                    *p.dc = q.dc(*p.dc);
                }
                for i in (numbins + wipe) as usize..numbins as usize {
                    p.bins[i] = q.bin(i, p.bins[i]);
                }
                *p.nyq = q.nyq(*p.nyq);
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvBinWipe`].
pub struct PvBinWipeCtor;

impl UnitDef for PvBinWipeCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvBinWipe { _pad: 0 }))
    }
}

/// `PV_RectComb2(bufferA, bufferB, numTeeth = 0, phase = 0, width = 0.5)`: `PV_RectComb`'s comb, but
/// the slots outside the teeth take `B`'s values instead of zero - the teeth show `A`, the gaps `B`.
///
/// The comb walks the same running phase as `PV_RectComb`, from the DC term through every bin to the
/// Nyquist term. Slots are copied as stored, without converting either frame, so each frame keeps
/// the coordinate form it arrived in (scsynth's `PV_RectComb2_next`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvRectComb2 {
    /// The unit is stateless; the state block must still be a non-zero-sized `Pod`.
    _pad: u32,
}

impl Unit for PvRectComb2 {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let num_teeth = ctx.ins.control(2);
        let start_phase = ctx.ins.control(3);
        let width = ctx.ins.control(4);
        // `PV_GET_BUF2`, then the raw frames (PV_UGens.cpp:888-896).
        pv::pv_pair(ctx, as_stored, |p, q| {
            let step = num_teeth / (p.bins.len() as i32 + 1) as f32;
            let mut phase = start_phase;
            if phase > width {
                *p.dc = q.dc(*p.dc);
            }
            phase = wrap_phase(phase + step);
            for i in 0..p.bins.len() {
                if phase > width {
                    p.bins[i] = q.bin(i, p.bins[i]);
                }
                phase = wrap_phase(phase + step);
            }
            if phase > width {
                *p.nyq = q.nyq(*p.nyq);
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvRectComb2`].
pub struct PvRectComb2Ctor;

impl UnitDef for PvRectComb2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 5 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvRectComb2 { _pad: 0 }))
    }
}

/// `PV_ConformalMap(buffer, areal = 0, aimag = 0)`: map every bin `z` through the conformal map
/// `(z - a) / (1 - z * conj(a))`, with `a = areal + i * aimag`.
///
/// The arithmetic is the reference's to the letter, including its reuse of the already-updated real
/// numerator when forming the imaginary one, and its floor of `0.001` on the squared denominator.
/// The DC and Nyquist terms are untouched. The op leaves the frame in Cartesian form.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvConformalMap {
    /// The unit is stateless; the state block must still be a non-zero-sized `Pod`.
    _pad: u32,
}

impl Unit for PvConformalMap {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let real2 = ctx.ins.control(1);
        let imag2 = ctx.ins.control(2);
        // `PV_GET_BUF`, then `ToComplexApx(buf)` (PV_ThirdParty.cpp:40-42).
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_complex(&mut buffer, ctx.fft.complex())
        {
            for bin in spectrum.bins.iter_mut() {
                let (real1, imag1) = (bin.x, bin.y);
                let mut numr = real1 - real2;
                let mut numi = imag1 - imag2;
                let mut denomr = 1.0 - (real1 * real2 + imag1 * imag2);
                let denomi = real1 * imag2 - real2 * imag1;

                numr = numr * denomr + numi * denomi;
                numi = numi * denomr - numr * denomi;

                // The squared modulus of the denominator, floored away from zero.
                denomr = denomr * denomr + denomi * denomi;
                if denomr < 0.001 {
                    denomr = 0.001;
                }
                denomr = 1.0 / denomr;

                bin.x = numr * denomr;
                bin.y = numi * denomr;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvConformalMap`].
pub struct PvConformalMapCtor;

impl UnitDef for PvConformalMapCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvConformalMap { _pad: 0 }))
    }
}
