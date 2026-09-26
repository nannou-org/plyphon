//! Shared phase-vocoder (`PV_*`) plumbing - plyphon's port of scsynth's `FFT_UGens.h` bin access.
//!
//! Every `PV_*` unit reads a frame-ready signal (input 0) carrying the FFT-chain buffer number (or
//! `< 0` between frames), edits that buffer's packed spectrum in place, and passes the signal on
//! (output 0) so the next unit in the chain sees the same frame. [`pv_frame`] does that read +
//! passthrough (scsynth's `PV_GET_BUF` preamble), and [`pv_pair`] the two-buffer form
//! (`PV_GET_BUF2`). The packed spectrum is scsynth's `[dc, nyq, x0, y0, x1, y1, ...]`; [`Spectrum`]
//! is a typed view over it, and [`to_polar`]/[`to_complex`] convert it in place - idempotently,
//! tracking the buffer's [`SpectrumCoord`] - with scsynth's lookup-table `ToPolarApx`/`ToComplexApx`
//! (see [`plyphon_dsp::complex`]).
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::unit::{self, LocalBufs, ProcessCtx};
use plyphon_dsp::buffer::{BufViewMut, BufferTable, SpectrumCoord};
use plyphon_dsp::complex::ComplexTables;

/// One spectral bin: a pair of floats whose meaning follows the buffer's [`SpectrumCoord`] -
/// `(re, im)` when [`Complex`](SpectrumCoord::Complex), `(mag, phase)` when
/// [`Polar`](SpectrumCoord::Polar). Memory-compatible with scsynth's `SCComplex`/`SCPolar`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Bin {
    /// `re` (complex) or `mag` (polar).
    pub x: f32,
    /// `im` (complex) or `phase` (polar).
    pub y: f32,
}

/// A mutable view over a packed spectrum `[dc, nyq, bins...]` - scsynth's `SCComplexBuf`/`SCPolarBuf`.
/// `dc` and `nyq` are the two purely-real terms; `bins` are the `(samples - 2) / 2` complex/polar
/// pairs.
pub struct Spectrum<'a> {
    /// The DC (0 Hz) term, purely real.
    pub dc: &'a mut f32,
    /// The Nyquist term, purely real.
    pub nyq: &'a mut f32,
    /// The bins, `(samples - 2) / 2` of them.
    pub bins: &'a mut [Bin],
}

impl<'a> Spectrum<'a> {
    /// View the packed slice `[dc, nyq, x0, y0, ...]`. `None` if it is shorter than two samples or
    /// its bin region is not an even number of floats (so it cannot pack into [`Bin`]s).
    fn new(data: &'a mut [f32]) -> Option<Spectrum<'a>> {
        let (dc, rest) = data.split_first_mut()?;
        let (nyq, bin_floats) = rest.split_first_mut()?;
        let bins = bytemuck::try_cast_slice_mut(bin_floats).ok()?;
        Some(Spectrum { dc, nyq, bins })
    }
}

/// A packed view of `buf`'s spectrum *without* converting its coordinate form. For coord-independent
/// edits (zeroing or copying whole bins, e.g. `PV_BrickWall`) that read neither magnitude nor phase.
pub fn spectrum<'a>(buf: &'a mut BufViewMut<'_>) -> Option<Spectrum<'a>> {
    Spectrum::new(buf.data_mut())
}

/// Read the frame-ready signal (input 0), pass it to output 0 (so the chain continues), and return
/// the chain buffer index when a frame is ready. `None` between frames (`< 0`, normalised to `-1` on
/// the output, like scsynth) - the unit returns without touching a buffer. scsynth's `PV_GET_BUF`
/// preamble.
pub fn pv_frame(ctx: &mut ProcessCtx<'_>) -> Option<usize> {
    let fbufnum = ctx.ins.control(0);
    *ctx.outs.control(0) = if fbufnum >= 0.0 { fbufnum } else { -1.0 };
    (fbufnum >= 0.0).then_some(fbufnum as usize)
}

/// Convert `buf` to polar form in place if it is currently complex (idempotent), then return its
/// packed view. scsynth's `ToPolarApx` (`FFT_UGens.h`), bin by bin with the lookup-table
/// [`ComplexTables::to_polar`] (the engine's, from `ctx.fft.complex()`). `None` if the buffer
/// cannot be viewed as a packed spectrum.
pub fn to_polar<'a>(buf: &'a mut BufViewMut<'_>, tables: &ComplexTables) -> Option<Spectrum<'a>> {
    if buf.coord() == SpectrumCoord::Complex {
        for bin in Spectrum::new(buf.data_mut())?.bins {
            (bin.x, bin.y) = tables.to_polar(bin.x, bin.y);
        }
        buf.set_coord(SpectrumCoord::Polar);
    }
    Spectrum::new(buf.data_mut())
}

/// Convert `buf` to complex (Cartesian) form in place if it is currently polar (idempotent), then
/// return its packed view. scsynth's `ToComplexApx` (`FFT_UGens.h`), bin by bin with the lookup-table
/// [`ComplexTables::to_complex`] (the engine's, from `ctx.fft.complex()`). `IFFT` and Cartesian
/// `PV_*` units call this so they read `(re, im)` regardless of what an upstream polar unit left
/// behind.
pub fn to_complex<'a>(buf: &'a mut BufViewMut<'_>, tables: &ComplexTables) -> Option<Spectrum<'a>> {
    if buf.coord() == SpectrumCoord::Polar {
        for bin in Spectrum::new(buf.data_mut())?.bins {
            (bin.x, bin.y) = tables.to_complex(bin.x, bin.y);
        }
        buf.set_coord(SpectrumCoord::Complex);
    }
    Spectrum::new(buf.data_mut())
}

/// Buffer `B` of a two-buffer op, as [`pv_pair`] hands it over (already converted).
pub enum Second<'a> {
    /// `B` is a different buffer: its DC and Nyquist terms and its bins, read-only.
    Other {
        /// `B`'s DC term.
        dc: f32,
        /// `B`'s Nyquist term.
        nyq: f32,
        /// `B`'s bins, as many as `A` has.
        bins: &'a [Bin],
    },
    /// Both inputs name one buffer, so `B` is `A` itself: every read of `B` sees what the op has
    /// written to `A` so far, as in scsynth, where both pointers alias the one buffer.
    Same,
}

impl Second<'_> {
    /// Whether `B` is `A` itself.
    pub fn is_same(&self) -> bool {
        matches!(self, Second::Same)
    }

    /// `B`'s DC term, given `A`'s as it is now.
    pub fn dc(&self, a_dc: f32) -> f32 {
        match self {
            Second::Other { dc, .. } => *dc,
            Second::Same => a_dc,
        }
    }

    /// `B`'s Nyquist term, given `A`'s as it is now.
    pub fn nyq(&self, a_nyq: f32) -> f32 {
        match self {
            Second::Other { nyq, .. } => *nyq,
            Second::Same => a_nyq,
        }
    }

    /// `B`'s bin `i`, given `A`'s bin `i` as it is now.
    pub fn bin(&self, i: usize, a_bin: Bin) -> Bin {
        match self {
            // `pv_pair` only pairs buffers of equal length, so `B` has every bin `A` has.
            Second::Other { bins, .. } => bins[i],
            Second::Same => a_bin,
        }
    }
}

/// A converter for [`pv_pair`]: [`to_polar`], [`to_complex`], or [`unconverted`].
pub type Convert = for<'b, 'c> fn(&'b mut BufViewMut<'c>, &ComplexTables) -> Option<Spectrum<'b>>;

/// The [`Convert`] for a two-buffer op that moves whole bins and converts neither buffer
/// (`PV_RandWipe`): the packed view as it stands.
pub fn unconverted<'a>(buf: &'a mut BufViewMut<'_>, _: &ComplexTables) -> Option<Spectrum<'a>> {
    spectrum(buf)
}

/// [`pv_pair`]'s preamble on its own - scsynth's `PV_GET_BUF2`, then both buffers converted with
/// `convert` - for a unit that must reach the rest of its context (memory, the random stream)
/// between the preamble and its op. Returns `A`'s and `B`'s buffer numbers when the op should run,
/// with output 0 already written, as [`pv_pair`] describes.
pub fn pv_pair_frame(ctx: &mut ProcessCtx<'_>, convert: Convert) -> Option<(usize, usize)> {
    let fbufnum1 = ctx.ins.control(0);
    let fbufnum2 = ctx.ins.control(1);
    if fbufnum1 < 0.0 || fbufnum2 < 0.0 {
        *ctx.outs.control(0) = -1.0;
        return None;
    }
    *ctx.outs.control(0) = fbufnum1;
    let (a, b) = (fbufnum1 as usize, fbufnum2 as usize);
    let samples = |i| unit::buffer_at(ctx.buffers, &ctx.local_bufs, i).map(|buf| buf.data().len());
    match (samples(a), samples(b)) {
        (Some(samples_a), Some(samples_b)) if samples_a == samples_b => {}
        _ => return None,
    }
    let tables = ctx.fft.complex();
    for i in [a, b] {
        if let Some(mut buf) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, i) {
            convert(&mut buf, tables);
        }
    }
    Some((a, b))
}

/// The two-buffer preamble - scsynth's `PV_GET_BUF2` followed by the unit converting `buf1` then
/// `buf2` with `convert` - then `op` on `A`'s spectrum and [`Second`] `B`, the result going into `A`.
/// The conversions use the World's lookup tables, `ctx.fft.complex()`.
///
/// Inputs 0 and 1 carry the frame signals of `A` and `B`. Unless both carry a frame, output 0 is
/// `-1` and nothing else happens. Otherwise output 0 passes `A`'s number on; if the two buffers
/// hold different numbers of samples the unit stops there, as scsynth does. Both buffers are
/// converted in place, so a chain continuing from `B` sees it in the new form too, and when both
/// inputs name the same buffer the op still runs, on that buffer alone ([`Second::Same`]).
pub fn pv_pair(
    ctx: &mut ProcessCtx<'_>,
    convert: Convert,
    op: impl FnOnce(Spectrum<'_>, Second<'_>),
) {
    if let Some((a, b)) = pv_pair_frame(ctx, convert) {
        pv_pair_op(ctx.buffers, &mut ctx.local_bufs, a, b, op);
    }
}

/// [`pv_pair`]'s op on its own: run `op` on buffer `a`'s spectrum and [`Second`] `b`, for buffers
/// that [`pv_pair_frame`] accepted. It takes the buffer tables rather than the whole context, so
/// `op` may borrow the unit's memory.
pub fn pv_pair_op(
    buffers: &mut BufferTable,
    local_bufs: &mut LocalBufs<'_>,
    a: usize,
    b: usize,
    op: impl FnOnce(Spectrum<'_>, Second<'_>),
) {
    if a == b {
        if let Some(mut buf) = unit::buffer_at_mut(buffers, local_bufs, a)
            && let Some(p) = Spectrum::new(buf.data_mut())
        {
            op(p, Second::Same);
        }
    } else if let Some((mut buf_a, buf_b)) = unit::buffer_pair_mut(buffers, local_bufs, a, b)
        && let Some(p) = Spectrum::new(buf_a.data_mut())
        && let [dc, nyq, bin_floats @ ..] = buf_b.data()
        && let Ok(bins) = bytemuck::try_cast_slice(bin_floats)
    {
        let q = Second::Other {
            dc: *dc,
            nyq: *nyq,
            bins,
        };
        op(p, q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plyphon_dsp::buffer::Buffer;

    /// The coord flag tracks the current form, so a second conversion to the same form is a no-op,
    /// and a round trip through the lookup tables lands close to where it started.
    #[test]
    fn conversion_is_idempotent() {
        // N = 8: [dc, nyq, re0, im0, re1, im1, re2, im2].
        let mut buf =
            Buffer::from_interleaved(vec![1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 0.5, -0.5], 1, 48_000.0);
        assert_eq!(buf.coord(), SpectrumCoord::Complex);
        let original = buf.data().to_vec();
        let tables = ComplexTables::new();

        // Complex -> polar: bin0 (3, 4) has magnitude 5, to the tables' precision.
        to_polar(&mut buf.view_mut(), &tables);
        assert_eq!(buf.coord(), SpectrumCoord::Polar);
        let mag0 = buf.data()[2];
        assert!((mag0 - 5.0).abs() < 1e-2, "mag of (3,4) is 5, got {mag0}");

        // A second to_polar is a no-op (already polar): the data is unchanged.
        let before = buf.data().to_vec();
        to_polar(&mut buf.view_mut(), &tables);
        assert_eq!(buf.data(), before.as_slice());

        // Back to complex lands near the original bins; dc/nyq are untouched throughout.
        to_complex(&mut buf.view_mut(), &tables);
        assert_eq!(buf.coord(), SpectrumCoord::Complex);
        assert_eq!(buf.data()[..2], original[..2]);
        for (got, want) in buf.data().iter().zip(&original) {
            assert!(
                (got - want).abs() < 1e-2,
                "round trip: got {got}, want {want}"
            );
        }
    }
}
