//! Shared phase-vocoder (`PV_*`) plumbing - plyphon's port of scsynth's `FFT_UGens.h` bin access.
//!
//! Every `PV_*` unit reads a frame-ready signal (input 0) carrying the FFT-chain buffer number (or
//! `< 0` between frames), edits that buffer's packed spectrum in place, and passes the signal on
//! (output 0) so the next unit in the chain sees the same frame. [`pv_frame`] does that read +
//! passthrough (scsynth's `PV_GET_BUF` preamble). The packed spectrum is scsynth's
//! `[dc, nyq, x0, y0, x1, y1, ...]`; [`Spectrum`] is a typed view over it, and
//! [`to_polar`]/[`to_complex`] convert it in place - idempotently, tracking the buffer's
//! [`SpectrumCoord`] - with scsynth's lookup-table `ToPolarApx`/`ToComplexApx` (see
//! [`plyphon_dsp::complex`]).
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::unit::ProcessCtx;
use plyphon_dsp::buffer::{BufViewMut, SpectrumCoord};
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

/// The magnitude of bin `b`, reading it in whatever form `coord` says `b` is stored: the table
/// magnitude of `(re, im)` for a complex bin, or `mag` directly for a polar one. Lets a unit read
/// another buffer's magnitudes without converting (mutating) it.
pub fn bin_magnitude(coord: SpectrumCoord, b: Bin, tables: &ComplexTables) -> f32 {
    bin_as_polar(coord, b, tables).x
}

/// Read bin `b` (stored in form `coord`) as a complex `(re, im)` pair, without mutating its buffer -
/// for a two-buffer complex op reading its read-only second buffer.
pub fn bin_as_complex(coord: SpectrumCoord, b: Bin, tables: &ComplexTables) -> Bin {
    match coord {
        SpectrumCoord::Complex => b,
        SpectrumCoord::Polar => {
            let (x, y) = tables.to_complex(b.x, b.y);
            Bin { x, y }
        }
    }
}

/// Read bin `b` (stored in form `coord`) as a polar `(mag, phase)` pair, without mutating its buffer.
pub fn bin_as_polar(coord: SpectrumCoord, b: Bin, tables: &ComplexTables) -> Bin {
    match coord {
        SpectrumCoord::Polar => b,
        SpectrumCoord::Complex => {
            let (x, y) = tables.to_polar(b.x, b.y);
            Bin { x, y }
        }
    }
}

/// A read-only packed view of the bins (skipping `dc`/`nyq`), for the second buffer of a two-buffer
/// op (`PV_MagMul` reads `B` while rewriting `A`). Empty if the slice is too short or odd.
pub fn bins(data: &[f32]) -> &[Bin] {
    if data.len() < 2 {
        return &[];
    }
    bytemuck::try_cast_slice(&data[2..]).unwrap_or(&[])
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
