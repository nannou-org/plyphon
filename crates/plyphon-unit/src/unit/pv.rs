//! Shared phase-vocoder (`PV_*`) plumbing - plyphon's port of scsynth's `FFT_UGens.h` bin access.
//!
//! Every `PV_*` unit reads a frame-ready signal (input 0) carrying the FFT-chain buffer number (or
//! `< 0` between frames), edits that buffer's packed spectrum in place, and passes the signal on
//! (output 0) so the next unit in the chain sees the same frame. [`pv_frame`] does that read +
//! passthrough (scsynth's `PV_GET_BUF` preamble). The packed spectrum is scsynth's
//! `[dc, nyq, x0, y0, x1, y1, ...]`; [`Spectrum`] is a typed view over it, and
//! [`to_polar`]/[`to_complex`] convert it in place - idempotently, tracking the buffer's
//! [`SpectrumCoord`] - the analogues of scsynth's `ToPolarApx`/`ToComplexApx` (plyphon uses exact
//! `hypot`/`atan2` where scsynth uses a lookup-table approximation).
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::unit::ProcessCtx;
use plyphon_dsp::buffer::{BufViewMut, SpectrumCoord};
use plyphon_dsp::math;

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
/// packed view. scsynth's `ToPolarApx`, with exact `hypot`/`atan2` rather than a lookup-table
/// approximation. `None` if the buffer cannot be viewed as a packed spectrum.
pub fn to_polar<'a>(buf: &'a mut BufViewMut<'_>) -> Option<Spectrum<'a>> {
    if buf.coord() == SpectrumCoord::Complex {
        for bin in Spectrum::new(buf.data_mut())?.bins {
            let (re, im) = (bin.x, bin.y);
            bin.x = math::hypot(im, re);
            bin.y = math::atan2(im, re);
        }
        buf.set_coord(SpectrumCoord::Polar);
    }
    Spectrum::new(buf.data_mut())
}

/// Convert `buf` to complex (Cartesian) form in place if it is currently polar (idempotent), then
/// return its packed view. scsynth's `ToComplexApx`. `IFFT` and Cartesian `PV_*` units call this so
/// they read `(re, im)` regardless of what an upstream polar unit left behind.
pub fn to_complex<'a>(buf: &'a mut BufViewMut<'_>) -> Option<Spectrum<'a>> {
    if buf.coord() == SpectrumCoord::Polar {
        for bin in Spectrum::new(buf.data_mut())?.bins {
            let (mag, phase) = (bin.x, bin.y);
            bin.x = mag * math::cos(phase);
            bin.y = mag * math::sin(phase);
        }
        buf.set_coord(SpectrumCoord::Complex);
    }
    Spectrum::new(buf.data_mut())
}

/// The magnitude of bin `b`, reading it in whatever form `coord` says `b` is stored: `hypot(re, im)`
/// for a complex bin, or `mag` directly for a polar one. Lets a unit read another buffer's
/// magnitudes without converting (mutating) it.
pub fn bin_magnitude(coord: SpectrumCoord, b: Bin) -> f32 {
    match coord {
        SpectrumCoord::Complex => math::hypot(b.y, b.x),
        SpectrumCoord::Polar => b.x,
    }
}

/// Read bin `b` (stored in form `coord`) as a complex `(re, im)` pair, without mutating its buffer -
/// for a two-buffer complex op reading its read-only second buffer.
pub fn bin_as_complex(coord: SpectrumCoord, b: Bin) -> Bin {
    match coord {
        SpectrumCoord::Complex => b,
        SpectrumCoord::Polar => Bin {
            x: b.x * math::cos(b.y),
            y: b.x * math::sin(b.y),
        },
    }
}

/// Read bin `b` (stored in form `coord`) as a polar `(mag, phase)` pair, without mutating its buffer.
pub fn bin_as_polar(coord: SpectrumCoord, b: Bin) -> Bin {
    match coord {
        SpectrumCoord::Polar => b,
        SpectrumCoord::Complex => Bin {
            x: math::hypot(b.y, b.x),
            y: math::atan2(b.y, b.x),
        },
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

    /// Round-tripping complex -> polar -> complex is the identity (within float tolerance), and the
    /// coord flag tracks the current form so a second conversion is a no-op.
    #[test]
    fn polar_round_trip_is_idempotent() {
        // N = 8: [dc, nyq, re0, im0, re1, im1, re2, im2].
        let mut buf =
            Buffer::from_interleaved(vec![1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 0.5, -0.5], 1, 48_000.0);
        assert_eq!(buf.coord(), SpectrumCoord::Complex);
        let original = buf.data().to_vec();

        // Complex -> polar: bin0 (3, 4) has magnitude 5.
        to_polar(&mut buf.view_mut());
        assert_eq!(buf.coord(), SpectrumCoord::Polar);
        let mag0 = buf.data()[2];
        assert!((mag0 - 5.0).abs() < 1e-5, "mag of (3,4) is 5, got {mag0}");

        // A second to_polar is a no-op (already polar): the data is unchanged.
        let before = buf.data().to_vec();
        to_polar(&mut buf.view_mut());
        assert_eq!(buf.data(), before.as_slice());

        // Back to complex restores the original bins (dc/nyq untouched throughout).
        to_complex(&mut buf.view_mut());
        assert_eq!(buf.coord(), SpectrumCoord::Complex);
        for (got, want) in buf.data().iter().zip(&original) {
            assert!(
                (got - want).abs() < 1e-4,
                "round trip: got {got}, want {want}"
            );
        }
    }
}
