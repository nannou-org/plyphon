//! `PV_MagMul` - a phase-vocoder unit that multiplies the magnitudes of two spectra, plyphon's port of
//! scsynth's `PV_MagMul`. Compiled only with the `fft` feature; the first of the `PV_*` family.
//!
//! It reads buffer `B`'s spectrum and rewrites buffer `A`'s in place, scaling each of `A`'s bins by the
//! magnitude of `B`'s (so the result is `A` shaped by `B`'s spectral envelope, keeping `A`'s phases).
//! Both inputs are the frame-ready signals from upstream `FFT`s (a buffer number, or `< 0` when no
//! frame is ready); `A`'s signal is passed through so a downstream `IFFT`/`PV_*` sees the frame.
//!
//! The two-buffer access (read `B`, write `A` at once) goes through
//! [`buffer_pair_mut`](crate::unit::buffer_pair_mut), which hands out the two slots as disjoint
//! borrows - the seam every two-buffer `PV_*` unit shares.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `PV_MagMul(bufferA, bufferB)`: `magA *= magB`, keeping `A`'s phases (stateless; operates on the
/// buffers each frame). The `Pod` state is just padding for a non-zero slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagMul {
    _pad: u32,
}

impl PvMagMul {
    const BUFFER_A: usize = 0;
    const BUFFER_B: usize = 1;
}

impl Unit for PvMagMul {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let fbuf_a = ctx.ins.control(Self::BUFFER_A);
        let fbuf_b = ctx.ins.control(Self::BUFFER_B);
        if fbuf_a >= 0.0
            && fbuf_b >= 0.0
            && let Some((buf_a, buf_b)) =
                unit::buffer_pair_mut(ctx.buffers, fbuf_a as usize, fbuf_b as usize)
            && buf_a.num_frames() == buf_b.num_frames()
        {
            mag_mul(buf_a.data_mut(), buf_b.data());
        }
        // Pass `A`'s frame-ready signal downstream.
        *ctx.outs.control(0) = fbuf_a;
        DoneAction::Nothing
    }
}

/// Scale each of `a`'s bins by the magnitude of `b`'s, in scsynth's packed layout
/// `[DC, Nyquist, re1, im1, ...]`. DC and Nyquist are real-only.
fn mag_mul(a: &mut [f32], b: &[f32]) {
    let n = a.len();
    if n < 2 || b.len() != n {
        return;
    }
    a[0] *= b[0].abs();
    a[1] *= b[1].abs();
    for k in 1..n / 2 {
        let (re, im) = (b[2 * k], b[2 * k + 1]);
        let mag_b = math::sqrt(re * re + im * im);
        a[2 * k] *= mag_b;
        a[2 * k + 1] *= mag_b;
    }
}

/// Constructor for [`PvMagMul`].
pub struct PvMagMulCtor;

impl UnitDef for PvMagMulCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvMagMul { _pad: 0 }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mag_mul_scales_each_bin_by_b_magnitude() {
        // N = 8: packed [DC, Nyquist, re1, im1, re2, im2, re3, im3].
        let mut a = [1.0f32, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let b = [2.0f32, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0];
        mag_mul(&mut a, &b);
        assert_eq!(a[0], 2.0); // DC: 1 * |2|
        assert_eq!(a[1], 3.0); // Nyquist: 1 * |3|
        assert_eq!([a[2], a[3]], [4.0, 0.0]); // bin1: magB = |0+4i| = 4
        assert_eq!([a[4], a[5]], [5.0, 0.0]); // bin2: magB = |5+0i| = 5
        assert_eq!([a[6], a[7]], [0.0, 0.0]); // bin3: magB = 0 -> killed
    }

    #[test]
    fn mag_mul_ignores_mismatched_lengths() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0];
        let before = a;
        mag_mul(&mut a, &[1.0, 1.0]); // wrong length: no-op
        assert_eq!(a, before);
    }
}
