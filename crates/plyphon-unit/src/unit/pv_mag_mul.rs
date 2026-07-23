//! `PV_MagMul` - a phase-vocoder unit that multiplies the magnitudes of two spectra, plyphon's port of
//! scsynth's `PV_MagMul`. Compiled only with the `fft` feature.
//!
//! It reads buffer `B`'s spectrum and rewrites buffer `A`'s in place, scaling each of `A`'s bins by the
//! magnitude of `B`'s (so the result is `A` shaped by `B`'s spectral envelope, keeping `A`'s phases).
//! Both inputs are the frame-ready signals from upstream `FFT`s (a buffer number, or `< 0` when no
//! frame is ready); `A`'s signal is passed through so a downstream `IFFT`/`PV_*` sees the frame.
//!
//! `A` is converted to polar (so the multiply scales magnitudes and keeps phases, as scsynth does).
//! `B`'s magnitudes are read in whatever form `B` is currently in (via [`pv::bin_magnitude`]), so -
//! unlike scsynth, which leaves `B` polar as a side effect - plyphon leaves `B` untouched; the
//! resulting audio is the same. The two-buffer access goes through
//! [`buffer_pair_mut`](crate::unit::buffer_pair_mut) - the seam every two-buffer `PV_*` unit shares.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec};

/// `PV_MagMul(bufferA, bufferB)`: `magA *= magB`, keeping `A`'s phases (stateless; operates on the
/// buffers each frame). The `Pod` state is just padding for a non-zero slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagMul {
    _pad: u32,
}

impl PvMagMul {
    const BUFFER_B: usize = 1;
}

impl Unit for PvMagMul {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `pv_frame` reads `A`'s frame signal (input 0) and passes it downstream (output 0).
        let fbuf_b = ctx.ins.control(Self::BUFFER_B);
        if let Some(a_idx) = pv::pv_frame(ctx)
            && fbuf_b >= 0.0
            && let Some((mut buf_a, buf_b)) =
                unit::buffer_pair_mut(ctx.buffers, &mut ctx.local_bufs, a_idx, fbuf_b as usize)
            && buf_a.num_frames() == buf_b.num_frames()
            // A frame needs at least its `[dc, nyq]` header; a shorter buffer (`LocalBuf(1, 1)`)
            // must not panic the audio thread on the raw reads below.
            && buf_b.data().len() >= 2
        {
            // Read `B`'s real DC/Nyquist and its bins without converting it.
            let coord_b = buf_b.coord();
            let (b_dc, b_nyq) = (buf_b.data()[0], buf_b.data()[1]);
            let b_bins = pv::bins(buf_b.data());
            if let Some(a) = pv::to_polar(&mut buf_a) {
                *a.dc *= b_dc;
                *a.nyq *= b_nyq;
                for (a_bin, &b_bin) in a.bins.iter_mut().zip(b_bins) {
                    a_bin.x *= pv::bin_magnitude(coord_b, b_bin);
                }
            }
        }
        DoneAction::Nothing
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
