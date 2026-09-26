//! `PV_MagMul` - a phase-vocoder unit that multiplies the magnitudes of two spectra, plyphon's port of
//! scsynth's `PV_MagMul` (`PV_UGens.cpp`). Compiled only with the `fft` feature.
//!
//! It reads buffer `B`'s spectrum and rewrites buffer `A`'s in place, scaling each of `A`'s bins by the
//! magnitude of `B`'s (so the result is `A` shaped by `B`'s spectral envelope, keeping `A`'s phases).
//! Both inputs are the frame-ready signals from upstream `FFT`s (a buffer number, or `< 0` when no
//! frame is ready); `A`'s signal is passed through so a downstream `IFFT`/`PV_*` sees the frame.
//!
//! Both buffers are converted to polar form in place, as scsynth does, through the shared two-buffer
//! preamble [`pv::pv_pair`].

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec};

/// `PV_MagMul(bufferA, bufferB)`: `magA *= magB`, keeping `A`'s phases (stateless; operates on the
/// buffers each frame). The `Pod` state is just padding for a non-zero slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagMul {
    _pad: u32,
}

impl Unit for PvMagMul {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        pv::pv_pair(ctx, pv::to_polar, |p, q| {
            *p.dc *= q.dc(*p.dc);
            *p.nyq *= q.nyq(*p.nyq);
            for (i, bin) in p.bins.iter_mut().enumerate() {
                bin.x *= q.bin(i, *bin).x;
            }
        });
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
