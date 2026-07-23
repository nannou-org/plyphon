//! `PV_MagSquared` - squares every bin's magnitude, plyphon's port of scsynth's `PV_MagSquared`. The
//! first *polar* `PV_*` unit, exercising the shared polar conversion in [`pv`].
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec};

/// `PV_MagSquared(buffer)`: square the magnitude of the DC, Nyquist, and every bin, leaving phases
/// (stateless; operates on the chain buffer each frame). The `Pod` state is padding for a non-zero
/// slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagSquared {
    _pad: u32,
}

impl Unit for PvMagSquared {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::to_polar(&mut buffer)
        {
            // DC and Nyquist are real magnitudes; the bins carry (mag, phase) in polar form.
            *spectrum.dc *= *spectrum.dc;
            *spectrum.nyq *= *spectrum.nyq;
            for bin in spectrum.bins.iter_mut() {
                bin.x *= bin.x;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagSquared`].
pub struct PvMagSquaredCtor;

impl UnitDef for PvMagSquaredCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvMagSquared { _pad: 0 }))
    }
}
