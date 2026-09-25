//! `Dreset` - a pass-through demand source that resets its input on a trigger, plyphon's port of
//! scsynth's `Dreset` (`DemandUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};

/// `Dreset(in, reset)`: on each demand, pulls `in` and then `reset`, and yields the value from `in`.
/// When `reset` crosses from `<= 0` to `> 0`, `in` is reset after its value was pulled, so the reset
/// takes effect from the next demand. Inputs are `[in, reset]`.
///
/// A `NaN` from `in` is yielded as is, without recording that demand's `reset` value. Resetting the
/// unit resets `in` only.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dreset {
    /// The previous `reset` value, for rising-edge detection.
    prev_reset: f32,
}

impl Dreset {
    const IN: usize = 0;
    const RESET: usize = 1;
}

impl DemandUnit for Dreset {
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        self.prev_reset = 0.0;
        self.reset(ctx);
    }

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        ctx.reset(Self::IN);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let x = ctx.demand(Self::IN);
        let reset = ctx.demand(Self::RESET);
        if x.is_nan() {
            return f32::NAN;
        }
        if reset > 0.0 && self.prev_reset <= 0.0 {
            ctx.reset(Self::IN);
        }
        self.prev_reset = reset;
        x
    }
}

/// Constructor for [`Dreset`].
pub struct DresetCtor;

impl DemandUnitDef for DresetCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dreset { prev_reset: 0.0 }))
    }
}
