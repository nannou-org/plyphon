//! `Dgeom` - a geometric-series demand source, plyphon's port of scsynth's `Dgeom`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Dgeom(length, start, grow)`: on each demand yields `start * grow^n` for `length` values, then
/// `NaN`. `length` and `start` are latched on the first demand; `grow` is re-read every demand (so it
/// can itself be a demand source). Inputs are in scsynth's server-side order: `[length, start, grow]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dgeom {
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted this run.
    repeat_count: u32,
    /// The running value, `start * grow^n`.
    value: f32,
    /// The most recent (non-NaN) growth factor.
    grow: f32,
}

impl Dgeom {
    const LENGTH: usize = 0;
    const START: usize = 1;
    const GROW: usize = 2;
}

impl DemandUnit for Dgeom {
    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let grow = ctx.demand(Self::GROW);
        if !grow.is_nan() {
            self.grow = grow;
        }
        if self.repeats < 0.0 {
            let len = ctx.demand(Self::LENGTH);
            self.repeats = if len.is_nan() {
                0.0
            } else {
                math::floor(len + 0.5)
            };
            self.value = ctx.demand(Self::START);
        }
        if self.repeat_count as f32 >= self.repeats {
            return f32::NAN;
        }
        self.repeat_count += 1;
        let out = self.value;
        self.value *= self.grow;
        out
    }
}

/// Constructor for [`Dgeom`].
pub struct DgeomCtor;

impl DemandUnitDef for DgeomCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dgeom {
            repeats: -1.0,
            repeat_count: 0,
            value: 0.0,
            grow: 0.0,
        }))
    }
}
