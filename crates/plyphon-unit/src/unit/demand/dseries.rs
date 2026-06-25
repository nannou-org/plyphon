//! `Dseries` - an arithmetic series demand source, plyphon's port of scsynth's `Dseries`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Dseries(length, start, step)`: on each demand, yields `start`, `start + step`, `start + 2*step`,
/// ... for `length` values, then `NaN`. `start` and `length` are latched on the first demand; `step`
/// is re-read every demand, so a modulating demand source can drive it (matching scsynth). Inputs are
/// in scsynth's server-side order: `[length, start, step]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dseries {
    /// The next value to emit.
    value: f64,
    /// The current step.
    step: f64,
    /// Latched length; `-1` until the first demand latches it (scsynth's `m_repeats < 0` sentinel).
    repeats: f64,
    /// How many values have been emitted this run.
    repeat_count: u32,
    _pad: u32,
}

impl Dseries {
    const LENGTH: usize = 0;
    const START: usize = 1;
    const STEP: usize = 2;
}

impl DemandUnit for Dseries {
    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        // `step` is re-read every demand so it can be modulated by a demand source.
        let step = ctx.demand(Self::STEP);
        if !step.is_nan() {
            self.step = step as f64;
        }
        // Latch `length` and `start` on the first demand of the run.
        if self.repeats < 0.0 {
            let len = ctx.demand(Self::LENGTH);
            self.repeats = if len.is_nan() {
                0.0
            } else {
                math::floor(len as f64 + 0.5)
            };
            self.value = ctx.demand(Self::START) as f64;
        }
        if (self.repeat_count as f64) >= self.repeats {
            return f32::NAN;
        }
        let out = self.value as f32;
        self.value += self.step;
        self.repeat_count += 1;
        out
    }
}

/// Constructor for [`Dseries`].
pub struct DseriesCtor;

impl DemandUnitDef for DseriesCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dseries {
            value: 0.0,
            step: 0.0,
            repeats: -1.0,
            repeat_count: 0,
            _pad: 0,
        }))
    }
}
