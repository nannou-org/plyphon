//! `Dconst` - a demand source that sums its input up to a fixed total, plyphon's port of scsynth's
//! `Dconst` (`DemandUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};

/// `Dconst(sum, in, tolerance)`: yields values from `in` until their running sum would reach `sum`,
/// then yields the remainder that makes the sum exact and ends (`NaN`). Inputs are
/// `[sum, in, tolerance]`.
///
/// `sum` and `tolerance` are latched together on the first demand (and demanded again while the
/// latched `sum` is negative); if either is `NaN` the unit yields `NaN`. Each demand then pulls `in`:
/// a `NaN` value yields `NaN`; a value that takes the running sum past `sum`, or to within
/// `tolerance` of it, yields `sum` minus the running sum so far and ends the sequence. A reset
/// restarts the sum and resets all three inputs.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dconst {
    /// The latched total; negative until the first demand latches it.
    total: f32,
    /// The sum of the values yielded so far; `NaN` once the sequence has ended.
    running_sum: f32,
    /// The latched tolerance.
    tolerance: f32,
}

impl Dconst {
    const SUM: usize = 0;
    const IN: usize = 1;
    const TOLERANCE: usize = 2;
}

impl DemandUnit for Dconst {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.total = -1.0;
        self.running_sum = 0.0;
        ctx.reset(Self::SUM);
        ctx.reset(Self::IN);
        ctx.reset(Self::TOLERANCE);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        if self.running_sum.is_nan() {
            return f32::NAN;
        }
        if self.total < 0.0 {
            let total = ctx.demand(Self::SUM);
            let tolerance = ctx.demand(Self::TOLERANCE);
            if total.is_nan() || tolerance.is_nan() {
                return f32::NAN;
            }
            self.total = total;
            self.tolerance = tolerance;
        }
        let total = self.total;
        let tolerance = self.tolerance;
        let val = ctx.demand(Self::IN);
        if val.is_nan() {
            return f32::NAN;
        }
        let running_sum = self.running_sum + val;
        if running_sum > total || (total - running_sum).abs() <= tolerance {
            let out = total - self.running_sum;
            // End the sequence: the next demand yields `NaN`.
            self.running_sum = f32::NAN;
            out
        } else {
            self.running_sum = running_sum;
            val
        }
    }
}

/// Constructor for [`Dconst`]. The unit is reset when the synth is constructed.
pub struct DconstCtor;

impl DemandUnitDef for DconstCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dconst {
            total: -1.0,
            running_sum: 0.0,
            tolerance: 0.0,
        }))
    }
}
