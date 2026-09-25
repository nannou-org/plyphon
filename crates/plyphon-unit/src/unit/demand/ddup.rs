//! `Ddup` and `Dstutter` - value-repeating demand sources, plyphon's port of scsynth's `Ddup` and
//! `Dstutter` (`DemandUGens.cpp`), whose calc functions are the same code.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Ddup(n, in)` (and the older `Dstutter(n, in)`): demands a value from `in` and a count from `n`,
/// yields that value `n` times, then demands the next pair. Inputs are `[n, in]`.
///
/// Once the current value has been yielded `n` times the next demand pulls `in` first, then `n`;
/// if either is `NaN` the unit yields `NaN` (and pulls both again on the next demand). A count is
/// rounded to the nearest integer (`floor(n + 0.5)`); a count of zero or less still yields the value
/// once. A reset restarts the count and resets both inputs.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Ddup {
    /// How many times the held value is yielded; `-1` until the first pair is pulled.
    repeats: f64,
    /// How many times the held value has been yielded.
    repeat_count: f64,
    /// The held value.
    value: f32,
    _pad: u32,
}

impl Ddup {
    const N: usize = 0;
    const IN: usize = 1;
}

impl DemandUnit for Ddup {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0.0;
        ctx.reset(Self::N);
        ctx.reset(Self::IN);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        if self.repeat_count >= self.repeats {
            let val = ctx.demand(Self::IN);
            let repeats = ctx.demand(Self::N);
            if repeats.is_nan() || val.is_nan() {
                return f32::NAN;
            }
            self.value = val;
            self.repeats = math::floor(repeats + 0.5) as f64;
            self.repeat_count = 0.0;
        }
        self.repeat_count += 1.0;
        self.value
    }
}

/// Constructor for [`Ddup`], registered as both `Ddup` and `Dstutter`. The unit is reset when the
/// synth is constructed.
pub struct DdupCtor;

impl DemandUnitDef for DdupCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Ddup {
            repeats: -1.0,
            repeat_count: 0.0,
            value: 0.0,
            _pad: 0,
        }))
    }
}
