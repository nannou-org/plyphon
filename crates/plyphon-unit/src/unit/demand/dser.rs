//! `Dser` - a serial (length-counted) sequence demand source, plyphon's port of scsynth's `Dser`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Dser(length, items...)`: cycles through the list items in order and yields exactly `length`
/// *values* (not `length` full passes - the difference from `Dseq`), then `NaN`. Input `0` is
/// `length`; inputs `1..` are the list items (a nested demand item is pulled until it yields `NaN`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dser {
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted so far.
    repeat_count: u32,
    /// Index of the current item input (`1..num_inputs`).
    index: u32,
    /// Whether the child at `index` should be reset before its next pull.
    need_reset_child: u32,
}

impl Dser {
    const LENGTH: usize = 0;
    const FIRST_ITEM: u32 = 1;
}

impl DemandUnit for Dser {
    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
        self.index = Self::FIRST_ITEM;
        self.need_reset_child = 1;
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let num_inputs = ctx.num_inputs();
        if num_inputs <= Self::FIRST_ITEM as usize {
            return f32::NAN;
        }
        if self.repeats < 0.0 {
            let x = ctx.demand(Self::LENGTH);
            self.repeats = if x.is_nan() {
                0.0
            } else {
                math::floor(x + 0.5)
            };
        }
        let guard_limit = num_inputs.saturating_mul(2) + 4;
        let mut guard = 0;
        loop {
            // Wrap to the start of the list, but - unlike `Dseq` - do NOT count a pass here; `Dser`
            // counts emitted values instead.
            if self.index as usize >= num_inputs {
                self.index = Self::FIRST_ITEM;
            }
            if self.repeat_count as f32 >= self.repeats {
                return f32::NAN;
            }
            let k = self.index as usize;
            if ctx.is_demand(k) {
                if self.need_reset_child != 0 {
                    self.need_reset_child = 0;
                    ctx.reset(k);
                }
                let x = ctx.demand(k);
                if x.is_nan() {
                    self.index += 1;
                    self.repeat_count += 1;
                    self.need_reset_child = 1;
                } else {
                    return x;
                }
            } else {
                let x = ctx.demand(k);
                self.index += 1;
                self.repeat_count += 1;
                self.need_reset_child = 1;
                return x;
            }
            guard += 1;
            if guard > guard_limit {
                return f32::NAN;
            }
        }
    }
}

/// Constructor for [`Dser`].
pub struct DserCtor;

impl DemandUnitDef for DserCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dser {
            repeats: -1.0,
            repeat_count: 0,
            index: Dser::FIRST_ITEM,
            need_reset_child: 1,
        }))
    }
}
