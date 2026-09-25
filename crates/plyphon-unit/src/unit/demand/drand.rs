//! `Drand` - a random-selection demand source, plyphon's port of scsynth's `Drand`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// A uniformly random item index in `1..num_inputs` (there are `num_inputs - 1` items after the
/// `length` input at `0`) - scsynth's `irand(mNumInputs - 1) + 1`, which draws even when there is no
/// item to choose.
fn pick(ctx: &mut DemandCtx<'_>) -> u32 {
    let n = ctx.num_inputs() as i32;
    (ctx.rgen().next_irand(n - 1) + 1).max(0) as u32
}

/// `Drand(length, items...)`: yields `length` values, each a uniformly random pick from the list
/// items (repeats allowed), then `NaN`. Input `0` is `length`; inputs `1..` are the items (a nested
/// demand item is pulled until it yields `NaN`, then a fresh item is picked). Picks are drawn from
/// the synth's random stream.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Drand {
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted so far.
    repeat_count: u32,
    /// Index of the current item input (`1..num_inputs`), chosen at random.
    index: u32,
    /// Whether the child at `index` should be reset before its next pull.
    need_reset_child: u32,
}

impl Drand {
    const LENGTH: usize = 0;
    const FIRST_ITEM: u32 = 1;
}

impl DemandUnit for Drand {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
        self.need_reset_child = 1;
        self.index = pick(ctx);
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
                    self.index = pick(ctx);
                    self.repeat_count += 1;
                    self.need_reset_child = 1;
                } else {
                    return x;
                }
            } else {
                let x = ctx.demand(k);
                self.index = pick(ctx);
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

/// Constructor for [`Drand`]. Its first item is picked when the synth is constructed (its reset).
pub struct DrandCtor;

impl DemandUnitDef for DrandCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Drand {
            repeats: -1.0,
            repeat_count: 0,
            index: Drand::FIRST_ITEM,
            need_reset_child: 1,
        }))
    }
}
