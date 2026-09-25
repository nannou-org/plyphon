//! `Dxrand` - a random-selection demand source with no immediate repeats, plyphon's port of
//! scsynth's `Dxrand`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// A random item index in `1..num_inputs` that is never equal to `current` - scsynth's
/// `irand(mNumInputs - 2) + 1`, remapped around the current index, which draws even when there is
/// no alternative. With one item the result can fall past the end, which the next pull wraps back
/// to the first item.
fn pick_skip(ctx: &mut DemandCtx<'_>, current: u32) -> u32 {
    let n = ctx.num_inputs() as i32;
    let newindex = ctx.rgen().next_irand(n - 2) + 1;
    let index = if newindex < current as i32 {
        newindex
    } else {
        newindex + 1
    };
    index.max(0) as u32
}

/// `Dxrand(length, items...)`: like [`Drand`](super::drand::Drand) but never picks the same item
/// twice in a row - yields `length` values, then `NaN`. Input `0` is `length`; inputs `1..` are the
/// items. Picks are drawn from the synth's random stream.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dxrand {
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted so far.
    repeat_count: u32,
    /// Index of the current item input (`1..num_inputs`). `0` before the constructor's first pick,
    /// where scsynth compares against memory it has not initialised.
    index: u32,
    /// Whether the child at `index` should be reset before its next pull.
    need_reset_child: u32,
}

impl Dxrand {
    const LENGTH: usize = 0;
    const FIRST_ITEM: u32 = 1;
}

impl DemandUnit for Dxrand {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
        self.need_reset_child = 1;
        self.index = pick_skip(ctx, self.index);
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
                    self.index = pick_skip(ctx, self.index);
                    self.repeat_count += 1;
                    self.need_reset_child = 1;
                } else {
                    return x;
                }
            } else {
                let x = ctx.demand(k);
                self.index = pick_skip(ctx, self.index);
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

/// Constructor for [`Dxrand`]. Its first item is picked when the synth is constructed (its reset).
pub struct DxrandCtor;

impl DemandUnitDef for DxrandCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dxrand {
            repeats: -1.0,
            repeat_count: 0,
            index: 0,
            need_reset_child: 1,
        }))
    }
}
