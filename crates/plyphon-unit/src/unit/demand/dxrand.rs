//! `Dxrand` - a random-selection demand source with no immediate repeats, plyphon's port of
//! scsynth's `Dxrand`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;
use plyphon_dsp::rng::Rng;

/// A uniformly random item index in `1..num_inputs` (used for the very first pick).
fn pick(rng: &mut Rng, num_inputs: usize) -> u32 {
    rng.next_irand((num_inputs - 1) as i32) as u32 + 1
}

/// A random item index in `1..num_inputs` that is never equal to `current` (scsynth's
/// `irand(n-2)+1`, remapped around the current index). With fewer than two items there is no
/// alternative, so the single item is returned.
fn pick_skip(rng: &mut Rng, num_inputs: usize, current: u32) -> u32 {
    if num_inputs <= Dxrand::FIRST_ITEM as usize + 1 {
        return Dxrand::FIRST_ITEM;
    }
    let newindex = rng.next_irand((num_inputs - 2) as i32) as u32 + 1;
    if newindex < current {
        newindex
    } else {
        newindex + 1
    }
}

/// `Dxrand(length, items...)`: like [`Drand`](super::drand::Drand) but never picks the same item
/// twice in a row - yields `length` values, then `NaN`. Input `0` is `length`; inputs `1..` are the
/// items. Carries its own [`Rng`].
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dxrand {
    rng: Rng,
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted so far.
    repeat_count: u32,
    /// Index of the current item input (`1..num_inputs`).
    index: u32,
    /// Whether the child at `index` should be reset before its next pull.
    need_reset_child: u32,
}

impl Dxrand {
    const LENGTH: usize = 0;
    const FIRST_ITEM: u32 = 1;
}

impl DemandUnit for Dxrand {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
        self.need_reset_child = 1;
        let n = ctx.num_inputs();
        self.index = if n > Self::FIRST_ITEM as usize {
            pick_skip(&mut self.rng, n, self.index)
        } else {
            Self::FIRST_ITEM
        };
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
            if self.index as usize >= num_inputs {
                self.index = pick_skip(&mut self.rng, num_inputs, self.index);
            }
            let k = self.index as usize;
            if ctx.is_demand(k) {
                if self.need_reset_child != 0 {
                    self.need_reset_child = 0;
                    ctx.reset(k);
                }
                let x = ctx.demand(k);
                if x.is_nan() {
                    self.index = pick_skip(&mut self.rng, num_inputs, self.index);
                    self.repeat_count += 1;
                    self.need_reset_child = 1;
                } else {
                    return x;
                }
            } else {
                let x = ctx.demand(k);
                self.index = pick_skip(&mut self.rng, num_inputs, self.index);
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

/// Constructor for [`Dxrand`].
pub struct DxrandCtor;

impl DemandUnitDef for DxrandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        let mut rng = Rng::new(ctx.seed);
        let n = ctx.input_rates.len();
        let index = if n > Dxrand::FIRST_ITEM as usize {
            pick(&mut rng, n)
        } else {
            Dxrand::FIRST_ITEM
        };
        Ok(demand_unit_spec(Dxrand {
            rng,
            repeats: -1.0,
            repeat_count: 0,
            index,
            need_reset_child: 1,
        }))
    }
}
