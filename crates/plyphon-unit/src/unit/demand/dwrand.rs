//! `Dwrand` - a weighted random-selection demand source, plyphon's port of scsynth's `Dwrand`
//! (`DemandUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Dwrand(list, weights, repeats)`: yields `repeats` values, each from an item picked at random
/// with the given weights, then `NaN`. Inputs are `[repeats, weightsSize, weights..., items...]`,
/// the layout the language builds (it pads `weights` with zeros to the list's size).
///
/// `weightsSize` is read once, when the synth is constructed; the items start after that many
/// weights. A pick draws one uniform `r` in `[0, 1)` from the synth's random stream and selects the
/// first item whose running weight sum reaches `r`, reading one weight per item; if no sum reaches
/// `r` the previous pick stands. An item is picked when the unit is constructed or reset, and again
/// after each value (a nested demand item is pulled until it yields `NaN`, and reset the next time it
/// is picked). `repeats` is latched on the first demand.
///
/// Weights and `weightsSize` are read without pulling them. A demand-rate input there reads as `0`,
/// where scsynth reads the source's last output. A pick that would read a weight past the last input
/// reads it as `0`. scsynth leaves the pick uninitialised when the constructor's pick selects
/// nothing; plyphon yields `NaN` until a pick succeeds.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dwrand {
    /// Latched repeat count; `-1` until the first demand latches it.
    repeats: f64,
    /// How many values have been yielded so far.
    repeat_count: i32,
    /// The picked item input; `-1` until a pick succeeds.
    index: i32,
    /// Whether the picked item should be reset before its next pull.
    need_reset_child: u32,
    /// The number of weights, read when the synth is constructed.
    weights_size: i32,
}

impl Dwrand {
    const REPEATS: usize = 0;
    const WEIGHTS_SIZE: usize = 1;
    const FIRST_WEIGHT: i32 = 2;

    /// Input `k`'s value without pulling it (scsynth's `IN0`).
    fn in0(ctx: &mut DemandCtx<'_>, k: i32) -> f32 {
        if k < 0 || k as usize >= ctx.num_inputs() || ctx.is_demand(k as usize) {
            0.0
        } else {
            ctx.demand(k as usize)
        }
    }

    /// Pick an item: draw `r` and select the first item whose running weight sum reaches it.
    fn pick(&mut self, ctx: &mut DemandCtx<'_>) {
        let offset = self.weights_size.wrapping_add(2);
        let num_items = (ctx.num_inputs() as i32).wrapping_sub(offset);
        let r = ctx.rgen().next_unipolar();
        let mut sum = 0.0f32;
        for i in 0..num_items {
            sum += Self::in0(ctx, Self::FIRST_WEIGHT + i);
            if sum >= r {
                self.index = i.wrapping_add(offset);
                break;
            }
        }
    }
}

impl DemandUnit for Dwrand {
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        self.weights_size = Self::in0(ctx, Self::WEIGHTS_SIZE as i32) as i32;
        self.reset(ctx);
    }

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
        self.need_reset_child = 1;
        self.pick(ctx);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        if self.repeats < 0.0 {
            let x = ctx.demand(Self::REPEATS);
            self.repeats = if x.is_nan() {
                0.0
            } else {
                math::floor(x + 0.5) as f64
            };
        }
        loop {
            if self.repeat_count as f64 >= self.repeats {
                return f32::NAN;
            }
            if self.index < 0 || self.index as usize >= ctx.num_inputs() {
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
                    self.pick(ctx);
                    self.repeat_count = self.repeat_count.wrapping_add(1);
                    self.need_reset_child = 1;
                } else {
                    return x;
                }
            } else {
                let x = ctx.demand(k);
                self.pick(ctx);
                self.repeat_count = self.repeat_count.wrapping_add(1);
                self.need_reset_child = 1;
                return x;
            }
        }
    }
}

/// Constructor for [`Dwrand`]. Its first item is picked when the synth is constructed.
pub struct DwrandCtor;

impl DemandUnitDef for DwrandCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dwrand {
            repeats: -1.0,
            repeat_count: 0,
            index: -1,
            need_reset_child: 1,
            weights_size: 0,
        }))
    }
}
