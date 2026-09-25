//! `Dswitch1` and `Dswitch` - index-selected demand sources, plyphon's port of scsynth's `Dswitch1`
//! and `Dswitch` (`DemandUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::{math, ops};

/// The index input; the list items occupy `1..num_inputs`.
const INDEX: usize = 0;

/// The item input an index value selects: `(int32)floor(x + 0.5f)` wrapped into `[0, hi]`, plus one
/// to skip the index input. The float-to-int conversion saturates (and maps `NaN` to `0`).
fn item(x: f32, hi: i32) -> i32 {
    ops::iwrap(math::floor(x + 0.5) as i32, 0, hi) + 1
}

/// Demand input `k`. Only an item index can land past the last input (see [`Dswitch`]); scsynth then
/// reads the unit's own output wire and recurses into itself without end, so plyphon yields `NaN`
/// there instead.
fn demand_item(ctx: &mut DemandCtx<'_>, k: i32) -> f32 {
    if (k as usize) < ctx.num_inputs() {
        ctx.demand(k as usize)
    } else {
        f32::NAN
    }
}

/// `Dswitch1(list, index)`: on each demand, pulls `index` and yields the next value of the item it
/// selects - one value, whether or not that item is a demand source. Inputs are `[index, items...]`.
///
/// The index is rounded to the nearest integer and wrapped into the list; a `NaN` index is yielded
/// as is. A reset resets every input. Unlike most demand sources it is not reset when the synth is
/// constructed. With no items the selection lands past the last input, where scsynth recurses into
/// itself without end; plyphon yields `NaN` instead.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dswitch1 {
    _pad: u32,
}

impl DemandUnit for Dswitch1 {
    fn init(&mut self, _ctx: &mut DemandCtx<'_>) {}

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        for k in 0..ctx.num_inputs() {
            ctx.reset(k);
        }
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let x = ctx.demand(INDEX);
        if x.is_nan() {
            return x;
        }
        let index = item(x, ctx.num_inputs() as i32 - 2);
        demand_item(ctx, index)
    }
}

/// Constructor for [`Dswitch1`].
pub struct Dswitch1Ctor;

impl DemandUnitDef for Dswitch1Ctor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dswitch1 { _pad: 0 }))
    }
}

/// `Dswitch(list, index)`: yields values from the selected item until it is exhausted, then pulls
/// `index` to select the next item. Inputs are `[index, items...]`.
///
/// The first item is selected when the synth is constructed, and again on every reset (after every
/// input is reset), from one value of `index`. On each demand the selected item is pulled; when it
/// yields `NaN`, `index` is pulled: a `NaN` index is yielded as is, otherwise the newly selected item
/// is pulled for the value, and then the previously selected item is reset.
///
/// scsynth wraps the constructor's and reset's index into `[0, num_inputs - 1]` but a demand's index
/// into `[0, num_inputs - 2]`, so the first selection can land one past the last item. scsynth then
/// reads the unit's own output wire and recurses into itself without end; plyphon treats that
/// selection as an exhausted item (whose reset is a no-op), so the demand pulls `index` again.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dswitch {
    /// The selected item input.
    index: i32,
}

impl Dswitch {
    /// Select the first item from one value of `index` (the constructor and reset).
    fn select_first(&mut self, ctx: &mut DemandCtx<'_>) {
        let x = ctx.demand(INDEX);
        self.index = item(x, ctx.num_inputs() as i32 - 1);
    }
}

impl DemandUnit for Dswitch {
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        self.select_first(ctx);
    }

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        for k in 0..ctx.num_inputs() {
            ctx.reset(k);
        }
        self.select_first(ctx);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let mut val = demand_item(ctx, self.index);
        if val.is_nan() {
            let ival = ctx.demand(INDEX);
            if ival.is_nan() {
                val = ival;
            } else {
                let index = item(ival, ctx.num_inputs() as i32 - 2);
                val = demand_item(ctx, index);
                if (self.index as usize) < ctx.num_inputs() {
                    ctx.reset(self.index as usize);
                }
                self.index = index;
            }
        }
        val
    }
}

/// Constructor for [`Dswitch`]. Its first item is selected when the synth is constructed.
pub struct DswitchCtor;

impl DemandUnitDef for DswitchCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dswitch { index: 1 }))
    }
}
