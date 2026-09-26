//! `Dshuf` - a shuffled-sequence demand source, plyphon's port of scsynth's `Dshuf`
//! (`DemandUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec_aux};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Dshuf(list, repeats)`: shuffles the list once, then yields its items in that order, `repeats`
/// times over, then `NaN`. Inputs are `[repeats, items...]`.
///
/// The order is a table of item input indices in the unit's aux memory, filled with `1..` when the
/// synth is constructed and shuffled by every reset (the constructor's included) - a Fisher-Yates
/// pass drawing from the synth's random stream, applied to the current order. `repeats` is latched
/// on the first demand. As in `Dseq`, a demand item is pulled until it yields `NaN` (and reset the
/// next time the sequence reaches it); any other item yields its value once. Every pass plays the
/// same order; only a reset shuffles again.
///
/// scsynth allocates the table in the constructor and silences the unit if that fails; plyphon
/// reserves it when the SynthDef is compiled, so there is no failure at run time.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dshuf {
    /// Latched repeat count; `-1` until the first demand latches it.
    repeats: f64,
    /// How many full passes over the list have completed.
    repeat_count: i32,
    /// Position in the shuffled order (`0..num_items`).
    index: i32,
    /// Whether the item at `index` should be reset before its next pull.
    need_reset_child: u32,
    _pad: u32,
}

impl Dshuf {
    const REPEATS: usize = 0;

    /// Shuffle the order in place (scsynth's `Dshuf_scramble`): for each position but the last, swap
    /// it with a uniformly drawn position at or after it.
    fn scramble(ctx: &mut DemandCtx<'_>) {
        let size = ctx.num_inputs() as i32 - 1;
        if size > 1 {
            let mut m = size;
            for i in 0..size - 1 {
                let j = i + ctx.rgen().next_irand(m);
                ctx.aux_mut::<i32>().swap(i as usize, j as usize);
                m -= 1;
            }
        }
    }
}

impl DemandUnit for Dshuf {
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        for (i, v) in ctx.aux_mut::<i32>().iter_mut().enumerate() {
            *v = i as i32 + 1;
        }
        self.reset(ctx);
    }

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
        self.need_reset_child = 1;
        self.index = 0;
        Self::scramble(ctx);
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
        let num_items = ctx.num_inputs() - 1;
        loop {
            if self.index as usize >= num_items {
                self.index = 0;
                self.repeat_count = self.repeat_count.wrapping_add(1);
            }
            if self.repeat_count as f64 >= self.repeats {
                self.index = 0;
                return f32::NAN;
            }
            // The table holds one entry per item, so this lookup always succeeds.
            let Some(&k) = ctx.aux_mut::<i32>().get(self.index as usize) else {
                return f32::NAN;
            };
            let k = k as usize;
            if ctx.is_demand(k) {
                if self.need_reset_child != 0 {
                    self.need_reset_child = 0;
                    ctx.reset(k);
                }
                let x = ctx.demand(k);
                if x.is_nan() {
                    self.index += 1;
                    self.need_reset_child = 1;
                } else {
                    return x;
                }
            } else {
                let x = ctx.demand(k);
                self.index += 1;
                self.need_reset_child = 1;
                return x;
            }
        }
    }
}

/// Constructor for [`Dshuf`]: reserves its order table, one `i32` per item.
pub struct DshufCtor;

impl DemandUnitDef for DshufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        let num_items = ctx
            .input_rates
            .len()
            .checked_sub(1)
            .ok_or(BuildError::WrongInputCount)?;
        Ok(demand_unit_spec_aux(
            Dshuf {
                repeats: -1.0,
                repeat_count: 0,
                index: 0,
                need_reset_child: 1,
                _pad: 0,
            },
            num_items * core::mem::size_of::<i32>(),
            core::mem::align_of::<i32>(),
        ))
    }
}
