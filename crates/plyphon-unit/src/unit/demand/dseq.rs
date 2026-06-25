//! `Dseq` - a sequence demand source, plyphon's port of scsynth's `Dseq`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Dseq(repeats, items...)`: on each demand, yields the items in order, looping `repeats` times, then
/// `NaN`. Inputs are in scsynth's server-side order: input `0` is `repeats`, inputs `1..` are the list
/// items. An item that is itself a demand source is *pulled until it yields `NaN`* (and reset the next
/// time the sequence reaches it), which is what lets sequences nest; a constant or wire item yields
/// its value once. This is plyphon's port of `Dseq_next`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dseq {
    /// Latched repeat count; `-1` until the first demand latches it (scsynth's `m_repeats < 0`).
    repeats: f32,
    /// How many full passes over the list have completed.
    repeat_count: u32,
    /// Index of the current item input (`1..num_inputs`).
    index: u32,
    /// Whether the child at `index` should be reset before its next pull (scsynth's
    /// `m_needToResetChild`).
    need_reset_child: u32,
}

impl Dseq {
    const REPEATS: usize = 0;
    /// The first item input; items occupy `1..num_inputs`.
    const FIRST_ITEM: u32 = 1;
}

impl DemandUnit for Dseq {
    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
        self.index = Self::FIRST_ITEM;
        self.need_reset_child = 1;
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let num_inputs = ctx.num_inputs();
        // No items (only the `repeats` input): nothing to sequence.
        if num_inputs <= Self::FIRST_ITEM as usize {
            return f32::NAN;
        }
        if self.repeats < 0.0 {
            let x = ctx.demand(Self::REPEATS);
            self.repeats = if x.is_nan() {
                0.0
            } else {
                math::floor(x + 0.5)
            };
        }
        // A degenerate list whose items all yield `NaN` immediately (e.g. zero-length children) would
        // spin forever under an infinite `repeats`; bound the scan so the audio thread never hangs.
        let guard_limit = num_inputs.saturating_mul(2) + 4;
        let mut guard = 0;
        loop {
            if self.index as usize >= num_inputs {
                self.index = Self::FIRST_ITEM;
                self.repeat_count += 1;
            }
            if (self.repeat_count as f32) >= self.repeats {
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
                    // Child exhausted: advance, and reset it the next time we reach it.
                    self.index += 1;
                    self.need_reset_child = 1;
                } else {
                    return x;
                }
            } else {
                // A scalar/wire item yields its value once, then we advance.
                let x = ctx.demand(k);
                self.index += 1;
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

/// Constructor for [`Dseq`].
pub struct DseqCtor;

impl DemandUnitDef for DseqCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dseq {
            repeats: -1.0,
            repeat_count: 0,
            index: Dseq::FIRST_ITEM,
            need_reset_child: 1,
        }))
    }
}
