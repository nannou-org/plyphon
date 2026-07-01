//! `Dibrown` - an integer Brownian-motion demand source, plyphon's port of scsynth's `Dibrown`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;
use plyphon_dsp::ops::ifold;
use plyphon_dsp::rng::Rng;

/// `Dibrown(length, lo, hi, step)`: like `Dbrown` but on the integers - each demand steps the value by
/// a random integer in `[-step, step]` and folds it back into `[lo, hi]`, for `length` values, then
/// `NaN`. `length` is latched on the first demand; `lo`/`hi`/`step` are re-read each demand. Inputs
/// are in scsynth's order `[length, lo, hi, step]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dibrown {
    rng: Rng,
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted this run.
    repeat_count: u32,
    /// The current walk value.
    val: i32,
    /// Current low/high bounds and step size (re-read each demand, truncated to integers).
    lo: i32,
    hi: i32,
    step: i32,
}

impl Dibrown {
    const LENGTH: usize = 0;
    const LO: usize = 1;
    const HI: usize = 2;
    const STEP: usize = 3;
}

impl DemandUnit for Dibrown {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let lo = ctx.demand(Self::LO);
        if !lo.is_nan() {
            self.lo = lo as i32;
        }
        let hi = ctx.demand(Self::HI);
        if !hi.is_nan() {
            self.hi = hi as i32;
        }
        let step = ctx.demand(Self::STEP);
        if !step.is_nan() {
            self.step = step as i32;
        }
        if self.repeats < 0.0 {
            let len = ctx.demand(Self::LENGTH);
            self.repeats = if len.is_nan() {
                0.0
            } else {
                math::floor(len + 0.5)
            };
            self.val = self.rng.next_irand(self.hi - self.lo + 1) + self.lo;
        }
        if self.repeat_count as f32 >= self.repeats {
            return f32::NAN;
        }
        self.repeat_count += 1;
        let out = self.val;
        let stepped = self.val + self.rng.next_irand2(self.step);
        self.val = ifold(stepped, self.lo, self.hi);
        out as f32
    }
}

/// Constructor for [`Dibrown`].
pub struct DibrownCtor;

impl DemandUnitDef for DibrownCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dibrown {
            rng: Rng::new(ctx.seed),
            repeats: -1.0,
            repeat_count: 0,
            val: 0,
            lo: 0,
            hi: 0,
            step: 0,
        }))
    }
}
