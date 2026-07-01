//! `Dbrown` - a Brownian-motion demand source, plyphon's port of scsynth's `Dbrown`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;
use plyphon_dsp::ops;
use plyphon_dsp::rng::Rng;

/// `Dbrown(length, lo, hi, step)`: a bounded random walk - each demand steps the value by up to
/// `±step` (uniformly) and folds it back into `[lo, hi]`, for `length` values, then `NaN`. `length`
/// is latched on the first demand; `lo`/`hi`/`step` are re-read each demand. Inputs are in scsynth's
/// order `[length, lo, hi, step]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dbrown {
    rng: Rng,
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted this run.
    repeat_count: u32,
    /// The current walk value.
    val: f32,
    /// Current low/high bounds and step size (re-read each demand).
    lo: f32,
    hi: f32,
    step: f32,
}

impl Dbrown {
    const LENGTH: usize = 0;
    const LO: usize = 1;
    const HI: usize = 2;
    const STEP: usize = 3;
}

impl DemandUnit for Dbrown {
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
            self.lo = lo;
        }
        let hi = ctx.demand(Self::HI);
        if !hi.is_nan() {
            self.hi = hi;
        }
        let step = ctx.demand(Self::STEP);
        if !step.is_nan() {
            self.step = step;
        }
        if self.repeats < 0.0 {
            let len = ctx.demand(Self::LENGTH);
            self.repeats = if len.is_nan() {
                0.0
            } else {
                math::floor(len + 0.5)
            };
            self.val = self.lo + self.rng.next_unipolar() * (self.hi - self.lo);
        }
        if self.repeat_count as f32 >= self.repeats {
            return f32::NAN;
        }
        self.repeat_count += 1;
        let out = self.val;
        let stepped = self.val + self.rng.next_bipolar() * self.step;
        self.val = ops::fold(stepped, self.lo, self.hi);
        out
    }
}

/// Constructor for [`Dbrown`].
pub struct DbrownCtor;

impl DemandUnitDef for DbrownCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dbrown {
            rng: Rng::new(ctx.seed),
            repeats: -1.0,
            repeat_count: 0,
            val: 0.0,
            lo: 0.0,
            hi: 0.0,
            step: 0.0,
        }))
    }
}
