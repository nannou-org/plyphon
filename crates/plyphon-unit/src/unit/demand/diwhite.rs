//! `Diwhite` - a uniform integer-random demand source, plyphon's port of scsynth's `Diwhite`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;
use plyphon_dsp::rng::Rng;

/// `Diwhite(length, lo, hi)`: like `Dwhite` but yields integers drawn uniformly from `[lo, hi]`
/// (inclusive, rounded), for `length` values, then `NaN`. `length` is latched on the first demand;
/// `lo`/`hi` are re-read each demand. Inputs are in scsynth's order `[length, lo, hi]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Diwhite {
    rng: Rng,
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted this run.
    repeat_count: u32,
    /// The current low bound (rounded to an integer).
    lo: i32,
    /// The inclusive range width `hi - lo + 1`.
    range: i32,
}

impl Diwhite {
    const LENGTH: usize = 0;
    const LO: usize = 1;
    const HI: usize = 2;
}

impl DemandUnit for Diwhite {
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
            self.lo = math::floor(lo + 0.5) as i32;
        }
        let hi = ctx.demand(Self::HI);
        if !hi.is_nan() {
            self.range = math::floor(hi + 0.5) as i32 - self.lo + 1;
        }
        if self.repeats < 0.0 {
            let len = ctx.demand(Self::LENGTH);
            self.repeats = if len.is_nan() {
                0.0
            } else {
                math::floor(len + 0.5)
            };
        }
        if self.repeat_count as f32 >= self.repeats {
            return f32::NAN;
        }
        self.repeat_count += 1;
        (self.rng.next_irand(self.range) + self.lo) as f32
    }
}

/// Constructor for [`Diwhite`].
pub struct DiwhiteCtor;

impl DemandUnitDef for DiwhiteCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Diwhite {
            rng: Rng::new(ctx.seed),
            repeats: -1.0,
            repeat_count: 0,
            lo: 0,
            range: 1,
        }))
    }
}
