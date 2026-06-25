//! `Dwhite` - a uniform-random demand source, plyphon's port of scsynth's `Dwhite`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;
use plyphon_dsp::rng::Rng;

/// `Dwhite(length, lo, hi)`: on each demand, yields a value drawn uniformly from `[lo, hi)`, for
/// `length` values, then `NaN`. `length` is latched on the first demand; `lo`/`hi` are re-read each
/// demand. Inputs are in scsynth's server-side order: `[length, lo, hi]`. Like `WhiteNoise` it carries
/// its own [`Rng`], re-seeded per instance so two synths of the same def decorrelate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dwhite {
    rng: Rng,
    /// Latched length; `-1` until the first demand latches it.
    repeats: f32,
    /// How many values have been emitted this run.
    repeat_count: u32,
}

impl Dwhite {
    const LENGTH: usize = 0;
    const LO: usize = 1;
    const HI: usize = 2;
}

impl DemandUnit for Dwhite {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn reset(&mut self, _ctx: &mut DemandCtx<'_>) {
        self.repeats = -1.0;
        self.repeat_count = 0;
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        if self.repeats < 0.0 {
            let len = ctx.demand(Self::LENGTH);
            self.repeats = if len.is_nan() {
                0.0
            } else {
                math::floor(len + 0.5)
            };
        }
        if (self.repeat_count as f32) >= self.repeats {
            return f32::NAN;
        }
        self.repeat_count += 1;
        let lo = ctx.demand(Self::LO);
        let hi = ctx.demand(Self::HI);
        lo + self.rng.next_unipolar() * (hi - lo)
    }
}

/// Constructor for [`Dwhite`].
pub struct DwhiteCtor;

impl DemandUnitDef for DwhiteCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec(Dwhite {
            rng: Rng::new(ctx.seed),
            repeats: -1.0,
            repeat_count: 0,
        }))
    }
}
