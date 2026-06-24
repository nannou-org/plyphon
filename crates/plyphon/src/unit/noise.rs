//! `WhiteNoise` - uniform white noise, plyphon's port of scsynth's `WhiteNoise`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::rate::Rate;
use crate::rng::Rng;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// `WhiteNoise.ar/kr`: samples drawn uniformly from `[-1, 1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WhiteNoise {
    rng: Rng,
    /// `0`/`1`: audio-rate (a full block) vs control-rate (one value).
    audio: u32,
}

impl Unit for WhiteNoise {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.audio != 0 {
            for o in ctx.outs.audio(0).iter_mut() {
                *o = self.rng.next_bipolar();
            }
        } else {
            *ctx.outs.control(0) = self.rng.next_bipolar();
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`WhiteNoise`].
pub struct WhiteNoiseCtor;

impl UnitDef for WhiteNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(WhiteNoise {
            rng: Rng::new(ctx.seed),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
