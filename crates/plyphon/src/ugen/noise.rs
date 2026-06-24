//! `WhiteNoise` - uniform white noise, plyphon's port of scsynth's `WhiteNoise`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::rate::Rate;
use crate::rng::Rng;
use crate::ugen::registry::{BuildContext, UgenDef};
use crate::ugen::{BuiltUgen, DoneAction, ProcessCtx, Ugen, ugen_spec};

/// `WhiteNoise.ar/kr`: samples drawn uniformly from `[-1, 1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WhiteNoise {
    rng: Rng,
    /// `0`/`1`: audio-rate (a full block) vs control-rate (one value).
    audio: u32,
}

impl Ugen for WhiteNoise {
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

impl UgenDef for WhiteNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError> {
        Ok(ugen_spec(WhiteNoise {
            rng: Rng::new(ctx.seed),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
