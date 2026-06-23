//! `WhiteNoise` - uniform white noise, plyphon's port of scsynth's `WhiteNoise`.

use crate::error::BuildError;
use crate::rate::Rate;
use crate::rng::Rng;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{DoneAction, ProcessCtx, Ugen};

/// `WhiteNoise.ar/kr`: samples drawn uniformly from `[-1, 1)`.
pub struct WhiteNoise {
    rng: Rng,
    audio: bool,
}

impl Ugen for WhiteNoise {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.audio {
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

impl UgenCtor for WhiteNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        Ok(Box::new(WhiteNoise {
            rng: Rng::new(ctx.seed),
            audio: ctx.rate == Rate::Audio,
        }))
    }
}
