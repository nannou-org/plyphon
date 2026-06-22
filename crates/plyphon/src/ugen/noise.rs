//! `WhiteNoise` - uniform white noise, plyphon's port of scsynth's `WhiteNoise`.

use crate::bus::AudioBus;
use crate::error::BuildError;
use crate::rate::Rate;
use crate::rng::Rng;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{Inputs, Outputs, ProcessContext, Ugen};

/// `WhiteNoise.ar/kr`: samples drawn uniformly from `[-1, 1)`.
pub struct WhiteNoise {
    rng: Rng,
    audio: bool,
}

impl Ugen for WhiteNoise {
    fn process(
        &mut self,
        _ctx: &ProcessContext<'_>,
        _ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        _out_bus: &mut AudioBus,
    ) {
        if self.audio {
            for o in outs.audio(0).iter_mut() {
                *o = self.rng.next_bipolar();
            }
        } else {
            *outs.control(0) = self.rng.next_bipolar();
        }
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
