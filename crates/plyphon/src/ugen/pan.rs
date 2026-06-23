//! `Pan2` - equal-power stereo panner, plyphon's port of scsynth's `Pan2`.

use std::f32::consts::FRAC_PI_4;

use crate::error::BuildError;
use crate::io::Io;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{DoneAction, Inputs, Outputs, ProcessContext, Ugen};

/// `Pan2.ar(in, pos, level)`: pan a mono signal across two channels with an equal-power law - `pos`
/// runs -1 (hard left) to +1 (hard right), `level` (default 1) scales. Has two outputs (left, right);
/// `pos`/`level` are taken at control rate (constant over the block).
pub struct Pan2;

impl Pan2 {
    const IN: usize = 0;
    const POS: usize = 1;
    const LEVEL: usize = 2;
}

impl Ugen for Pan2 {
    fn process(
        &mut self,
        _ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        _io: &mut Io,
    ) -> DoneAction {
        let pos = ins.control(Self::POS).clamp(-1.0, 1.0);
        let level = if ins.len() > Self::LEVEL {
            ins.control(Self::LEVEL)
        } else {
            1.0
        };
        // pos -1 -> angle 0 (all left); pos +1 -> angle pi/2 (all right).
        let angle = (pos + 1.0) * FRAC_PI_4;
        let (left_gain, right_gain) = (angle.cos() * level, angle.sin() * level);
        let input = ins.audio(Self::IN);
        for (o, &x) in outs.audio(0).iter_mut().zip(input) {
            *o = x * left_gain;
        }
        for (o, &x) in outs.audio(1).iter_mut().zip(input) {
            *o = x * right_gain;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Pan2`].
pub struct Pan2Ctor;

impl UgenCtor for Pan2Ctor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        Ok(Box::new(Pan2))
    }
}
