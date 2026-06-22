//! `Line` - a line generator that ramps from a start to an end value over a duration.

use crate::bus::AudioBus;
use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{Inputs, Outputs, ProcessContext, Ugen};

/// `Line.ar/kr(start, end, dur)`: ramps linearly from `start` to `end` over `dur` seconds, then
/// holds at `end`. The arguments are latched on the first block (as in SuperCollider). At control
/// rate it advances once per block (producing one output value); at audio rate, once per sample.
pub struct Line {
    audio: bool,
    started: bool,
    value: f64,
    end: f64,
    slope: f64,
    remaining: f64,
}

impl Line {
    const START: usize = 0;
    const END: usize = 1;
    const DUR: usize = 2;

    fn advance(&mut self) {
        if self.remaining > 0.0 {
            self.value += self.slope;
            self.remaining -= 1.0;
        } else {
            self.value = self.end;
        }
    }
}

impl Ugen for Line {
    fn process(
        &mut self,
        ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        _out_bus: &mut AudioBus,
    ) {
        if !self.started {
            let start = ins.control(Self::START) as f64;
            let end = ins.control(Self::END) as f64;
            let dur = (ins.control(Self::DUR) as f64).max(0.0);
            // Frames to ramp over: samples at audio rate, control blocks at control rate.
            let rate = if self.audio {
                ctx.audio.sample_rate
            } else {
                ctx.control.sample_rate
            };
            let frames = (dur * rate).max(1.0);
            self.value = start;
            self.end = end;
            self.slope = (end - start) / frames;
            self.remaining = frames;
            self.started = true;
        }
        if self.audio {
            let out = outs.audio(0);
            for o in out.iter_mut() {
                *o = self.value as f32;
                self.advance();
            }
        } else {
            *outs.control(0) = self.value as f32;
            self.advance();
        }
    }
}

/// Constructor for [`Line`].
pub struct LineCtor;

impl UgenCtor for LineCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        // start, end, dur (SuperCollider's trailing doneAction input is accepted but ignored).
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(Box::new(Line {
            audio: ctx.rate == Rate::Audio,
            started: false,
            value: 0.0,
            end: 0.0,
            slope: 0.0,
            remaining: 0.0,
        }))
    }
}
