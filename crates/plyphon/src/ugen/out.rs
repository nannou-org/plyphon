//! `Out` - writes signals to audio or control bus channels, plyphon's port of scsynth's `Out`.

use crate::bus::Buses;
use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{DoneAction, Inputs, Outputs, ProcessContext, Ugen};

/// `Out.ar(bus, signals)` / `Out.kr(bus, signals)`: writes each signal input to a consecutive bus
/// channel starting at `bus`, summing with anything already written to that channel this block.
/// `Out.ar` targets the audio bus bank, `Out.kr` the control bus bank, chosen by the UGen's rate.
pub struct Out {
    audio: bool,
}

impl Ugen for Out {
    fn process(
        &mut self,
        ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        _outs: &mut Outputs<'_>,
        buses: &mut Buses,
    ) -> DoneAction {
        if ins.is_empty() {
            return DoneAction::Nothing;
        }
        // Input 0 is the starting bus channel; the rest are signals to write.
        let base = ins.control(0) as usize;
        if self.audio {
            for k in 1..ins.len() {
                buses
                    .audio_mut()
                    .write_accumulate(base + (k - 1), ctx.buf_counter, ins.audio(k));
            }
        } else {
            for k in 1..ins.len() {
                buses.control_mut().write_accumulate(
                    base + (k - 1),
                    ctx.buf_counter,
                    ins.control(k),
                );
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Out`].
pub struct OutCtor;

impl UgenCtor for OutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        Ok(Box::new(Out {
            audio: ctx.rate == Rate::Audio,
        }))
    }
}
