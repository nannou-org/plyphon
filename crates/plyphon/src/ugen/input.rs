//! `In` - reads signals from audio or control bus channels, plyphon's port of scsynth's `In`.

use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{self, DoneAction, ProcessCtx, Ugen};

/// `In.ar(bus, numChannels)` / `In.kr(bus, numChannels)`: reads `numChannels` consecutive bus
/// channels starting at `bus`, one per output. `In.ar` reads the audio bus bank, `In.kr` the
/// control bus bank, chosen by the UGen's rate. The number of channels is fixed at build time (it
/// determines how many outputs the UGen has). Channels past the end of the bus read as zero.
pub struct In {
    audio: bool,
    num_channels: usize,
}

impl In {
    const BUS: usize = 0;
}

impl Ugen for In {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let base = ctx.ins.control(Self::BUS) as usize;
        if self.audio {
            for o in 0..self.num_channels {
                let dst = ctx.outs.audio(o);
                let chan = ugen::audio_in(ctx.buses, base + o);
                if chan.len() == dst.len() {
                    dst.copy_from_slice(chan);
                } else {
                    dst.fill(0.0);
                }
            }
        } else {
            for o in 0..self.num_channels {
                *ctx.outs.control(o) = ugen::control_in(ctx.buses, base + o);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`In`].
pub struct InCtor;

impl UgenCtor for InCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        Ok(Box::new(In {
            audio: ctx.rate == Rate::Audio,
            num_channels: ctx.num_outputs,
        }))
    }
}
