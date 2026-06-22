//! `Out` - writes signals to output-bus channels, plyphon's port of scsynth's `Out`.

use crate::bus::AudioBus;
use crate::error::BuildError;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{Inputs, Outputs, ProcessContext, Ugen};

/// `Out.ar(bus, channelsArray)`: writes each signal input to a consecutive output-bus channel
/// starting at `bus`, summing with anything already written to that channel this block.
pub struct Out;

impl Ugen for Out {
    fn process(
        &mut self,
        ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        _outs: &mut Outputs<'_>,
        out_bus: &mut AudioBus,
    ) {
        if ins.is_empty() {
            return;
        }
        // Input 0 is the starting bus channel; the rest are signals to write.
        let base = ins.control(0) as usize;
        let num_channels = out_bus.num_channels();
        for k in 1..ins.len() {
            let ch = base + (k - 1);
            if ch < num_channels {
                let signal = ins.audio(k);
                out_bus.write_accumulate(ch, ctx.buf_counter, signal);
            }
        }
    }
}

/// Constructor for [`Out`].
pub struct OutCtor;

impl UgenCtor for OutCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        Ok(Box::new(Out))
    }
}
