//! `Out` - writes signals to audio or control bus channels, plyphon's port of scsynth's `Out`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenDef};
use crate::ugen::{self, BuiltUgen, DoneAction, ProcessCtx, Ugen, ugen_spec};

/// `Out.ar(bus, signals)` / `Out.kr(bus, signals)`: writes each signal input to a consecutive bus
/// channel starting at `bus`, summing with anything already written to that channel this block.
/// `Out.ar` targets the audio bus bank, `Out.kr` the control bus bank, chosen by the UGen's rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Out {
    /// `0`/`1`: whether this writes the audio (`Out.ar`) or control (`Out.kr`) bus bank.
    audio: u32,
}

impl Ugen for Out {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.ins.is_empty() {
            return DoneAction::Nothing;
        }
        // Input 0 is the starting bus channel; the rest are signals to write.
        let base = ctx.ins.control(0) as usize;
        if self.audio != 0 {
            for k in 1..ctx.ins.len() {
                ugen::audio_out(ctx.buses, ctx.buf_counter, base + (k - 1), ctx.ins.audio(k));
            }
        } else {
            for k in 1..ctx.ins.len() {
                ugen::control_out(
                    ctx.buses,
                    ctx.buf_counter,
                    base + (k - 1),
                    ctx.ins.control(k),
                );
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Out`].
pub struct OutCtor;

impl UgenDef for OutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError> {
        Ok(ugen_spec(Out {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
