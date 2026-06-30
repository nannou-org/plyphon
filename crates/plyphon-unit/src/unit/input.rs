//! `In` - reads signals from audio or control bus channels, plyphon's port of scsynth's `In`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// `In.ar(bus, numChannels)` / `In.kr(bus, numChannels)`: reads `numChannels` consecutive bus
/// channels starting at `bus`, one per output. `In.ar` reads the audio bus bank, `In.kr` the
/// control bus bank, chosen by the unit's rate. The number of channels is fixed at build time (it
/// determines how many outputs the unit has). Channels past the end of the bus read as zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct In {
    num_channels: u32,
    /// `0`/`1`: whether this reads the audio (`In.ar`) or control (`In.kr`) bus bank.
    audio: u32,
}

impl In {
    const BUS: usize = 0;
}

impl Unit for In {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let base = ctx.ins.control(Self::BUS) as usize;
        let num_channels = self.num_channels as usize;
        if self.audio != 0 {
            let factor = ctx.resample_factor;
            for o in 0..num_channels {
                let dst = ctx.outs.audio(o);
                // This sub-block tick reads its `dst.len() / factor` World-rate samples of the
                // World-block-wide bus channel and zero-order-holds them up to the wire's full length.
                // For an ordinary graph (`tick` 0, `factor` 1) this is a straight copy of the channel.
                let world_samples = dst.len() / factor;
                let offset = ctx.tick * world_samples;
                let chan = unit::audio_in(ctx.buses, base + o);
                if chan.len() >= offset + world_samples {
                    for (j, slot) in dst.iter_mut().enumerate() {
                        *slot = chan[offset + j / factor];
                    }
                } else {
                    dst.fill(0.0);
                }
            }
        } else {
            for o in 0..num_channels {
                *ctx.outs.control(o) = unit::control_in(ctx.buses, base + o);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`In`].
pub struct InCtor;

impl UnitDef for InCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(In {
            num_channels: ctx.num_outputs as u32,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
