//! `Out`/`OffsetOut` - write signals to audio or control bus channels, plyphon's ports of scsynth's
//! `Out` and `OffsetOut`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// `Out.ar(bus, signals)` / `Out.kr(bus, signals)`: writes each signal input to a consecutive bus
/// channel starting at `bus`, summing with anything already written to that channel this block.
/// `Out.ar` targets the audio bus bank, `Out.kr` the control bus bank, chosen by the unit's rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Out {
    /// `0`/`1`: whether this writes the audio (`Out.ar`) or control (`Out.kr`) bus bank.
    audio: u32,
}

impl Unit for Out {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.ins.is_empty() {
            return DoneAction::Nothing;
        }
        // Input 0 is the starting bus channel; the rest are signals to write.
        let base = ctx.ins.control(0) as usize;
        if self.audio != 0 {
            for k in 1..ctx.ins.len() {
                unit::audio_out(ctx.buses, ctx.buf_counter, base + (k - 1), ctx.ins.audio(k));
            }
        } else {
            for k in 1..ctx.ins.len() {
                unit::control_out(
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

impl UnitDef for OutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Out {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `OffsetOut.ar(bus, signals)`: like [`Out`], but a synth created partway into a control block - by
/// a scheduled, time-tagged command - mutes the samples before its creation offset on its first
/// block, so the synth becomes audible at exactly the scheduled sample. plyphon's port of scsynth's
/// `OffsetOut`.
///
/// The offset is the within-block sample at which the synth was created (scsynth's `mSampleOffset`),
/// delivered as [`ProcessCtx::sample_offset`] - non-zero only on the synth's first block. Where
/// scsynth's `OffsetOut` shifts the whole signal by the offset for the synth's life (a one-block
/// delay that keeps every sample), plyphon's gates only the first block: the leading `offset`
/// samples are silenced rather than carried. For a synth that begins from silence - the usual
/// enveloped voice - the two are audibly identical, and gating needs no per-instance delay buffer.
/// `OffsetOut.kr` ignores the offset, since a control block is a single value.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OffsetOut {
    /// `0`/`1`: whether this writes the audio (`OffsetOut.ar`) or control (`OffsetOut.kr`) bus bank.
    audio: u32,
}

impl Unit for OffsetOut {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.ins.is_empty() {
            return DoneAction::Nothing;
        }
        let base = ctx.ins.control(0) as usize;
        if self.audio != 0 {
            for k in 1..ctx.ins.len() {
                // Stage the masked block (silence before the offset, signal after) in channel-0
                // scratch, reused per channel, then accumulate it onto the bus like `Out`.
                let signal = ctx.ins.audio(k);
                let staged = ctx.outs.audio(0);
                let offset = ctx.sample_offset.min(staged.len());
                staged[..offset].fill(0.0);
                staged[offset..].copy_from_slice(&signal[offset..]);
                unit::audio_out(ctx.buses, ctx.buf_counter, base + (k - 1), staged);
            }
        } else {
            for k in 1..ctx.ins.len() {
                unit::control_out(
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

/// Constructor for [`OffsetOut`].
pub struct OffsetOutCtor;

impl UnitDef for OffsetOutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(OffsetOut {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
