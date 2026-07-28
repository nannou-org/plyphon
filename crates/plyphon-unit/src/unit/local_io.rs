//! `LocalIn` / `LocalOut` - a synth's private feedback bus, plyphon's port of scsynth's local
//! buffers.
//!
//! `LocalOut.ar(signals)` writes the synth's private bus; `LocalIn.ar(numChannels)` reads it. The bus
//! lives in the per-instance pool block and persists across blocks, so a `LocalIn` reads what
//! `LocalOut` wrote on the *previous* block - a one-block feedback delay. The one-block delay falls
//! out of calc order: `LocalIn` (a source, ordered before `LocalOut`) reads the bus before
//! `LocalOut` overwrites it. The channel count is fixed by the single `LocalIn` (its output
//! count); a `LocalOut` whose input count differs builds with zero channels and writes nothing,
//! matching measured scsynth mismatch behavior.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// `LocalIn.ar(numChannels)`: reads the synth's private feedback bus - the value `LocalOut` wrote on
/// the previous block. The output count fixes the bus width for the whole synth.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LocalIn {
    num_channels: u32,
}

impl Unit for LocalIn {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        for ch in 0..self.num_channels as usize {
            let dst = ctx.outs.audio(ch);
            let src = unit::local_in(&ctx.local, ch);
            if src.len() == dst.len() {
                dst.copy_from_slice(src);
            } else {
                dst.fill(0.0);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`LocalIn`].
pub struct LocalInCtor;

impl UnitDef for LocalInCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(LocalIn {
            num_channels: ctx.num_outputs as u32,
        }))
    }
}

/// `LocalOut.ar(signals)`: writes its inputs to the synth's private feedback bus, overwriting last
/// block's contents. Makes no sound itself; a `LocalIn` reads the written values next block.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LocalOut {
    num_channels: u32,
}

impl Unit for LocalOut {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        for ch in 0..self.num_channels as usize {
            let src = ctx.ins.audio(ch);
            unit::local_out(&mut ctx.local, ch, src);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`LocalOut`]. Its inputs are the signals to write, one per local channel.
///
/// A `LocalOut` whose input count differs from the def's resolved local-bus width — including a
/// `LocalOut` with no `LocalIn` at all (width 0) — builds with zero channels and writes nothing.
/// This matches measured scsynth behavior: a mismatched `LocalOut` is a complete no-op (no
/// channel is partially written, wrapped, or truncated) while the rest of the definition still
/// compiles and renders, its `LocalIn` reading silence.
pub struct LocalOutCtor;

impl UnitDef for LocalOutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let inputs = ctx.input_rates.len();
        let num_channels = if inputs == ctx.local_channels {
            inputs as u32
        } else {
            0
        };
        Ok(unit_spec(LocalOut { num_channels }))
    }
}
