//! Rate-conversion units - plyphon's ports of scsynth's `DC`, `K2A`, `A2K`, and `T2A`.
//!
//! These bridge calc rates: `DC` emits a constant signal, `K2A` lifts a control signal to audio rate
//! (linearly interpolating between blocks), `A2K` samples an audio signal down to control rate (its
//! first sample), and `T2A` places a control-rate trigger at a sample-accurate position in an audio
//! block.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};

/// `DC.ar/kr(value)`: a constant signal. `value` is taken at scalar rate (the same every block).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dc {
    _pad: u32,
}

impl Dc {
    const VALUE: usize = 0;
}

impl Unit for Dc {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let value = ctx.ins.control(Self::VALUE);
        ctx.outs.audio(0).fill(value);
        DoneAction::Nothing
    }
}

/// Constructor for [`Dc`].
pub struct DcCtor;

impl UnitDef for DcCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Dc { _pad: 0 }))
    }
}

/// `K2A.ar(in)`: convert a control-rate signal to audio rate, linearly interpolating from the
/// previous block's value to this block's value across the block (scsynth's `CALCSLOPE` ramp).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct K2A {
    /// Last block's input value, the ramp's starting point.
    prev: f32,
}

impl K2A {
    const IN: usize = 0;
}

impl Unit for K2A {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Start at the input value so the first block holds steady instead of ramping up from zero.
        self.prev = ctx.ins.control(Self::IN);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let cur = ctx.ins.control(Self::IN);
        let prev = self.prev;
        let out = ctx.outs.audio(0);
        let slope = (cur - prev) / out.len() as f32;
        let mut level = prev;
        for o in out.iter_mut() {
            *o = level;
            level += slope;
        }
        self.prev = cur;
        DoneAction::Nothing
    }
}

/// Constructor for [`K2A`].
pub struct K2ACtor;

impl UnitDef for K2ACtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(K2A { prev: 0.0 }))
    }
}

/// `A2K.kr(in)`: convert an audio-rate signal to control rate by taking the first sample of the
/// block (scsynth's `ZOUT0(0) = ZIN0(0)`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct A2K {
    _pad: u32,
}

impl A2K {
    const IN: usize = 0;
}

impl Unit for A2K {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `control` collapses an audio input to its first sample, which is exactly A2K's behaviour.
        let value = ctx.ins.control(Self::IN);
        *ctx.outs.control(0) = value;
        DoneAction::Nothing
    }
}

/// Constructor for [`A2K`].
pub struct A2KCtor;

impl UnitDef for A2KCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(A2K { _pad: 0 }))
    }
}

/// `T2A.ar(in, offset)`: convert a control-rate trigger to a sample-accurate audio trigger. On a
/// rising edge of `in`, writes `in`'s value at sample `offset` (clamped into the block) and zero
/// elsewhere; otherwise the block is silent.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct T2A {
    /// Last block's trigger level, for rising-edge detection.
    prev: f32,
}

impl T2A {
    const IN: usize = 0;
    const OFFSET: usize = 1;
}

impl Unit for T2A {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let level = ctx.ins.control(Self::IN);
        let offset = ctx.ins.control(Self::OFFSET);
        let prev = self.prev;
        let out = ctx.outs.audio(0);
        out.fill(0.0);
        if prev <= 0.0 && level > 0.0 {
            let last = out.len().saturating_sub(1);
            let at = (offset as i32).clamp(0, last as i32) as usize;
            out[at] = level;
        }
        self.prev = level;
        DoneAction::Nothing
    }
}

/// Constructor for [`T2A`].
pub struct T2ACtor;

impl UnitDef for T2ACtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(T2A { prev: 0.0 }))
    }
}
