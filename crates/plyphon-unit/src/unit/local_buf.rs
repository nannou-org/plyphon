//! Graph-owned buffers - plyphon's ports of scsynth's `LocalBuf`, `MaxLocalBufs`, `SetBuf` and
//! `ClearBuf` (`DelayUGens.cpp`).
//!
//! A `LocalBuf` gives its enclosing synth a private buffer that any buffer consumer (`BufWr`,
//! `BufRd`, `FFT`, ...) can address by number: the buffer's number is `buffer-table capacity +
//! declaration index` (scsynth's `world->mNumSndBufs + i`), which the buffer io free fns resolve to
//! the graph-local storage. Where scsynth `RTAlloc`s each local buffer at ctor, plyphon sizes the
//! storage at SynthDef compile time - the `numChannels`/`numFrames` inputs must be constants - and
//! carves it from the synth's single pool block, so instantiation stays one allocation.
//!
//! One deliberate, benign divergence from scsynth: local-buffer memory is **zeroed once at synth
//! spawn** (scsynth's `RTAlloc` leaves it uninitialised, so a scsynth local buffer starts with
//! whatever the pool held). Deterministic silence is strictly safer and costs one bounded memset
//! per spawn.
//!
//! `SetBuf` and `ClearBuf` apply their write on the unit's **first process**. scsynth applies it at
//! ctor - before any unit's first calc - so a consumer ordered *before* the writer would see the
//! write one block earlier there; sclang always orders these writers before the consumers that read
//! the buffer, where the two schedules agree. Both output a constant `0` as scsynth does; sclang
//! reads the buffer number from the `LocalBuf` itself (`.set`/`.clear` return the receiver), never
//! from these units' outputs.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec, unit_spec_local_buf};

/// `LocalBuf(numChannels, numFrames)`: declares a graph-local buffer and outputs its buffer number
/// (`buffer-table capacity + declaration index`, held every block). Scalar rate; both inputs must be
/// compile-time constants, since they size the per-graph block. The storage lives beside the synth's
/// other state, persists across blocks, and is freed with the synth.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LocalBuf {
    /// This buffer's declaration index within the def (assigned at compile, in unit order).
    index: u32,
}

impl Unit for LocalBuf {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The number every consumer resolves back through the buffer io fns: table capacity + index
        // (scsynth's `bufnum + world->mNumSndBufs`). Read from the live table so the def stays
        // engine-agnostic (the capacity is an engine option, unknown at compile).
        *ctx.outs.control(0) = (unit::num_buffers(ctx.buffers) + self.index as usize) as f32;
        DoneAction::Nothing
    }
}

/// Constructor for [`LocalBuf`]: bakes the declaration index (from the running per-def count) and
/// declares the constant `channels * frames` storage for the compile loop to carve.
pub struct LocalBufCtor;

impl UnitDef for LocalBufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        // scsynth reads both at ctor (`IN0(0)` channels, `IN0(1)` frames); here they size pool
        // storage, so like a delay's `maxdelaytime` they must be baked constants.
        let channels = ctx
            .const_input(0)
            .ok_or(BuildError::AuxRequiresConstant { input: 0 })?;
        let frames = ctx
            .const_input(1)
            .ok_or(BuildError::AuxRequiresConstant { input: 1 })?;
        Ok(unit_spec_local_buf(
            LocalBuf {
                index: ctx.local_bufs_so_far as u32,
            },
            channels.max(0.0) as usize,
            frames.max(0.0) as usize,
        ))
    }
}

/// `MaxLocalBufs(count)`: sclang's automatic declaration of a def's `LocalBuf` count. In scsynth it
/// pre-allocates the graph's `SndBuf` array; plyphon sizes the storage from the actual `LocalBuf`
/// units at compile time, so this unit has no allocation role - it consumes its input and outputs
/// `0` (scsynth never writes this output, and an untouched scsynth wire reads `0`). Kept so
/// sclang-compiled defs load unchanged.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MaxLocalBufs {
    /// Padding for a non-zero state slot.
    _pad: u32,
}

impl Unit for MaxLocalBufs {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }
}

/// Constructor for [`MaxLocalBufs`].
pub struct MaxLocalBufsCtor;

impl UnitDef for MaxLocalBufsCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(MaxLocalBufs { _pad: 0 }))
    }
}

/// `ClearBuf(buf)`: zeroes every sample of buffer `buf` once, on the unit's first process. Outputs
/// a constant `0` (scsynth's `OUT0(0) = 0.f`). Works on world and graph-local buffers alike (they
/// share the buffer io resolution). A missing buffer is a no-op, like scsynth's "no valid buffer".
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ClearBuf {
    /// `0` until the one-time clear has run.
    done: u32,
}

impl Unit for ClearBuf {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // Clamp negative buffer numbers to 0, as scsynth's `CTOR_GET_BUF` does.
        let bufnum = ctx.ins.control(0).max(0.0) as usize;
        if self.done == 0 {
            self.done = 1;
            if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            {
                buffer.data_mut().fill(0.0);
            }
        }
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }
}

/// Constructor for [`ClearBuf`].
pub struct ClearBufCtor;

impl UnitDef for ClearBufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(ClearBuf { done: 0 }))
    }
}

/// `SetBuf(buf, offset, numValues, values...)`: writes `numValues` values into buffer `buf` starting
/// at flat (interleaved) sample `offset`, once, on the unit's first process. Outputs a constant `0`
/// (scsynth's `OUT0(0) = 0.f`). The input layout matches scsynth's `SetBuf_Ctor` (`IN0(1)` offset,
/// `IN0(2)` count, values from `IN0(3)`); the write is clamped to the buffer's samples
/// (`sc_min(buf->samples, ...)`) and to the values actually supplied. Works on world and graph-local
/// buffers alike. A missing buffer is a no-op.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SetBuf {
    /// `0` until the one-time write has run.
    done: u32,
}

impl SetBuf {
    const BUFNUM: usize = 0;
    const OFFSET: usize = 1;
    const NUM_VALUES: usize = 2;
    /// First value input index.
    const FIRST_VALUE: usize = 3;
}

impl Unit for SetBuf {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`, so we can also take the buffer.
        // Clamp negative buffer numbers to 0, as scsynth's `CTOR_GET_BUF` does.
        let bufnum = ins.control(Self::BUFNUM).max(0.0) as usize;
        if self.done == 0 {
            self.done = 1;
            if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            {
                // A negative offset is clamped to 0 (scsynth would index below the buffer).
                let offset = ins.control(Self::OFFSET).max(0.0) as usize;
                let count = (ins.control(Self::NUM_VALUES).max(0.0) as usize)
                    .min(ins.len().saturating_sub(Self::FIRST_VALUE));
                let data = buffer.data_mut();
                let end = data.len().min(offset.saturating_add(count));
                for (j, slot) in data[offset.min(end)..end].iter_mut().enumerate() {
                    *slot = ins.control(Self::FIRST_VALUE + j);
                }
            }
        }
        *ctx.outs.control(0) = 0.0;
        DoneAction::Nothing
    }
}

/// Constructor for [`SetBuf`].
pub struct SetBufCtor;

impl UnitDef for SetBufCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < SetBuf::FIRST_VALUE {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(SetBuf { done: 0 }))
    }
}
