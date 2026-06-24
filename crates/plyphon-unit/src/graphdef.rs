//! The compiled synth definition - plyphon's port of scsynth's `GraphDef`.
//!
//! A [`GraphDef`] is the immutable, shareable template a `SynthDef` compiles to (off the audio
//! thread, once). Like scsynth's `GraphDef` it is system-allocated and long-lived - *not* in the
//! rt-pool - and many live `Graph`s reference one via `Arc` (the `Arc` count is plyphon's
//! `mRefCount`). It holds the per-unit calc/seed vtable, the
//! wiring, the layout of the per-graph pool block, and the images needed to construct an instance on
//! the audio thread with a single allocation and a few `memcpy`s.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::unit::{InitFn, InputSource, ProcessFn, ReseedFn};
use plyphon_dsp::rate::Rate;

/// Where a unit output is published: an audio wire (a full block in the World's shared wire scratch)
/// or a control wire (one value in the per-graph control wires).
#[derive(Copy, Clone, Debug)]
pub struct OutputWire {
    /// The output's calculation rate.
    pub rate: Rate,
    /// Index into the synth's audio wires (audio rate) or control wires (control/scalar rate).
    pub wire: u32,
}

/// One unit's compiled record: its calc/seed vtable, resolved wiring, and state slot in the arena -
/// plyphon's per-unit `UnitSpec` plus `mCalcFunc`.
pub struct UnitVtbl {
    /// Per-block calc function over the state slot.
    pub process: ProcessFn,
    /// One-time first-block seeding function over the state slot.
    pub init: InitFn,
    /// Per-instance re-seed function over the state slot (no-op for units without randomness).
    pub reseed: ReseedFn,
    /// Resolved input sources, in order.
    pub inputs: Box<[InputSource]>,
    /// Where each output is published.
    pub outputs: Box<[OutputWire]>,
    /// Byte offset of this unit's state within the state arena.
    pub state_offset: usize,
    /// Exactly `size_of::<T>()` - the bytes this unit's state occupies.
    pub state_size: usize,
}

/// A byte sub-range within the per-graph pool block.
#[derive(Copy, Clone, Debug)]
pub struct Span {
    /// Byte offset from the start of the block's payload.
    pub off: usize,
    /// Length in bytes.
    pub len: usize,
}

impl Span {
    /// The span as a `Range`, for slicing the block (`buf[span.range()]`).
    pub fn range(self) -> core::ops::Range<usize> {
        self.off..self.off + self.len
    }
}

/// How a per-graph pool block is carved. The block holds only the per-instance mutable state; audio
/// wire buffers and per-unit output scratch are World-shared and live outside the block (matching
/// scsynth, which keeps those in `mWireBufSpace`, not the per-graph allocation).
///
/// Laid out so every span is correctly aligned given a 64-byte-aligned block base: the state arena
/// (alignment up to 8, for `f64` state) comes first, then the 4-byte-aligned `f32` control wires and
/// `u32` param maps. The spans are contiguous, hence disjoint - so `get_disjoint_mut` over them never
/// fails, and the `bytemuck` casts never hit an alignment error.
#[derive(Copy, Clone, Debug)]
pub struct BlockLayout {
    /// Heterogeneous unit state (each unit's `Pod` bytes at its `state_offset`).
    pub state: Span,
    /// Control wires (`f32`): the parameters first, then control-rate unit outputs.
    pub control: Span,
    /// Per-parameter control-bus map (`u32`; `u32::MAX` = unmapped).
    pub pmaps: Span,
    /// Total block size in bytes.
    pub total: usize,
}

/// The compiled, immutable, shareable synth definition (scsynth's `GraphDef`). Built once by
/// `SynthDef` compilation off the audio thread and shared via `Arc` so it can ride in a command to
/// the real-time engine; its parts are read back through the accessors below.
pub struct GraphDef {
    /// Per-unit vtable + wiring, in topological calc order.
    units: Box<[UnitVtbl]>,
    /// How a per-graph pool block is carved.
    layout: BlockLayout,
    /// The initial state-arena image: each unit's initial state bytes packed at its offset. Copied
    /// into a fresh block when a synth is built on the audio thread.
    state_image: Box<[u8]>,
    /// Initial control-wire values: parameter defaults in the first `num_params` slots, then zeros.
    control_defaults: Box<[f32]>,
    /// Control-parameter index -> control wire index.
    param_wires: Box<[u32]>,
    /// Number of control parameters.
    num_params: usize,
    /// Samples per control block.
    block_size: usize,
}

impl GraphDef {
    /// Assemble a compiled def from its parts - the output of `SynthDef` compilation. The parts must
    /// be mutually consistent (`layout` describes how `state_image` and each per-instance block are
    /// carved), so compilation is the only place that builds one.
    pub fn new(
        units: Box<[UnitVtbl]>,
        layout: BlockLayout,
        state_image: Box<[u8]>,
        control_defaults: Box<[f32]>,
        param_wires: Box<[u32]>,
        num_params: usize,
        block_size: usize,
    ) -> Self {
        GraphDef {
            units,
            layout,
            state_image,
            control_defaults,
            param_wires,
            num_params,
            block_size,
        }
    }

    /// The per-unit vtables and wiring, in topological calc order.
    pub fn units(&self) -> &[UnitVtbl] {
        &self.units
    }

    /// How a per-graph pool block is carved.
    pub fn layout(&self) -> BlockLayout {
        self.layout
    }

    /// The initial state-arena image, copied into a fresh block when a synth is instantiated.
    pub fn state_image(&self) -> &[u8] {
        &self.state_image
    }

    /// Initial control-wire values: parameter defaults first, then zeros.
    pub fn control_defaults(&self) -> &[f32] {
        &self.control_defaults
    }

    /// Control-parameter index -> control wire index.
    pub fn param_wires(&self) -> &[u32] {
        &self.param_wires
    }

    /// Number of control parameters.
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Samples per control block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Round `x` up to a multiple of `align` (a power of two).
const fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Compute the per-graph [`BlockLayout`] and each unit's state offset from the units' `(size, align)`
/// slots, the control-wire count, and the parameter count.
///
/// The state arena packs the slots in order (each bumped to its own alignment), then the control
/// wires and param maps follow on 4-byte boundaries. Because the block base is 64-byte aligned, every
/// resulting span is aligned for its element type.
pub fn build_layout(
    state_slots: &[(usize, usize)],
    num_control_wires: usize,
    num_params: usize,
) -> (BlockLayout, Vec<usize>) {
    let mut offsets = Vec::with_capacity(state_slots.len());
    let mut cursor = 0usize;
    for &(size, align) in state_slots {
        let align = align.max(1);
        cursor = align_up(cursor, align);
        offsets.push(cursor);
        cursor += size;
    }
    let state = Span {
        off: 0,
        len: align_up(cursor, 4),
    };
    let control = Span {
        off: state.off + state.len,
        len: num_control_wires * 4,
    };
    let pmaps = Span {
        off: control.off + control.len,
        len: num_params * 4,
    };
    let total = pmaps.off + pmaps.len;
    (
        BlockLayout {
            state,
            control,
            pmaps,
            total,
        },
        offsets,
    )
}
