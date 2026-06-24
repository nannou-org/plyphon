//! The compiled synth definition - plyphon's port of scsynth's `GraphDef`.
//!
//! A [`GraphDef`] is the immutable, shareable template a [`SynthDef`](crate::synthdef::SynthDef)
//! compiles to (off the audio thread, once). Like scsynth's `GraphDef` it is system-allocated and
//! long-lived - *not* in the rt-pool - and many live [`Graph`](crate::graph::Graph)s reference one
//! via `Arc` (the `Arc` count is plyphon's `mRefCount`). It holds the per-UGen calc/seed vtable, the
//! wiring, the layout of the per-graph pool block, and the images needed to construct an instance on
//! the audio thread with a single allocation and a few `memcpy`s.

use crate::rate::Rate;
use crate::ugen::{InitFn, InputSource, ProcessFn, ReseedFn};

/// Where a UGen output is published: an audio wire (a full block in the World's shared wire scratch)
/// or a control wire (one value in the per-graph control wires).
#[derive(Copy, Clone, Debug)]
pub(crate) struct OutputWire {
    /// The output's calculation rate.
    pub rate: Rate,
    /// Index into the synth's audio wires (audio rate) or control wires (control/scalar rate).
    pub wire: u32,
}

/// One UGen's compiled record: its calc/seed vtable, resolved wiring, and state slot in the arena -
/// plyphon's per-unit `UnitSpec` plus `mCalcFunc`.
pub(crate) struct UgenVtbl {
    /// Per-block calc function over the state slot.
    pub process: ProcessFn,
    /// One-time first-block seeding function over the state slot.
    pub init: InitFn,
    /// Per-instance re-seed function over the state slot (no-op for UGens without randomness).
    pub reseed: ReseedFn,
    /// Resolved input sources, in order.
    pub inputs: Box<[InputSource]>,
    /// Where each output is published.
    pub outputs: Box<[OutputWire]>,
    /// Byte offset of this UGen's state within the state arena.
    pub state_offset: usize,
    /// Exactly `size_of::<T>()` - the bytes this UGen's state occupies.
    pub state_size: usize,
}

/// A byte sub-range within the per-graph pool block.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Span {
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
/// wire buffers and per-UGen output scratch are World-shared and live outside the block (matching
/// scsynth, which keeps those in `mWireBufSpace`, not the per-graph allocation).
///
/// Laid out so every span is correctly aligned given a 64-byte-aligned block base: the state arena
/// (alignment up to 8, for `f64` state) comes first, then the 4-byte-aligned `f32` control wires and
/// `u32` param maps. The spans are contiguous, hence disjoint - so `get_disjoint_mut` over them never
/// fails, and the `bytemuck` casts never hit an alignment error.
#[derive(Copy, Clone, Debug)]
pub(crate) struct BlockLayout {
    /// Heterogeneous UGen state (each UGen's `Pod` bytes at its `state_offset`).
    pub state: Span,
    /// Control wires (`f32`): the parameters first, then control-rate UGen outputs.
    pub control: Span,
    /// Per-parameter control-bus map (`u32`; `u32::MAX` = unmapped).
    pub pmaps: Span,
    /// Total block size in bytes.
    pub total: usize,
}

/// The compiled, immutable, shareable synth definition (scsynth's `GraphDef`). Public so it can ride
/// in a [`Command`](crate::command::Command); its fields are crate-internal.
pub struct GraphDef {
    /// Per-UGen vtable + wiring, in topological calc order.
    pub(crate) ugens: Box<[UgenVtbl]>,
    /// How a per-graph pool block is carved.
    pub(crate) layout: BlockLayout,
    /// The initial state-arena image: each UGen's initial state bytes packed at its offset. Copied
    /// into a fresh block when a synth is built on the audio thread.
    pub(crate) state_image: Box<[u8]>,
    /// Initial control-wire values: parameter defaults in the first `num_params` slots, then zeros.
    pub(crate) control_defaults: Box<[f32]>,
    /// Control-parameter index -> control wire index.
    pub(crate) param_wires: Box<[u32]>,
    /// Number of control parameters.
    pub(crate) num_params: usize,
    /// Samples per control block.
    pub(crate) block_size: usize,
}

/// Round `x` up to a multiple of `align` (a power of two).
const fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Compute the per-graph [`BlockLayout`] and each UGen's state offset from the UGens' `(size, align)`
/// slots, the control-wire count, and the parameter count.
///
/// The state arena packs the slots in order (each bumped to its own alignment), then the control
/// wires and param maps follow on 4-byte boundaries. Because the block base is 64-byte aligned, every
/// resulting span is aligned for its element type.
pub(crate) fn build_layout(
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
