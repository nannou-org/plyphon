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

use crate::unit::demand::DemandVtbl;
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
/// Laid out so every span is correctly aligned given a 64-byte-aligned block base: the two state
/// arenas (alignment up to 8, for `f64` state) come first, then the 4-byte-aligned `f32` control
/// wires, `u32` param maps, and `u32` done flags. The spans are contiguous, hence disjoint - so `get_disjoint_mut` over
/// them never fails, and the `bytemuck` casts never hit an alignment error. The calc-unit and
/// demand-unit state are *separate* spans so the audio thread can hold a calc unit's `&mut` state
/// slot and the `&mut` demand arena at once (the latter is pulled re-entrantly while the former runs).
#[derive(Copy, Clone, Debug)]
pub struct BlockLayout {
    /// Heterogeneous calc-unit state (each calc unit's `Pod` bytes at its `state_offset`).
    pub state: Span,
    /// Heterogeneous demand-unit state (each demand unit's `Pod` bytes at its `state_offset`). Empty
    /// when the def has no demand units.
    pub demand_state: Span,
    /// Control wires (`f32`): the parameters first, then control-rate unit outputs.
    pub control: Span,
    /// Per-parameter control-bus map (`u32`; `u32::MAX` = unmapped).
    pub pmaps: Span,
    /// Per-calc-unit "done" flag (`u32`; non-zero = the unit has finished) - plyphon's port of
    /// scsynth's per-`Unit` `mDone`, written by producers (`EnvGen`/`Line`/`PlayBuf`) and read by the
    /// done-watching units (`Done`/`FreeSelfWhenDone`/`PauseSelfWhenDone`). One slot per calc unit,
    /// indexed by calc-unit position in [`GraphDef::units`].
    pub done_flags: Span,
    /// Total block size in bytes.
    pub total: usize,
}

/// The compiled, immutable, shareable synth definition (scsynth's `GraphDef`). Built once by
/// `SynthDef` compilation off the audio thread and shared via `Arc` so it can ride in a command to
/// the real-time engine; its parts are read back through the accessors below.
pub struct GraphDef {
    /// Per-unit vtable + wiring, in topological calc order.
    units: Box<[UnitVtbl]>,
    /// The demand plan: per-demand-unit pull/reset/seed vtable + wiring. Not in the per-block calc
    /// list - each is driven on demand by a consuming unit. Empty when the def has no demand units.
    demand_units: Box<[DemandVtbl]>,
    /// How a per-graph pool block is carved.
    layout: BlockLayout,
    /// The initial state-arena image: each unit's initial state bytes packed at its offset. Copied
    /// into a fresh block when a synth is built on the audio thread.
    state_image: Box<[u8]>,
    /// The initial demand-state image: each demand unit's initial state bytes packed at its offset.
    /// Copied into the block's `demand_state` span when a synth is built. Empty without demand units.
    demand_state_image: Box<[u8]>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        units: Box<[UnitVtbl]>,
        demand_units: Box<[DemandVtbl]>,
        layout: BlockLayout,
        state_image: Box<[u8]>,
        demand_state_image: Box<[u8]>,
        control_defaults: Box<[f32]>,
        param_wires: Box<[u32]>,
        num_params: usize,
        block_size: usize,
    ) -> Self {
        GraphDef {
            units,
            demand_units,
            layout,
            state_image,
            demand_state_image,
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

    /// The demand plan: per-demand-unit vtables and wiring, indexed by demand-plan index.
    pub fn demand_units(&self) -> &[DemandVtbl] {
        &self.demand_units
    }

    /// How a per-graph pool block is carved.
    pub fn layout(&self) -> BlockLayout {
        self.layout
    }

    /// The initial state-arena image, copied into a fresh block when a synth is instantiated.
    pub fn state_image(&self) -> &[u8] {
        &self.state_image
    }

    /// The initial demand-state image, copied into the block's `demand_state` span on instantiation.
    pub fn demand_state_image(&self) -> &[u8] {
        &self.demand_state_image
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

/// Compute the per-graph [`BlockLayout`], each calc unit's state offset, and each demand unit's state
/// offset (relative to the `demand_state` span) from the units' `(size, align)` slots, the
/// control-wire count, and the parameter count.
///
/// The two state arenas pack their slots in order (each bumped to its own alignment), then the
/// control wires and param maps follow on 4-byte boundaries. The calc-state arena is padded to 8 so
/// the demand-state arena that follows starts 8-aligned (it may hold `f64` state). Because the block
/// base is 64-byte aligned, every resulting span is aligned for its element type.
pub fn build_layout(
    state_slots: &[(usize, usize)],
    demand_state_slots: &[(usize, usize)],
    num_control_wires: usize,
    num_params: usize,
) -> (BlockLayout, Vec<usize>, Vec<usize>) {
    let pack = |slots: &[(usize, usize)]| -> (Vec<usize>, usize) {
        let mut offsets = Vec::with_capacity(slots.len());
        let mut cursor = 0usize;
        for &(size, align) in slots {
            let align = align.max(1);
            cursor = align_up(cursor, align);
            offsets.push(cursor);
            cursor += size;
        }
        (offsets, cursor)
    };
    let (offsets, state_end) = pack(state_slots);
    let (demand_offsets, demand_end) = pack(demand_state_slots);
    // Pad the calc-state arena to 8 so the demand-state arena starts aligned for `f64` state.
    let state = Span {
        off: 0,
        len: align_up(state_end, 8),
    };
    let demand_state = Span {
        off: state.off + state.len,
        len: align_up(demand_end, 4),
    };
    let control = Span {
        off: demand_state.off + demand_state.len,
        len: num_control_wires * 4,
    };
    let pmaps = Span {
        off: control.off + control.len,
        len: num_params * 4,
    };
    // One `u32` done flag per calc unit (`state_slots.len()`), `u32`-aligned after the param maps.
    let done_flags = Span {
        off: pmaps.off + pmaps.len,
        len: state_slots.len() * 4,
    };
    let total = done_flags.off + done_flags.len;
    (
        BlockLayout {
            state,
            demand_state,
            control,
            pmaps,
            done_flags,
            total,
        },
        offsets,
        demand_offsets,
    )
}
