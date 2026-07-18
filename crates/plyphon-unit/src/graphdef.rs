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
use plyphon_dsp::rate::{Rate, RateInfo};

/// Where a unit output is published: an audio wire (a full block in the World's shared wire scratch)
/// or a control wire (one value in the per-graph control wires).
#[derive(Copy, Clone, Debug)]
pub struct OutputWire {
    /// The output's calculation rate.
    pub rate: Rate,
    /// Index into the synth's audio wires (audio rate) or control wires (control/scalar rate).
    pub wire: u32,
}

/// An audio-rate parameter (`AudioControl`). Its stored value lives in a control wire (`value_slot`,
/// the `/n_set`/`/n_map` target) and is lifted to an audio wire (`wire`) at the start of every block,
/// so it can feed audio-rate inputs; `/n_mapa` instead fills the wire from an audio bus.
#[derive(Copy, Clone, Debug)]
pub struct AudioParam {
    /// The parameter index (indexes the `amaps` audio-bus map for `/n_mapa`).
    pub param: u32,
    /// Control-wire index holding the parameter's value.
    pub value_slot: u32,
    /// Audio-wire index the value is lifted to each block (what consumers of the param read).
    pub wire: u32,
}

/// A lagged parameter (`LagControl`). Its stored value lives in a control wire (`value_slot`, the
/// `/n_set`/`/n_map` target); each block a one-pole with coefficient `b1` smooths it into a separate
/// `wire` (what consumers read), with the per-instance state kept in the `lag_state` span.
#[derive(Copy, Clone, Debug)]
pub struct LagParam {
    /// Control-wire index holding the parameter's (un-lagged) value.
    pub value_slot: u32,
    /// Control-wire index of the lagged output (what consumers of the param read).
    pub wire: u32,
    /// One-pole coefficient (`exp(ln(0.001) / (lagTime * controlRate))`), precomputed at compile.
    pub b1: f32,
}

/// One unit's compiled record: its calc/seed vtable, resolved wiring, and state slot in the arena -
/// plyphon's per-unit `UnitSpec` plus `mCalcFunc`.
pub struct UnitVtbl {
    /// The unit's own calculation rate (scsynth's `unit->mCalcRate`): `Audio` units compute a full
    /// block per tick, `Control`/`Scalar` units one sample. Selects the unit's [`Outputs`] length
    /// and its own-rate constants ([`ProcessCtx::own`]) in the process loop.
    ///
    /// [`Outputs`]: crate::unit::Outputs
    /// [`ProcessCtx::own`]: crate::unit::ProcessCtx::own
    pub rate: Rate,
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
    /// Byte offset of this unit's auxiliary memory (a delay line) within the `aux` arena. Unused when
    /// `aux_size == 0`.
    pub aux_offset: usize,
    /// Bytes this unit's aux region occupies in the `aux` arena (`0` for units with no aux memory).
    pub aux_size: usize,
}

/// The compiled shape of one graph-local buffer (a `LocalBuf`), indexed by declaration order.
/// The sample storage itself lives in the per-graph block's `local_bufs` span; this records where and
/// how big. A local buffer's number as seen by consumers is `buffer-table capacity + index`
/// (scsynth's `world->mNumSndBufs + i`), and its sample rate is the graph's audio rate.
#[derive(Copy, Clone, Debug)]
pub struct LocalBufSpec {
    /// Number of interleaved channels.
    pub channels: u32,
    /// Number of frames (samples per channel).
    pub frames: u32,
    /// Offset in `f32` samples within the block's `local_bufs` span.
    pub offset: usize,
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
/// arenas (alignment up to 8, for `f64` state) come first, then the 8-byte-aligned `aux` arena, then
/// the 4-byte-aligned `f32` control
/// wires, `u32` param maps, `u32` done flags, the `f32` local feedback bus, the `u32` local-buffer
/// coord tags, the `f32` local-buffer samples, the `u32` audio-bus maps, and the `f32` lag state.
/// The spans are contiguous, hence disjoint - so `get_disjoint_mut` over
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
    /// Heterogeneous per-unit auxiliary memory (delay lines / circular buffers): each unit's bytes at
    /// its `aux_offset`. Empty when no unit declared aux memory. Sized at compile time (e.g. from a
    /// delay's `maxdelaytime`) and, unlike the state arenas, **not** initialized on instantiation - a
    /// unit guards its own cold start. This is plyphon's fold of scsynth's per-unit `RTAlloc`'d delay
    /// buffers into the single per-graph block, keeping one allocation (and one free) per synth.
    pub aux: Span,
    /// Control wires (`f32`): the parameters first, then control-rate unit outputs.
    pub control: Span,
    /// Per-parameter control-bus map (`u32`; `u32::MAX` = unmapped).
    pub pmaps: Span,
    /// Per-calc-unit "done" flag (`u32`; non-zero = the unit has finished) - plyphon's port of
    /// scsynth's per-`Unit` `mDone`, written by producers (`EnvGen`/`Line`/`PlayBuf`) and read by the
    /// done-watching units (`Done`/`FreeSelfWhenDone`/`PauseSelfWhenDone`). One slot per calc unit,
    /// indexed by calc-unit position in [`GraphDef::units`].
    pub done_flags: Span,
    /// Per-synth local feedback bus (`f32`, channel-major: `num_local_channels * block_size`) -
    /// scsynth's `LocalIn`/`LocalOut` private buffer. Persists across blocks (never cleared), so a
    /// `LocalIn` reads the value the `LocalOut` wrote last block (a one-block feedback delay). Empty
    /// when the def has no `LocalIn`/`LocalOut`.
    pub local: Span,
    /// One `u32` coordinate tag (see `SpectrumCoord::to_tag`) per graph-local buffer (`LocalBuf`),
    /// indexed by declaration order - a local buffer's stand-in for `Buffer`'s coord field, so the
    /// FFT chain can run over local buffers. Empty when the def has no `LocalBuf`.
    pub local_buf_coords: Span,
    /// Graph-local buffer samples (`f32`, interleaved frame-major): each `LocalBuf`'s storage at the
    /// offset its [`LocalBufSpec`] records. Persists across blocks. Empty when the def has no
    /// `LocalBuf`.
    pub local_bufs: Span,
    /// Per-parameter audio-bus map (`u32`; `u32::MAX` = unmapped) for `/n_mapa`. Only audio-rate
    /// parameters read their slot; control params' slots are unused.
    pub amaps: Span,
    /// One-pole state (`f32`) for each `LagControl` parameter, indexed by lag-param position. Empty
    /// when the def has no lagged params.
    pub lag_state: Span,
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
    /// Control-parameter index -> its value-slot control wire index (the `/n_set`/`/n_map` target,
    /// for every parameter regardless of rate).
    param_wires: Box<[u32]>,
    /// Audio-rate parameters (`AudioControl`): each one's `(value_slot, audio_wire)`. Empty when the
    /// def has none. The process loop lifts each value slot to its audio wire every block.
    audio_params: Box<[AudioParam]>,
    /// Value-slot (control wire) indices of `TrigControl` parameters. The process loop zeros each one
    /// after the unit walk, so a `/n_set` is seen for exactly one block (scsynth's output-then-zero).
    trig_params: Box<[u32]>,
    /// Lagged parameters (`LagControl`): each one's `(value_slot, lagged_wire, b1)`. Indexed by
    /// position into the `lag_state` span (one `f32` of one-pole state per lag param).
    lag_params: Box<[LagParam]>,
    /// The graph-local buffers (`LocalBuf`), in declaration order: each one's shape and its sample
    /// offset within the block's `local_bufs` span. Empty when the def has no `LocalBuf`.
    local_bufs: Box<[LocalBufSpec]>,
    /// Number of control parameters.
    num_params: usize,
    /// The graph's audio-rate timing. The World's rate for an ordinary def; for a reblocked/resampled
    /// def, a smaller block (and/or higher sample rate). The units were built with this rate and
    /// process at it - the engine hands it to each `ProcessCtx` (scsynth's per-`Graph` `mFullRate`).
    audio: RateInfo,
    /// The graph's control-rate timing (block size 1; its sample rate is the graph's audio rate over
    /// the graph block size - smaller for a reblocked def, the finer control granularity).
    control: RateInfo,
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
        audio_params: Box<[AudioParam]>,
        trig_params: Box<[u32]>,
        lag_params: Box<[LagParam]>,
        local_bufs: Box<[LocalBufSpec]>,
        num_params: usize,
        audio: RateInfo,
        control: RateInfo,
    ) -> Self {
        GraphDef {
            units,
            demand_units,
            layout,
            state_image,
            demand_state_image,
            control_defaults,
            param_wires,
            audio_params,
            trig_params,
            lag_params,
            local_bufs,
            num_params,
            audio,
            control,
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

    /// Control-parameter index -> its value-slot control wire index.
    pub fn param_wires(&self) -> &[u32] {
        &self.param_wires
    }

    /// The audio-rate parameters (`AudioControl`), each as `(value_slot, audio_wire)`.
    pub fn audio_params(&self) -> &[AudioParam] {
        &self.audio_params
    }

    /// Value-slot control-wire indices of `TrigControl` parameters (zeroed after each block).
    pub fn trig_params(&self) -> &[u32] {
        &self.trig_params
    }

    /// The lagged parameters (`LagControl`), each as `(value_slot, lagged_wire, b1)`. Indexed by
    /// position into the `lag_state` span.
    pub fn lag_params(&self) -> &[LagParam] {
        &self.lag_params
    }

    /// The graph-local buffers (`LocalBuf`), in declaration order.
    pub fn local_buf_specs(&self) -> &[LocalBufSpec] {
        &self.local_bufs
    }

    /// Number of control parameters.
    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Samples per control block (the graph's audio block size).
    pub fn block_size(&self) -> usize {
        self.audio.block_size
    }

    /// The graph's audio-rate timing (the rate its units were built with and process at).
    pub fn audio_rate(&self) -> &RateInfo {
        &self.audio
    }

    /// The graph's control-rate timing.
    pub fn control_rate(&self) -> &RateInfo {
        &self.control
    }
}

/// Round `x` up to a multiple of `align` (a power of two).
const fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Compute the per-graph [`BlockLayout`], each calc unit's state offset, each demand unit's state
/// offset (relative to the `demand_state` span), and each calc unit's aux offset (relative to the
/// `aux` span) from the units' `(size, align)` slots, the control-wire count, and the parameter count.
///
/// The two state arenas pack their slots in order (each bumped to its own alignment), then the `aux`
/// arena, then the control wires and param maps on 4-byte boundaries. Each arena is padded to 8 so
/// the next one starts 8-aligned (state and aux may hold `f64`/`f32` data). Because the block base is
/// 64-byte aligned, every resulting span is aligned for its element type.
#[allow(clippy::too_many_arguments)]
pub fn build_layout(
    state_slots: &[(usize, usize)],
    demand_state_slots: &[(usize, usize)],
    aux_slots: &[(usize, usize)],
    num_control_wires: usize,
    num_params: usize,
    num_local_channels: usize,
    num_local_bufs: usize,
    local_buf_samples: usize,
    num_lag_params: usize,
    block_size: usize,
) -> (BlockLayout, Vec<usize>, Vec<usize>, Vec<usize>) {
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
    let (aux_offsets, aux_end) = pack(aux_slots);
    // Pad the calc-state arena to 8 so the demand-state arena starts aligned for `f64` state.
    let state = Span {
        off: 0,
        len: align_up(state_end, 8),
    };
    // Pad the demand-state arena to 8 so the `aux` arena that follows starts 8-aligned.
    let demand_state = Span {
        off: state.off + state.len,
        len: align_up(demand_end, 8),
    };
    // The per-unit aux arena (delay lines), padded to 8 so the `f32` control wires that follow stay
    // aligned. Empty (`len == 0`) when no unit declared aux memory.
    let aux = Span {
        off: demand_state.off + demand_state.len,
        len: align_up(aux_end, 8),
    };
    let control = Span {
        off: aux.off + aux.len,
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
    // The per-synth local feedback bus: `num_local_channels * block_size` `f32`s, channel-major,
    // 4-byte-aligned after the done flags.
    let local = Span {
        off: done_flags.off + done_flags.len,
        len: num_local_channels * block_size * 4,
    };
    // One `u32` coord tag per graph-local buffer (`LocalBuf`), `u32`-aligned after the local bus.
    let local_buf_coords = Span {
        off: local.off + local.len,
        len: num_local_bufs * 4,
    };
    // The graph-local buffer samples: `local_buf_samples` `f32`s, each buffer at its spec's offset,
    // 4-byte-aligned after the coord tags.
    let local_bufs = Span {
        off: local_buf_coords.off + local_buf_coords.len,
        len: local_buf_samples * 4,
    };
    // One `u32` audio-bus map per parameter (`/n_mapa`), `u32`-aligned after the local buffers.
    let amaps = Span {
        off: local_bufs.off + local_bufs.len,
        len: num_params * 4,
    };
    // One `f32` one-pole state per lagged param (`LagControl`), 4-byte-aligned after the audio maps.
    let lag_state = Span {
        off: amaps.off + amaps.len,
        len: num_lag_params * 4,
    };
    let total = lag_state.off + lag_state.len;
    (
        BlockLayout {
            state,
            demand_state,
            aux,
            control,
            pmaps,
            done_flags,
            local,
            local_buf_coords,
            local_bufs,
            amaps,
            lag_state,
            total,
        },
        offsets,
        demand_offsets,
        aux_offsets,
    )
}

#[cfg(test)]
mod tests {
    use super::{align_up, build_layout};

    /// Lay out a block with a single 8-byte state slot and the given aux slots, returning the layout
    /// and the per-unit aux offsets. Everything but `aux_slots` is held fixed.
    fn layout(aux_slots: &[(usize, usize)]) -> (super::BlockLayout, alloc::vec::Vec<usize>) {
        let (l, _state, _demand, aux) =
            build_layout(&[(8, 4)], &[], aux_slots, 4, 1, 0, 0, 0, 0, 64);
        (l, aux)
    }

    #[test]
    fn aux_empty_when_no_unit_declares_it() {
        let (l, aux) = layout(&[]);
        assert_eq!(l.aux.len, 0);
        assert!(aux.is_empty());
        // The empty aux span sits exactly where control begins; no downstream span is displaced.
        assert_eq!(l.aux.off, l.control.off);
        assert_eq!(l.aux.off, l.demand_state.off + l.demand_state.len);
    }

    #[test]
    fn aux_single_slot_is_packed_and_8_aligned() {
        let (l, aux) = layout(&[(100, 4)]);
        assert_eq!(aux, alloc::vec![0]);
        assert_eq!(l.aux.off % 8, 0);
        assert_eq!(l.aux.len, align_up(100, 8));
        assert_eq!(l.control.off, l.aux.off + l.aux.len);
    }

    #[test]
    fn aux_two_slots_pack_contiguously() {
        let (l, aux) = layout(&[(100, 4), (40, 4)]);
        assert_eq!(aux, alloc::vec![0, align_up(100, 4)]);
        assert_eq!(l.aux.len, align_up(100 + 40, 8));
    }

    #[test]
    fn aux_respects_slot_alignment() {
        let (_l, aux) = layout(&[(4, 4), (8, 8)]);
        assert_eq!(aux, alloc::vec![0, align_up(4, 8)]);
    }

    #[test]
    fn aux_grows_total_by_its_arena() {
        let base = layout(&[]).0;
        let with = layout(&[(256, 4)]).0;
        assert_eq!(with.total, base.total + align_up(256, 8));
    }
}
