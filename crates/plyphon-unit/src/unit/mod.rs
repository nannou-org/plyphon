//! Unit generators - plyphon's port of scsynth's `Unit`/`UnitCalcFunc`.
//!
//! A [`Unit`] is constructed off the audio thread (it may allocate) and then [`Unit::process`]ed
//! once per control block on the audio thread, where it must not allocate or block. Everything a
//! unit reads from the wider engine arrives in one [`ProcessCtx`] argument - the read-only
//! [`Inputs`], the writable [`Outputs`], the engine constants, and the shared buses/buffers - so
//! there is no global state.
//!
//! `ProcessCtx` is a plain field aggregate, and the operations on the shared buses/buffers are free
//! fns in the [`io`] submodule that take only the field they need (e.g. `io::audio_in(&ctx.buses,
//! ..)`). That keeps them borrow-friendly: because `ins`, `outs`, and `buses` are disjoint fields, a
//! unit can read an input and write an output (or a bus) in the same expression - the safe
//! equivalent of scsynth's raw aliasing `float*` wires.

pub mod amp_comp;
pub mod band_limited;
pub mod bank;
pub mod binary_op;
pub mod buf_rd;
pub mod buf_wr;
pub mod chaos;
#[cfg(feature = "fft")]
pub mod convolution;
pub mod decay;
pub mod delay;
pub mod deltap;
pub mod demand;
pub mod disk_in;
pub mod disk_out;
pub mod dynamics;
pub mod env;
pub mod eq;
#[cfg(feature = "fft")]
pub mod fft;
pub mod filter;
pub mod filter_simple;
pub mod formant;
pub mod freeverb;
pub mod gendy;
pub mod grain;
pub mod grain_tap;
pub mod gverb;
pub mod hilbert;
pub mod info;
pub mod input;
pub mod io;
pub mod lf;
pub mod lf_noise;
pub mod line;
pub mod linen;
pub mod local_buf;
pub mod local_io;
pub mod measure;
pub mod median;
pub mod moog;
pub mod node_ctl;
pub mod noise;
pub mod one_pole;
pub mod out;
#[cfg(feature = "fft")]
pub mod pack_fft;
pub mod pan;
pub mod physical;
pub mod pitch;
pub mod pitch_shift;
pub mod play_buf;
pub mod pluck;
pub mod poll;
pub mod psin_grain;
#[cfg(feature = "fft")]
pub mod pv;
#[cfg(feature = "fft")]
pub mod pv_combine;
#[cfg(feature = "fft")]
pub mod pv_mag_mul;
#[cfg(feature = "fft")]
pub mod pv_mag_squared;
#[cfg(feature = "fft")]
pub mod pv_ops;
pub mod ramp;
pub mod rand;
pub mod rate_conv;
pub mod record_buf;
pub mod registry;
pub mod resonant;
pub mod scope_out;
pub mod section;
pub mod select;
pub mod send_peak_rms;
pub mod send_reply;
pub mod send_trig;
pub mod shape;
pub mod sin_osc;
pub mod test;
pub mod timing;
pub mod trigger;
pub mod two_pole;
pub mod unary_op;
pub mod util;
pub mod vdisk_in;
pub mod vibrato;
pub mod wavetable_osc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use bytemuck::Pod;

use crate::graphdef::LocalBufMeta;
use plyphon_dsp::buffer::{BufView, BufViewMut, BufferTable, SpectrumCoord};
use plyphon_dsp::bus::Buses;
use plyphon_dsp::fft::FftTables;
use plyphon_dsp::rate::{Rate, RateInfo};
use plyphon_dsp::rng::Rng;
use plyphon_dsp::wavetable::Wavetables;

/// What a unit asks the engine to do with its enclosing synth when it finishes - scsynth's full set
/// of done-action codes (0-14). The discriminant is the scsynth code, and the variants are declared
/// in code order so the derived `Ord` lets the strongest action win when several units in one synth
/// finish together (every code `>= 2` frees self, so the neighbour/group variants outrank a plain
/// free, and a free outranks a pause).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub enum DoneAction {
    /// Keep running (no action). Code 0.
    #[default]
    Nothing,
    /// Pause the enclosing synth. Code 1.
    PauseSelf,
    /// Free the enclosing synth. Code 2.
    FreeSelf,
    /// Free this synth and the preceding node. Code 3.
    FreeSelfAndPrev,
    /// Free this synth and the following node. Code 4.
    FreeSelfAndNext,
    /// Free this synth; `g_freeAll` the preceding node if it is a group, else free it. Code 5.
    FreeSelfAndFreeAllPrev,
    /// Free this synth; `g_freeAll` the following node if it is a group, else free it. Code 6.
    FreeSelfAndFreeAllNext,
    /// Free this synth and every preceding node in its group. Code 7.
    FreeSelfToHead,
    /// Free this synth and every following node in its group. Code 8.
    FreeSelfToTail,
    /// Free this synth and pause the preceding node. Code 9.
    FreeSelfPausePrev,
    /// Free this synth and pause the following node. Code 10.
    FreeSelfPauseNext,
    /// Free this synth; `g_deepFree` the preceding node if it is a group, else free it. Code 11.
    FreeSelfAndDeepFreePrev,
    /// Free this synth; `g_deepFree` the following node if it is a group, else free it. Code 12.
    FreeSelfAndDeepFreeNext,
    /// Free this synth and every other node in its group. Code 13.
    FreeAllInGroup,
    /// Free the enclosing group and every node within it (this synth included). Code 14.
    FreeGroup,
}

impl DoneAction {
    /// Map a scsynth done-action code to a [`DoneAction`]. Out-of-range codes (`< 0` or `> 14`) fall
    /// back to [`FreeSelf`](DoneAction::FreeSelf), matching scsynth's "free on anything unexpected".
    fn from_index(code: i32) -> DoneAction {
        match code {
            0 => DoneAction::Nothing,
            1 => DoneAction::PauseSelf,
            2 => DoneAction::FreeSelf,
            3 => DoneAction::FreeSelfAndPrev,
            4 => DoneAction::FreeSelfAndNext,
            5 => DoneAction::FreeSelfAndFreeAllPrev,
            6 => DoneAction::FreeSelfAndFreeAllNext,
            7 => DoneAction::FreeSelfToHead,
            8 => DoneAction::FreeSelfToTail,
            9 => DoneAction::FreeSelfPausePrev,
            10 => DoneAction::FreeSelfPauseNext,
            11 => DoneAction::FreeSelfAndDeepFreePrev,
            12 => DoneAction::FreeSelfAndDeepFreeNext,
            13 => DoneAction::FreeAllInGroup,
            14 => DoneAction::FreeGroup,
            _ => DoneAction::FreeSelf,
        }
    }

    /// Map a scsynth done-action code (carried as a float unit input) to a [`DoneAction`].
    pub fn from_code(code: f32) -> DoneAction {
        DoneAction::from_index(code as i32)
    }

    /// Encode as a small integer tag (the scsynth code), so a unit can hold a `DoneAction` in its
    /// `Pod` state.
    pub fn to_tag(self) -> u32 {
        self as u32
    }

    /// Decode a tag produced by [`DoneAction::to_tag`] (any out-of-range tag maps to `FreeSelf`).
    pub fn from_tag(tag: u32) -> DoneAction {
        DoneAction::from_index(tag as i32)
    }
}

pub use band_limited::{Pulse, Saw};
pub use binary_op::BinaryOp;
pub use buf_wr::BufWr;
pub use delay::{Delay, FeedbackDelay};
pub use demand::{
    Dbufrd, Dbufwr, Demand, DemandAccess, DemandCtx, DemandUnit, DemandVtbl, DemandWorld, Dpoll,
    Dseq, Dseries, Duty, Dwhite, demand_next, demand_reset,
};
pub use disk_in::DiskIn;
pub use disk_out::DiskOut;
pub use env::EnvGen;
#[cfg(feature = "fft")]
pub use fft::{Fft, Ifft};
pub use filter::Butter;
pub use info::{BufInfo, BufInfoKind, Info, InfoKind};
pub use input::In;
pub use io::{
    audio_channel_mut, audio_in, audio_in_touched, audio_out, audio_out_decimated,
    audio_replace_decimated, audio_touch, buffer_at, buffer_at_mut, buffer_pair_mut,
    control_crossfade, control_in, control_in_touched, control_out, control_replace, local_in,
    local_out, num_audio_buses, num_buffers, num_control_buses, num_input_buses, num_output_buses,
    recording_at_mut, stream_at_mut,
};
pub use lf::{Impulse, LFPulse, LFSaw};
pub use line::Line;
pub use local_buf::{ClearBuf, LocalBuf, MaxLocalBufs, SetBuf};
pub use local_io::{LocalIn, LocalOut};
pub use node_ctl::{Done, Free, Pause, SelfTrig, WhenDone};
pub use noise::WhiteNoise;
pub use out::{OffsetOut, Out};
pub use pan::Pan2;
pub use play_buf::PlayBuf;
#[cfg(feature = "fft")]
pub use pv_mag_mul::PvMagMul;
#[cfg(feature = "fft")]
pub use pv_mag_squared::PvMagSquared;
pub use rate_conv::{A2K, Dc, K2A, T2A};
pub use record_buf::RecordBuf;
pub use registry::{BuildContext, DemandUnitDef, UnitDef, UnitRegistry};
pub use send_reply::SendReply;
pub use send_trig::SendTrig;
pub use sin_osc::SinOsc;
pub use unary_op::UnaryOp;
pub use util::{Amplitude, Lag, MulAdd};

/// A trigger a `SendTrig` unit fires on a rising edge: the enclosing node's id, the user-supplied
/// trigger id, and the value sampled at the edge. The engine surfaces each as a `/tr` message.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Trigger {
    /// The enclosing synth's node id.
    pub node: i32,
    /// The user-supplied trigger id (`SendTrig`'s second argument).
    pub id: i32,
    /// The value sampled at the trigger (`SendTrig`'s third argument).
    pub value: f32,
}

/// A bounded, allocation-free sink a unit pushes [`Trigger`]s into during one control block. It wraps
/// a caller-owned `Vec` that the engine drains after the tree walk. Pushes past `capacity` are
/// dropped - a `/tr` is best-effort, like scsynth's trigger FIFO under load - so the audio thread
/// never reallocates.
pub struct TriggerSink<'a> {
    buf: &'a mut Vec<Trigger>,
    capacity: usize,
}

impl<'a> TriggerSink<'a> {
    /// Wrap `buf`, capping the block at `capacity` triggers.
    pub fn new(buf: &'a mut Vec<Trigger>, capacity: usize) -> Self {
        TriggerSink { buf, capacity }
    }

    /// Record `trigger`, unless the block's capacity is already reached (then drop it).
    pub fn push(&mut self, trigger: Trigger) {
        if self.buf.len() < self.capacity {
            self.buf.push(trigger);
        }
    }
}

/// Maximum bytes in a [`NodeMsg`] label (an OSC path for `SendReply`). A unit whose label is longer
/// is rejected at compile time - plyphon bounds the carrier so the audio thread never allocates.
pub const MAX_LABEL: usize = 32;
/// Maximum values a [`NodeMsg`] carries (`SendReply`'s value count). Over-long is rejected at build.
pub const MAX_VALUES: usize = 32;

/// What kind of host message a [`NodeMsg`] is - which decides how the dispatcher surfaces it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NodeMsgKind {
    /// `SendReply`: emit an OSC message `/<label> [node, reply_id, values...]`.
    Reply,
    /// `Poll`/`Dpoll`: post `label: value` to the host's console (no OSC form). `reply_id` carries the
    /// optional trigger id (scsynth's `trigid`); the polled value is `values[0]`.
    Poll,
}

/// A message a unit emits from the audio thread for the host - `SendReply`'s analogue of [`Trigger`],
/// but carrying a custom OSC path (`label`) and a bounded number of `values`. Unlike scsynth's
/// `NodeReplyMsg` (a pointer to RT-allocated memory), the path and values are **inline**, so the whole
/// message is one `Copy` value pushed onto a ring with no audio-thread allocation. The path is a
/// compile-time constant baked into the emitting unit, copied in here when it fires.
#[derive(Copy, Clone, Debug)]
pub struct NodeMsg {
    /// The enclosing synth's node id.
    pub node: i32,
    /// `SendReply`'s reply id (echoed in the OSC reply).
    pub reply_id: i32,
    /// How the host should surface this message.
    pub kind: NodeMsgKind,
    /// The OSC path bytes (UTF-8), the first `label_len` of which are valid.
    pub label: [u8; MAX_LABEL],
    /// Valid byte length of `label` (`<= MAX_LABEL`).
    pub label_len: u32,
    /// The emitted values, the first `num_values` of which are valid.
    pub values: [f32; MAX_VALUES],
    /// Valid length of `values` (`<= MAX_VALUES`).
    pub num_values: u32,
}

/// A bounded, allocation-free sink a unit pushes [`NodeMsg`]s into during one control block - the
/// custom-path analogue of [`TriggerSink`]. The engine drains it to a ring after the tree walk;
/// pushes past `capacity` are dropped (best-effort, like `/tr`) so the audio thread never reallocates.
pub struct NodeMsgSink<'a> {
    buf: &'a mut Vec<NodeMsg>,
    capacity: usize,
}

impl<'a> NodeMsgSink<'a> {
    /// Wrap `buf`, capping the block at `capacity` messages.
    pub fn new(buf: &'a mut Vec<NodeMsg>, capacity: usize) -> Self {
        NodeMsgSink { buf, capacity }
    }

    /// Record `msg`, unless the block's capacity is already reached (then drop it).
    pub fn push(&mut self, msg: NodeMsg) {
        if self.buf.len() < self.capacity {
            self.buf.push(msg);
        }
    }

    /// A shorter-lived view over the same buffer, so the sink can be threaded through a nested borrow
    /// (e.g. into a demand pull, via [`DemandCtx`]) without moving it.
    pub fn reborrow(&mut self) -> NodeMsgSink<'_> {
        NodeMsgSink {
            buf: &mut *self.buf,
            capacity: self.capacity,
        }
    }
}

/// A unit's window onto every unit's "done" flag for the block - plyphon's port of scsynth's
/// per-`Unit` `mDone`. A producer marks *its own* completion with [`mark_done`](Self::mark_done); a
/// watcher (`Done`/`FreeSelfWhenDone`/`PauseSelfWhenDone`) reads a source unit's flag with
/// [`is_done`](Self::is_done), using the source unit index the compiler captured. Flags live in the
/// rt-pool block and persist across blocks (the process loop carries each unit's flag forward), so a
/// unit that finishes stays done. A source is calc-ordered before its watcher, so the watcher reads
/// the current block's value.
pub struct DoneState<'a> {
    /// Every calc unit's done flag, indexed by calc-unit position (read-only).
    flags: &'a [u32],
    /// This unit's own flag; the process loop persists it back into `flags` after the unit runs.
    own: &'a mut u32,
}

impl<'a> DoneState<'a> {
    /// Wrap the block's done flags and this unit's own slot. Used by the synth process loop.
    pub fn new(flags: &'a [u32], own: &'a mut u32) -> Self {
        DoneState { flags, own }
    }

    /// Mark this unit done (scsynth's `unit->mDone = true`). Idempotent.
    pub fn mark_done(&mut self) {
        *self.own = 1;
    }

    /// Clear this unit's done flag (scsynth's `unit->mDone = false` on retrigger) - a retriggered
    /// producer (e.g. `EnvGen` on a rising gate) is no longer done. Idempotent.
    pub fn clear_done(&mut self) {
        *self.own = 0;
    }

    /// Whether calc unit `index` has finished (scsynth's `src->mDone`). Out of range reads `false`.
    pub fn is_done(&self, index: usize) -> bool {
        self.flags.get(index).is_some_and(|&flag| flag != 0)
    }
}

/// What a `Free`/`Pause` unit asks the engine to do to *another* node (by id), applied after the
/// block - the analogue of scsynth's `NodeEnd`/`NodeRun` calls from within `Free`/`Pause`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NodeOpKind {
    /// Free the node (scsynth's `Free`).
    Free,
    /// Set the node's run state: `false` pauses, `true` resumes (scsynth's `Pause`).
    Run(bool),
}

/// A deferred node operation a `Free`/`Pause` unit emits this block: the target node id and what to
/// do to it. The engine applies these after the tree walk (it cannot relink the tree mid-walk).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeOp {
    /// The target node id.
    pub node: i32,
    /// The operation to apply.
    pub kind: NodeOpKind,
}

/// A bounded, allocation-free sink for [`NodeOp`]s emitted during one control block - the by-id
/// analogue of [`TriggerSink`]. The engine drains it after the tree walk; pushes past `capacity` are
/// dropped so the audio thread never reallocates.
pub struct NodeOpSink<'a> {
    buf: &'a mut Vec<NodeOp>,
    capacity: usize,
}

impl<'a> NodeOpSink<'a> {
    /// Wrap `buf`, capping the block at `capacity` node ops.
    pub fn new(buf: &'a mut Vec<NodeOp>, capacity: usize) -> Self {
        NodeOpSink { buf, capacity }
    }

    /// Record `op`, unless the block's capacity is already reached (then drop it).
    pub fn push(&mut self, op: NodeOp) {
        if self.buf.len() < self.capacity {
            self.buf.push(op);
        }
    }
}

/// A synth's private feedback bus for `LocalIn`/`LocalOut` - scsynth's local buffers. It lives in the
/// per-instance pool block and **persists across blocks**, so a `LocalIn` reads the value the
/// `LocalOut` wrote *last* block (a one-block feedback delay). Channel-major: channel `ch` occupies
/// `data[ch*block_size .. (ch+1)*block_size]`. Units touch it only through the crate-private
/// [`io::local_in`]/[`io::local_out`] free fns.
pub struct LocalBus<'a> {
    data: &'a mut [f32],
    block_size: usize,
}

impl<'a> LocalBus<'a> {
    /// Wrap the block's local-bus span. Used by the synth process loop.
    pub fn new(data: &'a mut [f32], block_size: usize) -> Self {
        LocalBus { data, block_size }
    }

    /// Number of local channels (0 when the synth has no `LocalIn`/`LocalOut`).
    pub fn num_channels(&self) -> usize {
        self.data.len().checked_div(self.block_size).unwrap_or(0)
    }

    /// Local channel `ch` for this block (read), or an empty slice if out of range.
    pub(crate) fn channel(&self, ch: usize) -> &[f32] {
        if ch < self.num_channels() {
            let start = ch * self.block_size;
            &self.data[start..start + self.block_size]
        } else {
            &[]
        }
    }

    /// Local channel `ch` for this block (write), or `None` if out of range.
    pub(crate) fn channel_mut(&mut self, ch: usize) -> Option<&mut [f32]> {
        if ch < self.num_channels() {
            let start = ch * self.block_size;
            Some(&mut self.data[start..start + self.block_size])
        } else {
            None
        }
    }
}

/// A synth's graph-local buffers (`LocalBuf`) for the block - plyphon's port of scsynth's
/// `parent->mLocalSndBufs`. The sample storage and each buffer's [`LocalBufMeta`] live in the
/// per-instance pool block and **persist across blocks**; each `LocalBuf` records its shape there
/// when the synth starts. A local buffer's number is `buffer-table capacity + index`, and the io
/// free fns ([`buffer_at`]/[`buffer_at_mut`]/[`buffer_pair_mut`]) resolve such a number here, so
/// every buffer consumer works on local buffers unchanged. Empty for synths with no `LocalBuf`.
pub struct LocalBufs<'a> {
    /// Each local buffer's record, in declaration order.
    meta: &'a mut [LocalBufMeta],
    /// The sample storage: every local buffer packed at its record's offset.
    samples: &'a mut [f32],
    /// The graph's audio sample rate - a local buffer's own rate (scsynth's `FULLRATE`).
    sample_rate: f64,
}

impl<'a> LocalBufs<'a> {
    /// Wrap the block's local-buffer spans. Used by the synth process loop.
    pub fn new(meta: &'a mut [LocalBufMeta], samples: &'a mut [f32], sample_rate: f64) -> Self {
        LocalBufs {
            meta,
            samples,
            sample_rate,
        }
    }

    /// A shorter-lived handle over the same storage, so it can be threaded through a nested borrow
    /// (a demand pull, via [`DemandWorld`]) without moving it.
    pub fn reborrow(&mut self) -> LocalBufs<'_> {
        LocalBufs {
            meta: &mut *self.meta,
            samples: &mut *self.samples,
            sample_rate: self.sample_rate,
        }
    }

    /// Number of graph-local buffers.
    pub fn len(&self) -> usize {
        self.meta.len()
    }

    /// Whether the synth declared no local buffers.
    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// Whether local buffer `index` has storage: its `LocalBuf` has started and its allocation
    /// succeeded.
    pub fn is_live(&self, index: usize) -> bool {
        self.meta.get(index).is_some_and(|meta| meta.live != 0)
    }

    /// The record and sample range of local buffer `index`, if it has storage. The bounds checks
    /// keep the accessors below panic-free on the audio thread even for a malformed record.
    fn range(&self, index: usize) -> Option<(LocalBufMeta, core::ops::Range<usize>)> {
        let meta = *self.meta.get(index)?;
        if meta.live == 0 {
            return None;
        }
        let len = (meta.channels as usize).checked_mul(meta.frames as usize)?;
        let start = meta.offset as usize;
        let range = start..start.checked_add(len)?;
        (range.end <= self.samples.len()).then_some((meta, range))
    }

    /// Local buffer `index` as a read-only view, or `None` if it has no storage.
    pub(crate) fn view(&self, index: usize) -> Option<BufView<'_>> {
        let (meta, range) = self.range(index)?;
        Some(BufView::from_parts(
            &self.samples[range],
            meta.frames as usize,
            meta.channels as usize,
            self.sample_rate,
            SpectrumCoord::from_tag(meta.coord),
        ))
    }

    /// Local buffer `index` as a mutable view, or `None` if it has no storage.
    pub(crate) fn view_mut(&mut self, index: usize) -> Option<BufViewMut<'_>> {
        let (meta, range) = self.range(index)?;
        let samples = &mut self.samples[range];
        Some(BufViewMut::from_tagged_parts(
            samples,
            meta.frames as usize,
            meta.channels as usize,
            self.sample_rate,
            &mut self.meta[index].coord,
        ))
    }

    /// Local buffer `a` mutably and local buffer `b` read-only, as disjoint borrows - the local
    /// counterpart of `BufferTable::pair_mut`. `None` unless `a != b` and both have storage.
    pub(crate) fn pair_mut(&mut self, a: usize, b: usize) -> Option<(BufViewMut<'_>, BufView<'_>)> {
        if a == b {
            return None;
        }
        let (a_meta, a_range) = self.range(a)?;
        let (b_meta, b_range) = self.range(b)?;
        // Each buffer is appended at a distinct offset, so the two ranges are disjoint by
        // construction and the split never fails.
        let [a_samples, b_samples] = self.samples.get_disjoint_mut([a_range, b_range]).ok()?;
        Some((
            BufViewMut::from_tagged_parts(
                a_samples,
                a_meta.frames as usize,
                a_meta.channels as usize,
                self.sample_rate,
                &mut self.meta[a].coord,
            ),
            BufView::from_parts(
                b_samples,
                b_meta.frames as usize,
                b_meta.channels as usize,
                self.sample_rate,
                SpectrumCoord::from_tag(b_meta.coord),
            ),
        ))
    }
}

/// A unit's private auxiliary memory for the block (a delay line, a circular buffer) - the safe
/// stand-in for scsynth's `RTAlloc`'d `float* m_dlybuf`. It comes from one of two places:
///
/// - a fixed-size region reserved at build time in the per-instance pool block (see
///   [`unit_spec_aux`]), for memory whose size does not depend on any input;
/// - a region the unit allocates itself, sized from live inputs, through [`Aux::alloc`] (see
///   [`unit_spec_pool`] and [`Unit::alloc`]), exactly as a scsynth constructor calls `RTAlloc`.
///
/// Either way the bytes **persist across blocks**, so a delay reads back what earlier blocks wrote,
/// and are **not** zeroed (they may hold a previous tenant's data), so a unit guards its first reads
/// with a cold-start counter in its own state - exactly as scsynth's `_z` calc variants do. Empty for
/// units with no memory, and for a pool-sized unit that has not allocated yet.
pub struct Aux<'a> {
    inner: AuxInner<'a>,
}

enum AuxInner<'a> {
    /// The unit's memory, already resolved to its bytes.
    Bytes(&'a mut [u8]),
    /// A pool-sized unit that has not allocated yet, reaching the engine's allocator.
    Pending(&'a mut dyn AuxAlloc),
}

/// The engine's side of a pool-sized unit's not-yet-allocated [`Aux`]: a one-time allocation from
/// the engine's real-time pool (scsynth's `RTAlloc`). Implemented by the engine; units only reach it
/// through [`Aux::alloc`].
pub trait AuxAlloc {
    /// Allocate `bytes` for this unit unless it already has memory. Returns whether the unit now has
    /// memory; `false` means the pool could not satisfy the request, and the engine silences the unit
    /// (scsynth's `ClearUnitOnMemFailed`).
    fn alloc(&mut self, bytes: usize) -> bool;

    /// This unit's memory: empty until [`alloc`](Self::alloc) succeeds.
    fn bytes(&mut self) -> &mut [u8];
}

impl<'a> Aux<'a> {
    /// Wrap this unit's resolved memory. Used by the synth process loop.
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Aux {
            inner: AuxInner::Bytes(bytes),
        }
    }

    /// Wrap a pool-sized unit's allocator before it has allocated. Used by the synth process loop.
    pub fn pending(alloc: &'a mut dyn AuxAlloc) -> Self {
        Aux {
            inner: AuxInner::Pending(alloc),
        }
    }

    /// Allocate `bytes` of memory for this unit, once - scsynth's `RTAlloc`, for memory sized from
    /// live inputs. Only a unit built with [`unit_spec_pool`] can allocate; for it, a later call once
    /// memory exists is a no-op returning `true`. Returns `false` when the pool cannot satisfy the
    /// request (or the unit cannot allocate): the unit then outputs silence for the rest of its life,
    /// as scsynth's `ClearUnitIfMemFailed` does.
    pub fn alloc(&mut self, bytes: usize) -> bool {
        match &mut self.inner {
            AuxInner::Bytes(b) => !b.is_empty(),
            AuxInner::Pending(a) => a.alloc(bytes),
        }
    }

    fn bytes(&mut self) -> &mut [u8] {
        match &mut self.inner {
            AuxInner::Bytes(b) => b,
            AuxInner::Pending(a) => a.bytes(),
        }
    }

    /// Whether this unit has no memory (none declared, or not allocated yet).
    pub fn is_empty(&mut self) -> bool {
        self.bytes().is_empty()
    }

    /// The memory as an `f32` slice (the usual delay-line element type). Its length is the region's
    /// bytes over 4; an empty region yields an empty slice. Never panics - a malformed region (the
    /// architecture rules this out for a well-built unit) yields an empty slice rather than aborting
    /// the audio thread.
    pub fn f32_mut(&mut self) -> &mut [f32] {
        self.cast_mut()
    }

    /// The memory as a slice of `Pod` elements `T` (e.g. a bank of grains). Its length is the
    /// region's bytes over `size_of::<T>()`; a region whose size or alignment does not fit `T` yields
    /// an empty slice rather than aborting the audio thread. A reserved region is sized and aligned
    /// for `T` by its unit (`aux_bytes`/`aux_align` in [`unit_spec_aux`]); an allocated one is
    /// 64-byte aligned, so only its size matters.
    pub fn cast_mut<T: bytemuck::Pod>(&mut self) -> &mut [T] {
        bytemuck::try_cast_slice_mut(self.bytes()).unwrap_or(&mut [])
    }
}

/// Everything a unit touches while processing one control block - plyphon's safe decomposition of
/// scsynth's `unit` (which reaches inputs, outputs, and the world through one pointer).
///
/// The signal ports ([`ins`](Self::ins)/[`outs`](Self::outs)) and engine constants are plain fields.
/// The shared [`buses`](Self::buses)/[`buffers`](Self::buffers) are fields too, but their dangerous
/// mutators are crate-private - a unit touches them only through the audited free fns in
/// [`io`], so it cannot resize a bus or swap a buffer. Those fns take individual
/// fields rather than `&self`, so reading `ins` and writing `buses` in one expression borrows
/// disjoint fields.
pub struct ProcessCtx<'a> {
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// *This unit's* rate constants (scsynth's `unit->mRate`): [`audio`](Self::audio) for an `.ar`
    /// unit, [`control`](Self::control) for a `.kr` one. Step sizing and seconds-to-samples
    /// conversions use this, so a unit advancing once per control period counts control periods;
    /// `audio`/`control` remain for units that genuinely need a specific rate (boundary I/O, FFT
    /// hop math, the `Info` units).
    pub own: &'a RateInfo,
    /// Shared wavetables (sine, ...), owned by the engine.
    pub wavetables: &'a Wavetables,
    /// Shared FFT plans + windows (`FFT`/`IFFT`/`PV_*`); empty without the `fft` feature. Most units
    /// ignore it.
    pub fft: &'a FftTables,
    /// This unit's inputs for the block (read-only).
    pub ins: Inputs<'a>,
    /// This unit's output scratch for the block.
    pub outs: Outputs<'a>,
    /// The World's shared buses, via the [`io`] free fns (`In`/`Out`).
    pub buses: &'a mut Buses,
    /// The World's shared buffer table, via the [`io`] free fns (`PlayBuf`/`DiskIn`).
    pub buffers: &'a mut BufferTable,
    /// The current block counter (stamps bus writes: the first writer clears, the rest sum).
    pub buf_counter: u64,
    /// Which sub-block tick this is, for a reblocked graph: `0..num_ticks`. Always `0` for an ordinary
    /// def (one tick per World block). The boundary I/O units (`In`/`Out`) use it with their block
    /// size (`audio.block_size`) and [`resample_factor`](Self::resample_factor) to find this tick's
    /// slice of the World-block-wide bus channel; every other unit ignores it.
    pub tick: usize,
    /// The graph's oversample factor (scsynth's `Resample(n)`): the graph runs at `factor`x the World
    /// sample rate. `1` for an ordinary def. The boundary I/O units use it to decimate (`Out`) or
    /// zero-order-hold (`In`) between the World-rate bus and the graph-rate wire; others ignore it.
    pub resample_factor: usize,
    /// The sample offset within this block at which the enclosing synth was created (scsynth's
    /// `mSampleOffset`). It is non-zero only on the first block of a synth scheduled mid-block, and
    /// only `OffsetOut` acts on it - to delay the onset to that exact sample. Most units ignore it.
    pub sample_offset: usize,
    /// The fractional (sub-sample) part of [`sample_offset`](Self::sample_offset) (scsynth's
    /// `mSubsampleOffset`), in `[0, 1)`. Like `sample_offset` it is non-zero only on the first block
    /// of a synth scheduled mid-block; `SubsampleOffset` is its only reader (it snapshots the value
    /// for the synth's life). Most units ignore it.
    pub subsample_offset: f32,
    /// Handle to the synth's demand plan. A demand-rate consumer (`Demand`/`Duty`) pulls demand
    /// sources through this with the [`demand_next`] / [`demand_reset`] free fns; other units ignore
    /// it. Empty for synths with no demand units.
    pub demand: DemandAccess<'a>,
    /// The enclosing synth's node id (`-1` if unknown), so a side-effecting unit (`SendTrig`) can tag
    /// its `/tr` with the node that fired it. Most units ignore it.
    pub node_id: i32,
    /// Sink for triggers a unit fires this block (`SendTrig`). Most units ignore it.
    pub triggers: TriggerSink<'a>,
    /// Sink for custom-path host messages a unit emits this block (`SendReply`). Most units ignore it.
    pub node_msgs: NodeMsgSink<'a>,
    /// Number of synths running at the start of this block (`NumRunningSynths`), snapshotted before
    /// the tree walk. Most units ignore it.
    pub running_synths: usize,
    /// This block's per-unit done flags (scsynth's `mDone`): a producer marks itself done, a watcher
    /// reads a source unit's flag. Most units ignore it.
    pub done: DoneState<'a>,
    /// Sink for node operations (`Free`/`Pause` by id) a unit emits this block, applied after the
    /// tree walk. Most units ignore it.
    pub node_ops: NodeOpSink<'a>,
    /// The synth's private feedback bus (`LocalIn`/`LocalOut`). Empty for synths with no local bus;
    /// most units ignore it.
    pub local: LocalBus<'a>,
    /// The synth's graph-local buffers (`LocalBuf`). The buffer io free fns take this alongside
    /// [`buffers`](Self::buffers) so a past-capacity buffer number resolves here transparently.
    /// Empty for synths with no `LocalBuf`; most units only pass it through.
    pub local_bufs: LocalBufs<'a>,
    /// This unit's private auxiliary memory (a delay line). Empty for units that declared none; most
    /// units ignore it.
    pub aux: Aux<'a>,
    /// The random stream the synth draws from (scsynth's `mParent->mRGen`): one of the World's
    /// streams, shared with every synth drawing from the same one. Every random unit draws from it
    /// and `RandSeed` re-seeds it. A unit drawing through a block can copy it into a local and
    /// write it back, as scsynth's `RGET`/`RPUT` do.
    pub rgen: &'a mut Rng,
    /// Which of the World's random streams the synth draws from, for `RandID` to change.
    pub rgen_id: RgenId<'a>,
}

/// Which of the World's random streams a synth draws from - scsynth's `mParent->mRGen`, a pointer
/// into `world->mRGen`. `RandID` repoints it; the units after it draw from the new stream.
pub struct RgenId<'a> {
    index: &'a mut u32,
    count: u32,
}

impl<'a> RgenId<'a> {
    /// A handle over the synth's stream `index`, among the World's `count` streams. Used by the
    /// synth process loop.
    pub fn new(index: &'a mut u32, count: u32) -> Self {
        RgenId { index, count }
    }

    /// Draw from stream `id` from now on. An `id` the World does not have is ignored, as
    /// scsynth's `RandID` ignores one at or beyond `mNumRGens`.
    pub fn select(&mut self, id: u32) {
        if id < self.count {
            *self.index = id;
        }
    }
}

/// What a unit may touch while sizing its memory on the first block - see [`Unit::alloc`].
///
/// Like [`ProcessCtx`] but read-only on the world and without [`outs`](ProcessCtx::outs): `alloc`
/// sizes the unit's memory from live inputs; it does not produce output or mutate the world.
pub struct InitCtx<'a> {
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// *This unit's* rate constants (scsynth's `unit->mRate`) - see [`ProcessCtx::own`].
    pub own: &'a RateInfo,
    /// Shared wavetables.
    pub wavetables: &'a Wavetables,
    /// Shared FFT plans + windows (empty without the `fft` feature).
    pub fft: &'a FftTables,
    /// This unit's inputs for the block (read-only).
    pub ins: Inputs<'a>,
    /// The World's shared buses (read-only), via the [`io`] free fns.
    pub buses: &'a Buses,
    /// The World's shared buffer table (read-only), via the [`io`] free fns.
    pub buffers: &'a BufferTable,
    /// The synth's graph-local buffers (`LocalBuf`), so an init-time buffer resolution (a
    /// buffer-backed delay clamping to its line) sees local buffers too. Read-only here: the handle's
    /// mutators need `&mut`, unreachable through the shared `InitCtx` a unit's `init` receives.
    pub local_bufs: LocalBufs<'a>,
    /// The current block counter.
    pub buf_counter: u64,
}

/// How a single unit input is sourced. Resolved once at build time from the SynthDef.
#[derive(Copy, Clone, Debug)]
pub enum InputSource {
    /// A constant baked into the SynthDef.
    Constant(f32),
    /// A control-rate wire (index into the synth's control wires).
    Control(u32),
    /// An audio-rate wire (index into the synth's audio wires).
    Audio(u32),
    /// A demand-rate unit (index into the synth's demand plan). Such an input has no wire: a consumer
    /// reads it with the [`demand_next`]/[`demand_reset`] free fns, which pull the source on the audio
    /// thread.
    Demand(u32),
}

impl InputSource {
    /// The calculation rate this source presents to a consuming unit.
    pub fn rate(self) -> Rate {
        match self {
            InputSource::Constant(_) => Rate::Scalar,
            InputSource::Control(_) => Rate::Control,
            InputSource::Audio(_) => Rate::Audio,
            InputSource::Demand(_) => Rate::Demand,
        }
    }
}

/// Read-only view of a unit's inputs for one block.
///
/// A small bundle of borrows (hence `Copy`). Audio wires are stored flat; wire `w` occupies
/// `audio_wires[w*bs .. (w+1)*bs]`.
#[derive(Copy, Clone)]
pub struct Inputs<'a> {
    sources: &'a [InputSource],
    audio_wires: &'a [f32],
    control_wires: &'a [f32],
    block_size: usize,
    /// Samples an audio input yields: `block_size`, or one during the constructor calc.
    calc_len: usize,
}

impl<'a> Inputs<'a> {
    /// Construct an input view. Used by the synth process loop.
    pub fn new(
        sources: &'a [InputSource],
        audio_wires: &'a [f32],
        control_wires: &'a [f32],
        block_size: usize,
    ) -> Self {
        Inputs {
            sources,
            audio_wires,
            control_wires,
            block_size,
            calc_len: block_size,
        }
    }

    /// The same view yielding `len` samples per audio input instead of a whole block - the
    /// constructor calc, which scsynth runs for exactly one sample.
    pub fn with_len(mut self, len: usize) -> Self {
        self.calc_len = len.min(self.block_size);
        self
    }

    /// Number of inputs.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether there are no inputs.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// The calculation rate of input `i`.
    pub fn rate(&self, i: usize) -> Rate {
        self.sources[i].rate()
    }

    /// How input `i` is sourced (constant, wire, or demand unit). A consumer uses this to route a
    /// demand input through the [`demand_next`] free fn rather than reading a wire.
    pub fn source(&self, i: usize) -> InputSource {
        self.sources[i]
    }

    /// Audio-rate input `i` as a slice of the calc length: `block_size`, or one sample during the
    /// constructor calc.
    ///
    /// Only meaningful when input `i` is audio-rate; units select by [`Inputs::rate`] (they chose
    /// their calc variant at build time from these same rates), so a correctly-built graph never
    /// calls this on a non-audio input. A non-audio input yields an empty slice rather than panic.
    pub fn audio(&self, i: usize) -> &'a [f32] {
        match self.sources[i] {
            InputSource::Audio(w) => {
                let start = w as usize * self.block_size;
                &self.audio_wires[start..start + self.calc_len]
            }
            _ => &self.audio_wires[..0],
        }
    }

    /// The single value of a constant or control-rate input `i`.
    ///
    /// An audio-rate input collapses to its first sample (scsynth's `IN0`). A demand-rate input has
    /// no wire to read - it yields 0; a consumer must pull it via the [`demand_next`] free fn instead.
    pub fn control(&self, i: usize) -> f32 {
        match self.sources[i] {
            InputSource::Constant(v) => v,
            InputSource::Control(w) => self.control_wires[w as usize],
            InputSource::Audio(w) => self.audio_wires[w as usize * self.block_size],
            InputSource::Demand(_) => 0.0,
        }
    }
}

/// Mutable view of a unit's output wires for one block.
///
/// Outputs are written into pre-allocated scratch (disjoint from the input wires), then the synth
/// process loop copies them into the arena. Output `i` occupies `scratch[i*len .. (i+1)*len]`,
/// where `len` is the unit's calc length: the full block for an audio-rate unit, **one sample**
/// for a control-rate one (scsynth's `inNumSamples == 1`) - so a `.kr` unit's block loop runs
/// once, computing exactly the one value it publishes.
pub struct Outputs<'a> {
    scratch: &'a mut [f32],
    block_size: usize,
}

impl<'a> Outputs<'a> {
    /// Construct an output view over `scratch` at the owning unit's calc length. Used by the synth
    /// process loop.
    pub fn new(scratch: &'a mut [f32], block_size: usize) -> Self {
        Outputs {
            scratch,
            block_size,
        }
    }

    /// Audio-rate output `i` as a mutable slice to write into: `block_size` samples for an
    /// audio-rate unit, a 1-sample slice for a control-rate one (whose per-sample loop then runs
    /// exactly once).
    pub fn audio(&mut self, i: usize) -> &mut [f32] {
        let start = i * self.block_size;
        &mut self.scratch[start..start + self.block_size]
    }

    /// Control-rate output `i` as a single mutable value to write (the output's first scratch
    /// slot, which the synth process loop publishes to its control wire).
    pub fn control(&mut self, i: usize) -> &mut f32 {
        &mut self.scratch[i * self.block_size]
    }

    /// Two distinct audio outputs as simultaneous mutable slices, so an accumulate spanning two
    /// channels (a stereo-panned grain) binds both once per render rather than re-slicing per
    /// sample. `None` when `a == b` or either output is out of scratch range.
    pub fn audio_pair(&mut self, a: usize, b: usize) -> Option<(&mut [f32], &mut [f32])> {
        let bs = self.block_size;
        let [sa, sb] = self
            .scratch
            .get_disjoint_mut([a * bs..(a + 1) * bs, b * bs..(b + 1) * bs])
            .ok()?;
        Some((sa, sb))
    }
}

/// A unit generator - plyphon's `Unit` is scsynth's server-side `Unit` (the language-side `UGen` has
/// no plyphon analogue; we consume compiled SynthDefs directly). Its state must be [`Pod`] so it can
/// live as bytes in the rt-pool and be reinterpreted without `unsafe`; behaviour is invoked through
/// the [`ProcessFn`]/[`InitFn`] vtable a [`UnitDef`] builds via [`unit_spec`].
pub trait Unit: Pod {
    /// Construct the unit - scsynth's `*_Ctor`.
    ///
    /// scsynth's `Graph_FirstCalc` runs every unit's constructor, in SynthDef order, before any
    /// unit's first calc. plyphon does the same on the synth's first block: each unit's `init` runs
    /// on the audio thread, in SynthDef order, right after [`Unit::alloc`], before any
    /// [`Unit::process`]. `ctx` views exactly one sample, as a constructor's calc does
    /// (`inNumSamples == 1`): every input reads its real starting value - constants, control
    /// parameters (including `/s_new` args and `/n_map`ped buses), and the sample each earlier
    /// unit's constructor wrote - which is what a constructor reads with `ZIN0`. The one sample
    /// `init` writes to `ctx.outs` is what later constructors read; the first block then
    /// overwrites it.
    ///
    /// Stateful units seed here, so their first block is already correct - e.g. a smoother starts
    /// *at* its input rather than ramping up from zero, which is what avoids onset clicks. Most
    /// scsynth constructors then run the calc for one sample, and so does the default. A unit whose
    /// constructor does anything else overrides it to match: writing a value without advancing,
    /// clearing its output, or running the calc and putting its state back.
    ///
    /// Like [`Unit::process`], `init` must not allocate, block, or take locks; memory sized from
    /// inputs is allocated just before, in [`Unit::alloc`].
    #[must_use]
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.process(ctx)
    }

    /// Allocate the unit's input-sized memory - scsynth's constructor-time `RTAlloc`.
    ///
    /// Called once, in SynthDef order on the synth's first block, immediately before
    /// [`Unit::init`], for units built
    /// with [`unit_spec_pool`]. Like `init` it sees live inputs, so a size read from `ctx.ins` is the
    /// first-sample value a scsynth constructor reads with `ZIN0`. The unit computes its size, calls
    /// [`Aux::alloc`], and may prepare the fresh memory (or derived state such as a wrap mask). If the
    /// allocation fails the engine silences the unit for the rest of its life. The default is a no-op.
    fn alloc(&mut self, _ctx: &InitCtx<'_>, _aux: &mut Aux<'_>) {}

    /// Compute one control block.
    ///
    /// Reads `ctx.ins`, writes `ctx.outs`, and (for I/O units like `In`/`Out`/`PlayBuf`) reads or
    /// writes the World's shared buses and buffers via the [`io`] free fns. Must
    /// not allocate, block, or take locks. Returns the [`DoneAction`] the unit wants applied to its
    /// enclosing synth (almost always [`DoneAction::Nothing`]).
    #[must_use]
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction;
}

/// A type-erased per-block calc function over a unit's pool-resident state bytes - plyphon's
/// `UnitCalcFunc`/`mCalcFunc`. `state` is exactly `size_of::<T>()` bytes, aligned for `T`.
pub type ProcessFn = fn(&mut [u8], &mut ProcessCtx<'_>) -> DoneAction;

/// A type-erased constructor over a unit's pool-resident state bytes (see [`Unit::init`]).
pub type InitFn = fn(&mut [u8], &mut ProcessCtx<'_>) -> DoneAction;

/// A type-erased one-time allocation function over a unit's pool-resident state bytes (see
/// [`Unit::alloc`]).
pub type AllocFn = fn(&mut [u8], &InitCtx<'_>, &mut Aux<'_>);

/// Reinterpret `bytes` as `T` and run its [`Unit::process`]. Monomorphised per `T` and coerced to a
/// [`ProcessFn`]; the cast cannot fail because the slot is sized and aligned for `T` by construction.
fn process_thunk<T: Unit>(bytes: &mut [u8], ctx: &mut ProcessCtx<'_>) -> DoneAction {
    bytemuck::from_bytes_mut::<T>(bytes).process(ctx)
}

/// As [`process_thunk`], for [`Unit::init`].
fn init_thunk<T: Unit>(bytes: &mut [u8], ctx: &mut ProcessCtx<'_>) -> DoneAction {
    bytemuck::from_bytes_mut::<T>(bytes).init(ctx)
}

/// As [`process_thunk`], for [`Unit::alloc`].
fn alloc_thunk<T: Unit>(bytes: &mut [u8], ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
    bytemuck::from_bytes_mut::<T>(bytes).alloc(ctx, aux);
}

/// The body of [`Unit::init`] for a scsynth constructor that runs the calc for one sample and then
/// puts the state it seeded back - `LFSaw_next_k(unit, 1); unit->mPhase = initPhase;` - so later
/// constructors read the sample while the unit's first block starts from the seeded state.
pub fn calc_and_restore<T: Unit>(unit: &mut T, ctx: &mut ProcessCtx<'_>) -> DoneAction {
    let seeded = *unit;
    let action = unit.process(ctx);
    *unit = seeded;
    action
}

/// A built unit: its calc/seed vtable plus the initial state image to copy into the pool. Produced
/// off the audio thread by a [`UnitDef`] (via [`unit_spec`]) and baked into a
/// [`GraphDef`](crate::graphdef::GraphDef).
pub struct BuiltUnit {
    /// Per-block calc function.
    pub process: ProcessFn,
    /// Constructor, run once in SynthDef order on the synth's first block.
    pub init: InitFn,
    /// One-time allocation function for input-sized memory (no-op unless `pool_aux`).
    pub alloc: AllocFn,
    /// Whether this unit allocates its memory from the engine's pool when the synth starts, sized
    /// from live inputs (see [`unit_spec_pool`]). Such a unit reserves no `aux_bytes`.
    pub pool_aux: bool,
    /// The value written to every output of a unit silenced because its allocation failed: `0` for
    /// most units (scsynth's `ClearUnitOutputs`), `-1` for an FFT chain unit, so the units reading
    /// its chain see no ready frame (scsynth's `FFT_ClearUnitOutputs`).
    pub cleared_output: f32,
    /// `size_of::<T>()` - the bytes this unit's state occupies in the arena.
    pub size: usize,
    /// `align_of::<T>()` - the alignment its state slot needs.
    pub align: usize,
    /// The initial state, as bytes to `copy_from_slice` into the slot when a synth is built on-RT.
    pub init_bytes: Box<[u8]>,
    /// Bytes of per-instance auxiliary memory whose size is fixed at build time (a reverb's fixed
    /// lines), summed into the block's `aux` arena at compile time. `0` for units with no such
    /// memory, including units that allocate at synth start (`pool_aux`). Handed to the unit each
    /// block as [`ProcessCtx::aux`].
    pub aux_bytes: usize,
    /// Alignment the aux region needs (e.g. `align_of::<f32>()` for an `f32` delay line). Ignored
    /// when `aux_bytes == 0`.
    pub aux_align: usize,
    /// `Some(shape)` when this unit declares a graph-local buffer (`LocalBuf`): the function that
    /// reads the buffer's `(channels, frames)` from the unit's inputs when the synth starts. The
    /// compile loop numbers the declarations in unit order; on the first block the synth appends
    /// each one's storage to its block. `None` for every other unit.
    pub local_buf: Option<LocalBufShapeFn>,
}

/// Reads a `LocalBuf`'s `(channels, frames)` from its inputs as its constructor runs (scsynth's
/// `(int)IN0(0), (int)IN0(1)` in `LocalBuf_Ctor`).
pub type LocalBufShapeFn = fn(&Inputs<'_>) -> (f32, f32);

/// Build a [`BuiltUnit`] from an initial unit state. The thunks are monomorphised for `T` here, so a
/// [`UnitDef`] only constructs its initial state and hands it to this helper.
pub fn unit_spec<T: Unit>(state: T) -> BuiltUnit {
    BuiltUnit {
        process: process_thunk::<T>,
        init: init_thunk::<T>,
        alloc: alloc_thunk::<T>,
        pool_aux: false,
        cleared_output: 0.0,
        size: core::mem::size_of::<T>(),
        align: core::mem::align_of::<T>(),
        init_bytes: bytemuck::bytes_of(&state).to_vec().into_boxed_slice(),
        aux_bytes: 0,
        aux_align: 1,
        local_buf: None,
    }
}

/// Build a [`BuiltUnit`] that also reserves `aux_bytes` of per-instance auxiliary memory aligned to
/// `aux_align` - memory whose size is fixed at build time, independent of any input (a reverb's
/// fixed lines). Memory sized from inputs uses [`unit_spec_pool`] instead. The unit receives the
/// region as [`ProcessCtx::aux`] each block; it lives in the per-instance pool block and persists
/// across blocks.
///
/// The region is **not** zeroed at instantiation (a large delay line would make that an unbounded
/// audio-thread memset at `/s_new`); like scsynth's `RTAlloc`'d delay buffers, a unit must treat its
/// aux as initially undefined and guard cold-start reads (e.g. with a written-sample counter).
pub fn unit_spec_aux<T: Unit>(state: T, aux_bytes: usize, aux_align: usize) -> BuiltUnit {
    BuiltUnit {
        aux_bytes,
        aux_align: aux_align.max(1),
        ..unit_spec(state)
    }
}

/// Build a [`BuiltUnit`] whose memory is sized from live inputs when the synth starts - scsynth's
/// constructor `RTAlloc`, for a delay's `maxdelaytime` or a pitch shifter's window. The unit
/// allocates in [`Unit::alloc`] through [`Aux::alloc`], from the engine's pool, and receives the
/// region as [`ProcessCtx::aux`] every block after. Unlike [`unit_spec_aux`], the size need not be
/// known at build time: any input, wired or constant, can drive it.
pub fn unit_spec_pool<T: Unit>(state: T) -> BuiltUnit {
    BuiltUnit {
        pool_aux: true,
        ..unit_spec(state)
    }
}

/// Build a [`BuiltUnit`] that declares a graph-local buffer - what a `LocalBuf` returns from its
/// build. `shape` reads the buffer's `(channels, frames)` from the unit's inputs on the synth's
/// first block, just before the unit runs, and the synth appends that much storage to its block
/// then, as scsynth's `LocalBuf_Ctor` `RTAlloc`s it; any input, wired or constant, can size it.
pub fn unit_spec_local_buf<T: Unit>(state: T, shape: LocalBufShapeFn) -> BuiltUnit {
    BuiltUnit {
        local_buf: Some(shape),
        ..unit_spec(state)
    }
}
