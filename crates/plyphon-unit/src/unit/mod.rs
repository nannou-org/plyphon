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

pub mod band_limited;
pub mod binary_op;
pub mod buf_wr;
pub mod chaos;
pub mod decay;
pub mod delay;
pub mod demand;
pub mod disk_in;
pub mod disk_out;
pub mod dynamics;
pub mod env;
#[cfg(feature = "fft")]
pub mod fft;
pub mod filter;
pub mod filter_simple;
pub mod info;
pub mod input;
pub mod io;
pub mod lf;
pub mod line;
pub mod local_io;
pub mod node_ctl;
pub mod noise;
pub mod one_pole;
pub mod out;
pub mod pan;
pub mod play_buf;
#[cfg(feature = "fft")]
pub mod pv;
#[cfg(feature = "fft")]
pub mod pv_mag_mul;
#[cfg(feature = "fft")]
pub mod pv_mag_squared;
pub mod rate_conv;
pub mod record_buf;
pub mod registry;
pub mod resonant;
pub mod scope_out;
pub mod send_reply;
pub mod send_trig;
pub mod shape;
pub mod sin_osc;
pub mod timing;
pub mod trigger;
pub mod two_pole;
pub mod unary_op;
pub mod util;

use alloc::boxed::Box;
use alloc::vec::Vec;

use bytemuck::Pod;

use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::bus::Buses;
use plyphon_dsp::fft::FftTables;
use plyphon_dsp::rate::{Rate, RateInfo};
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
pub use delay::DelayN;
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
    audio_in, audio_out, audio_out_decimated, buffer_at, buffer_at_mut, buffer_pair_mut,
    control_in, control_out, local_in, local_out, num_audio_buses, num_buffers, num_control_buses,
    num_input_buses, num_output_buses, recording_at_mut, stream_at_mut,
};
pub use lf::{Impulse, LFPulse, LFSaw};
pub use line::Line;
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

/// A unit's private auxiliary memory for the block - a delay line / circular buffer sized at build
/// time (see [`unit_spec_aux`]). It is the safe stand-in for scsynth's `RTAlloc`'d `float* m_dlybuf`:
/// the bytes live in the per-instance pool block (so there is still one allocation per synth) and
/// **persist across blocks**, so a delay reads back what earlier blocks wrote.
///
/// Empty for units that declared no aux memory. The bytes are **not** zeroed at instantiation (a
/// recycled block carries a previous tenant's data), so a unit must guard its first reads with a
/// cold-start counter in its own state - exactly as scsynth's `_z` calc variants do.
pub struct Aux<'a> {
    bytes: &'a mut [u8],
}

impl<'a> Aux<'a> {
    /// Wrap this unit's aux byte region. Used by the synth process loop.
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Aux { bytes }
    }

    /// Whether this unit declared no aux memory.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The aux region as an `f32` slice (the usual delay-line element type). Its length is
    /// `aux_bytes / 4`; an empty region yields an empty slice. Never panics - a malformed region
    /// (the architecture rules this out for a well-built unit) yields an empty slice rather than
    /// aborting the audio thread.
    pub fn f32_mut(&mut self) -> &mut [f32] {
        bytemuck::try_cast_slice_mut(self.bytes).unwrap_or(&mut [])
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
    /// This unit's private auxiliary memory (a delay line). Empty for units that declared none; most
    /// units ignore it.
    pub aux: Aux<'a>,
}

/// What a unit may touch while *seeding* state on the first block - see [`Unit::init`].
///
/// Like [`ProcessCtx`] but read-only on the world and without [`outs`](ProcessCtx::outs): `init`
/// seeds the unit's own state from live inputs; it does not produce output or mutate the world.
pub struct InitCtx<'a> {
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
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
        }
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

    /// Audio-rate input `i` as a `block_size` slice.
    ///
    /// Only meaningful when input `i` is audio-rate; units select by [`Inputs::rate`] (they chose
    /// their calc variant at build time from these same rates), so a correctly-built graph never
    /// calls this on a non-audio input. A non-audio input yields an empty slice rather than panic.
    pub fn audio(&self, i: usize) -> &'a [f32] {
        match self.sources[i] {
            InputSource::Audio(w) => {
                let start = w as usize * self.block_size;
                &self.audio_wires[start..start + self.block_size]
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
/// process loop copies them into the arena. Output `i` occupies `scratch[i*bs .. (i+1)*bs]`.
pub struct Outputs<'a> {
    scratch: &'a mut [f32],
    block_size: usize,
}

impl<'a> Outputs<'a> {
    /// Construct an output view over `scratch`. Used by the synth process loop.
    pub fn new(scratch: &'a mut [f32], block_size: usize) -> Self {
        Outputs {
            scratch,
            block_size,
        }
    }

    /// Audio-rate output `i` as a mutable `block_size` slice to write into.
    pub fn audio(&mut self, i: usize) -> &mut [f32] {
        let start = i * self.block_size;
        &mut self.scratch[start..start + self.block_size]
    }

    /// Control-rate output `i` as a single mutable value to write (the first scratch slot, which the
    /// synth process loop publishes to the output's control wire).
    pub fn control(&mut self, i: usize) -> &mut f32 {
        &mut self.scratch[i * self.block_size]
    }
}

/// A unit generator - plyphon's `Unit` is scsynth's server-side `Unit` (the language-side `UGen` has
/// no plyphon analogue; we consume compiled SynthDefs directly). Its state must be [`Pod`] so it can
/// live as bytes in the rt-pool and be reinterpreted without `unsafe`; behaviour is invoked through
/// the [`ProcessFn`]/[`InitFn`] vtable a [`UnitDef`] builds via [`unit_spec`].
pub trait Unit: Pod {
    /// Re-seed any per-instance randomness from `seed`, called once when the synth is constructed on
    /// the audio thread (before the first block). The default is a no-op; units with an
    /// [`Rng`](plyphon_dsp::rng::Rng) override it so that two instances of the same def decorrelate -
    /// plyphon's stand-in for scsynth seeding each `Graph`'s `RGen`. Must not allocate or block.
    fn reseed(&mut self, _seed: u64) {}

    /// Seed state from the unit's initial inputs.
    ///
    /// Called once, on the first control block, in topological order immediately before this unit's
    /// first [`Unit::process`] - on the audio thread, where inputs are live. By then every input is
    /// readable at its real starting value: constants, control parameters (including `/s_new` args
    /// and `/n_map`ped buses), and the first-block outputs of upstream units. Stateful units seed
    /// here so their first block is already correct - e.g. a smoother starts *at* its input rather
    /// than ramping up from zero - which is what avoids onset clicks.
    ///
    /// This mirrors the seeding an scsynth `*_Ctor` does at its first calc; *allocation*, by
    /// contrast, happens earlier and off the audio thread when the unit is built. Like
    /// [`Unit::process`] it must not allocate, block, or take locks. The default is a no-op.
    fn init(&mut self, _ctx: &InitCtx<'_>) {}

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

/// A type-erased one-time seeding function over a unit's pool-resident state bytes (see
/// [`Unit::init`]).
pub type InitFn = fn(&mut [u8], &InitCtx<'_>);

/// A type-erased per-instance re-seed function over a unit's pool-resident state bytes (see
/// [`Unit::reseed`]).
pub type ReseedFn = fn(&mut [u8], u64);

/// Reinterpret `bytes` as `T` and run its [`Unit::process`]. Monomorphised per `T` and coerced to a
/// [`ProcessFn`]; the cast cannot fail because the slot is sized and aligned for `T` by construction.
fn process_thunk<T: Unit>(bytes: &mut [u8], ctx: &mut ProcessCtx<'_>) -> DoneAction {
    bytemuck::from_bytes_mut::<T>(bytes).process(ctx)
}

/// As [`process_thunk`], for [`Unit::init`].
fn init_thunk<T: Unit>(bytes: &mut [u8], ctx: &InitCtx<'_>) {
    bytemuck::from_bytes_mut::<T>(bytes).init(ctx);
}

/// As [`process_thunk`], for [`Unit::reseed`].
fn reseed_thunk<T: Unit>(bytes: &mut [u8], seed: u64) {
    bytemuck::from_bytes_mut::<T>(bytes).reseed(seed);
}

/// A built unit: its calc/seed vtable plus the initial state image to copy into the pool. Produced
/// off the audio thread by a [`UnitDef`] (via [`unit_spec`]) and baked into a
/// [`GraphDef`](crate::graphdef::GraphDef).
pub struct BuiltUnit {
    /// Per-block calc function.
    pub process: ProcessFn,
    /// One-time first-block seeding function.
    pub init: InitFn,
    /// Per-instance re-seed function (no-op for units without randomness).
    pub reseed: ReseedFn,
    /// `size_of::<T>()` - the bytes this unit's state occupies in the arena.
    pub size: usize,
    /// `align_of::<T>()` - the alignment its state slot needs.
    pub align: usize,
    /// The initial state, as bytes to `copy_from_slice` into the slot when a synth is built on-RT.
    pub init_bytes: Box<[u8]>,
    /// Bytes of per-instance auxiliary memory (a delay line / circular buffer) this unit needs,
    /// summed into the block's `aux` arena at compile time. `0` for units with no aux memory. Unlike
    /// `init_bytes` (a fixed image), aux memory is sized per build (e.g. from a delay's
    /// `maxdelaytime`) and handed to the unit each block as [`ProcessCtx::aux`].
    pub aux_bytes: usize,
    /// Alignment the aux region needs (e.g. `align_of::<f32>()` for an `f32` delay line). Ignored
    /// when `aux_bytes == 0`.
    pub aux_align: usize,
}

/// Build a [`BuiltUnit`] from an initial unit state. The thunks are monomorphised for `T` here, so a
/// [`UnitDef`] only constructs its initial state and hands it to this helper.
pub fn unit_spec<T: Unit>(state: T) -> BuiltUnit {
    BuiltUnit {
        process: process_thunk::<T>,
        init: init_thunk::<T>,
        reseed: reseed_thunk::<T>,
        size: core::mem::size_of::<T>(),
        align: core::mem::align_of::<T>(),
        init_bytes: bytemuck::bytes_of(&state).to_vec().into_boxed_slice(),
        aux_bytes: 0,
        aux_align: 1,
    }
}

/// Build a [`BuiltUnit`] that also reserves `aux_bytes` of per-instance auxiliary memory aligned to
/// `aux_align` - a delay line / circular buffer whose size a `UnitDef` computes at build time (e.g.
/// from a delay's scalar `maxdelaytime`). The unit receives the region as [`ProcessCtx::aux`] each
/// block; it lives in the per-instance pool block and persists across blocks.
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
