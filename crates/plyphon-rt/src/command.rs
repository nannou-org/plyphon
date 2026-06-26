//! The messages crossing the control/RT boundary.
//!
//! [`Command`]s flow control-side -> RT-side over a lock-free ring. A [`GraphDef`] is compiled
//! control-side and installed once via [`Command::DefineGraphDef`]; thereafter `s_new` just names a
//! `def_id`, and the audio thread constructs the synth from the resident def. Three streams flow back
//! RT-side -> NRT-side, all drained by the [`Nrt`](crate::nrt::Nrt): [`Trash`] carries freed `Box`es
//! (buffers/streams) to be dropped off the audio thread - freed synths return their pool block
//! directly, with no trash - [`Event`] carries node notifications, and [`Reply`] carries query
//! answers (the getters), each reassembled into an OSC reply control-side.

use alloc::boxed::Box;
use alloc::sync::Arc;

use crate::tree::AddAction;
use plyphon_dsp::buffer::Buffer;
use plyphon_dsp::stream::StreamPlayback;
use plyphon_unit::graphdef::GraphDef;

/// A command from the `Controller` to the
/// [`World`](crate::world::World).
pub enum Command {
    /// Install (or replace) the compiled def at `def_id`, resident in the World's def table so
    /// `s_new` can reference it by id (scsynth's `GraphDef_Recv`/`/d_recv`).
    DefineGraphDef {
        /// Def-table slot to install into.
        def_id: u32,
        /// The compiled def (built off the audio thread), shared via `Arc`.
        def: Arc<GraphDef>,
    },
    /// Free the resident def at `def_id`, emptying its def-table slot (scsynth's `/d_free`). The
    /// slot's `Arc<GraphDef>` is dropped here, but the Controller retains its own `Arc`, so this is a
    /// non-final refcount decrement - the heavy drop never lands on the audio thread.
    FreeGraphDef {
        /// Def-table slot to empty.
        def_id: u32,
    },
    /// Construct a synth from the def at `def_id` (on the audio thread) and link it under `target`.
    AddSynth {
        /// Client id for the new synth.
        id: i32,
        /// The resident def to instantiate.
        def_id: u32,
        /// Target group's client id.
        target: i32,
        /// Placement within the target group.
        action: AddAction,
    },
    /// Create an empty group under group `target`.
    AddGroup {
        /// Client id for the new group.
        id: i32,
        /// Target group's client id.
        target: i32,
        /// Placement within the target group.
        action: AddAction,
    },
    /// Set control parameter `param` of node `node` to `value`.
    SetControl {
        /// Target node's client id.
        node: i32,
        /// Parameter index.
        param: usize,
        /// New value.
        value: f32,
    },
    /// Set control bus channel `bus` to `value` (scsynth's `/c_set`).
    SetControlBus {
        /// Control bus channel index.
        bus: u32,
        /// New value.
        value: f32,
    },
    /// Map control parameter `param` of node `node` to a control bus, or unmap it (`bus = None`).
    ///
    /// While mapped, the parameter reads the bus's value at the start of every control block
    /// (scsynth's `/n_map`).
    MapControl {
        /// Target node's client id.
        node: i32,
        /// Parameter index.
        param: usize,
        /// Control bus channel to read from, or `None` to unmap.
        bus: Option<u32>,
    },
    /// Free node `node` (deeply for a group), trashing any owned synths.
    FreeNode {
        /// Target node's client id.
        node: i32,
    },
    /// Move node `node` to `target`/`action` (scsynth's `/g_head`/`/g_tail`/`/n_before`/`/n_after`).
    MoveNode {
        /// The node to move.
        node: i32,
        /// The target node or group.
        target: i32,
        /// Where to place `node` relative to `target`.
        action: AddAction,
    },
    /// Free every node in group `group`, leaving it empty (scsynth's `/g_freeAll`).
    FreeAll {
        /// Target group's client id.
        group: i32,
    },
    /// Free every synth in group `group` and its subgroups, keeping the groups (`/g_deepFree`).
    DeepFree {
        /// Target group's client id.
        group: i32,
    },
    /// Pause or resume node `node` (scsynth's `/n_run`).
    NodeRun {
        /// Target node's client id.
        node: i32,
        /// Run the node (`true`) or pause it (`false`).
        run: bool,
    },
    /// Install (or replace) the buffer at `index` with an already-built buffer (scsynth's
    /// `/b_alloc`/`/b_allocRead` stage that swaps the new buffer into the live table). Any buffer
    /// previously at `index` is routed to the trash ring.
    SetBuffer {
        /// Buffer table index.
        index: usize,
        /// The pre-built buffer (all allocation and loading already done off the audio thread).
        buffer: Box<Buffer>,
    },
    /// Install (or replace) a disk-streaming endpoint at buffer `index` (scsynth's
    /// `Buffer.cueSoundFile`). Any slot previously at `index` is routed to the trash ring.
    CueStream {
        /// Buffer table index.
        index: usize,
        /// The pre-built RT-side stream endpoint (its rings allocated off the audio thread).
        playback: Box<StreamPlayback>,
    },
    /// Free the buffer at `index` (scsynth's `/b_free`), routing any slot to the trash ring.
    FreeBuffer {
        /// Buffer table index.
        index: usize,
    },
    /// Overwrite one sample of the buffer at `index`, in place (scsynth's `/b_set`/`/b_setn`).
    /// `sample` is a flat interleaved index (`frame * num_channels + channel`).
    SetBufferSample {
        /// Buffer table index.
        index: usize,
        /// Flat (interleaved) sample index within the buffer.
        sample: usize,
        /// New sample value.
        value: f32,
    },
    /// Fill `count` consecutive samples of the buffer at `index` with `value`, starting at flat
    /// (interleaved) index `start` (scsynth's `/b_fill`).
    FillBuffer {
        /// Buffer table index.
        index: usize,
        /// First flat (interleaved) sample index to write.
        start: usize,
        /// Number of consecutive samples to write.
        count: usize,
        /// Value written to every sample in the range.
        value: f32,
    },
    /// Overwrite the sample-rate metadata of the buffer at `index` (scsynth's `/b_setSampleRate`).
    SetBufferSampleRate {
        /// Buffer table index.
        index: usize,
        /// New sample rate in Hz.
        sample_rate: f64,
    },
    /// Copy `count` interleaved samples from buffer `src` (flat index `src_start`) into buffer `dst`
    /// (flat index `dst_start`), overlap-safe (scsynth's `/b_gen "copy"`).
    CopyBufferRegion {
        /// Destination buffer table index.
        dst: usize,
        /// First flat sample index written in `dst`.
        dst_start: usize,
        /// Source buffer table index.
        src: usize,
        /// First flat sample index read from `src`.
        src_start: usize,
        /// Number of samples to copy.
        count: usize,
    },
    /// Clear every command still pending in the World's scheduler (scsynth's `/clearSched`). Any
    /// scheduled command that owns a `Box` is routed to the trash ring rather than dropped here.
    ClearSched,

    // --- Queries (getters). Each reads live engine state and answers over the reply ring; one
    // command is issued per queried element, and the dispatcher reassembles the grouped OSC reply. ---
    /// `/sync`: a command-stream barrier. Applied in FIFO order, so when it runs every earlier
    /// command's effect is in place; answers [`Reply::Synced`].
    QuerySync {
        /// The id echoed back in `/synced`.
        id: i32,
    },
    /// `/status`: engine counts + sample rate. Answers [`Reply::Status`].
    QueryStatus,
    /// `/rtMemoryStatus`: rt-pool free/largest-chunk. Answers [`Reply::RtMemoryStatus`].
    QueryRtMemory,
    /// `/n_query` (one per node id): node tree position. Answers [`Reply::NodeInfo`]/[`Reply::NodeNotFound`].
    QueryNode {
        /// The node to describe.
        node: i32,
    },
    /// `/c_get` (one per bus): a control bus value. Answers [`Reply::ControlValue`].
    QueryControlBus {
        /// Control bus channel.
        bus: u32,
    },
    /// `/c_getn` (one per range): a run of control bus values. Answers a
    /// [`Reply::ControlRangeHeader`] then `count` [`Reply::RangeValue`].
    QueryControlBusRange {
        /// First control bus channel.
        start: u32,
        /// Number of consecutive channels.
        count: u32,
    },
    /// `/s_get` (one per control): a synth control value. Answers [`Reply::SGetValue`]/[`Reply::SGetMissing`].
    QuerySynthControl {
        /// Target synth's client id.
        node: i32,
        /// Parameter index.
        control: usize,
    },
    /// `/s_getn` (one per range): a run of synth control values. Answers a
    /// [`Reply::SGetRangeHeader`] then `count` [`Reply::RangeValue`].
    QuerySynthControlRange {
        /// Target synth's client id.
        node: i32,
        /// First parameter index.
        control: usize,
        /// Number of consecutive parameters.
        count: usize,
    },
    /// `/b_get` (one per index): a buffer sample. Answers [`Reply::BufferValue`].
    QueryBuffer {
        /// Buffer table index.
        buf: usize,
        /// Flat (interleaved) sample index.
        index: usize,
    },
    /// `/b_getn` (one per range): a run of buffer samples. Answers a [`Reply::BufferRangeHeader`]
    /// then `count` [`Reply::RangeValue`].
    QueryBufferRange {
        /// Buffer table index.
        buf: usize,
        /// First flat sample index.
        index: usize,
        /// Number of consecutive samples.
        count: usize,
    },
    /// `/g_queryTree`: stream the subtree under `group`. Answers a [`Reply::QueryTreeHeader`], a
    /// pre-order body stream, then [`Reply::QueryTreeEnd`].
    QueryTree {
        /// Root of the subtree to dump.
        group: i32,
        /// Include control values.
        flag: bool,
    },
    /// `/g_dumpTree`: like [`QueryTree`](Self::QueryTree) but opened with a [`Reply::DumpTreeHeader`]
    /// so the dispatcher routes it to a text sink instead of an OSC reply.
    DumpTree {
        /// Root of the subtree to dump.
        group: i32,
        /// Include control values.
        flag: bool,
    },
}

/// When a [`Command`] should take effect on the audio thread - plyphon's port of scsynth's OSC
/// bundle time tag.
///
/// An [`Immediate`](Self::Immediate) command is applied as the World drains it (scsynth's time-tag
/// `0`/`1`, and any already-late tag); an [`At`](Self::At) command is held in the World's scheduler
/// until its time is reached. The time is absolute OSC/NTP time: the 32.32 fixed-point
/// `(seconds << 32) | fraction` since 1900 that OSC bundles carry, compared on the audio thread
/// against the World's drift-corrected clock.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CommandTime {
    /// Apply as soon as the World drains the command.
    Immediate,
    /// Apply when the World's clock reaches this absolute OSC/NTP time.
    At(u64),
}

/// A [`Command`] paired with [the time](CommandTime) it should take effect - the item the
/// control -> RT ring carries.
///
/// Flat by value: the scheduler holds it directly, with no `Box`/`Vec`, so a scheduled command
/// never forces a heap free on the audio thread.
pub struct TimedCommand {
    /// When to apply the command.
    pub time: CommandTime,
    /// The command to apply.
    pub command: Command,
}

/// Heap-owning values handed back to the NRT side to be dropped off the audio thread. Freed synths
/// no longer appear here: their state lives in the rt-pool and is reclaimed by `dealloc` on the audio
/// thread (a cheap free-list return), and their `Arc<GraphDef>` is a non-final refcount decrement.
pub enum Trash {
    /// A freed or replaced buffer.
    Buffer(Box<Buffer>),
    /// A freed or replaced streaming endpoint (its rings drop off the audio thread).
    Stream(Box<StreamPlayback>),
}

/// A notification flowing RT-side -> NRT-side, surfaced to the consumer by the
/// [`Nrt`](crate::nrt::Nrt). Each mirrors a scsynth node-notification message.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Event {
    /// A node was added to the tree (`/n_go`).
    NodeStarted {
        /// The node's client id.
        id: i32,
    },
    /// A node was freed (`/n_end`), whether explicitly or by a done action.
    NodeEnded {
        /// The node's client id.
        id: i32,
    },
    /// A node was paused (`/n_off`).
    NodePaused {
        /// The node's client id.
        id: i32,
    },
    /// A node was resumed (`/n_on`).
    NodeResumed {
        /// The node's client id.
        id: i32,
    },
    /// A node was moved to a new tree position (`/n_move`), carrying the same fields as `/n_info`.
    /// Neighbour/parent ids are `-1` when absent; `head`/`tail` are `-1` for a synth.
    NodeMoved {
        /// The moved node's client id.
        node: i32,
        /// Parent group's client id, or `-1`.
        parent: i32,
        /// Previous sibling's client id, or `-1`.
        prev: i32,
        /// Next sibling's client id, or `-1`.
        next: i32,
        /// `1` if a group, else `0`.
        is_group: i32,
        /// First child's client id (groups only), or `-1`.
        head: i32,
        /// Last child's client id (groups only), or `-1`.
        tail: i32,
    },
    /// An `s_new` could not be realised - the def-table slot was empty or the rt-pool was exhausted -
    /// so no node with this id was created.
    SynthFailed {
        /// The client id that would have been assigned.
        id: i32,
    },
}

/// An answer to a query, flowing RT-side -> NRT-side over the reply ring, drained by
/// [`Nrt::poll_reply`](crate::nrt::Nrt::poll_reply) and reassembled into an OSC reply by the
/// dispatcher.
///
/// Fixed-size and `Copy` like [`Event`] - it never allocates on the audio thread. The RT side only
/// ever returns numeric indices and `f32`/`f64` values; every name (defNames, control names) is
/// resolved control-side. Variable-length answers are a `*Header` carrying a count followed by
/// exactly that many [`RangeValue`](Self::RangeValue) (or, for the tree, a body stream terminated by
/// [`QueryTreeEnd`](Self::QueryTreeEnd)); the dispatcher consumes the reply stream as a FIFO queue,
/// in the same order the queries were issued.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Reply {
    /// `/sync <id>` reached: answers `/synced <id>`.
    Synced {
        /// The id to echo.
        id: i32,
    },
    /// `/status.reply` payload.
    Status {
        /// Total live unit generators.
        num_ugens: i32,
        /// Live synths.
        num_synths: i32,
        /// Live groups (including the root).
        num_groups: i32,
        /// Resident synthdefs.
        num_synthdefs: i32,
        /// Average CPU load (not measured; `0.0`).
        avg_cpu: f32,
        /// Peak CPU load (not measured; `0.0`).
        peak_cpu: f32,
        /// Nominal sample rate.
        nominal_sr: f64,
        /// Actual sample rate.
        actual_sr: f64,
    },
    /// `/rtMemoryStatus.reply` payload.
    RtMemoryStatus {
        /// Total free rt-pool bytes.
        total_free: i32,
        /// Largest free rt-pool chunk in bytes.
        largest_free: i32,
    },
    /// One node's `/n_info` row. Neighbour/parent ids are `-1` when absent; `head`/`tail` are `-1`
    /// for a synth (`is_group == 0`).
    NodeInfo {
        /// The queried node's client id.
        node: i32,
        /// Parent group's client id, or `-1`.
        parent: i32,
        /// Previous sibling's client id, or `-1`.
        prev: i32,
        /// Next sibling's client id, or `-1`.
        next: i32,
        /// `1` if a group, else `0`.
        is_group: i32,
        /// First child's client id (groups only), or `-1`.
        head: i32,
        /// Last child's client id (groups only), or `-1`.
        tail: i32,
    },
    /// The queried node did not exist (`/n_query` of an unknown id).
    NodeNotFound {
        /// The queried client id.
        node: i32,
    },
    /// One `(bus, value)` pair for `/c_get`.
    ControlValue {
        /// Control bus channel.
        bus: i32,
        /// Its current value.
        value: f32,
    },
    /// Opens a `/c_getn` run: `count` [`RangeValue`](Self::RangeValue) bodies follow.
    ControlRangeHeader {
        /// First control bus channel.
        start: i32,
        /// Number of values that follow.
        count: i32,
    },
    /// One value of a range answer (`/c_getn`, `/s_getn`, `/b_getn`), following its header.
    RangeValue {
        /// The value.
        value: f32,
    },
    /// One synth control value for `/s_get`. `control` is the resolved parameter index; the
    /// dispatcher re-echoes the as-given control token.
    SGetValue {
        /// The synth's client id.
        node: i32,
        /// Parameter index.
        control: i32,
        /// Its current value.
        value: f32,
    },
    /// The `/s_get`/`/s_getn` target was not a live synth.
    SGetMissing {
        /// The queried client id.
        node: i32,
    },
    /// Opens an `/s_getn` run: `count` [`RangeValue`](Self::RangeValue) bodies follow.
    SGetRangeHeader {
        /// The synth's client id.
        node: i32,
        /// First parameter index.
        control: i32,
        /// Number of values that follow.
        count: i32,
    },
    /// One `(index, value)` sample for `/b_get`.
    BufferValue {
        /// Buffer table index.
        buf: i32,
        /// Flat sample index.
        index: i32,
        /// Its value.
        value: f32,
    },
    /// Opens a `/b_getn` run: `count` [`RangeValue`](Self::RangeValue) bodies follow.
    BufferRangeHeader {
        /// Buffer table index.
        buf: i32,
        /// First flat sample index.
        index: i32,
        /// Number of values that follow.
        count: i32,
    },
    /// Opens a `/g_queryTree.reply` body stream.
    QueryTreeHeader {
        /// Whether control values are included.
        flag: i32,
    },
    /// Opens a `/g_dumpTree` body stream (routed to the dispatcher's text sink).
    DumpTreeHeader {
        /// Whether control values are included.
        flag: i32,
    },
    /// One node in a tree body stream (pre-order). `num_children` is `-1` for a synth.
    QueryTreeNode {
        /// The node's client id.
        node: i32,
        /// Direct child count, or `-1` for a synth.
        num_children: i32,
    },
    /// Follows a synth's [`QueryTreeNode`](Self::QueryTreeNode): its control count (when the tree
    /// flag is set, each control then follows as [`QueryTreeControl`](Self::QueryTreeControl)).
    QueryTreeSynth {
        /// Number of controls that follow (0 unless the flag was set).
        num_controls: i32,
    },
    /// One control of the preceding synth in a tree body stream.
    QueryTreeControl {
        /// Parameter index.
        index: i32,
        /// Its current value.
        value: f32,
    },
    /// Terminates a tree body stream.
    QueryTreeEnd,
}
