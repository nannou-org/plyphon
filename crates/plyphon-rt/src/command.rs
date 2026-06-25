//! The messages crossing the control/RT boundary.
//!
//! [`Command`]s flow control-side -> RT-side over a lock-free ring. A [`GraphDef`] is compiled
//! control-side and installed once via [`Command::DefineGraphDef`]; thereafter `s_new` just names a
//! `def_id`, and the audio thread constructs the synth from the resident def. Two streams flow back
//! RT-side -> NRT-side, both drained by the [`Nrt`](crate::nrt::Nrt): [`Trash`] carries freed `Box`es
//! (buffers/streams) to be dropped off the audio thread - freed synths return their pool block
//! directly, with no trash - and [`Event`] carries notifications for the consumer.

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
    /// Clear every command still pending in the World's scheduler (scsynth's `/clearSched`). Any
    /// scheduled command that owns a `Box` is routed to the trash ring rather than dropped here.
    ClearSched,
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
    /// An `s_new` could not be realised - the def-table slot was empty or the rt-pool was exhausted -
    /// so no node with this id was created.
    SynthFailed {
        /// The client id that would have been assigned.
        id: i32,
    },
}
