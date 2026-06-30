//! A SuperCollider-compatible OSC front-end for the plyphon engine.
//!
//! [`OscDispatcher`] applies the OSC server commands a typical SuperCollider client sends,
//! translating them into calls on a plyphon [`Controller`] the host lends it per call (the host owns
//! the `Controller` alongside its `Nrt`/`World`, and passes `&mut Controller`/`&Controller` into each
//! [`apply`](OscDispatcher::apply)/[`reply`](OscDispatcher::reply)). The commands handled:
//!
//! - **SynthDefs:** `/d_recv`, `/d_load`, `/d_loadDir`, `/d_free`, `/d_freeAll`.
//! - **Synths & nodes:** `/s_new`, `/s_noid`, `/n_set`, `/n_setn`, `/n_fill`, `/n_free`, `/n_run`,
//!   `/n_trace` (a per-unit I/O dump to a text sink), the control mappers `/n_map`/`/n_mapn`.
//! - **Groups & node tree:** `/g_new`, `/p_new`, `/g_head`/`/g_tail`/`/n_before`/`/n_after`/
//!   `/n_order`/`/g_freeAll`/`/g_deepFree`.
//! - **Control buses:** `/c_set`/`/c_setn`/`/c_fill`.
//! - **Buffers:** `/b_alloc`, `/b_allocRead`, `/b_read`, `/b_allocReadChannel`, `/b_readChannel`,
//!   `/b_write`, `/b_close`, `/b_free`, `/b_zero`, `/b_query`, `/b_set`, `/b_setn`, `/b_fill`,
//!   `/b_setSampleRate`, `/b_gen`.
//! - **Server admin:** `/clearSched`, `/error`.
//! - **Host commands** (deferred to the host - see below): `/cmd`, `/u_cmd`.
//! - **Getters** (engine state reads; answered asynchronously - see below): `/sync`, `/status`,
//!   `/rtMemoryStatus`, `/n_query`, `/c_get`/`/c_getn`, `/s_get`/`/s_getn`, `/b_get`/`/b_getn`,
//!   `/g_queryTree`, `/g_dumpTree`.
//!
//! OSC handling is strictly control-side; the audio thread is never involved. `/s_new`, `/n_set`,
//! `/n_setn`, `/n_fill`, `/n_map`, and `/s_get` accept a string control name, resolved against the
//! node's SynthDef, so the dispatcher tracks which definition each node was created from.
//!
//! Commands about the *connection* rather than the engine - `/notify` (a client's per-connection
//! subscription to node notifications), `/quit`, `/dumpOSC`, `/version` - are deliberately *not*
//! handled here; they belong to the host/transport layer that knows about clients (see the
//! `plyphon-cli` server). This front-end always emits the node notifications; who receives them is
//! the host's decision. The getters above *are* engine-state reads, so they live here.
//!
//! # Replies and notifications
//!
//! Commands that report back - `/b_query` (`/b_info`), the asynchronous buffer loads (`/done`), and
//! failures (`/fail`) - queue OSC packets the transport drains with [`OscDispatcher::take_replies`].
//! Node lifecycle is reported the same way: feed the engine [`Event`]s drained from the
//! [`Nrt`](plyphon::Nrt) to [`OscDispatcher::notify`], which queues the matching `/n_go`/`/n_end`/
//! `/n_off`/`/n_on` reply - so a self-freeing synth's `/n_end` reaches the client over OSC too.
//!
//! Getter replies are *asynchronous*: a getter `apply` pushes a query the engine answers a block
//! later over a reply ring. Drain those answers with [`Render::poll_reply`](plyphon::Render::poll_reply)/
//! `Nrt::poll_reply` and feed each to [`OscDispatcher::reply`], which reassembles them (strictly in
//! the FIFO order the getters were issued) into the matching `/n_info`/`/c_set`/`/g_queryTree.reply`/…
//! message, queued for [`take_replies`](OscDispatcher::take_replies). So a getter's reply arrives a
//! render or two after the command, not synchronously.
//!
//! # Time-tag scheduling
//!
//! A bundle's OSC time tag is honoured: a future tag schedules every message in the bundle (and
//! nested bundles) for that absolute time instead of applying it at once. The engine resolves the
//! tag sample-accurately on the audio thread against a drift-corrected clock, and `OffsetOut` places
//! a scheduled synth's onset on the exact sample. The "immediately" tags `0`/`1`, and any already-
//! past time, apply at once. For this to track wall-clock time the host drives the engine with
//! [`World::fill_at`](plyphon::World::fill_at) (passing each buffer's OSC time); otherwise the
//! engine's clock free-runs at the nominal rate.
//!
//! # Asynchronous buffer loading
//!
//! `/b_allocRead` and `/b_read` read sound files, which plyphon keeps off the OSC-handling path:
//! `apply` *queues* the load and the host drives queued loads on its own executor with
//! [`OscDispatcher::run_pending`], lending the `Controller` and the [`BufferSource`] to read through
//! (the dispatcher owns neither). It installs the buffer, runs the command's completion message, and
//! queues `/done` - or `/fail` if no source is given. (Sources are decoded the host's way - see
//! `plyphon-buffers`.)

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[macro_use]
extern crate alloc;

pub mod args;
mod bgen;
pub mod encode;
pub mod score;

pub use args::{bus_index, count_arg, float_arg, index_arg, int_arg, last_blob, map_bus, str_arg};
pub use encode::{
    encode_event, encode_node_info, encode_node_msg, encode_rt_memory, encode_status,
    encode_synced, encode_trigger, node_info_args,
};
pub use score::{ScoreEntry, ScoreError, ScoreReader, parse_score};

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;

use plyphon::controller::SynthNewError;
use plyphon::synthdef::read::ReadError;
use plyphon::{
    AddAction, CommandTime, Controller, Event, NodeMsg, Rate, Render, RenderUntil, Reply, Trigger,
};
use plyphon_buffers::{
    BufFuture, BufferData, BufferSink, BufferSinkStream, BufferSource, DefSource, ReadRegion,
    StreamDrainer, StreamInfo,
};
use plyphon_dsp::buffer::Buffer;
use rosc::{OscMessage, OscPacket, OscTime, OscType};
use thiserror::Error;

/// An error applying an OSC command.
#[derive(Debug, Error)]
pub enum OscError {
    /// The bytes failed to decode as an OSC packet.
    //
    // Not `#[from]`: `rosc::OscError` only implements `Error` under rosc's `std` feature, so a
    // `#[source]`/`#[from]` field would not compile without `std`. The detail is surfaced through
    // `Display` instead (rosc implements that unconditionally), and decode sites map explicitly.
    #[error("OSC decode error: {0}")]
    Decode(rosc::OscError),
    /// The command address is not supported.
    #[error("unsupported OSC command: {0}")]
    UnsupportedCommand(String),
    /// The arguments did not match the command.
    #[error("bad OSC arguments: {0}")]
    BadArguments(&'static str),
    /// The `addAction` code is not one plyphon supports (only head/tail for now).
    #[error("unsupported addAction: {0}")]
    UnsupportedAddAction(i32),
    /// A `/d_recv` payload failed to load as a SynthDef.
    #[error("bad SynthDef")]
    SynthDef(#[from] ReadError),
    /// A `/s_new` failed to instantiate.
    #[error("s_new failed")]
    SynthNew(#[from] SynthNewError),
    /// A command ring was full.
    #[error("command queue full")]
    QueueFull,
    /// A control name was not found on the target node's SynthDef.
    #[error("unknown control: {0}")]
    UnknownParam(String),
    /// A node id was referenced whose SynthDef the dispatcher does not know.
    #[error("unknown node: {0}")]
    UnknownNode(i32),
}

/// A buffer's dimensions, mirrored control-side so `/b_query` and `/b_zero` need no RT round-trip.
#[derive(Clone, Copy)]
struct BufferInfo {
    num_frames: usize,
    num_channels: usize,
    sample_rate: f64,
}

/// Deinterleave `data`, keeping only `channels` (in order) - scsynth's `CopyChannels` for
/// `/b_allocReadChannel`/`/b_readChannel`. A channel index `< 0` or `>= data.num_channels` yields
/// silence. The result is `channels.len()` wide.
fn select_channels(data: &BufferData, channels: &[i32]) -> BufferData {
    let src_channels = data.num_channels.max(1);
    let frames = data.samples.len() / src_channels;
    let out_channels = channels.len();
    let mut samples = vec![0.0f32; frames * out_channels];
    for frame in 0..frames {
        for (ci, &c) in channels.iter().enumerate() {
            if c >= 0 && (c as usize) < src_channels {
                samples[frame * out_channels + ci] =
                    data.samples[frame * src_channels + c as usize];
            }
        }
    }
    BufferData {
        samples,
        num_channels: out_channels,
        sample_rate: data.sample_rate,
    }
}

/// The trailing channel-index list of a `*Channel` buffer command: the consecutive `Int` args from
/// `start` (scsynth's `InitChannels`, which slurps every trailing `'i'` until a non-int - the
/// completion blob stops it). An empty result means "all channels".
fn channel_list(args: &[OscType], start: usize) -> Vec<i32> {
    args.get(start..)
        .unwrap_or(&[])
        .iter()
        .map_while(|a| match a {
            OscType::Int(c) => Some(*c),
            _ => None,
        })
        .collect()
}

/// The `i32` of an `OscType::Int` (else `0`) - for reading the `/n_trace` dump's flat records.
fn int_field(arg: &OscType) -> i32 {
    match arg {
        OscType::Int(v) => *v,
        _ => 0,
    }
}

/// The `f32` of an `OscType::Float` (else `0.0`) - for reading the `/n_trace` dump's flat records.
fn float_field(arg: Option<&OscType>) -> f32 {
    match arg {
        Some(OscType::Float(v)) => *v,
        _ => 0.0,
    }
}

/// A queued asynchronous buffer load (`/b_allocRead`, `/b_read`, and their `*Channel` forms), run by
/// [`OscDispatcher::run_pending`].
struct PendingLoad {
    command: &'static str,
    bufnum: i32,
    key: String,
    region: ReadRegion,
    /// Selected source channels (`/b_allocReadChannel`/`/b_readChannel`), kept in order; an index `< 0`
    /// or `>= the file's channel count` reads as silence (scsynth's `CopyChannels`). `None` for the
    /// plain loads, and an empty list means "all channels" (the `*Channel` command with no selection).
    channels: Option<Vec<i32>>,
    /// The raw OSC completion message to run once the load finishes, if any.
    completion: Option<Vec<u8>>,
    /// The client this load answers to; replayed in `run_pending` so `/done`/`/fail` and any reply the
    /// completion message emits all route back to it.
    target: ReplyTarget,
}

/// A queued asynchronous SynthDef load (`/d_load`, `/d_loadDir`), run by
/// [`OscDispatcher::run_pending`] through the host's [`DefSource`].
struct PendingDef {
    command: &'static str,
    /// `true` for `/d_loadDir` (every def file under `key`), `false` for `/d_load` (one file).
    is_dir: bool,
    key: String,
    completion: Option<Vec<u8>>,
    target: ReplyTarget,
}

/// Chunk size (frames) and pool depth for a `/b_write` copy-out's recording stream. A generous pool
/// (16 chunks of 4096 frames) lets each `run_pending` tick drain a large buffer in few passes; the
/// engine only fills what the pool allows per block, so this bounds the per-tick copy without
/// affecting correctness.
const WRITE_CHUNK_FRAMES: usize = 4096;
const WRITE_CHUNKS: usize = 16;

/// An in-flight `/b_write` buffer copy-out, driven across [`OscDispatcher::run_pending`] ticks: the
/// engine streams the buffer's samples into a recording the `drainer` pulls and writes to the host
/// `sink`. It is multi-tick by necessity (the copy spans many control blocks) and must stay non-
/// blocking - those same ticks recycle drained chunks back to the RT producer, so a busy-wait would
/// starve it and the copy would never complete.
struct PendingWrite {
    command: &'static str,
    bufnum: i32,
    /// The sink key (a path) to open on the first drive.
    key: String,
    /// Header metadata (channels, sample rate) the sink needs up front.
    info: StreamInfo,
    /// Pulls recorded chunks from the engine.
    drainer: StreamDrainer,
    /// The opened sink, or `None` until the first drive opens it.
    sink: Option<Box<dyn BufferSinkStream>>,
    /// `true` for `/b_write leaveOpen=1`: once the sink opens, hand the stream to [`OpenWrite`] (a
    /// `DiskOut` keeps filling it) instead of draining the buffer snapshot to completion.
    leave_open: bool,
    completion: Option<Vec<u8>>,
    target: ReplyTarget,
}

/// A `/b_write leaveOpen=1` stream the host left open for `DiskOut` to fill: drained each
/// [`OscDispatcher::run_pending`] tick until a `/b_close` (or a drain error) ends it. The engine's
/// `DiskOut` recording slot at this bufnum is the source.
struct OpenWrite {
    /// Pulls recorded chunks from the engine's recording slot.
    drainer: StreamDrainer,
    /// The open sink the chunks are written to.
    sink: Box<dyn BufferSinkStream>,
    /// The client a mid-stream `/fail` routes to (the opener of the stream).
    target: ReplyTarget,
}

/// A queued `/b_close`, run by [`OscDispatcher::run_pending`] (its final drain + close is async).
struct PendingClose {
    bufnum: i32,
    completion: Option<Vec<u8>>,
    target: ReplyTarget,
}

/// A queued plugin/unit command (`/cmd`, `/u_cmd`), run by [`OscDispatcher::run_pending`] through the
/// host's [`CommandHost`].
struct PendingCommand {
    command: &'static str,
    /// What the command addresses (a plugin, or a unit within a node).
    cmd_target: CmdTarget,
    /// The command name the host's registry resolves.
    name: String,
    /// The command's trailing arguments (everything after its addressing/name fields).
    args: Vec<OscType>,
    /// The client this command answers to; replayed in `run_pending` so any reply routes back to it.
    target: ReplyTarget,
}

/// One outstanding getter, in the FIFO order queries were issued. As [`Reply`] items arrive (in that
/// same order), the matching entry accumulates them and, when complete, emits exactly one OSC reply
/// message - so a getter that issued several per-element queries answers with one grouped message.
enum PendingQuery {
    /// `/sync` -> `/synced`.
    Sync,
    /// `/status` -> `/status.reply`.
    Status,
    /// `/rtMemoryStatus` -> `/rtMemoryStatus.reply`.
    RtMemory,
    /// One `/n_query` node -> one `/n_info`.
    Node,
    /// `/c_get`/`/b_get`: collect `remaining` `(id, value)` pairs into one `addr` message.
    Pairs {
        addr: &'static str,
        remaining: usize,
        args: Vec<OscType>,
    },
    /// `/c_getn`/`/b_getn`: collect `remaining` `(start, count, value…)` runs into one `addr` message.
    Ranges {
        addr: &'static str,
        remaining: usize,
        in_range: usize,
        args: Vec<OscType>,
    },
    /// `/s_get` -> `/n_set`: like [`Pairs`](Self::Pairs) but echoing the as-given control tokens.
    SGet {
        controls: VecDeque<OscType>,
        remaining: usize,
        args: Vec<OscType>,
    },
    /// `/s_getn` -> `/n_setn`.
    SGetN {
        controls: VecDeque<OscType>,
        remaining: usize,
        in_range: usize,
        args: Vec<OscType>,
    },
    /// `/g_queryTree` (`dump=false`) or `/g_dumpTree` (`dump=true`): accumulate the pre-order body
    /// stream, resolving def/control names control-side, then emit `/g_queryTree.reply` or feed the
    /// text sink.
    Tree {
        dump: bool,
        flag: bool,
        last_node: i32,
        last_def: Option<String>,
        args: Vec<OscType>,
    },
}

/// A host text sink for `/g_dumpTree` (scsynth prints to stdout; plyphon is headless).
pub type DumpSink = Box<dyn FnMut(&str)>;

/// Where an outbound reply should be delivered.
///
/// scsynth copies the requester's reply address into each command at receive time and carries it
/// through every stage (including completion messages), so replies are self-addressed. plyphon mirrors
/// that: the dispatcher tags every reply it produces - `Broadcast` for node notifications (scsynth's
/// `mUsers` set), or `Requester` for an answer owed to the client that issued the command. The host
/// sets [`OscDispatcher::set_reply_target`] before each [`apply`](OscDispatcher::apply) and routes by
/// the tag (see [`take_replies_targeted`](OscDispatcher::take_replies_targeted)); the `u64` is an opaque
/// routing handle the host assigns and interprets - the dispatcher only stores and echoes it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReplyTarget {
    /// Deliver to every client the host considers a notification subscriber.
    Broadcast,
    /// Deliver to the one client identified by this opaque host-assigned handle.
    Requester(u64),
}

/// The target of a host command (`/cmd`/`/u_cmd`).
///
/// (scsynth's `/n_cmd` is unimplemented - commented out in its command table, `SC_MiscCmds.cpp` - so
/// there is no `Node` target; plyphon matches the live surface.)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CmdTarget {
    /// `/cmd`: a plugin command, by name (scsynth dispatches to a loaded `.scx`; plyphon to the host).
    Plugin,
    /// `/u_cmd`: a command addressed to a specific unit within a node.
    Unit {
        /// The enclosing node id.
        node: i32,
        /// The unit's index within the node's def.
        index: i32,
    },
}

/// A host handler for plugin/unit commands (`/cmd`/`/u_cmd`). plyphon has no built-in command
/// plugins, so an embedding app supplies this to interpret app-specific commands (a command registry);
/// an absent handler makes those commands fail. It mirrors how [`BufferSource`] defers buffer I/O - the
/// dispatcher only routes, never interprets.
pub trait CommandHost {
    /// Run command `name` against `target` with `args`, optionally returning a reply packet to send to
    /// the requester. scsynth's `PlugIn_DoCmd`/`Unit_DoCmd` leave any reply entirely to the command
    /// function (no automatic `/done`), so the returned packet *is* the command's reply. `Err` becomes a
    /// `/fail`.
    fn command<'a>(
        &'a self,
        target: CmdTarget,
        name: &'a str,
        args: &'a [OscType],
    ) -> BufFuture<'a, Result<Option<OscPacket>, String>>;
}

/// The bundle of deferred host capabilities the dispatcher drives through
/// [`run_pending`](OscDispatcher::run_pending) - plyphon's seam for surfacing host-owned I/O (sound
/// files, def files, plugin commands, buffer extraction) upward to the embedding app. Each accessor
/// defaults to "unsupported"; a host returns `Some` for what it provides, and the dispatcher fails any
/// action whose capability is absent.
pub trait Host {
    /// The source for `/b_allocRead`/`/b_read` sound-file loads.
    fn buffer_source(&self) -> Option<&dyn BufferSource> {
        None
    }
    /// The sink for `/b_write` buffer saves.
    fn buffer_sink(&self) -> Option<&dyn BufferSink> {
        None
    }
    /// The source for `/d_load`/`/d_loadDir` SynthDef-file loads.
    fn def_source(&self) -> Option<&dyn DefSource> {
        None
    }
    /// The handler for `/cmd`/`/u_cmd` plugin/unit commands.
    fn commands(&self) -> Option<&dyn CommandHost> {
        None
    }
}

/// Applies SuperCollider OSC commands to a plyphon [`Controller`] lent per call by the host.
pub struct OscDispatcher {
    /// Tracks the SynthDef each live node was created from, for control-name resolution.
    node_defs: HashMap<i32, String>,
    /// Control-side mirror of each buffer's dimensions, for `/b_query` and `/b_zero`.
    buffers: HashMap<i32, BufferInfo>,
    /// Buffer loads queued by `apply`, awaiting [`OscDispatcher::run_pending`] (which the host drives
    /// with the [`Host`] it lends - the dispatcher holds no I/O itself).
    pending: Vec<PendingLoad>,
    /// SynthDef loads (`/d_load`/`/d_loadDir`) queued by `apply`, awaiting `run_pending`.
    pending_defs: Vec<PendingDef>,
    /// Plugin/unit commands (`/cmd`/`/u_cmd`) queued by `apply`, awaiting `run_pending`.
    pending_commands: Vec<PendingCommand>,
    /// `/b_write` buffer copy-outs in progress, advanced each `run_pending` tick until the engine
    /// finishes streaming the buffer out (see [`PendingWrite`]).
    writes_in_progress: Vec<PendingWrite>,
    /// `/b_write leaveOpen=1` streams left open for `DiskOut`, keyed by bufnum, drained each tick and
    /// closed by `/b_close`.
    open_writes: HashMap<i32, OpenWrite>,
    /// `/b_close` requests awaiting their async final-drain-and-close in `run_pending`.
    pending_closes: Vec<PendingClose>,
    /// Outbound replies, each tagged with its destination, drained by
    /// [`take_replies`](OscDispatcher::take_replies)/[`take_replies_targeted`](OscDispatcher::take_replies_targeted).
    replies: Vec<(ReplyTarget, OscPacket)>,
    /// The destination stamped onto replies produced right now (set by the host before each `apply`,
    /// and replayed by `reply`/`run_pending` so answers cross the async gap to the right client).
    current_target: ReplyTarget,
    /// Permanent error-posting mode (scsynth's `/error 0|1`; default on).
    error_perm: bool,
    /// Bundle-local error-posting override (scsynth's `/error -1|-2`), saved/restored per bundle.
    error_bundle: Option<bool>,
    /// Outstanding getters, FIFO, reassembled as their [`Reply`]s arrive via [`OscDispatcher::reply`].
    /// Each carries the requester captured when the query was issued.
    pending_queries: VecDeque<(ReplyTarget, PendingQuery)>,
    /// Optional host text sink for `/g_dumpTree` (no OSC reply); a no-op when unset.
    dump_sink: Option<DumpSink>,
    /// Optional host text sink for `/n_trace` dumps (no OSC reply); a no-op when unset.
    trace_sink: Option<DumpSink>,
    /// The `/n_trace` dump currently being reassembled from the reply ring (one node at a time, between
    /// a `TraceHeader` and its `TraceEnd`); outside the FIFO getter queue since trace records are
    /// node-tagged and self-delimited.
    trace_accum: Option<TraceAccum>,
}

/// A `/n_trace` dump being reassembled: the traced node and the flat `[index, nin, nout, values…]`
/// records accumulated since its `TraceHeader`.
struct TraceAccum {
    node: i32,
    records: Vec<OscType>,
}

impl Default for OscDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl OscDispatcher {
    /// A fresh dispatcher. The host lends it the [`Controller`] per call, and the [`BufferSource`] for
    /// asynchronous `/b_allocRead`/`/b_read` loads when it drives [`run_pending`](Self::run_pending).
    pub fn new() -> Self {
        OscDispatcher {
            node_defs: HashMap::new(),
            buffers: HashMap::new(),
            pending: Vec::new(),
            pending_defs: Vec::new(),
            pending_commands: Vec::new(),
            writes_in_progress: Vec::new(),
            open_writes: HashMap::new(),
            pending_closes: Vec::new(),
            replies: Vec::new(),
            current_target: ReplyTarget::Broadcast,
            error_perm: true,
            error_bundle: None,
            pending_queries: VecDeque::new(),
            dump_sink: None,
            trace_sink: None,
            trace_accum: None,
        }
    }

    /// Install a text sink for `/g_dumpTree` (scsynth prints the tree to stdout; plyphon is headless,
    /// so a host that wants the dump provides a sink). Unset by default - `/g_dumpTree` is then a
    /// no-op. `/g_queryTree` is unaffected (it always answers over OSC).
    pub fn set_dump_sink(&mut self, sink: DumpSink) {
        self.dump_sink = Some(sink);
    }

    /// Install a text sink for `/n_trace` dumps (scsynth prints them to stdout; plyphon is headless).
    /// Unset by default - `/n_trace` is then a no-op. Separate from the `/g_dumpTree` sink, since a
    /// trace consumer differs from a tree-dump consumer.
    pub fn set_trace_sink(&mut self, sink: DumpSink) {
        self.trace_sink = Some(sink);
    }

    /// Reassemble a query [`Reply`] (drained from [`Render::poll_reply`](plyphon::Render::poll_reply)/
    /// `Nrt::poll_reply`) into its OSC reply, queued for [`take_replies`](Self::take_replies). Feed
    /// every reply in order, alongside [`notify`](Self::notify); replies arrive in the same FIFO order
    /// the getters were issued, so each is matched against the oldest outstanding query.
    pub fn reply(&mut self, controller: &Controller, reply: Reply) {
        // `/n_trace` records are node-tagged and self-delimited, so they reassemble to the trace sink
        // outside the FIFO getter queue (the trace is answered during the walk, not in `apply`, so it
        // cannot ride the FIFO that assumes answer-in-`apply`).
        if self.handle_trace_reply(controller, &reply) {
            return;
        }
        let Some((target, mut pending)) = self.pending_queries.pop_front() else {
            return; // a stray reply with nothing outstanding (e.g. after a reset); ignore.
        };
        // Replay the requester captured when the query was issued, so the reassembled message routes
        // back to it (the success `/n_info` overrides this to `Broadcast` itself).
        self.current_target = target;
        if !self.apply_reply(controller, &mut pending, reply) {
            self.pending_queries.push_front((target, pending));
        }
    }

    /// Reassemble a `/n_trace` dump record. Returns `true` if `reply` was a `Trace*` variant (consumed
    /// here); `false` otherwise (a normal getter reply). On `TraceEnd` the accumulated records are
    /// formatted - resolving each calc-unit index to its UGen name via the node's def - and fed to the
    /// trace sink (no OSC reply).
    fn handle_trace_reply(&mut self, controller: &Controller, reply: &Reply) -> bool {
        match *reply {
            Reply::TraceHeader { node } => {
                // A fresh dump; a lingering incomplete one (truncated by an overflow) is discarded.
                self.trace_accum = Some(TraceAccum {
                    node,
                    records: Vec::new(),
                });
                true
            }
            Reply::TraceUnit {
                index,
                num_inputs,
                num_outputs,
            } => {
                if let Some(acc) = self.trace_accum.as_mut() {
                    acc.records.push(OscType::Int(index));
                    acc.records.push(OscType::Int(num_inputs));
                    acc.records.push(OscType::Int(num_outputs));
                }
                true
            }
            Reply::TraceValue { value } => {
                if let Some(acc) = self.trace_accum.as_mut() {
                    acc.records.push(OscType::Float(value));
                }
                true
            }
            Reply::TraceEnd => {
                if let Some(acc) = self.trace_accum.take() {
                    let text = self.format_trace(controller, acc.node, &acc.records);
                    if let Some(sink) = self.trace_sink.as_mut() {
                        sink(&text);
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Format one node's `/n_trace` dump as text: a header, then one line per calc unit with its name
    /// (resolved by calc-order index from the node's def, skipping demand units - which are not in the
    /// block walk) and its inputs'/outputs' first samples.
    fn format_trace(&self, controller: &Controller, node: i32, records: &[OscType]) -> String {
        let def_name = self.node_defs.get(&node);
        // Calc-order unit names: the def's units minus the demand-rate ones (the compile splits those
        // out of the calc list), so the calc index lines up with this filtered list.
        let names: Vec<&str> = def_name
            .and_then(|name| controller.synthdef(name))
            .map(|def| {
                def.units
                    .iter()
                    .filter(|u| u.rate != Rate::Demand)
                    .map(|u| u.name.as_str())
                    .collect()
            })
            .unwrap_or_default();
        let mut text = alloc::format!("TRACE node {node}");
        if let Some(name) = def_name {
            text.push_str(&alloc::format!(" ({name})"));
        }
        text.push('\n');
        let mut pos = 0;
        while pos + 3 <= records.len() {
            let index = int_field(&records[pos]);
            let num_inputs = int_field(&records[pos + 1]).max(0) as usize;
            let num_outputs = int_field(&records[pos + 2]).max(0) as usize;
            pos += 3;
            let ins: Vec<f32> = (0..num_inputs)
                .map(|k| float_field(records.get(pos + k)))
                .collect();
            let outs: Vec<f32> = (0..num_outputs)
                .map(|k| float_field(records.get(pos + num_inputs + k)))
                .collect();
            pos += num_inputs + num_outputs;
            let name = names.get(index as usize).copied().unwrap_or("?");
            text.push_str(&alloc::format!(
                "  {index} {name}  in: {ins:?}  out: {outs:?}\n"
            ));
        }
        text
    }

    /// Set the destination stamped onto replies the dispatcher produces next. The host calls this with
    /// the issuing client before each [`apply`](Self::apply); getters and async loads capture it so
    /// their later answers (and any reply a completion message emits) reach the same client. Defaults
    /// to [`ReplyTarget::Broadcast`], which is all an in-process single-client host needs.
    pub fn set_reply_target(&mut self, target: ReplyTarget) {
        self.current_target = target;
    }

    /// Take the OSC replies queued since the last call (`/done`, `/b_info`, `/fail`, getter answers,
    /// and the `/n_*` node notifications), dropping their destination tags - for single-client hosts.
    /// Multi-client hosts want [`take_replies_targeted`](Self::take_replies_targeted) instead.
    pub fn take_replies(&mut self) -> Vec<OscPacket> {
        core::mem::take(&mut self.replies)
            .into_iter()
            .map(|(_, packet)| packet)
            .collect()
    }

    /// Take the queued replies paired with their [`ReplyTarget`], for a host that routes per client:
    /// `Broadcast` to its notification subscribers, `Requester` to the one client the handle names.
    pub fn take_replies_targeted(&mut self) -> Vec<(ReplyTarget, OscPacket)> {
        core::mem::take(&mut self.replies)
    }

    /// Translate an engine [`Event`] into the matching SuperCollider node-notification reply and
    /// queue it for [`take_replies`](Self::take_replies): `/n_go` (started), `/n_end` (freed),
    /// `/n_off` (paused), `/n_on` (resumed), `/n_move` (moved).
    ///
    /// Feed this the events drained from the [`Nrt`](plyphon::Nrt), so node lifecycle - including
    /// synths that free themselves via a done action - is reported back over OSC alongside the
    /// command replies. Node notifications are tagged [`ReplyTarget::Broadcast`] (scsynth fans them out
    /// to its `mUsers` set); which clients that set contains (scsynth's `/notify` subscription) is the
    /// host/transport's job. The lifecycle events carry only the node id; `/n_move` carries the full
    /// `/n_info`-shaped position (parent/prev/next/isGroup, plus head/tail for a group).
    pub fn notify(&mut self, event: Event) {
        // The only stateful effect: reclaim def tracking for a node that never reaches `/n_free` (a
        // self-freed synth's `NodeEnded`) or never existed (a failed `/s_new`). The OSC mapping
        // itself is pure - see [`encode::encode_event`].
        if let Event::NodeEnded { id } | Event::SynthFailed { id } = event {
            self.node_defs.remove(&id);
        }
        self.push_packet(ReplyTarget::Broadcast, encode::encode_event(event));
    }

    /// Translate a `SendTrig` [`Trigger`] into a `/tr [nodeID, id, value]` message and queue it for
    /// [`take_replies`](Self::take_replies). Feed this the triggers drained from
    /// [`Nrt::poll_trigger`](plyphon::Nrt::poll_trigger). Broadcast, like the node notifications.
    pub fn notify_trigger(&mut self, trigger: Trigger) {
        self.push_packet(ReplyTarget::Broadcast, encode::encode_trigger(trigger));
    }

    /// Translate a `SendReply` [`NodeMsg`] into its OSC message and queue it for
    /// [`take_replies`](Self::take_replies). For `SendReply` that is `<path> [nodeID, replyID,
    /// values...]`, where the path is the unit's `cmdName` verbatim (scsynth's path includes its
    /// leading `/`). Feed this the messages drained from
    /// [`Nrt::poll_node_msg`](plyphon::Nrt::poll_node_msg). Broadcast, like `/tr`.
    pub fn notify_node_msg(&mut self, msg: NodeMsg) {
        if let Some(packet) = encode::encode_node_msg(msg) {
            self.push_packet(ReplyTarget::Broadcast, packet);
        }
    }

    /// Run the host actions queued by `apply`, in order, driving each through the [`Host`] the caller
    /// lends (the dispatcher owns no I/O).
    ///
    /// Buffer loads (`/b_allocRead`, `/b_read`): load through `host.buffer_source()`, install the
    /// buffer, run the command's completion message, and queue `/done` - or `/fail` if the capability
    /// is absent or the load errors. (Later actions - `/d_load`, `/cmd`, `/b_write` - drain here too.)
    /// Drive this on whatever executor suits the host (a background thread natively, `spawn_local` on
    /// the web); it never touches the audio thread.
    pub async fn run_pending(&mut self, controller: &mut Controller, host: Option<&dyn Host>) {
        let source = host.and_then(|h| h.buffer_source());
        for load in core::mem::take(&mut self.pending) {
            // Answer this load (its `/done`/`/fail`, and anything its completion message emits) back to
            // the client that issued it - exactly as scsynth stamps completion packets with the
            // command's stored reply address.
            self.current_target = load.target;
            let result = match source {
                Some(source) => Some(source.load(&load.key, load.region).await),
                None => None,
            };
            match result {
                None => self.fail(load.command, "no buffer source configured"),
                Some(Err(err)) => self.fail(load.command, &err.to_string()),
                Some(Ok(data)) => {
                    // `*Channel` loads keep only the selected source channels (scsynth reads the whole
                    // file, then `CopyChannels` deinterleaves); an empty selection means "all".
                    let data = match &load.channels {
                        Some(channels) if !channels.is_empty() => select_channels(&data, channels),
                        _ => data,
                    };
                    let num_channels = data.num_channels.max(1);
                    let info = BufferInfo {
                        num_frames: data.samples.len() / num_channels,
                        num_channels,
                        sample_rate: data.sample_rate,
                    };
                    if controller
                        .buffer_set(load.bufnum as usize, Box::new(data.into()))
                        .is_err()
                    {
                        self.fail(load.command, "command queue full");
                        continue;
                    }
                    self.buffers.insert(load.bufnum, info);
                    self.run_completion_bytes(controller, load.completion.as_deref());
                    self.done(load.command, load.bufnum);
                }
            }
        }

        // SynthDef loads (`/d_load`/`/d_loadDir`): read the SCgf bytes through the host's `DefSource`,
        // parse and register each def, run the completion message, and reply `/done /<command>`.
        let def_source = host.and_then(|h| h.def_source());
        for load in core::mem::take(&mut self.pending_defs) {
            self.current_target = load.target;
            let Some(source) = def_source else {
                self.fail(load.command, "no def source configured");
                continue;
            };
            let blobs = if load.is_dir {
                source.read_def_dir(&load.key).await
            } else {
                source.read_def(&load.key).await.map(|bytes| vec![bytes])
            };
            match blobs {
                Err(err) => self.fail(load.command, &err.to_string()),
                Ok(blobs) => {
                    let mut error = None;
                    for blob in &blobs {
                        match plyphon::synthdef::read::parse(blob) {
                            Ok(defs) => {
                                for def in defs {
                                    controller.add_synthdef(def);
                                }
                            }
                            Err(err) => {
                                error = Some(err.to_string());
                                break;
                            }
                        }
                    }
                    match error {
                        Some(err) => self.fail(load.command, &err),
                        None => {
                            self.run_completion_bytes(controller, load.completion.as_deref());
                            self.done_command(load.command);
                        }
                    }
                }
            }
        }

        // Plugin/unit commands (`/cmd`/`/u_cmd`): the host's `CommandHost` interprets each (plyphon has
        // no built-in command plugins) and owns any reply - scsynth's `PlugIn_DoCmd`/`Unit_DoCmd` send no
        // automatic `/done`, so the returned packet *is* the command's reply. An absent handler fails.
        let commands = host.and_then(|h| h.commands());
        for command in core::mem::take(&mut self.pending_commands) {
            self.current_target = command.target;
            let Some(handler) = commands else {
                self.fail(command.command, "no command host configured");
                continue;
            };
            match handler
                .command(command.cmd_target, &command.name, &command.args)
                .await
            {
                Ok(Some(packet)) => self.reply_packet(packet),
                Ok(None) => {}
                Err(err) => self.fail(command.command, &err),
            }
        }

        // `/b_write` copy-outs in progress: open each sink on first sight, drain whatever the engine
        // produced since the last tick, and finish (close + `/done`) once the copy completes. This is
        // multi-tick and deliberately non-blocking - draining recycles chunks back to the RT producer,
        // so busy-waiting here would starve it and the copy would never finish.
        let sink_host = host.and_then(|h| h.buffer_sink());
        let mut i = 0;
        while i < self.writes_in_progress.len() {
            self.current_target = self.writes_in_progress[i].target;
            // Open the sink on the first drive; a missing sink host or an open failure fails the
            // command and drops the copy.
            if self.writes_in_progress[i].sink.is_none() {
                let opened = match sink_host {
                    None => Err("no buffer sink configured".to_string()),
                    Some(sink_host) => {
                        let write = &self.writes_in_progress[i];
                        sink_host
                            .open_write(&write.key, write.info)
                            .await
                            .map_err(|err| err.to_string())
                    }
                };
                match opened {
                    Ok(sink) => self.writes_in_progress[i].sink = Some(sink),
                    Err(err) => {
                        let write = self.writes_in_progress.swap_remove(i);
                        self.fail(write.command, &err);
                        continue;
                    }
                }
            }
            // A `leaveOpen=1` write hands off to the `open_writes` registry once its sink is open: the
            // engine's `DiskOut` recording slot keeps filling, drained each tick below until `/b_close`.
            if self.writes_in_progress[i].leave_open {
                let write = self.writes_in_progress.swap_remove(i);
                let sink = write.sink.expect("sink opened above");
                self.run_completion_bytes(controller, write.completion.as_deref());
                self.done(write.command, write.bufnum);
                self.open_writes.insert(
                    write.bufnum,
                    OpenWrite {
                        drainer: write.drainer,
                        sink,
                        target: write.target,
                    },
                );
                continue;
            }
            // Drain whatever the engine produced since the last tick; a write error fails the copy.
            let drained = {
                let write = &mut self.writes_in_progress[i];
                let sink = write.sink.as_mut().expect("sink opened above");
                write.drainer.drain(sink.as_mut()).await
            };
            if let Err(err) = drained {
                let write = self.writes_in_progress.swap_remove(i);
                self.fail(write.command, &err.to_string());
                continue;
            }
            // Not finished (the engine is still copying): revisit on the next tick.
            if !self.writes_in_progress[i].drainer.is_done() {
                i += 1;
                continue;
            }
            // Finished: close the sink, run the completion message, reply `/done /b_write <bufnum>`.
            let mut write = self.writes_in_progress.swap_remove(i);
            let closed = match write.sink.as_mut() {
                Some(sink) => sink.close().await,
                None => Ok(()),
            };
            match closed {
                Ok(()) => {
                    self.run_completion_bytes(controller, write.completion.as_deref());
                    self.done(write.command, write.bufnum);
                }
                Err(err) => self.fail(write.command, &err.to_string()),
            }
        }

        // Keep each `leaveOpen=1` stream flowing: drain whatever its `DiskOut` produced this tick. A
        // drain error ends the stream (and fails it to its opener); `/b_close` ends it cleanly below.
        let mut drain_errors: Vec<(i32, String)> = Vec::new();
        for (bufnum, open) in self.open_writes.iter_mut() {
            if let Err(err) = open.drainer.drain(open.sink.as_mut()).await {
                drain_errors.push((*bufnum, err.to_string()));
            }
        }
        for (bufnum, err) in drain_errors {
            if let Some(open) = self.open_writes.remove(&bufnum) {
                self.current_target = open.target;
                self.fail("/b_write", &err);
                let _ = controller.buffer_free(bufnum as usize);
                self.buffers.remove(&bufnum);
            }
        }

        // `/b_close`: final-drain and close each left-open stream, free its recording slot, reply
        // `/done /b_close <bufnum>` (an unopened bufnum fails). The close is async, hence the queue.
        for close in core::mem::take(&mut self.pending_closes) {
            self.current_target = close.target;
            let Some(mut open) = self.open_writes.remove(&close.bufnum) else {
                self.fail("/b_close", "buffer not open for writing");
                continue;
            };
            let result = open.drainer.finish(open.sink.as_mut()).await;
            // The recording slot is the streaming ring (no flat data), so drop it on close.
            let _ = controller.buffer_free(close.bufnum as usize);
            self.buffers.remove(&close.bufnum);
            match result {
                Ok(()) => {
                    self.run_completion_bytes(controller, close.completion.as_deref());
                    self.done("/b_close", close.bufnum);
                }
                Err(err) => self.fail("/b_close", &err.to_string()),
            }
        }
    }

    /// Decode and apply a single OSC packet from raw bytes.
    pub fn apply_bytes(
        &mut self,
        controller: &mut Controller,
        data: &[u8],
    ) -> Result<(), OscError> {
        let (_, packet) = rosc::decoder::decode_udp(data).map_err(OscError::Decode)?;
        self.apply(controller, &packet)
    }

    /// Apply a decoded OSC packet: a message immediately, or every message in a bundle at the
    /// bundle's time tag.
    ///
    /// A future time tag schedules the bundle's messages (and any nested bundles) for that absolute
    /// OSC/NTP time; the engine maps the tag to a sample-exact block on the audio thread, against a
    /// drift-corrected clock. The "immediately" tags `0`/`1` (and any already-past time) apply now.
    pub fn apply(
        &mut self,
        controller: &mut Controller,
        packet: &OscPacket,
    ) -> Result<(), OscError> {
        match packet {
            OscPacket::Message(message) => self.message(controller, message),
            OscPacket::Bundle(bundle) => {
                let prev = controller.begin_scheduled(bundle_command_time(bundle.timetag));
                // A bundle-local `/error -1|-2` override is scoped to this bundle (and its nested
                // bundles); save it here and restore on exit, exactly like the schedule window.
                let prev_error = self.error_bundle;
                let mut result = Ok(());
                for inner in &bundle.content {
                    result = self.apply(controller, inner);
                    if result.is_err() {
                        break;
                    }
                }
                // Restore the enclosing window and error scope (Immediate / inherited at the top
                // level), even on error.
                self.error_bundle = prev_error;
                controller.begin_scheduled(prev);
                result
            }
        }
    }

    fn message(
        &mut self,
        controller: &mut Controller,
        message: &OscMessage,
    ) -> Result<(), OscError> {
        match message.addr.as_str() {
            "/d_recv" => self.d_recv(controller, &message.args),
            "/d_load" => self.d_load(&message.args),
            "/d_loadDir" => self.d_load_dir(&message.args),
            "/cmd" => self.cmd(&message.args),
            "/u_cmd" => self.u_cmd(&message.args),
            "/d_free" => self.d_free(controller, &message.args),
            "/d_freeAll" => self.d_free_all(controller),
            "/s_new" => self.s_new(controller, &message.args),
            "/s_noid" => self.s_noid(&message.args),
            "/n_set" => self.n_set(controller, &message.args),
            "/n_setn" => self.n_setn(controller, &message.args),
            "/n_fill" => self.n_fill(controller, &message.args),
            "/n_free" => self.n_free(controller, &message.args),
            "/n_run" => self.n_run(controller, &message.args),
            "/g_new" => self.g_new(controller, &message.args),
            // scsynth emulates parallel groups with ordinary groups; same triple layout as `/g_new`.
            "/p_new" => self.g_new(controller, &message.args),
            "/c_set" => self.c_set(controller, &message.args),
            "/c_setn" => self.c_setn(controller, &message.args),
            "/c_fill" => self.c_fill(controller, &message.args),
            "/n_map" => self.n_map(controller, &message.args),
            "/n_mapn" => self.n_mapn(controller, &message.args),
            "/n_mapa" => self.n_mapa(controller, &message.args),
            "/n_mapan" => self.n_mapan(controller, &message.args),
            "/b_alloc" => self.b_alloc(controller, &message.args),
            "/b_free" => self.b_free(controller, &message.args),
            "/b_zero" => self.b_zero(controller, &message.args),
            "/b_query" => self.b_query(&message.args),
            "/b_set" => self.b_set(controller, &message.args),
            "/b_setn" => self.b_setn(controller, &message.args),
            "/b_fill" => self.b_fill(controller, &message.args),
            "/b_setSampleRate" => self.b_set_sample_rate(controller, &message.args),
            "/b_gen" => self.b_gen(controller, &message.args),
            "/b_allocRead" => self.b_alloc_read(&message.args),
            "/b_read" => self.b_read(&message.args),
            "/b_allocReadChannel" => self.b_alloc_read_channel(&message.args),
            "/b_readChannel" => self.b_read_channel(&message.args),
            "/b_write" => self.b_write(controller, &message.args),
            "/b_close" => self.b_close(&message.args),
            "/g_head" => self.group_moves(controller, &message.args, AddAction::Head),
            "/g_tail" => self.group_moves(controller, &message.args, AddAction::Tail),
            "/n_before" => self.node_moves(controller, &message.args, AddAction::Before),
            "/n_after" => self.node_moves(controller, &message.args, AddAction::After),
            "/n_order" => self.n_order(controller, &message.args),
            "/g_freeAll" => self.g_free_all(controller, &message.args),
            "/g_deepFree" => self.g_deep_free(controller, &message.args),
            "/clearSched" => self.clear_sched(controller),
            "/error" => self.error_cmd(&message.args),
            "/sync" => self.sync(controller, &message.args),
            "/status" => self.status(controller),
            "/rtMemoryStatus" => self.rt_memory_status(controller),
            "/n_query" => self.n_query(controller, &message.args),
            "/n_trace" => self.n_trace(controller, &message.args),
            "/c_get" => self.c_get(controller, &message.args),
            "/c_getn" => self.c_getn(controller, &message.args),
            "/s_get" => self.s_get(controller, &message.args),
            "/s_getn" => self.s_getn(controller, &message.args),
            "/b_get" => self.b_get(controller, &message.args),
            "/b_getn" => self.b_getn(controller, &message.args),
            "/g_queryTree" => self.g_query_tree(controller, &message.args, false),
            "/g_dumpTree" => self.g_query_tree(controller, &message.args, true),
            other => Err(OscError::UnsupportedCommand(other.to_string())),
        }
    }

    fn d_recv(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let blob = match args.first() {
            Some(OscType::Blob(bytes)) => bytes,
            _ => return Err(OscError::BadArguments("d_recv expects a blob")),
        };
        let defs = plyphon::synthdef::read::parse(blob).map_err(OscError::SynthDef)?;
        for def in defs {
            controller.add_synthdef(def);
        }
        Ok(())
    }

    /// `/d_free <name>...`: free each named synth definition (a later `/s_new` of it then fails until
    /// it is re-sent).
    fn d_free(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let name = str_arg(arg)?;
            controller.free_def(name).map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/d_freeAll`: free every registered synth definition.
    fn d_free_all(&mut self, controller: &mut Controller) -> Result<(), OscError> {
        controller.free_all_defs().map_err(|_| OscError::QueueFull)
    }

    fn s_new(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        if args.len() < 4 {
            return Err(OscError::BadArguments(
                "s_new expects name, id, addAction, target",
            ));
        }
        let name = str_arg(&args[0])?.to_string();
        let id = int_arg(&args[1])?;
        let action = add_action(int_arg(&args[2])?)?;
        let target = int_arg(&args[3])?;

        let id = if id < 0 {
            controller
                .synth_new(&name, target, action)
                .map_err(OscError::SynthNew)?
        } else {
            controller
                .synth_new_with_id(id, &name, target, action)
                .map_err(OscError::SynthNew)?;
            id
        };
        self.node_defs.insert(id, name.clone());
        self.apply_controls(controller, id, Some(&name), &args[4..])
    }

    /// `/s_noid <nodeID>...`: detach each node's def tracking so it is no longer addressed by control
    /// *name* (partial vs scsynth: the node keeps running and stays reachable by control *index*,
    /// since plyphon does not reassign a hidden id).
    fn s_noid(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let node = int_arg(arg)?;
            self.node_defs.remove(&node);
        }
        Ok(())
    }

    fn n_set(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_set expects a node"))?,
        )?;
        self.apply_controls(controller, node, None, &args[1..])
    }

    /// `/n_setn nodeID (control, count, value...)...`: set contiguous ranges of a node's controls.
    fn n_setn(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_setn expects a node"))?,
        )?;
        let rest = &args[1..];
        let mut i = 0;
        while i < rest.len() {
            let start = self.control_index(controller, node, &rest[i])?;
            let count = count_arg(rest.get(i + 1))?;
            i += 2;
            if i + count > rest.len() {
                return Err(OscError::BadArguments(
                    "n_setn value count exceeds arguments",
                ));
            }
            for (j, arg) in rest[i..i + count].iter().enumerate() {
                controller
                    .set_control(node, start + j, float_arg(arg)?)
                    .map_err(|_| OscError::QueueFull)?;
            }
            i += count;
        }
        Ok(())
    }

    /// `/n_fill nodeID (control, count, value)...`: fill contiguous ranges of a node's controls.
    fn n_fill(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_fill expects a node"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "n_fill expects control/count/value triples",
            ));
        }
        for triple in rest.chunks_exact(3) {
            let start = self.control_index(controller, node, &triple[0])?;
            let count = count_arg(Some(&triple[1]))?;
            let value = float_arg(&triple[2])?;
            for j in 0..count {
                controller
                    .set_control(node, start + j, value)
                    .map_err(|_| OscError::QueueFull)?;
            }
        }
        Ok(())
    }

    fn n_free(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let node = int_arg(arg)?;
            controller.free(node).map_err(|_| OscError::QueueFull)?;
            self.node_defs.remove(&node);
        }
        Ok(())
    }

    /// `/n_run (nodeID, flag)...`: pause (flag 0) or resume (flag 1) each node.
    fn n_run(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("n_run expects node/flag pairs"));
        }
        for pair in args.chunks_exact(2) {
            let node = int_arg(&pair[0])?;
            let run = int_arg(&pair[1])? != 0;
            controller
                .node_run(node, run)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn g_new(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        if args.is_empty() || !args.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "g_new expects id, addAction, target triples",
            ));
        }
        for triple in args.chunks_exact(3) {
            let id = int_arg(&triple[0])?;
            let action = add_action(int_arg(&triple[1])?)?;
            let target = int_arg(&triple[2])?;
            if id < 0 {
                controller
                    .new_group(target, action)
                    .map_err(|_| OscError::QueueFull)?;
            } else {
                controller
                    .new_group_with_id(id, target, action)
                    .map_err(|_| OscError::QueueFull)?;
            }
        }
        Ok(())
    }

    /// `/g_head`/`/g_tail`: `(group, node)` pairs - move each node to the group's head/tail.
    fn group_moves(
        &mut self,
        controller: &mut Controller,
        args: &[OscType],
        action: AddAction,
    ) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("expects group/node pairs"));
        }
        for pair in args.chunks_exact(2) {
            let group = int_arg(&pair[0])?;
            let node = int_arg(&pair[1])?;
            controller
                .move_node(node, group, action)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/n_before`/`/n_after`: `(node, target)` pairs - move each node before/after its target.
    fn node_moves(
        &mut self,
        controller: &mut Controller,
        args: &[OscType],
        action: AddAction,
    ) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("expects node/target pairs"));
        }
        for pair in args.chunks_exact(2) {
            let node = int_arg(&pair[0])?;
            let target = int_arg(&pair[1])?;
            controller
                .move_node(node, target, action)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/n_order addAction target node...`: place the nodes consecutively, in order, at the location.
    fn n_order(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        if args.len() < 3 {
            return Err(OscError::BadArguments(
                "n_order expects addAction, target, nodes",
            ));
        }
        let mut anchor = int_arg(&args[1])?;
        let mut action = add_action(int_arg(&args[0])?)?;
        for arg in &args[2..] {
            let node = int_arg(arg)?;
            controller
                .move_node(node, anchor, action)
                .map_err(|_| OscError::QueueFull)?;
            // Subsequent nodes follow the previous one, preserving the given order.
            anchor = node;
            action = AddAction::After;
        }
        Ok(())
    }

    /// `/g_freeAll group...`: empty each group, keeping the group.
    fn g_free_all(
        &mut self,
        controller: &mut Controller,
        args: &[OscType],
    ) -> Result<(), OscError> {
        for arg in args {
            let group = int_arg(arg)?;
            controller
                .free_all(group)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/g_deepFree group...`: free each group's synths recursively, keeping the groups.
    fn g_deep_free(
        &mut self,
        controller: &mut Controller,
        args: &[OscType],
    ) -> Result<(), OscError> {
        for arg in args {
            let group = int_arg(arg)?;
            controller
                .deep_free(group)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/clearSched`: clear the engine scheduler's pending time-tagged commands.
    fn clear_sched(&mut self, controller: &mut Controller) -> Result<(), OscError> {
        controller.clear_sched().map_err(|_| OscError::QueueFull)
    }

    /// `/error <mode>`: set the error-posting mode. `0`/`1` set the permanent mode; `-1`/`-2` set a
    /// bundle-local override (scoped to the enclosing bundle).
    fn error_cmd(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let mode = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("error expects a mode"))?,
        )?;
        match mode {
            0 => self.error_perm = false,
            1 => self.error_perm = true,
            -2 => self.error_bundle = Some(false),
            -1 => self.error_bundle = Some(true),
            _ => return Err(OscError::BadArguments("error mode must be -2..=1")),
        }
        Ok(())
    }

    // --- Getters. Each issues one query per element (so commands stay flat) and records a
    // `PendingQuery`; `reply` reassembles the answers into one OSC message per command (one `/n_info`
    // per node for `/n_query`). ---

    /// `/sync <id>` -> `/synced <id>`.
    fn sync(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let id = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("sync expects an id"))?,
        )?;
        controller.query_sync(id).map_err(|_| OscError::QueueFull)?;
        self.push_query(PendingQuery::Sync);
        Ok(())
    }

    /// `/status` -> `/status.reply`.
    fn status(&mut self, controller: &mut Controller) -> Result<(), OscError> {
        controller.query_status().map_err(|_| OscError::QueueFull)?;
        self.push_query(PendingQuery::Status);
        Ok(())
    }

    /// `/rtMemoryStatus` -> `/rtMemoryStatus.reply`.
    fn rt_memory_status(&mut self, controller: &mut Controller) -> Result<(), OscError> {
        controller
            .query_rt_memory()
            .map_err(|_| OscError::QueueFull)?;
        self.push_query(PendingQuery::RtMemory);
        Ok(())
    }

    /// `/n_query <node>...` -> one `/n_info` per node.
    fn n_query(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let node = int_arg(arg)?;
            controller
                .query_node(node)
                .map_err(|_| OscError::QueueFull)?;
            self.push_query(PendingQuery::Node);
        }
        Ok(())
    }

    /// `/n_trace <node>...`: dump each synth's per-unit inputs/outputs for one block to the trace sink
    /// (scsynth's `meth_n_trace`; like `/g_dumpTree` there is no OSC reply). The dump streams back over
    /// the reply ring and is reassembled by [`handle_trace_reply`](Self::handle_trace_reply).
    fn n_trace(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let node = int_arg(arg)?;
            controller
                .trace_node(node)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/c_get <bus>...` -> one `/c_set`.
    fn c_get(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let bus = bus_index(arg)?;
            controller
                .query_control_bus(bus)
                .map_err(|_| OscError::QueueFull)?;
        }
        if args.is_empty() {
            self.reply_msg("/c_set", Vec::new());
        } else {
            self.push_query(PendingQuery::Pairs {
                addr: "/c_set",
                remaining: args.len(),
                args: Vec::new(),
            });
        }
        Ok(())
    }

    /// `/c_getn (start, count)...` -> one `/c_setn`.
    fn c_getn(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("c_getn expects start/count pairs"));
        }
        for pair in args.chunks_exact(2) {
            let start = bus_index(&pair[0])?;
            let count = count_arg(Some(&pair[1]))? as u32;
            controller
                .query_control_bus_range(start, count)
                .map_err(|_| OscError::QueueFull)?;
        }
        let ranges = args.len() / 2;
        if ranges == 0 {
            self.reply_msg("/c_setn", Vec::new());
        } else {
            self.push_query(PendingQuery::Ranges {
                addr: "/c_setn",
                remaining: ranges,
                in_range: 0,
                args: Vec::new(),
            });
        }
        Ok(())
    }

    /// `/s_get <node> <control>...` -> one `/n_set` (echoing the as-given control tokens).
    fn s_get(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("s_get expects a node"))?,
        )?;
        let rest = &args[1..];
        let mut controls = VecDeque::with_capacity(rest.len());
        for arg in rest {
            let control = self.control_index(controller, node, arg)?;
            controller
                .query_synth_control(node, control)
                .map_err(|_| OscError::QueueFull)?;
            controls.push_back(arg.clone());
        }
        if rest.is_empty() {
            self.reply_msg("/n_set", vec![OscType::Int(node)]);
        } else {
            self.push_query(PendingQuery::SGet {
                remaining: rest.len(),
                controls,
                args: vec![OscType::Int(node)],
            });
        }
        Ok(())
    }

    /// `/s_getn <node> (control, count)...` -> one `/n_setn`.
    fn s_getn(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("s_getn expects a node"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("s_getn expects control/count pairs"));
        }
        let mut controls = VecDeque::with_capacity(rest.len() / 2);
        for pair in rest.chunks_exact(2) {
            let control = self.control_index(controller, node, &pair[0])?;
            let count = count_arg(Some(&pair[1]))?;
            controller
                .query_synth_control_range(node, control, count)
                .map_err(|_| OscError::QueueFull)?;
            controls.push_back(pair[0].clone());
        }
        let ranges = rest.len() / 2;
        if ranges == 0 {
            self.reply_msg("/n_setn", vec![OscType::Int(node)]);
        } else {
            self.push_query(PendingQuery::SGetN {
                remaining: ranges,
                in_range: 0,
                controls,
                args: vec![OscType::Int(node)],
            });
        }
        Ok(())
    }

    /// `/b_get <buf> <index>...` -> one `/b_set`.
    fn b_get(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let buf = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_get expects a bufnum"))?,
        )?;
        let rest = &args[1..];
        for arg in rest {
            let index = index_arg(arg)?;
            controller
                .query_buffer(buf as usize, index)
                .map_err(|_| OscError::QueueFull)?;
        }
        if rest.is_empty() {
            self.reply_msg("/b_set", vec![OscType::Int(buf)]);
        } else {
            self.push_query(PendingQuery::Pairs {
                addr: "/b_set",
                remaining: rest.len(),
                args: vec![OscType::Int(buf)],
            });
        }
        Ok(())
    }

    /// `/b_getn <buf> (index, count)...` -> one `/b_setn`.
    fn b_getn(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let buf = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_getn expects a bufnum"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("b_getn expects index/count pairs"));
        }
        for pair in rest.chunks_exact(2) {
            let index = index_arg(&pair[0])?;
            let count = count_arg(Some(&pair[1]))?;
            controller
                .query_buffer_range(buf as usize, index, count)
                .map_err(|_| OscError::QueueFull)?;
        }
        let ranges = rest.len() / 2;
        if ranges == 0 {
            self.reply_msg("/b_setn", vec![OscType::Int(buf)]);
        } else {
            self.push_query(PendingQuery::Ranges {
                addr: "/b_setn",
                remaining: ranges,
                in_range: 0,
                args: vec![OscType::Int(buf)],
            });
        }
        Ok(())
    }

    /// `/g_queryTree <group> [flag]` (`dump=false`) / `/g_dumpTree` (`dump=true`).
    fn g_query_tree(
        &mut self,
        controller: &mut Controller,
        args: &[OscType],
        dump: bool,
    ) -> Result<(), OscError> {
        let group = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("queryTree expects a group"))?,
        )?;
        let flag = matches!(args.get(1), Some(OscType::Int(f)) if *f != 0);
        if dump {
            controller
                .dump_tree(group, flag)
                .map_err(|_| OscError::QueueFull)?;
        } else {
            controller
                .query_tree(group, flag)
                .map_err(|_| OscError::QueueFull)?;
        }
        self.push_query(PendingQuery::Tree {
            dump,
            flag,
            last_node: -1,
            last_def: None,
            args: Vec::new(),
        });
        Ok(())
    }

    /// Reassemble one query [`Reply`] into `pending`, returning whether `pending` is now complete
    /// (its OSC message emitted / dump routed).
    fn apply_reply(
        &mut self,
        controller: &Controller,
        pending: &mut PendingQuery,
        reply: Reply,
    ) -> bool {
        match pending {
            PendingQuery::Sync => {
                if let Reply::Synced { id } = reply {
                    self.reply_packet(encode::encode_synced(id));
                }
                true
            }
            PendingQuery::Status => {
                if let Some(packet) = encode::encode_status(&reply) {
                    self.reply_packet(packet);
                }
                true
            }
            PendingQuery::RtMemory => {
                if let Some(packet) = encode::encode_rt_memory(&reply) {
                    self.reply_packet(packet);
                }
                true
            }
            PendingQuery::Node => {
                if let Some(packet) = encode::encode_node_info(&reply) {
                    // scsynth answers `/n_query` by broadcasting `/n_info` to all registered clients
                    // (Server-Command-Reference: "sent to all registered clients"), not just the
                    // asker - so an unregistered querier receives nothing.
                    self.push_packet(ReplyTarget::Broadcast, packet);
                } else if let Reply::NodeNotFound { node } = reply {
                    // scsynth returns kSCErr_NodeNotFound, which the dispatcher reports as a `/fail`
                    // back to the requester (errors are not broadcast).
                    self.fail("/n_query", &alloc::format!("Node {node} not found"));
                }
                true
            }
            PendingQuery::Pairs {
                addr,
                remaining,
                args,
            } => {
                match reply {
                    Reply::ControlValue { bus, value } => {
                        args.push(OscType::Int(bus));
                        args.push(OscType::Float(value));
                    }
                    Reply::BufferValue { index, value, .. } => {
                        args.push(OscType::Int(index));
                        args.push(OscType::Float(value));
                    }
                    _ => {}
                }
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    self.reply_msg(addr, core::mem::take(args));
                    true
                } else {
                    false
                }
            }
            PendingQuery::Ranges {
                addr,
                remaining,
                in_range,
                args,
            } => {
                match reply {
                    Reply::ControlRangeHeader { start, count } => {
                        args.push(OscType::Int(start));
                        args.push(OscType::Int(count));
                        *in_range = count.max(0) as usize;
                    }
                    Reply::BufferRangeHeader { index, count, .. } => {
                        args.push(OscType::Int(index));
                        args.push(OscType::Int(count));
                        *in_range = count.max(0) as usize;
                    }
                    Reply::RangeValue { value } => {
                        args.push(OscType::Float(value));
                        *in_range = in_range.saturating_sub(1);
                    }
                    _ => {}
                }
                if *in_range == 0 {
                    *remaining = remaining.saturating_sub(1);
                }
                if *remaining == 0 {
                    self.reply_msg(addr, core::mem::take(args));
                    true
                } else {
                    false
                }
            }
            PendingQuery::SGet {
                controls,
                remaining,
                args,
            } => match reply {
                Reply::SGetValue { value, .. } => {
                    if let Some(token) = controls.pop_front() {
                        args.push(token);
                    }
                    args.push(OscType::Float(value));
                    *remaining = remaining.saturating_sub(1);
                    if *remaining == 0 {
                        self.reply_msg("/n_set", core::mem::take(args));
                        true
                    } else {
                        false
                    }
                }
                Reply::SGetMissing { .. } => {
                    self.fail("/s_get", "node not found");
                    true
                }
                _ => false,
            },
            PendingQuery::SGetN {
                controls,
                remaining,
                in_range,
                args,
            } => match reply {
                Reply::SGetRangeHeader { count, .. } => {
                    if let Some(token) = controls.pop_front() {
                        args.push(token);
                    }
                    args.push(OscType::Int(count));
                    *in_range = count.max(0) as usize;
                    if *in_range == 0 {
                        *remaining = remaining.saturating_sub(1);
                    }
                    if *remaining == 0 {
                        self.reply_msg("/n_setn", core::mem::take(args));
                        true
                    } else {
                        false
                    }
                }
                Reply::RangeValue { value } => {
                    args.push(OscType::Float(value));
                    *in_range = in_range.saturating_sub(1);
                    if *in_range == 0 {
                        *remaining = remaining.saturating_sub(1);
                    }
                    if *remaining == 0 {
                        self.reply_msg("/n_setn", core::mem::take(args));
                        true
                    } else {
                        false
                    }
                }
                Reply::SGetMissing { .. } => {
                    self.fail("/s_getn", "node not found");
                    true
                }
                _ => false,
            },
            PendingQuery::Tree {
                dump,
                flag,
                last_node,
                last_def,
                args,
            } => match reply {
                Reply::QueryTreeHeader { flag: f } | Reply::DumpTreeHeader { flag: f } => {
                    args.clear();
                    args.push(OscType::Int(f));
                    false
                }
                Reply::QueryTreeNode { node, num_children } => {
                    args.push(OscType::Int(node));
                    args.push(OscType::Int(num_children));
                    *last_node = node;
                    false
                }
                Reply::QueryTreeSynth { num_controls } => {
                    let name = self
                        .node_defs
                        .get(last_node)
                        .cloned()
                        .unwrap_or_else(|| "?".to_string());
                    args.push(OscType::String(name.clone()));
                    if *flag {
                        args.push(OscType::Int(num_controls));
                    }
                    *last_def = Some(name);
                    false
                }
                Reply::QueryTreeControl { index, value } => {
                    let token = last_def
                        .as_deref()
                        .and_then(|d| controller.synthdef(d))
                        .and_then(|def| def.params.get(index as usize))
                        .map(|p| OscType::String(p.name.clone()))
                        .unwrap_or(OscType::Int(index));
                    args.push(token);
                    args.push(OscType::Float(value));
                    false
                }
                Reply::QueryTreeEnd => {
                    if *dump {
                        let text = format_tree(args);
                        if let Some(sink) = self.dump_sink.as_mut() {
                            sink(&text);
                        }
                    } else {
                        self.reply_msg("/g_queryTree.reply", core::mem::take(args));
                    }
                    true
                }
                _ => false,
            },
        }
    }

    fn c_set(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        if args.is_empty() || !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("c_set expects bus/value pairs"));
        }
        for pair in args.chunks_exact(2) {
            let bus = bus_index(&pair[0])?;
            let value = float_arg(&pair[1])?;
            controller
                .set_control_bus(bus, value)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn c_setn(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let mut i = 0;
        while i < args.len() {
            let start = bus_index(&args[i])?;
            let count = count_arg(args.get(i + 1))?;
            i += 2;
            if i + count > args.len() {
                return Err(OscError::BadArguments(
                    "c_setn value count exceeds arguments",
                ));
            }
            for (j, arg) in args[i..i + count].iter().enumerate() {
                controller
                    .set_control_bus(start + j as u32, float_arg(arg)?)
                    .map_err(|_| OscError::QueueFull)?;
            }
            i += count;
        }
        Ok(())
    }

    /// `/c_fill (bus, count, value)...`: set each contiguous range of control buses to `value`.
    fn c_fill(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        if args.is_empty() || !args.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "c_fill expects bus/count/value triples",
            ));
        }
        for triple in args.chunks_exact(3) {
            let start = bus_index(&triple[0])?;
            let count = count_arg(Some(&triple[1]))?;
            let value = float_arg(&triple[2])?;
            for j in 0..count {
                controller
                    .set_control_bus(start + j as u32, value)
                    .map_err(|_| OscError::QueueFull)?;
            }
        }
        Ok(())
    }

    fn n_map(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_map expects a node"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("n_map expects control/bus pairs"));
        }
        for pair in rest.chunks_exact(2) {
            let control = self.control_index(controller, node, &pair[0])?;
            let bus = map_bus(&pair[1])?;
            controller
                .map_control(node, control, bus)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn n_mapn(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_mapn expects a node"))?,
        )?;
        let rest = &args[1..];
        if rest.is_empty() || !rest.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "n_mapn expects control/bus/count triples",
            ));
        }
        for triple in rest.chunks_exact(3) {
            let control = self.control_index(controller, node, &triple[0])?;
            let bus = map_bus(&triple[1])?;
            let count = count_arg(Some(&triple[2]))?;
            controller
                .map_control_n(node, control, bus, count)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn n_mapa(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_mapa expects a node"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("n_mapa expects control/bus pairs"));
        }
        for pair in rest.chunks_exact(2) {
            let control = self.control_index(controller, node, &pair[0])?;
            let bus = map_bus(&pair[1])?;
            controller
                .map_control_audio(node, control, bus)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn n_mapan(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_mapan expects a node"))?,
        )?;
        let rest = &args[1..];
        if rest.is_empty() || !rest.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "n_mapan expects control/bus/count triples",
            ));
        }
        for triple in rest.chunks_exact(3) {
            let control = self.control_index(controller, node, &triple[0])?;
            let bus = map_bus(&triple[1])?;
            let count = count_arg(Some(&triple[2]))?;
            controller
                .map_control_audio_n(node, control, bus, count)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn b_alloc(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_alloc expects a bufnum"))?,
        )?;
        let num_frames = count_arg(args.get(1))?;
        let num_channels = match args.get(2) {
            Some(OscType::Int(c)) => (*c).max(1) as usize,
            _ => 1,
        };
        let sample_rate = controller.sample_rate();
        controller
            .buffer_alloc(bufnum as usize, num_frames, num_channels, sample_rate)
            .map_err(|_| OscError::QueueFull)?;
        self.buffers.insert(
            bufnum,
            BufferInfo {
                num_frames,
                num_channels,
                sample_rate,
            },
        );
        self.run_completion_bytes(controller, last_blob(args));
        self.done("/b_alloc", bufnum);
        Ok(())
    }

    fn b_free(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_free expects a bufnum"))?,
        )?;
        controller
            .buffer_free(bufnum as usize)
            .map_err(|_| OscError::QueueFull)?;
        self.buffers.remove(&bufnum);
        self.run_completion_bytes(controller, last_blob(args));
        self.done("/b_free", bufnum);
        Ok(())
    }

    fn b_zero(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_zero expects a bufnum"))?,
        )?;
        // Zero by re-allocating the same dimensions (the new buffer is zeroed); the old one is
        // dropped off the audio thread, the same as `/b_alloc`.
        match self.buffers.get(&bufnum).copied() {
            Some(info) => {
                controller
                    .buffer_alloc(
                        bufnum as usize,
                        info.num_frames,
                        info.num_channels,
                        info.sample_rate,
                    )
                    .map_err(|_| OscError::QueueFull)?;
                self.run_completion_bytes(controller, last_blob(args));
                self.done("/b_zero", bufnum);
            }
            None => self.fail("/b_zero", "unknown buffer"),
        }
        Ok(())
    }

    fn b_query(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let mut info = Vec::with_capacity(args.len() * 4);
        for arg in args {
            let bufnum = int_arg(arg)?;
            let (frames, channels, sample_rate) = match self.buffers.get(&bufnum) {
                Some(b) => (
                    b.num_frames as i32,
                    b.num_channels as i32,
                    b.sample_rate as f32,
                ),
                None => (0, 0, 0.0),
            };
            info.push(OscType::Int(bufnum));
            info.push(OscType::Int(frames));
            info.push(OscType::Int(channels));
            info.push(OscType::Float(sample_rate));
        }
        self.reply_msg("/b_info", info);
        Ok(())
    }

    /// `/b_set bufID (sampleIndex, value)...`: overwrite individual buffer samples (flat indices).
    fn b_set(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_set expects a bufnum"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("b_set expects sample/value pairs"));
        }
        for pair in rest.chunks_exact(2) {
            let sample = index_arg(&pair[0])?;
            let value = float_arg(&pair[1])?;
            controller
                .buffer_set_sample(bufnum as usize, sample, value)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/b_setn bufID (start, count, value...)...`: overwrite contiguous ranges of buffer samples.
    fn b_setn(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_setn expects a bufnum"))?,
        )?;
        let rest = &args[1..];
        let mut i = 0;
        while i < rest.len() {
            let start = index_arg(&rest[i])?;
            let count = count_arg(rest.get(i + 1))?;
            i += 2;
            if i + count > rest.len() {
                return Err(OscError::BadArguments(
                    "b_setn value count exceeds arguments",
                ));
            }
            for (j, arg) in rest[i..i + count].iter().enumerate() {
                controller
                    .buffer_set_sample(bufnum as usize, start + j, float_arg(arg)?)
                    .map_err(|_| OscError::QueueFull)?;
            }
            i += count;
        }
        Ok(())
    }

    /// `/b_fill bufID (start, count, value)...`: fill contiguous ranges of buffer samples.
    fn b_fill(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_fill expects a bufnum"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(3) {
            return Err(OscError::BadArguments(
                "b_fill expects start/count/value triples",
            ));
        }
        for triple in rest.chunks_exact(3) {
            let start = index_arg(&triple[0])?;
            let count = count_arg(Some(&triple[1]))?;
            let value = float_arg(&triple[2])?;
            controller
                .buffer_fill(bufnum as usize, start, count, value)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/b_setSampleRate bufID rate`: overwrite a buffer's sample-rate metadata.
    fn b_set_sample_rate(
        &mut self,
        controller: &mut Controller,
        args: &[OscType],
    ) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_setSampleRate expects a bufnum"))?,
        )?;
        let rate = float_arg(
            args.get(1)
                .ok_or(OscError::BadArguments("b_setSampleRate expects a rate"))?,
        )? as f64;
        controller
            .buffer_set_sample_rate(bufnum as usize, rate)
            .map_err(|_| OscError::QueueFull)?;
        // Keep the control-side mirror in step, so `/b_query` reports the new rate.
        if let Some(info) = self.buffers.get_mut(&bufnum) {
            info.sample_rate = rate;
        }
        Ok(())
    }

    /// `/b_gen <bufID> <genName> <flags> <args…> [completionMsg]`: fill a buffer from a generator.
    ///
    /// `sine1`/`sine2`/`sine3`/`cheby` are computed control-side into a fresh buffer and installed via
    /// the `/b_alloc` swap path (so no engine round-trip, and the old buffer trashes off-thread);
    /// `copy` reads a live source buffer, so it routes to the engine. Flags: `normalize`(1) supported;
    /// `wavetable`(2) rejected (the consuming `Osc` UGen is unimplemented); `clear`(4) is implicit -
    /// the dispatcher has no copy of the current samples, so it always generates fresh (a documented
    /// deviation from scsynth's accumulate-when-unset). Non-mono buffers are rejected.
    fn b_gen(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_gen expects a bufnum"))?,
        )?;
        let gen_name = str_arg(
            args.get(1)
                .ok_or(OscError::BadArguments("b_gen expects a gen name"))?,
        )?
        .to_string();
        let flags = int_arg(
            args.get(2)
                .ok_or(OscError::BadArguments("b_gen expects flags"))?,
        )?;
        // Gen args are everything after the flags, minus a trailing completion blob.
        let completion = last_blob(args);
        let gen_end = if completion.is_some() {
            args.len() - 1
        } else {
            args.len()
        };
        let gen_args = &args[3.min(gen_end)..gen_end];

        if flags & 2 != 0 {
            self.fail("/b_gen", "wavetable mode unsupported");
            return Ok(());
        }
        let Some(info) = self.buffers.get(&bufnum).copied() else {
            self.fail("/b_gen", "unknown buffer");
            return Ok(());
        };

        if gen_name == "copy" {
            return self.b_gen_copy(controller, bufnum, gen_args, completion);
        }
        if info.num_channels != 1 {
            self.fail("/b_gen", "b_gen requires a mono buffer");
            return Ok(());
        }
        let floats: Vec<f32> = gen_args.iter().map(float_arg).collect::<Result<_, _>>()?;
        let mut samples = vec![0.0f32; info.num_frames];
        match gen_name.as_str() {
            "sine1" => bgen::sine1(&mut samples, &floats),
            "sine2" => bgen::sine2(&mut samples, &to_pairs(&floats)),
            "sine3" => bgen::sine3(&mut samples, &to_triples(&floats)),
            "cheby" => bgen::cheby(&mut samples, &floats),
            _ => {
                self.fail("/b_gen", "unsupported gen");
                return Ok(());
            }
        }
        if flags & 1 != 0 {
            bgen::normalize(&mut samples);
        }
        let buffer = Box::new(Buffer::from_interleaved(samples, 1, info.sample_rate));
        controller
            .buffer_set(bufnum as usize, buffer)
            .map_err(|_| OscError::QueueFull)?;
        self.run_completion_bytes(controller, completion);
        self.done("/b_gen", bufnum);
        Ok(())
    }

    /// `/b_gen <buf> "copy" <dstStart> <srcBufID> <srcStart> <numSamples>`: copy a region from a live
    /// source buffer into the destination, on the audio thread.
    fn b_gen_copy(
        &mut self,
        controller: &mut Controller,
        bufnum: i32,
        gen_args: &[OscType],
        completion: Option<&[u8]>,
    ) -> Result<(), OscError> {
        let bad = || OscError::BadArguments("b_gen copy expects dstStart, srcBuf, srcStart, count");
        let dst_start = index_arg(gen_args.first().ok_or_else(bad)?)?;
        let src = int_arg(gen_args.get(1).ok_or_else(bad)?)? as usize;
        let src_start = index_arg(gen_args.get(2).ok_or_else(bad)?)?;
        let count = index_arg(gen_args.get(3).ok_or_else(bad)?)?;
        controller
            .buffer_copy_region(bufnum as usize, dst_start, src, src_start, count)
            .map_err(|_| OscError::QueueFull)?;
        self.run_completion_bytes(controller, completion);
        self.done("/b_gen", bufnum);
        Ok(())
    }

    fn b_alloc_read(&mut self, args: &[OscType]) -> Result<(), OscError> {
        self.queue_load("/b_allocRead", args, None)
    }

    fn b_read(&mut self, args: &[OscType]) -> Result<(), OscError> {
        // Simplified: reads the file region and replaces the buffer (the `bufStartFrame`/`leaveOpen`
        // arguments are ignored for now).
        self.queue_load("/b_read", args, None)
    }

    /// `/b_allocReadChannel bufnum path [startFrame numFrames] channels...`: allocate a buffer and read
    /// only the selected file channels into it (scsynth's `BufAllocReadChannelCmd`). The channel
    /// indices are the trailing `Int` args after `startFrame`/`numFrames` (at `args[4..]`).
    fn b_alloc_read_channel(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let channels = channel_list(args, 4);
        self.queue_load("/b_allocReadChannel", args, Some(channels))
    }

    /// `/b_readChannel bufnum path [startFrame numFrames bufOffset leaveOpen] channels...`: read only the
    /// selected file channels into the buffer (scsynth's `BufReadChannelCmd`). Like `/b_read`, plyphon
    /// replaces the whole buffer (`bufOffset`/`leaveOpen` ignored), so the channels are the trailing
    /// `Int` args at `args[6..]`.
    fn b_read_channel(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let channels = channel_list(args, 6);
        self.queue_load("/b_readChannel", args, Some(channels))
    }

    /// `/b_write bufnum path [header] [sample] [numFrames] [startFrame] [leaveOpen]` [completion]:
    /// snapshot the buffer at `bufnum` to `path`. The engine streams the buffer's samples out
    /// race-free (it shares no buffer memory with the host); this queues a [`PendingWrite`] that
    /// [`run_pending`](Self::run_pending) drives to completion across ticks, replying `/done /b_write
    /// <bufnum>`. Mirrors scsynth's `BufWriteCmd` (`SC_SequencedCommand.cpp`), but writes through the
    /// host's [`BufferSink`] rather than libsndfile - so the header/sample formats are the sink's
    /// choice (`path`'s extension), not these arguments.
    ///
    /// `leaveOpen = 0` writes the whole-buffer snapshot; `leaveOpen = 1` leaves the file open for a
    /// `DiskOut.ar(bufnum)` to stream into (installing a recording slot at `bufnum`, the streaming ring,
    /// so any flat data there is replaced - which is why the snapshot is the separate `leaveOpen = 0`
    /// path), closed later by `/b_close`. Partial `numFrames`/`startFrame` ranges remain deferred.
    fn b_write(&mut self, controller: &mut Controller, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_write expects a bufnum"))?,
        )?;
        let path = str_arg(
            args.get(1)
                .ok_or(OscError::BadArguments("b_write expects a path"))?,
        )?
        .to_string();
        // `leaveOpen` follows the two format strings and the numFrames/startFrame ints (scsynth's
        // positional layout); default 0. A non-zero value selects the open-for-streaming form.
        let leave_open = matches!(args.get(6), Some(OscType::Int(v)) if *v != 0);
        let Some(info) = self.buffers.get(&bufnum).copied() else {
            self.fail("/b_write", "buffer not allocated");
            return Ok(());
        };
        let consumer = if leave_open {
            // Install a `DiskOut` recording slot at `bufnum` (the slot becomes the streaming ring), to
            // be drained continuously until `/b_close`.
            controller.buffer_cue_write(
                bufnum as usize,
                info.num_channels,
                info.sample_rate,
                WRITE_CHUNK_FRAMES,
                WRITE_CHUNKS,
            )
        } else {
            // Snapshot: copy the buffer out race-free without disturbing the slot.
            controller.buffer_write_out(
                bufnum as usize,
                info.num_channels,
                info.sample_rate,
                WRITE_CHUNK_FRAMES,
                WRITE_CHUNKS,
            )
        }
        .map_err(|_| OscError::QueueFull)?;
        self.writes_in_progress.push(PendingWrite {
            command: "/b_write",
            bufnum,
            key: path,
            info: StreamInfo {
                num_channels: info.num_channels,
                sample_rate: info.sample_rate,
                // A left-open stream's total is unknown; a snapshot's is the buffer's frame count.
                total_frames: (!leave_open).then_some(info.num_frames as u64),
            },
            drainer: StreamDrainer::new(consumer),
            sink: None,
            leave_open,
            completion: last_blob(args).map(|bytes| bytes.to_vec()),
            target: self.current_target,
        });
        Ok(())
    }

    /// `/b_close bufnum` [completion]: close a file left open by `/b_write leaveOpen=1`, ending its
    /// stream and freeing the recording slot. Queued; `run_pending` does the final drain + close
    /// (async) and replies `/done /b_close <bufnum>` - an unopened bufnum fails. Mirrors scsynth's
    /// `BufCloseCmd`, except the recording slot is freed (it holds no flat data) and `DiskOut`'s final
    /// sub-chunk tail may be dropped, the same bounded tail loss continuous `DiskOut` recording has.
    fn b_close(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_close expects a bufnum"))?,
        )?;
        self.pending_closes.push(PendingClose {
            bufnum,
            completion: last_blob(args).map(|bytes| bytes.to_vec()),
            target: self.current_target,
        });
        Ok(())
    }

    /// `/d_load <path>` [completion]: load the SynthDef file at `path` - the host's [`DefSource`]
    /// reads the bytes in [`run_pending`](Self::run_pending), which parses and registers each def, then
    /// replies `/done /d_load`. (plyphon treats `path` as a single file; scsynth globs it.)
    fn d_load(&mut self, args: &[OscType]) -> Result<(), OscError> {
        self.queue_def("/d_load", false, args)
    }

    /// `/d_loadDir <dir>` [completion]: load every SynthDef file under `dir`, then reply `/done
    /// /d_loadDir`.
    fn d_load_dir(&mut self, args: &[OscType]) -> Result<(), OscError> {
        self.queue_def("/d_loadDir", true, args)
    }

    /// Queue an asynchronous def load of `path`, run later by [`Self::run_pending`].
    fn queue_def(
        &mut self,
        command: &'static str,
        is_dir: bool,
        args: &[OscType],
    ) -> Result<(), OscError> {
        let key = str_arg(
            args.first()
                .ok_or(OscError::BadArguments("def load expects a path"))?,
        )?
        .to_string();
        self.pending_defs.push(PendingDef {
            command,
            is_dir,
            key,
            completion: last_blob(args).map(|bytes| bytes.to_vec()),
            target: self.current_target,
        });
        Ok(())
    }

    /// `/cmd <name> [args...]`: a plugin command. plyphon has no built-in command plugins, so the host's
    /// [`CommandHost`] interprets it in [`run_pending`](Self::run_pending) (an absent handler fails the
    /// command). Mirrors scsynth's `PlugIn_DoCmd` (`SC_UnitDef.cpp`): the command name leads, the rest is
    /// the command's payload.
    fn cmd(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let name = str_arg(
            args.first()
                .ok_or(OscError::BadArguments("/cmd expects a command name"))?,
        )?
        .to_string();
        self.queue_command("/cmd", CmdTarget::Plugin, name, &args[1..]);
        Ok(())
    }

    /// `/u_cmd <node> <unit-index> <name> [args...]`: a unit command addressed to one unit within a
    /// node's def. Mirrors scsynth's `Unit_DoCmd` (`SC_UnitDef.cpp`): node id, unit index, command name,
    /// then the command's payload. (scsynth's `/n_cmd` is unimplemented, so plyphon omits it.)
    fn u_cmd(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("/u_cmd expects a node id"))?,
        )?;
        let index = int_arg(
            args.get(1)
                .ok_or(OscError::BadArguments("/u_cmd expects a unit index"))?,
        )?;
        let name = str_arg(
            args.get(2)
                .ok_or(OscError::BadArguments("/u_cmd expects a command name"))?,
        )?
        .to_string();
        self.queue_command("/u_cmd", CmdTarget::Unit { node, index }, name, &args[3..]);
        Ok(())
    }

    /// Queue an asynchronous plugin/unit command, run later by [`Self::run_pending`] through the host's
    /// [`CommandHost`].
    fn queue_command(
        &mut self,
        command: &'static str,
        cmd_target: CmdTarget,
        name: String,
        args: &[OscType],
    ) {
        self.pending_commands.push(PendingCommand {
            command,
            cmd_target,
            name,
            args: args.to_vec(),
            target: self.current_target,
        });
    }

    /// Queue an asynchronous load of `path` into `bufnum`, run later by [`Self::run_pending`].
    /// `channels` selects a subset of the file's channels (`*Channel` forms; `None` reads all).
    fn queue_load(
        &mut self,
        command: &'static str,
        args: &[OscType],
        channels: Option<Vec<i32>>,
    ) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("buffer read expects a bufnum"))?,
        )?;
        let key = str_arg(
            args.get(1)
                .ok_or(OscError::BadArguments("buffer read expects a path"))?,
        )?
        .to_string();
        let start_frame = match args.get(2) {
            Some(OscType::Int(s)) => (*s).max(0) as u64,
            _ => 0,
        };
        let num_frames = match args.get(3) {
            Some(OscType::Int(n)) if *n > 0 => Some(*n as u64),
            _ => None,
        };
        self.pending.push(PendingLoad {
            command,
            bufnum,
            key,
            region: ReadRegion {
                start_frame,
                num_frames,
            },
            channels,
            completion: last_blob(args).map(|bytes| bytes.to_vec()),
            target: self.current_target,
        });
        Ok(())
    }

    /// Apply an embedded OSC completion message (the trailing blob of an async command), if present.
    fn run_completion_bytes(&mut self, controller: &mut Controller, bytes: Option<&[u8]>) {
        if let Some(bytes) = bytes
            && let Ok((_, packet)) = rosc::decoder::decode_udp(bytes)
        {
            let _ = self.apply(controller, &packet);
        }
    }

    /// Queue an OSC reply for the current requester (see [`current_target`](Self::current_target)).
    fn reply_msg(&mut self, addr: &str, args: Vec<OscType>) {
        let target = self.current_target;
        self.push_reply(target, addr, args);
    }

    /// Queue a pre-built reply packet for the current requester - the packet twin of [`reply_msg`](
    /// Self::reply_msg), used by the getter arms that delegate to the pure [`encode`] encoders.
    fn reply_packet(&mut self, packet: OscPacket) {
        let target = self.current_target;
        self.push_packet(target, packet);
    }

    /// Queue an OSC reply for an explicit destination, regardless of the current requester (e.g. node
    /// notifications and the success `/n_info`, which broadcast).
    fn push_reply(&mut self, target: ReplyTarget, addr: &str, args: Vec<OscType>) {
        self.push_packet(
            target,
            OscPacket::Message(OscMessage {
                addr: addr.to_string(),
                args,
            }),
        );
    }

    /// Queue a pre-built OSC packet for `target` - the single leaf the reply/notify helpers and the
    /// pure [`encode`] encoders funnel through.
    fn push_packet(&mut self, target: ReplyTarget, packet: OscPacket) {
        self.replies.push((target, packet));
    }

    /// Record an outstanding getter against the current requester, in issue order, so its later
    /// [`Reply`]s reassemble into a message routed back to that client.
    fn push_query(&mut self, query: PendingQuery) {
        self.pending_queries.push_back((self.current_target, query));
    }

    /// Queue a `/done <command> <bufnum>` reply.
    fn done(&mut self, command: &str, bufnum: i32) {
        self.reply_msg(
            "/done",
            vec![OscType::String(command.to_string()), OscType::Int(bufnum)],
        );
    }

    /// Queue a `/done <command>` reply (no value), for actions that answer with just the command name
    /// (scsynth's `SendDone`, e.g. `/d_load`).
    fn done_command(&mut self, command: &str) {
        self.reply_msg("/done", vec![OscType::String(command.to_string())]);
    }

    /// Queue a `/fail <command> <error>` reply, unless error posting is currently suppressed
    /// (scsynth's `/error`).
    fn fail(&mut self, command: &str, error: &str) {
        if !self.errors_enabled() {
            return;
        }
        self.reply_msg(
            "/fail",
            vec![
                OscType::String(command.to_string()),
                OscType::String(error.to_string()),
            ],
        );
    }

    /// Whether `/fail` replies are currently queued: the bundle-local error override if set, else
    /// the permanent error-posting mode (scsynth's `/error`).
    fn errors_enabled(&self) -> bool {
        self.error_bundle.unwrap_or(self.error_perm)
    }

    /// Resolve a control argument (an `int` index or a `string` name) to a parameter index.
    fn control_index(
        &self,
        controller: &Controller,
        node: i32,
        arg: &OscType,
    ) -> Result<usize, OscError> {
        match arg {
            OscType::Int(idx) => {
                usize::try_from(*idx).map_err(|_| OscError::BadArguments("negative control index"))
            }
            OscType::String(name) => self.resolve_param(controller, node, None, name),
            _ => Err(OscError::BadArguments("control must be an int or string")),
        }
    }

    /// Apply `(control, value)` argument pairs to `node`. A control is an `int` index or a `string`
    /// name resolved against the node's SynthDef (`def_name` when known, else the tracked one).
    fn apply_controls(
        &mut self,
        controller: &mut Controller,
        node: i32,
        def_name: Option<&str>,
        args: &[OscType],
    ) -> Result<(), OscError> {
        let mut i = 0;
        while i + 1 < args.len() {
            let index = match &args[i] {
                OscType::Int(idx) => usize::try_from(*idx)
                    .map_err(|_| OscError::BadArguments("negative control index"))?,
                OscType::String(name) => self.resolve_param(controller, node, def_name, name)?,
                _ => return Err(OscError::BadArguments("control must be an int or string")),
            };
            let value = float_arg(&args[i + 1])?;
            controller
                .set_control(node, index, value)
                .map_err(|_| OscError::QueueFull)?;
            i += 2;
        }
        Ok(())
    }

    fn resolve_param(
        &self,
        controller: &Controller,
        node: i32,
        def_name: Option<&str>,
        name: &str,
    ) -> Result<usize, OscError> {
        let def_name = match def_name {
            Some(d) => d,
            None => self
                .node_defs
                .get(&node)
                .map(String::as_str)
                .ok_or(OscError::UnknownNode(node))?,
        };
        let def = controller
            .synthdef(def_name)
            .ok_or(OscError::UnknownNode(node))?;
        def.param_index(name)
            .ok_or_else(|| OscError::UnknownParam(name.to_string()))
    }
}

/// Fills one interleaved input block per control block for [`render_osc_score`] (the input buses for
/// `In.ar`). The block is zeroed before each call, so leaving the tail untouched zero-pads past
/// end-of-input.
pub type InputBlockFn<'a> = &'a mut dyn FnMut(&mut [f32]);

/// Render a parsed OSC `score` offline through `dispatcher`, writing each interleaved output block
/// to `sink` - the scsynth `-N` workflow over plyphon's [`Render`] driver.
///
/// `score` must be time-sorted (as [`parse_score`] returns it). Each control block, every entry due
/// by [`Render::block_end`] is applied through the dispatcher - so each command schedules at its
/// time tag and fires sample-accurately - then one block is rendered and handed to `sink`. Node
/// lifecycle events are forwarded to [`OscDispatcher::notify`] so `/n_go`/`/n_end` etc. queue as
/// replies. `input`, if given, fills one interleaved input block per call (see [`InputBlockFn`]).
/// Rendering runs until `until` (the last command's time plus a tail, or an explicit duration). The
/// render is deterministic - it drives [`Render::step`], never a resync.
pub fn render_osc_score(
    render: &mut Render,
    dispatcher: &mut OscDispatcher,
    controller: &mut Controller,
    score: &[ScoreEntry],
    mut input: Option<InputBlockFn<'_>>,
    mut sink: impl FnMut(&[f32]),
    until: RenderUntil,
) -> Result<(), OscError> {
    let max_time = score.iter().map(|e| e.osc_time).max().unwrap_or(0);
    let end_time = until.end_time(max_time);
    let mut in_block = vec![0.0f32; render.input_block_len()];
    let mut next = 0;
    while render.block_start() <= end_time {
        let cutoff = render.block_end();
        while next < score.len() && score[next].osc_time <= cutoff {
            dispatcher.apply(controller, &score[next].packet)?;
            next += 1;
        }
        let block = match input.as_deref_mut() {
            Some(fill) => {
                in_block.iter_mut().for_each(|s| *s = 0.0);
                fill(&mut in_block);
                render.step(&in_block)
            }
            None => render.step(&[]),
        };
        sink(block);
        while let Some(event) = render.poll() {
            dispatcher.notify(event);
        }
        while let Some(reply) = render.poll_reply() {
            dispatcher.reply(controller, reply);
        }
    }
    render.finish();
    Ok(())
}

/// Pack an [`OscTime`] into its 32.32 fixed-point OSC/NTP `u64` (`(seconds << 32) | fractional`).
pub(crate) fn pack_ntp(timetag: OscTime) -> u64 {
    ((timetag.seconds as u64) << 32) | timetag.fractional as u64
}

/// Map an OSC bundle time tag to a [`CommandTime`]. The special "immediately" tags `0` and `1`
/// apply now (scsynth's `PerformOSCPacket`); anything else schedules for that absolute OSC/NTP time
/// (the 32.32 fixed-point value since 1900), with the engine resolving a late tag to "as soon as
/// possible" on the audio thread.
fn bundle_command_time(timetag: OscTime) -> CommandTime {
    let ntp = pack_ntp(timetag);
    if ntp <= 1 {
        CommandTime::Immediate
    } else {
        CommandTime::At(ntp)
    }
}

/// Map a SuperCollider `addAction` code to a plyphon [`AddAction`] (only head/tail supported).
fn add_action(code: i32) -> Result<AddAction, OscError> {
    match code {
        0 => Ok(AddAction::Head),
        1 => Ok(AddAction::Tail),
        2 => Ok(AddAction::Before),
        3 => Ok(AddAction::After),
        // 4 is addReplace, not yet supported.
        other => Err(OscError::UnsupportedAddAction(other)),
    }
}

/// Group a flat float list into `(freq, amp)` pairs for `/b_gen sine2` (a trailing odd value drops).
fn to_pairs(floats: &[f32]) -> Vec<(f32, f32)> {
    floats.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

/// Group a flat float list into `(freq, amp, phase)` triples for `/b_gen sine3`.
fn to_triples(floats: &[f32]) -> Vec<(f32, f32, f32)> {
    floats.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect()
}

/// Format a reassembled `/g_queryTree.reply` arg list into an indented text tree for `/g_dumpTree`.
/// The args are `[flag, then per node: id, numChildren, (defName, [numControls, (control, value)…])
/// for a synth]`; the pre-order child counts reconstruct the nesting depth.
fn format_tree(args: &[OscType]) -> String {
    let int_at = |i: usize| match args.get(i) {
        Some(OscType::Int(v)) => *v,
        _ => 0,
    };
    let flag = int_at(0) != 0;
    let mut out = String::new();
    let mut stack: Vec<i32> = Vec::new(); // remaining children to read at each open group level
    let mut i = 1;
    while i + 1 < args.len() {
        let id = int_at(i);
        let num_children = int_at(i + 1);
        i += 2;
        for _ in 0..stack.len() {
            out.push_str("   ");
        }
        if num_children < 0 {
            // A synth: its def name, and (when `flag`) its controls.
            let name = match args.get(i) {
                Some(OscType::String(s)) => s.clone(),
                _ => String::from("?"),
            };
            i += 1;
            out.push_str(&alloc::format!("{id} synth {name}"));
            if flag {
                let n = int_at(i);
                i += 1;
                for _ in 0..n.max(0) {
                    let label = match args.get(i) {
                        Some(OscType::String(s)) => s.clone(),
                        Some(OscType::Int(v)) => alloc::format!("{v}"),
                        _ => String::new(),
                    };
                    let value = match args.get(i + 1) {
                        Some(OscType::Float(f)) => *f,
                        _ => 0.0,
                    };
                    out.push_str(&alloc::format!(" {label}: {value}"));
                    i += 2;
                }
            }
            out.push('\n');
            complete_node(&mut stack);
        } else {
            out.push_str(&alloc::format!("{id} group ({num_children} children)\n"));
            if num_children > 0 {
                stack.push(num_children); // descend; its children follow in pre-order
            } else {
                complete_node(&mut stack);
            }
        }
    }
    out
}

/// Account a completed leaf/empty node against its ancestors: decrement the open group's remaining
/// child count, and unwind every level that reaches zero (a completed group is itself one child).
fn complete_node(stack: &mut Vec<i32>) {
    while let Some(top) = stack.last_mut() {
        *top -= 1;
        if *top == 0 {
            stack.pop();
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use core::time::Duration;
    use plyphon::render::nominal_increment;
    use plyphon::{
        InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
        render::OSC_UNITS_PER_SEC,
    };
    use rosc::{OscBundle, OscTime};

    const SR: f64 = 48_000.0;
    const BLOCK: usize = 64;

    /// A click voice: 0.5 held for 5 ms then self-freed, onset placed by `OffsetOut`.
    fn click_def() -> SynthDef {
        SynthDef {
            name: "click".into(),
            params: vec![],
            units: vec![
                UnitSpec::new(
                    "Line",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.5),
                        InputRef::Constant(0.5),
                        InputRef::Constant(0.005),
                        InputRef::Constant(2.0),
                    ],
                    1,
                ),
                UnitSpec::new(
                    "OffsetOut",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.0),
                        InputRef::Unit { unit: 0, output: 0 },
                    ],
                    0,
                ),
            ],
        }
    }

    fn time_for_sample(s: usize) -> u64 {
        let block = (s / BLOCK) as u64;
        let off = (s % BLOCK) as f64;
        block * nominal_increment(SR, BLOCK) + (off * (OSC_UNITS_PER_SEC / SR)).round() as u64
    }

    fn unpack(ntp: u64) -> OscTime {
        OscTime {
            seconds: (ntp >> 32) as u32,
            fractional: ntp as u32,
        }
    }

    fn click_bundle(time: u64, id: i32) -> OscPacket {
        OscPacket::Bundle(OscBundle {
            timetag: unpack(time),
            content: vec![OscPacket::Message(OscMessage {
                addr: "/s_new".into(),
                args: vec![
                    OscType::String("click".into()),
                    OscType::Int(id),
                    OscType::Int(1),             // addAction tail
                    OscType::Int(ROOT_GROUP_ID), // target root
                ],
            })],
        })
    }

    fn encode_score(packets: &[OscPacket]) -> Vec<u8> {
        let mut out = Vec::new();
        for packet in packets {
            let bytes = rosc::encoder::encode(packet).expect("encode");
            out.extend_from_slice(&(bytes.len() as i32).to_be_bytes());
            out.extend_from_slice(&bytes);
        }
        out
    }

    fn assert_onsets(out: &[f32], targets: &[usize]) {
        let mut from = 0;
        for (k, &s) in targets.iter().enumerate() {
            let onset = (from..out.len())
                .find(|&i| out[i] != 0.0)
                .unwrap_or_else(|| panic!("click {k} never sounded"));
            assert_eq!(onset, s, "click {k} should onset at {s}, got {onset}");
            from = onset;
            while from < out.len() && out[from] != 0.0 {
                from += 1;
            }
        }
    }

    /// Render the click `targets` (submitted in `order`) through the binary-score + OSC path.
    fn render(targets: &[usize], order: &[usize]) -> Vec<f32> {
        let opts = Options {
            sample_rate: SR,
            output_channels: 1,
            ..Options::default()
        };
        let (mut controller, nrt, world) = engine(opts);
        controller.add_synthdef(click_def());
        let mut dispatcher = OscDispatcher::new();
        let mut render = Render::new(world, nrt, &opts);

        let packets: Vec<OscPacket> = order
            .iter()
            .map(|&i| click_bundle(time_for_sample(targets[i]), 1000 + i as i32))
            .collect();
        let blob = encode_score(&packets);
        let (score, _max) = parse_score(&blob).expect("parse");

        let mut out = Vec::new();
        render_osc_score(
            &mut render,
            &mut dispatcher,
            &mut controller,
            &score,
            None,
            |block| out.extend_from_slice(block),
            RenderUntil::EndOfScore {
                tail: Duration::from_millis(20),
            },
        )
        .expect("render score");
        out
    }

    #[test]
    fn osc_score_onsets_at_exact_samples() {
        let targets = [600usize, 1503, 2305, 3100];
        let order = [2usize, 0, 3, 1]; // out of order: time tags, not arrival, decide onset
        assert_onsets(&render(&targets, &order), &targets);
    }

    #[test]
    fn osc_score_render_is_deterministic() {
        let targets = [600usize, 1503, 2305, 3100];
        let order = [0usize, 1, 2, 3];
        assert_eq!(render(&targets, &order), render(&targets, &order));
    }

    #[test]
    fn osc_score_forwards_node_events() {
        // The clicks free themselves; their /n_go and /n_end must reach the dispatcher's replies.
        let opts = Options {
            sample_rate: SR,
            output_channels: 1,
            ..Options::default()
        };
        let (mut controller, nrt, world) = engine(opts);
        controller.add_synthdef(click_def());
        let mut dispatcher = OscDispatcher::new();
        let mut render = Render::new(world, nrt, &opts);
        let blob = encode_score(&[click_bundle(time_for_sample(600), 1000)]);
        let (score, _) = parse_score(&blob).expect("parse");
        render_osc_score(
            &mut render,
            &mut dispatcher,
            &mut controller,
            &score,
            None,
            |_| {},
            RenderUntil::EndOfScore {
                tail: Duration::from_millis(20),
            },
        )
        .expect("render");
        let replies = dispatcher.take_replies();
        let addrs: Vec<&str> = replies
            .iter()
            .filter_map(|p| match p {
                OscPacket::Message(m) => Some(m.addr.as_str()),
                _ => None,
            })
            .collect();
        assert!(addrs.contains(&"/n_go"), "expected /n_go in {addrs:?}");
        assert!(addrs.contains(&"/n_end"), "expected /n_end in {addrs:?}");
    }

    #[test]
    fn clear_sched_cancels_scheduled_commands() {
        let opts = Options {
            sample_rate: SR,
            output_channels: 1,
            ..Options::default()
        };
        let (mut controller, nrt, world) = engine(opts);
        controller.add_synthdef(click_def());
        let mut dispatcher = OscDispatcher::new();
        let mut render = Render::new(world, nrt, &opts);

        // Schedule a click for sample 2000, plus a buffer alloc whose `Box` exercises the
        // trash-on-clear path...
        let scheduled = OscPacket::Bundle(OscBundle {
            timetag: unpack(time_for_sample(2000)),
            content: vec![
                OscPacket::Message(OscMessage {
                    addr: "/s_new".into(),
                    args: vec![
                        OscType::String("click".into()),
                        OscType::Int(1000),
                        OscType::Int(1),
                        OscType::Int(ROOT_GROUP_ID),
                    ],
                }),
                OscPacket::Message(OscMessage {
                    addr: "/b_alloc".into(),
                    args: vec![OscType::Int(0), OscType::Int(64), OscType::Int(1)],
                }),
            ],
        });
        dispatcher
            .apply(&mut controller, &scheduled)
            .expect("schedule");
        // ...then clear the scheduler before any of it is due.
        dispatcher
            .apply(
                &mut controller,
                &OscPacket::Message(OscMessage {
                    addr: "/clearSched".into(),
                    args: vec![],
                }),
            )
            .expect("/clearSched");

        let mut out = Vec::new();
        while render.block_start() <= time_for_sample(3000) {
            out.extend_from_slice(render.step(&[]));
        }
        render.finish();
        assert!(
            out.iter().all(|s| *s == 0.0),
            "a cleared scheduled command must never fire"
        );
    }
}
