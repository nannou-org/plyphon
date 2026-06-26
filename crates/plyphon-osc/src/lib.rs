//! A SuperCollider-compatible OSC front-end for the plyphon engine.
//!
//! [`OscDispatcher`] wraps a plyphon [`Controller`] and applies the OSC server commands a typical
//! SuperCollider client sends, translating them into [`Controller`] calls:
//!
//! - **SynthDefs:** `/d_recv`, `/d_free`, `/d_freeAll`.
//! - **Synths & nodes:** `/s_new`, `/s_noid`, `/n_set`, `/n_setn`, `/n_fill`, `/n_free`, `/n_run`,
//!   the control mappers `/n_map`/`/n_mapn`.
//! - **Groups & node tree:** `/g_new`, `/p_new`, `/g_head`/`/g_tail`/`/n_before`/`/n_after`/
//!   `/n_order`/`/g_freeAll`/`/g_deepFree`.
//! - **Control buses:** `/c_set`/`/c_setn`/`/c_fill`.
//! - **Buffers:** `/b_alloc`, `/b_allocRead`, `/b_read`, `/b_free`, `/b_zero`, `/b_query`, `/b_set`,
//!   `/b_setn`, `/b_fill`, `/b_setSampleRate`, `/b_gen`.
//! - **Server admin:** `/clearSched`, `/error`.
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
//! configure the dispatcher with a [`BufferSource`] via [`OscDispatcher::with_buffer_source`], and
//! `apply` *queues* the load; the host drives queued loads on its own executor with
//! [`OscDispatcher::run_pending`], which installs the buffer, runs the command's completion message,
//! and queues `/done`. (Sources are decoded the host's way - see `plyphon-buffers`.)

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[macro_use]
extern crate alloc;

mod bgen;
pub mod score;

pub use score::{ScoreEntry, ScoreError, ScoreReader, parse_score};

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;

use plyphon::controller::SynthNewError;
use plyphon::synthdef::read::ReadError;
use plyphon::{AddAction, CommandTime, Controller, Event, Render, RenderUntil, Reply};
use plyphon_buffers::{BufferSource, ReadRegion};
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

/// A queued asynchronous buffer load (`/b_allocRead`, `/b_read`), run by [`OscDispatcher::run_pending`].
struct PendingLoad {
    command: &'static str,
    bufnum: i32,
    key: String,
    region: ReadRegion,
    /// The raw OSC completion message to run once the load finishes, if any.
    completion: Option<Vec<u8>>,
    /// The client this load answers to; replayed in `run_pending` so `/done`/`/fail` and any reply the
    /// completion message emits all route back to it.
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

/// Applies SuperCollider OSC commands to a plyphon [`Controller`].
pub struct OscDispatcher {
    controller: Controller,
    /// Tracks the SynthDef each live node was created from, for control-name resolution.
    node_defs: HashMap<i32, String>,
    /// Control-side mirror of each buffer's dimensions, for `/b_query` and `/b_zero`.
    buffers: HashMap<i32, BufferInfo>,
    /// Source for asynchronous buffer loads, if configured.
    source: Option<Box<dyn BufferSource>>,
    /// Loads queued by `apply`, awaiting [`OscDispatcher::run_pending`].
    pending: Vec<PendingLoad>,
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
}

impl OscDispatcher {
    /// Wrap a controller (no buffer source: `/b_allocRead`/`/b_read` will fail with `/fail`).
    pub fn new(controller: Controller) -> Self {
        OscDispatcher {
            controller,
            node_defs: HashMap::new(),
            buffers: HashMap::new(),
            source: None,
            pending: Vec::new(),
            replies: Vec::new(),
            current_target: ReplyTarget::Broadcast,
            error_perm: true,
            error_bundle: None,
            pending_queries: VecDeque::new(),
            dump_sink: None,
        }
    }

    /// Wrap a controller with a [`BufferSource`] for asynchronous `/b_allocRead`/`/b_read` loads.
    pub fn with_buffer_source(controller: Controller, source: Box<dyn BufferSource>) -> Self {
        OscDispatcher {
            source: Some(source),
            ..Self::new(controller)
        }
    }

    /// Access the wrapped controller (e.g. to add SynthDefs or register custom units).
    pub fn controller(&mut self) -> &mut Controller {
        &mut self.controller
    }

    /// Unwrap the controller.
    pub fn into_controller(self) -> Controller {
        self.controller
    }

    /// Install a text sink for `/g_dumpTree` (scsynth prints the tree to stdout; plyphon is headless,
    /// so a host that wants the dump provides a sink). Unset by default - `/g_dumpTree` is then a
    /// no-op. `/g_queryTree` is unaffected (it always answers over OSC).
    pub fn set_dump_sink(&mut self, sink: DumpSink) {
        self.dump_sink = Some(sink);
    }

    /// Reassemble a query [`Reply`] (drained from [`Render::poll_reply`](plyphon::Render::poll_reply)/
    /// `Nrt::poll_reply`) into its OSC reply, queued for [`take_replies`](Self::take_replies). Feed
    /// every reply in order, alongside [`notify`](Self::notify); replies arrive in the same FIFO order
    /// the getters were issued, so each is matched against the oldest outstanding query.
    pub fn reply(&mut self, reply: Reply) {
        let Some((target, mut pending)) = self.pending_queries.pop_front() else {
            return; // a stray reply with nothing outstanding (e.g. after a reset); ignore.
        };
        // Replay the requester captured when the query was issued, so the reassembled message routes
        // back to it (the success `/n_info` overrides this to `Broadcast` itself).
        self.current_target = target;
        if !self.apply_reply(&mut pending, reply) {
            self.pending_queries.push_front((target, pending));
        }
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
        let (addr, id) = match event {
            Event::NodeStarted { id } => ("/n_go", id),
            Event::NodeEnded { id } => ("/n_end", id),
            Event::NodePaused { id } => ("/n_off", id),
            Event::NodeResumed { id } => ("/n_on", id),
            // A move carries the moved node's new tree position, like `/n_info`, not just the id.
            Event::NodeMoved {
                node,
                parent,
                prev,
                next,
                is_group,
                head,
                tail,
            } => {
                let args = node_info_args(node, parent, prev, next, is_group, head, tail);
                self.push_reply(ReplyTarget::Broadcast, "/n_move", args);
                return;
            }
            // No node was created (empty def slot or pool exhaustion); report a `/s_new` failure,
            // mirroring scsynth's `/fail` reply, and drop any def tracking for the would-be id.
            Event::SynthFailed { id } => {
                self.node_defs.remove(&id);
                self.push_reply(
                    ReplyTarget::Broadcast,
                    "/fail",
                    vec![OscType::String("/s_new".to_string()), OscType::Int(id)],
                );
                return;
            }
        };
        // A freed node's def tracking is no longer needed; self-freed synths never reach `/n_free`,
        // so this is where their entry is reclaimed.
        if let Event::NodeEnded { id } = event {
            self.node_defs.remove(&id);
        }
        self.push_reply(ReplyTarget::Broadcast, addr, vec![OscType::Int(id)]);
    }

    /// Run the buffer loads queued by `apply` (`/b_allocRead`, `/b_read`), in order.
    ///
    /// For each: load through the configured [`BufferSource`], install the buffer, run the command's
    /// completion message, and queue `/done` - or queue `/fail` if there is no source or the load
    /// errors. Drive this on whatever executor suits the host (a background thread natively,
    /// `spawn_local` on the web); it never touches the audio thread.
    pub async fn run_pending(&mut self) {
        for load in core::mem::take(&mut self.pending) {
            // Answer this load (its `/done`/`/fail`, and anything its completion message emits) back to
            // the client that issued it - exactly as scsynth stamps completion packets with the
            // command's stored reply address.
            self.current_target = load.target;
            let result = match &self.source {
                Some(source) => Some(source.load(&load.key, load.region).await),
                None => None,
            };
            match result {
                None => self.fail(load.command, "no buffer source configured"),
                Some(Err(err)) => self.fail(load.command, &err.to_string()),
                Some(Ok(data)) => {
                    let num_channels = data.num_channels.max(1);
                    let info = BufferInfo {
                        num_frames: data.samples.len() / num_channels,
                        num_channels,
                        sample_rate: data.sample_rate,
                    };
                    if self
                        .controller
                        .buffer_set(load.bufnum as usize, Box::new(data.into()))
                        .is_err()
                    {
                        self.fail(load.command, "command queue full");
                        continue;
                    }
                    self.buffers.insert(load.bufnum, info);
                    self.run_completion_bytes(load.completion.as_deref());
                    self.done(load.command, load.bufnum);
                }
            }
        }
    }

    /// Decode and apply a single OSC packet from raw bytes.
    pub fn apply_bytes(&mut self, data: &[u8]) -> Result<(), OscError> {
        let (_, packet) = rosc::decoder::decode_udp(data).map_err(OscError::Decode)?;
        self.apply(&packet)
    }

    /// Apply a decoded OSC packet: a message immediately, or every message in a bundle at the
    /// bundle's time tag.
    ///
    /// A future time tag schedules the bundle's messages (and any nested bundles) for that absolute
    /// OSC/NTP time; the engine maps the tag to a sample-exact block on the audio thread, against a
    /// drift-corrected clock. The "immediately" tags `0`/`1` (and any already-past time) apply now.
    pub fn apply(&mut self, packet: &OscPacket) -> Result<(), OscError> {
        match packet {
            OscPacket::Message(message) => self.message(message),
            OscPacket::Bundle(bundle) => {
                let prev = self
                    .controller
                    .begin_scheduled(bundle_command_time(bundle.timetag));
                // A bundle-local `/error -1|-2` override is scoped to this bundle (and its nested
                // bundles); save it here and restore on exit, exactly like the schedule window.
                let prev_error = self.error_bundle;
                let mut result = Ok(());
                for inner in &bundle.content {
                    result = self.apply(inner);
                    if result.is_err() {
                        break;
                    }
                }
                // Restore the enclosing window and error scope (Immediate / inherited at the top
                // level), even on error.
                self.error_bundle = prev_error;
                self.controller.begin_scheduled(prev);
                result
            }
        }
    }

    fn message(&mut self, message: &OscMessage) -> Result<(), OscError> {
        match message.addr.as_str() {
            "/d_recv" => self.d_recv(&message.args),
            "/d_free" => self.d_free(&message.args),
            "/d_freeAll" => self.d_free_all(),
            "/s_new" => self.s_new(&message.args),
            "/s_noid" => self.s_noid(&message.args),
            "/n_set" => self.n_set(&message.args),
            "/n_setn" => self.n_setn(&message.args),
            "/n_fill" => self.n_fill(&message.args),
            "/n_free" => self.n_free(&message.args),
            "/n_run" => self.n_run(&message.args),
            "/g_new" => self.g_new(&message.args),
            // scsynth emulates parallel groups with ordinary groups; same triple layout as `/g_new`.
            "/p_new" => self.g_new(&message.args),
            "/c_set" => self.c_set(&message.args),
            "/c_setn" => self.c_setn(&message.args),
            "/c_fill" => self.c_fill(&message.args),
            "/n_map" => self.n_map(&message.args),
            "/n_mapn" => self.n_mapn(&message.args),
            "/b_alloc" => self.b_alloc(&message.args),
            "/b_free" => self.b_free(&message.args),
            "/b_zero" => self.b_zero(&message.args),
            "/b_query" => self.b_query(&message.args),
            "/b_set" => self.b_set(&message.args),
            "/b_setn" => self.b_setn(&message.args),
            "/b_fill" => self.b_fill(&message.args),
            "/b_setSampleRate" => self.b_set_sample_rate(&message.args),
            "/b_gen" => self.b_gen(&message.args),
            "/b_allocRead" => self.b_alloc_read(&message.args),
            "/b_read" => self.b_read(&message.args),
            "/g_head" => self.group_moves(&message.args, AddAction::Head),
            "/g_tail" => self.group_moves(&message.args, AddAction::Tail),
            "/n_before" => self.node_moves(&message.args, AddAction::Before),
            "/n_after" => self.node_moves(&message.args, AddAction::After),
            "/n_order" => self.n_order(&message.args),
            "/g_freeAll" => self.g_free_all(&message.args),
            "/g_deepFree" => self.g_deep_free(&message.args),
            "/clearSched" => self.clear_sched(),
            "/error" => self.error_cmd(&message.args),
            "/sync" => self.sync(&message.args),
            "/status" => self.status(),
            "/rtMemoryStatus" => self.rt_memory_status(),
            "/n_query" => self.n_query(&message.args),
            "/c_get" => self.c_get(&message.args),
            "/c_getn" => self.c_getn(&message.args),
            "/s_get" => self.s_get(&message.args),
            "/s_getn" => self.s_getn(&message.args),
            "/b_get" => self.b_get(&message.args),
            "/b_getn" => self.b_getn(&message.args),
            "/g_queryTree" => self.g_query_tree(&message.args, false),
            "/g_dumpTree" => self.g_query_tree(&message.args, true),
            other => Err(OscError::UnsupportedCommand(other.to_string())),
        }
    }

    fn d_recv(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let blob = match args.first() {
            Some(OscType::Blob(bytes)) => bytes,
            _ => return Err(OscError::BadArguments("d_recv expects a blob")),
        };
        let defs = plyphon::synthdef::read::parse(blob).map_err(OscError::SynthDef)?;
        for def in defs {
            self.controller.add_synthdef(def);
        }
        Ok(())
    }

    /// `/d_free <name>...`: free each named synth definition (a later `/s_new` of it then fails until
    /// it is re-sent).
    fn d_free(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let name = str_arg(arg)?;
            self.controller
                .free_def(name)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/d_freeAll`: free every registered synth definition.
    fn d_free_all(&mut self) -> Result<(), OscError> {
        self.controller
            .free_all_defs()
            .map_err(|_| OscError::QueueFull)
    }

    fn s_new(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            self.controller
                .synth_new(&name, target, action)
                .map_err(OscError::SynthNew)?
        } else {
            self.controller
                .synth_new_with_id(id, &name, target, action)
                .map_err(OscError::SynthNew)?;
            id
        };
        self.node_defs.insert(id, name.clone());
        self.apply_controls(id, Some(&name), &args[4..])
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

    fn n_set(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_set expects a node"))?,
        )?;
        self.apply_controls(node, None, &args[1..])
    }

    /// `/n_setn nodeID (control, count, value...)...`: set contiguous ranges of a node's controls.
    fn n_setn(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_setn expects a node"))?,
        )?;
        let rest = &args[1..];
        let mut i = 0;
        while i < rest.len() {
            let start = self.control_index(node, &rest[i])?;
            let count = count_arg(rest.get(i + 1))?;
            i += 2;
            if i + count > rest.len() {
                return Err(OscError::BadArguments(
                    "n_setn value count exceeds arguments",
                ));
            }
            for (j, arg) in rest[i..i + count].iter().enumerate() {
                self.controller
                    .set_control(node, start + j, float_arg(arg)?)
                    .map_err(|_| OscError::QueueFull)?;
            }
            i += count;
        }
        Ok(())
    }

    /// `/n_fill nodeID (control, count, value)...`: fill contiguous ranges of a node's controls.
    fn n_fill(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            let start = self.control_index(node, &triple[0])?;
            let count = count_arg(Some(&triple[1]))?;
            let value = float_arg(&triple[2])?;
            for j in 0..count {
                self.controller
                    .set_control(node, start + j, value)
                    .map_err(|_| OscError::QueueFull)?;
            }
        }
        Ok(())
    }

    fn n_free(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let node = int_arg(arg)?;
            self.controller
                .free(node)
                .map_err(|_| OscError::QueueFull)?;
            self.node_defs.remove(&node);
        }
        Ok(())
    }

    /// `/n_run (nodeID, flag)...`: pause (flag 0) or resume (flag 1) each node.
    fn n_run(&mut self, args: &[OscType]) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("n_run expects node/flag pairs"));
        }
        for pair in args.chunks_exact(2) {
            let node = int_arg(&pair[0])?;
            let run = int_arg(&pair[1])? != 0;
            self.controller
                .node_run(node, run)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn g_new(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
                self.controller
                    .new_group(target, action)
                    .map_err(|_| OscError::QueueFull)?;
            } else {
                self.controller
                    .new_group_with_id(id, target, action)
                    .map_err(|_| OscError::QueueFull)?;
            }
        }
        Ok(())
    }

    /// `/g_head`/`/g_tail`: `(group, node)` pairs - move each node to the group's head/tail.
    fn group_moves(&mut self, args: &[OscType], action: AddAction) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("expects group/node pairs"));
        }
        for pair in args.chunks_exact(2) {
            let group = int_arg(&pair[0])?;
            let node = int_arg(&pair[1])?;
            self.controller
                .move_node(node, group, action)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/n_before`/`/n_after`: `(node, target)` pairs - move each node before/after its target.
    fn node_moves(&mut self, args: &[OscType], action: AddAction) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("expects node/target pairs"));
        }
        for pair in args.chunks_exact(2) {
            let node = int_arg(&pair[0])?;
            let target = int_arg(&pair[1])?;
            self.controller
                .move_node(node, target, action)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/n_order addAction target node...`: place the nodes consecutively, in order, at the location.
    fn n_order(&mut self, args: &[OscType]) -> Result<(), OscError> {
        if args.len() < 3 {
            return Err(OscError::BadArguments(
                "n_order expects addAction, target, nodes",
            ));
        }
        let mut anchor = int_arg(&args[1])?;
        let mut action = add_action(int_arg(&args[0])?)?;
        for arg in &args[2..] {
            let node = int_arg(arg)?;
            self.controller
                .move_node(node, anchor, action)
                .map_err(|_| OscError::QueueFull)?;
            // Subsequent nodes follow the previous one, preserving the given order.
            anchor = node;
            action = AddAction::After;
        }
        Ok(())
    }

    /// `/g_freeAll group...`: empty each group, keeping the group.
    fn g_free_all(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let group = int_arg(arg)?;
            self.controller
                .free_all(group)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/g_deepFree group...`: free each group's synths recursively, keeping the groups.
    fn g_deep_free(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let group = int_arg(arg)?;
            self.controller
                .deep_free(group)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/clearSched`: clear the engine scheduler's pending time-tagged commands.
    fn clear_sched(&mut self) -> Result<(), OscError> {
        self.controller
            .clear_sched()
            .map_err(|_| OscError::QueueFull)
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
    fn sync(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let id = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("sync expects an id"))?,
        )?;
        self.controller
            .query_sync(id)
            .map_err(|_| OscError::QueueFull)?;
        self.push_query(PendingQuery::Sync);
        Ok(())
    }

    /// `/status` -> `/status.reply`.
    fn status(&mut self) -> Result<(), OscError> {
        self.controller
            .query_status()
            .map_err(|_| OscError::QueueFull)?;
        self.push_query(PendingQuery::Status);
        Ok(())
    }

    /// `/rtMemoryStatus` -> `/rtMemoryStatus.reply`.
    fn rt_memory_status(&mut self) -> Result<(), OscError> {
        self.controller
            .query_rt_memory()
            .map_err(|_| OscError::QueueFull)?;
        self.push_query(PendingQuery::RtMemory);
        Ok(())
    }

    /// `/n_query <node>...` -> one `/n_info` per node.
    fn n_query(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let node = int_arg(arg)?;
            self.controller
                .query_node(node)
                .map_err(|_| OscError::QueueFull)?;
            self.push_query(PendingQuery::Node);
        }
        Ok(())
    }

    /// `/c_get <bus>...` -> one `/c_set`.
    fn c_get(&mut self, args: &[OscType]) -> Result<(), OscError> {
        for arg in args {
            let bus = bus_index(arg)?;
            self.controller
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
    fn c_getn(&mut self, args: &[OscType]) -> Result<(), OscError> {
        if !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("c_getn expects start/count pairs"));
        }
        for pair in args.chunks_exact(2) {
            let start = bus_index(&pair[0])?;
            let count = count_arg(Some(&pair[1]))? as u32;
            self.controller
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
    fn s_get(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("s_get expects a node"))?,
        )?;
        let rest = &args[1..];
        let mut controls = VecDeque::with_capacity(rest.len());
        for arg in rest {
            let control = self.control_index(node, arg)?;
            self.controller
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
    fn s_getn(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            let control = self.control_index(node, &pair[0])?;
            let count = count_arg(Some(&pair[1]))?;
            self.controller
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
    fn b_get(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let buf = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_get expects a bufnum"))?,
        )?;
        let rest = &args[1..];
        for arg in rest {
            let index = index_arg(arg)?;
            self.controller
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
    fn b_getn(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            self.controller
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
    fn g_query_tree(&mut self, args: &[OscType], dump: bool) -> Result<(), OscError> {
        let group = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("queryTree expects a group"))?,
        )?;
        let flag = matches!(args.get(1), Some(OscType::Int(f)) if *f != 0);
        if dump {
            self.controller
                .dump_tree(group, flag)
                .map_err(|_| OscError::QueueFull)?;
        } else {
            self.controller
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
    fn apply_reply(&mut self, pending: &mut PendingQuery, reply: Reply) -> bool {
        match pending {
            PendingQuery::Sync => {
                if let Reply::Synced { id } = reply {
                    self.reply_msg("/synced", vec![OscType::Int(id)]);
                }
                true
            }
            PendingQuery::Status => {
                if let Reply::Status {
                    num_ugens,
                    num_synths,
                    num_groups,
                    num_synthdefs,
                    avg_cpu,
                    peak_cpu,
                    nominal_sr,
                    actual_sr,
                } = reply
                {
                    self.reply_msg(
                        "/status.reply",
                        vec![
                            OscType::Int(1),
                            OscType::Int(num_ugens),
                            OscType::Int(num_synths),
                            OscType::Int(num_groups),
                            OscType::Int(num_synthdefs),
                            OscType::Float(avg_cpu),
                            OscType::Float(peak_cpu),
                            OscType::Double(nominal_sr),
                            OscType::Double(actual_sr),
                        ],
                    );
                }
                true
            }
            PendingQuery::RtMemory => {
                if let Reply::RtMemoryStatus {
                    total_free,
                    largest_free,
                } = reply
                {
                    self.reply_msg(
                        "/rtMemoryStatus.reply",
                        vec![OscType::Int(total_free), OscType::Int(largest_free)],
                    );
                }
                true
            }
            PendingQuery::Node => {
                match reply {
                    Reply::NodeInfo {
                        node,
                        parent,
                        prev,
                        next,
                        is_group,
                        head,
                        tail,
                    } => {
                        let args = node_info_args(node, parent, prev, next, is_group, head, tail);
                        // scsynth answers `/n_query` by broadcasting `/n_info` to all registered
                        // clients (Server-Command-Reference: "sent to all registered clients"), not
                        // just the asker - so an unregistered querier receives nothing.
                        self.push_reply(ReplyTarget::Broadcast, "/n_info", args);
                    }
                    // scsynth returns kSCErr_NodeNotFound, which the dispatcher reports as a `/fail`
                    // back to the requester (errors are not broadcast).
                    Reply::NodeNotFound { node } => {
                        self.fail("/n_query", &alloc::format!("Node {node} not found"))
                    }
                    _ => {}
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
                        .and_then(|d| self.controller.synthdef(d))
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

    fn c_set(&mut self, args: &[OscType]) -> Result<(), OscError> {
        if args.is_empty() || !args.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("c_set expects bus/value pairs"));
        }
        for pair in args.chunks_exact(2) {
            let bus = bus_index(&pair[0])?;
            let value = float_arg(&pair[1])?;
            self.controller
                .set_control_bus(bus, value)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn c_setn(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
                self.controller
                    .set_control_bus(start + j as u32, float_arg(arg)?)
                    .map_err(|_| OscError::QueueFull)?;
            }
            i += count;
        }
        Ok(())
    }

    /// `/c_fill (bus, count, value)...`: set each contiguous range of control buses to `value`.
    fn c_fill(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
                self.controller
                    .set_control_bus(start + j as u32, value)
                    .map_err(|_| OscError::QueueFull)?;
            }
        }
        Ok(())
    }

    fn n_map(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_map expects a node"))?,
        )?;
        let rest = &args[1..];
        if !rest.len().is_multiple_of(2) {
            return Err(OscError::BadArguments("n_map expects control/bus pairs"));
        }
        for pair in rest.chunks_exact(2) {
            let control = self.control_index(node, &pair[0])?;
            let bus = map_bus(&pair[1])?;
            self.controller
                .map_control(node, control, bus)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn n_mapn(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            let control = self.control_index(node, &triple[0])?;
            let bus = map_bus(&triple[1])?;
            let count = count_arg(Some(&triple[2]))?;
            self.controller
                .map_control_n(node, control, bus, count)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    fn b_alloc(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_alloc expects a bufnum"))?,
        )?;
        let num_frames = count_arg(args.get(1))?;
        let num_channels = match args.get(2) {
            Some(OscType::Int(c)) => (*c).max(1) as usize,
            _ => 1,
        };
        let sample_rate = self.controller.sample_rate();
        self.controller
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
        self.run_completion_bytes(last_blob(args));
        self.done("/b_alloc", bufnum);
        Ok(())
    }

    fn b_free(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_free expects a bufnum"))?,
        )?;
        self.controller
            .buffer_free(bufnum as usize)
            .map_err(|_| OscError::QueueFull)?;
        self.buffers.remove(&bufnum);
        self.run_completion_bytes(last_blob(args));
        self.done("/b_free", bufnum);
        Ok(())
    }

    fn b_zero(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_zero expects a bufnum"))?,
        )?;
        // Zero by re-allocating the same dimensions (the new buffer is zeroed); the old one is
        // dropped off the audio thread, the same as `/b_alloc`.
        match self.buffers.get(&bufnum).copied() {
            Some(info) => {
                self.controller
                    .buffer_alloc(
                        bufnum as usize,
                        info.num_frames,
                        info.num_channels,
                        info.sample_rate,
                    )
                    .map_err(|_| OscError::QueueFull)?;
                self.run_completion_bytes(last_blob(args));
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
    fn b_set(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            self.controller
                .buffer_set_sample(bufnum as usize, sample, value)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/b_setn bufID (start, count, value...)...`: overwrite contiguous ranges of buffer samples.
    fn b_setn(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
                self.controller
                    .buffer_set_sample(bufnum as usize, start + j, float_arg(arg)?)
                    .map_err(|_| OscError::QueueFull)?;
            }
            i += count;
        }
        Ok(())
    }

    /// `/b_fill bufID (start, count, value)...`: fill contiguous ranges of buffer samples.
    fn b_fill(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            self.controller
                .buffer_fill(bufnum as usize, start, count, value)
                .map_err(|_| OscError::QueueFull)?;
        }
        Ok(())
    }

    /// `/b_setSampleRate bufID rate`: overwrite a buffer's sample-rate metadata.
    fn b_set_sample_rate(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let bufnum = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("b_setSampleRate expects a bufnum"))?,
        )?;
        let rate = float_arg(
            args.get(1)
                .ok_or(OscError::BadArguments("b_setSampleRate expects a rate"))?,
        )? as f64;
        self.controller
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
    fn b_gen(&mut self, args: &[OscType]) -> Result<(), OscError> {
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
            return self.b_gen_copy(bufnum, gen_args, completion);
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
        self.controller
            .buffer_set(bufnum as usize, buffer)
            .map_err(|_| OscError::QueueFull)?;
        self.run_completion_bytes(completion);
        self.done("/b_gen", bufnum);
        Ok(())
    }

    /// `/b_gen <buf> "copy" <dstStart> <srcBufID> <srcStart> <numSamples>`: copy a region from a live
    /// source buffer into the destination, on the audio thread.
    fn b_gen_copy(
        &mut self,
        bufnum: i32,
        gen_args: &[OscType],
        completion: Option<&[u8]>,
    ) -> Result<(), OscError> {
        let bad = || OscError::BadArguments("b_gen copy expects dstStart, srcBuf, srcStart, count");
        let dst_start = index_arg(gen_args.first().ok_or_else(bad)?)?;
        let src = int_arg(gen_args.get(1).ok_or_else(bad)?)? as usize;
        let src_start = index_arg(gen_args.get(2).ok_or_else(bad)?)?;
        let count = index_arg(gen_args.get(3).ok_or_else(bad)?)?;
        self.controller
            .buffer_copy_region(bufnum as usize, dst_start, src, src_start, count)
            .map_err(|_| OscError::QueueFull)?;
        self.run_completion_bytes(completion);
        self.done("/b_gen", bufnum);
        Ok(())
    }

    fn b_alloc_read(&mut self, args: &[OscType]) -> Result<(), OscError> {
        self.queue_load("/b_allocRead", args)
    }

    fn b_read(&mut self, args: &[OscType]) -> Result<(), OscError> {
        // Simplified: reads the file region and replaces the buffer (the `bufStartFrame`/`leaveOpen`
        // arguments are ignored for now).
        self.queue_load("/b_read", args)
    }

    /// Queue an asynchronous load of `path` into `bufnum`, run later by [`Self::run_pending`].
    fn queue_load(&mut self, command: &'static str, args: &[OscType]) -> Result<(), OscError> {
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
            completion: last_blob(args).map(|bytes| bytes.to_vec()),
            target: self.current_target,
        });
        Ok(())
    }

    /// Apply an embedded OSC completion message (the trailing blob of an async command), if present.
    fn run_completion_bytes(&mut self, bytes: Option<&[u8]>) {
        if let Some(bytes) = bytes
            && let Ok((_, packet)) = rosc::decoder::decode_udp(bytes)
        {
            let _ = self.apply(&packet);
        }
    }

    /// Queue an OSC reply for the current requester (see [`current_target`](Self::current_target)).
    fn reply_msg(&mut self, addr: &str, args: Vec<OscType>) {
        let target = self.current_target;
        self.push_reply(target, addr, args);
    }

    /// Queue an OSC reply for an explicit destination, regardless of the current requester (e.g. node
    /// notifications and the success `/n_info`, which broadcast).
    fn push_reply(&mut self, target: ReplyTarget, addr: &str, args: Vec<OscType>) {
        self.replies.push((
            target,
            OscPacket::Message(OscMessage {
                addr: addr.to_string(),
                args,
            }),
        ));
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
    fn control_index(&self, node: i32, arg: &OscType) -> Result<usize, OscError> {
        match arg {
            OscType::Int(idx) => {
                usize::try_from(*idx).map_err(|_| OscError::BadArguments("negative control index"))
            }
            OscType::String(name) => self.resolve_param(node, None, name),
            _ => Err(OscError::BadArguments("control must be an int or string")),
        }
    }

    /// Apply `(control, value)` argument pairs to `node`. A control is an `int` index or a `string`
    /// name resolved against the node's SynthDef (`def_name` when known, else the tracked one).
    fn apply_controls(
        &mut self,
        node: i32,
        def_name: Option<&str>,
        args: &[OscType],
    ) -> Result<(), OscError> {
        let mut i = 0;
        while i + 1 < args.len() {
            let index = match &args[i] {
                OscType::Int(idx) => usize::try_from(*idx)
                    .map_err(|_| OscError::BadArguments("negative control index"))?,
                OscType::String(name) => self.resolve_param(node, def_name, name)?,
                _ => return Err(OscError::BadArguments("control must be an int or string")),
            };
            let value = float_arg(&args[i + 1])?;
            self.controller
                .set_control(node, index, value)
                .map_err(|_| OscError::QueueFull)?;
            i += 2;
        }
        Ok(())
    }

    fn resolve_param(
        &self,
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
        let def = self
            .controller
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
            dispatcher.apply(&score[next].packet)?;
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
            dispatcher.reply(reply);
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

/// Build the arguments shared by `/n_info` (the `/n_query` answer) and `/n_move` (the node-move
/// notification): node, parent, prev, next, isGroup, plus head/tail when the node is a group.
fn node_info_args(
    node: i32,
    parent: i32,
    prev: i32,
    next: i32,
    is_group: i32,
    head: i32,
    tail: i32,
) -> Vec<OscType> {
    let mut args = vec![
        OscType::Int(node),
        OscType::Int(parent),
        OscType::Int(prev),
        OscType::Int(next),
        OscType::Int(is_group),
    ];
    if is_group == 1 {
        args.push(OscType::Int(head));
        args.push(OscType::Int(tail));
    }
    args
}

fn int_arg(arg: &OscType) -> Result<i32, OscError> {
    match arg {
        OscType::Int(i) => Ok(*i),
        _ => Err(OscError::BadArguments("expected an int")),
    }
}

/// A non-negative bus index (`/c_set`, `/c_setn`).
fn bus_index(arg: &OscType) -> Result<u32, OscError> {
    u32::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative bus index"))
}

/// A bus index for `/n_map`/`/n_mapn`, where a negative index means "unmap".
fn map_bus(arg: &OscType) -> Result<Option<u32>, OscError> {
    let bus = int_arg(arg)?;
    Ok(u32::try_from(bus).ok())
}

/// A non-negative count argument (`/c_setn`, `/n_mapn`).
fn count_arg(arg: Option<&OscType>) -> Result<usize, OscError> {
    let arg = arg.ok_or(OscError::BadArguments("expected a count"))?;
    usize::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative count"))
}

/// A non-negative `usize` index argument (`/b_set`, `/b_setn`, `/b_fill`).
fn index_arg(arg: &OscType) -> Result<usize, OscError> {
    usize::try_from(int_arg(arg)?).map_err(|_| OscError::BadArguments("negative index"))
}

fn float_arg(arg: &OscType) -> Result<f32, OscError> {
    match arg {
        OscType::Float(f) => Ok(*f),
        OscType::Int(i) => Ok(*i as f32),
        OscType::Double(d) => Ok(*d as f32),
        _ => Err(OscError::BadArguments("expected a number")),
    }
}

fn str_arg(arg: &OscType) -> Result<&str, OscError> {
    match arg {
        OscType::String(s) => Ok(s.as_str()),
        _ => Err(OscError::BadArguments("expected a string")),
    }
}

/// The trailing OSC completion blob of an async command, if the last argument is one.
fn last_blob(args: &[OscType]) -> Option<&[u8]> {
    match args.last() {
        Some(OscType::Blob(bytes)) => Some(bytes),
        _ => None,
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
        let mut dispatcher = OscDispatcher::new(controller);
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
        let mut dispatcher = OscDispatcher::new(controller);
        let mut render = Render::new(world, nrt, &opts);
        let blob = encode_score(&[click_bundle(time_for_sample(600), 1000)]);
        let (score, _) = parse_score(&blob).expect("parse");
        render_osc_score(
            &mut render,
            &mut dispatcher,
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
        let mut dispatcher = OscDispatcher::new(controller);
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
        dispatcher.apply(&scheduled).expect("schedule");
        // ...then clear the scheduler before any of it is due.
        dispatcher
            .apply(&OscPacket::Message(OscMessage {
                addr: "/clearSched".into(),
                args: vec![],
            }))
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
