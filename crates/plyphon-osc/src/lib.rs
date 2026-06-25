//! A SuperCollider-compatible OSC front-end for the plyphon engine.
//!
//! [`OscDispatcher`] wraps a plyphon [`Controller`] and applies the OSC server commands a typical
//! SuperCollider client sends - `/d_recv`, `/s_new`, `/n_set`, `/n_free`, `/g_new`, the node-tree ops
//! `/g_head`/`/g_tail`/`/n_before`/`/n_after`/`/n_order`/`/g_freeAll`/`/g_deepFree`, the control-bus
//! setters `/c_set`/`/c_setn`, the control mappers `/n_map`/`/n_mapn`, and the buffer commands
//! `/b_alloc`, `/b_allocRead`, `/b_read`, `/b_free`, `/b_zero`, and `/b_query` - translating them
//! into [`Controller`] calls. OSC handling is strictly control-side; the audio thread is never
//! involved. `/s_new`, `/n_set`, and `/n_map` accept a string control name, resolved against the
//! node's SynthDef, so the dispatcher tracks which definition each node was created from.
//!
//! # Replies and notifications
//!
//! Commands that report back - `/b_query` (`/b_info`), the asynchronous buffer loads (`/done`), and
//! failures (`/fail`) - queue OSC packets the transport drains with [`OscDispatcher::take_replies`].
//! Node lifecycle is reported the same way: feed the engine [`Event`]s drained from the
//! [`Nrt`](plyphon::Nrt) to [`OscDispatcher::notify`], which queues the matching `/n_go`/`/n_end`/
//! `/n_off`/`/n_on` reply - so a self-freeing synth's `/n_end` reaches the client over OSC too.
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

pub mod score;

pub use score::{ScoreEntry, ScoreError, ScoreReader, parse_score};

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use hashbrown::HashMap;

use plyphon::controller::SynthNewError;
use plyphon::synthdef::read::ReadError;
use plyphon::{AddAction, CommandTime, Controller, Event, Render, RenderUntil};
use plyphon_buffers::{BufferSource, ReadRegion};
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
    /// Outbound replies, drained by [`OscDispatcher::take_replies`].
    replies: Vec<OscPacket>,
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

    /// Take the OSC replies queued since the last call (`/done`, `/b_info`, `/fail`, and the
    /// `/n_*` node notifications) for the transport to send back to the client.
    pub fn take_replies(&mut self) -> Vec<OscPacket> {
        core::mem::take(&mut self.replies)
    }

    /// Translate an engine [`Event`] into the matching SuperCollider node-notification reply and
    /// queue it for [`take_replies`](Self::take_replies): `/n_go` (started), `/n_end` (freed),
    /// `/n_off` (paused), `/n_on` (resumed).
    ///
    /// Feed this the events drained from the [`Nrt`](plyphon::Nrt), so node lifecycle - including
    /// synths that free themselves via a done action - is reported back over OSC alongside the
    /// command replies. plyphon's events carry only the node id, so, unlike scsynth, the
    /// parent/sibling/group fields are omitted and the id is the lone argument.
    pub fn notify(&mut self, event: Event) {
        let (addr, id) = match event {
            Event::NodeStarted { id } => ("/n_go", id),
            Event::NodeEnded { id } => ("/n_end", id),
            Event::NodePaused { id } => ("/n_off", id),
            Event::NodeResumed { id } => ("/n_on", id),
            // No node was created (empty def slot or pool exhaustion); report a `/s_new` failure,
            // mirroring scsynth's `/fail` reply, and drop any def tracking for the would-be id.
            Event::SynthFailed { id } => {
                self.node_defs.remove(&id);
                self.reply_msg(
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
        self.reply_msg(addr, vec![OscType::Int(id)]);
    }

    /// Run the buffer loads queued by `apply` (`/b_allocRead`, `/b_read`), in order.
    ///
    /// For each: load through the configured [`BufferSource`], install the buffer, run the command's
    /// completion message, and queue `/done` - or queue `/fail` if there is no source or the load
    /// errors. Drive this on whatever executor suits the host (a background thread natively,
    /// `spawn_local` on the web); it never touches the audio thread.
    pub async fn run_pending(&mut self) {
        for load in core::mem::take(&mut self.pending) {
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
                let mut result = Ok(());
                for inner in &bundle.content {
                    result = self.apply(inner);
                    if result.is_err() {
                        break;
                    }
                }
                // Restore the enclosing window (Immediate at the top level), even on error.
                self.controller.begin_scheduled(prev);
                result
            }
        }
    }

    fn message(&mut self, message: &OscMessage) -> Result<(), OscError> {
        match message.addr.as_str() {
            "/d_recv" => self.d_recv(&message.args),
            "/s_new" => self.s_new(&message.args),
            "/n_set" => self.n_set(&message.args),
            "/n_free" => self.n_free(&message.args),
            "/g_new" => self.g_new(&message.args),
            "/c_set" => self.c_set(&message.args),
            "/c_setn" => self.c_setn(&message.args),
            "/n_map" => self.n_map(&message.args),
            "/n_mapn" => self.n_mapn(&message.args),
            "/b_alloc" => self.b_alloc(&message.args),
            "/b_free" => self.b_free(&message.args),
            "/b_zero" => self.b_zero(&message.args),
            "/b_query" => self.b_query(&message.args),
            "/b_allocRead" => self.b_alloc_read(&message.args),
            "/b_read" => self.b_read(&message.args),
            "/g_head" => self.group_moves(&message.args, AddAction::Head),
            "/g_tail" => self.group_moves(&message.args, AddAction::Tail),
            "/n_before" => self.node_moves(&message.args, AddAction::Before),
            "/n_after" => self.node_moves(&message.args, AddAction::After),
            "/n_order" => self.n_order(&message.args),
            "/g_freeAll" => self.g_free_all(&message.args),
            "/g_deepFree" => self.g_deep_free(&message.args),
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

    fn n_set(&mut self, args: &[OscType]) -> Result<(), OscError> {
        let node = int_arg(
            args.first()
                .ok_or(OscError::BadArguments("n_set expects a node"))?,
        )?;
        self.apply_controls(node, None, &args[1..])
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

    /// Queue an OSC reply for the transport to send back to the client.
    fn reply_msg(&mut self, addr: &str, args: Vec<OscType>) {
        self.replies.push(OscPacket::Message(OscMessage {
            addr: addr.to_string(),
            args,
        }));
    }

    /// Queue a `/done <command> <bufnum>` reply.
    fn done(&mut self, command: &str, bufnum: i32) {
        self.reply_msg(
            "/done",
            vec![OscType::String(command.to_string()), OscType::Int(bufnum)],
        );
    }

    /// Queue a `/fail <command> <error>` reply.
    fn fail(&mut self, command: &str, error: &str) {
        self.reply_msg(
            "/fail",
            vec![
                OscType::String(command.to_string()),
                OscType::String(error.to_string()),
            ],
        );
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
}
