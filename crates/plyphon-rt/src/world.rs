//! The real-time side of the engine - plyphon's port of scsynth's `World`/`World_Run`.
//!
//! `World` owns the rt-pool, the resident def table, the buses, the node tree, and the World-shared
//! wire/output scratch. The host's audio callback drives it via [`World::fill`], which reblocks the
//! engine's fixed control-block size to the host's arbitrary buffer size. Every per-block step is
//! O(1) link manipulation or a bounded loop over pre-allocated buffers; the only audio-thread
//! allocator is the rt-pool, used to build and free a synth's per-instance state block.
//!
//! Synths are constructed *here*, on the audio thread, from a resident [`GraphDef`] (scsynth's
//! `Graph_New`): one pool allocation, a few `memcpy`s, then linked into the tree. Freeing a synth
//! returns its block to the pool (a cheap free-list op) - no trash. Buffers and streams still flow to
//! the trash ring (drained by the [`Nrt`](crate::nrt::Nrt)) to drop off the audio thread, and node
//! notifications go to the events ring. Done actions are applied here after the tree runs.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;

use bytemuck::cast_slice_mut;
use rtrb::{Consumer, Producer, PushError};

use crate::command::{Command, CommandTime, Event, Reply, TimedCommand, Trash};
use crate::graph::{Block, Graph, Pool};
use crate::options::Options;
use crate::sched::{Clock, Scheduler};
use crate::tree::{AddAction, FreedNode, NodeTree};
use plyphon_dsp::buffer::{BufferSlot, BufferTable};
use plyphon_dsp::bus::Buses;
use plyphon_dsp::rate::RateInfo;
use plyphon_dsp::wavetable::Wavetables;
use plyphon_unit::graphdef::GraphDef;
use plyphon_unit::unit::{DoneAction, Trigger};

/// The seed the per-instance RNG counter starts from (any fixed non-zero value; keeps runs
/// deterministic while decorrelating distinct synth instances).
const SEED_INIT: u64 = 0x853c_49e6_748f_ea9b;

/// The golden-ratio odd constant used to spread per-instance and per-unit seeds.
const SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;

/// The most values a single range getter (`/c_getn`/`/s_getn`/`/b_getn`) answers, so one oversized
/// request cannot overflow the reply ring. scsynth similarly bounds its reply sizes.
pub const MAX_QUERY_RANGE: usize = 256;

/// The real-time engine half.
pub struct World {
    audio: RateInfo,
    control: RateInfo,
    wavetables: Wavetables,
    buses: Buses,
    buffers: BufferTable,
    tree: NodeTree,
    /// The rt-pool backing every synth's per-instance state block (scsynth's `mAllocPool`).
    pool: Pool,
    /// Resident compiled defs, indexed by `def_id` (scsynth's `gGraphDefLib`).
    def_table: Vec<Option<Arc<GraphDef>>>,
    /// World-shared audio wire scratch, reused per graph (`max_wire_bufs * block_size` f32).
    wire_scratch: Box<[f32]>,
    /// World-shared per-unit output scratch, reused per unit (`max_unit_outputs * block_size` f32).
    unit_scratch: Box<[f32]>,
    /// Per-instance RNG seed counter, advanced for each synth built.
    next_seed: u64,
    rx: Consumer<TimedCommand>,
    trash_tx: Producer<Trash>,
    events_tx: Producer<Event>,
    replies_tx: Producer<Reply>,
    triggers_tx: Producer<Trigger>,
    /// Freed items awaiting space in the trash ring (pre-allocated; never reallocates at runtime).
    pending_trash: Vec<Trash>,
    /// Events awaiting space in the events ring (pre-allocated; never reallocates at runtime).
    pending_events: Vec<Event>,
    /// Query answers awaiting space in the reply ring (pre-allocated; never reallocates at runtime).
    /// A `VecDeque` so the FIFO order getters rely on survives a ring-full backlog.
    pending_replies: VecDeque<Reply>,
    /// Scratch the `/g_queryTree` walk fills before draining into the reply ring (pre-allocated).
    tree_scratch: Vec<Reply>,
    /// Scratch list of `(slot index, action)` for nodes whose units requested a done action.
    done_nodes: Vec<(u32, DoneAction)>,
    /// Scratch sink for nodes removed by a free, so freeing a whole group allocates nothing.
    freed_nodes: Vec<FreedNode>,
    /// Scratch sink for node ids paused by a done action, drained into `/n_off` notifications.
    paused_nodes: Vec<i32>,
    /// Per-block sink for `SendTrig` triggers (pre-allocated to `trigger_cap`; never reallocates),
    /// drained into the trigger ring after the tree walk.
    trigger_buf: Vec<Trigger>,
    /// Cap on triggers per block; the sink and the trigger ring share it. Excess is dropped.
    trigger_cap: usize,
    /// The drift-correcting OSC/NTP clock (scsynth's `mOSCbuftime` + DLL).
    clock: Clock,
    /// Pending time-tagged commands awaiting their block (scsynth's `mScheduler`).
    scheduler: Scheduler,
    /// The within-block sample offset of the scheduled command currently being applied (scsynth's
    /// `mSampleOffset`); 0 for immediate commands. Recorded onto a synth created mid-block so its
    /// `OffsetOut` onsets sample-exactly.
    current_sample_offset: usize,
    buf_counter: u64,
    block_size: usize,
    /// How many frames of the current control block have already been emitted to the host.
    block_frames_emitted: usize,
}

impl World {
    // The command consumer plus the four RT->NRT producer rings the World writes to, wired once by
    // the engine builder.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        options: &Options,
        audio: RateInfo,
        control: RateInfo,
        rx: Consumer<TimedCommand>,
        trash_tx: Producer<Trash>,
        events_tx: Producer<Event>,
        replies_tx: Producer<Reply>,
        triggers_tx: Producer<Trigger>,
    ) -> Self {
        let capacity = options.max_nodes.max(1);
        let trigger_cap = options.max_triggers;
        let bs = options.block_size;
        // A `/g_queryTree` dump can emit several records per node (the node row, a synth row, and one
        // per control); size the scratch and reply backlog generously so the audio thread never grows
        // them (a truly huge dump is capped instead - see `query_tree`).
        let tree_capacity = capacity.saturating_mul(4).max(MAX_QUERY_RANGE + 2);
        World {
            audio,
            control,
            wavetables: Wavetables::new(),
            buses: Buses::new(
                options.output_channels,
                options.input_channels,
                options.audio_bus_channels,
                options.control_bus_channels,
                bs,
            ),
            buffers: BufferTable::new(options.max_buffers),
            tree: NodeTree::new(options.max_nodes, crate::options::ROOT_GROUP_ID),
            pool: Pool::with_capacity_bytes(options.pool_bytes),
            def_table: vec![None; options.max_synthdefs],
            wire_scratch: vec![0.0f32; options.max_wire_bufs * bs].into_boxed_slice(),
            unit_scratch: vec![0.0f32; options.max_unit_outputs * bs].into_boxed_slice(),
            next_seed: SEED_INIT,
            rx,
            trash_tx,
            events_tx,
            replies_tx,
            triggers_tx,
            pending_trash: Vec::with_capacity(capacity),
            pending_events: Vec::with_capacity(capacity),
            pending_replies: VecDeque::with_capacity(tree_capacity),
            tree_scratch: Vec::with_capacity(tree_capacity),
            done_nodes: Vec::with_capacity(capacity),
            freed_nodes: Vec::with_capacity(capacity),
            paused_nodes: Vec::with_capacity(capacity),
            trigger_buf: Vec::with_capacity(trigger_cap),
            trigger_cap,
            clock: Clock::new(options.sample_rate, bs),
            scheduler: Scheduler::new(options.max_scheduled),
            current_sample_offset: 0,
            buf_counter: 0,
            block_size: bs,
            // Force a fresh control block on the first fill.
            block_frames_emitted: bs,
        }
    }

    /// Fill `output` (interleaved, `out_channels` wide) with synthesized audio.
    ///
    /// Reblocks the fixed control-block size to `output`'s arbitrary length. RT-safe.
    pub fn fill(&mut self, output: &mut [f32], out_channels: usize) {
        self.fill_duplex(output, out_channels, &[], 0);
    }

    /// Like [`World::fill`], but drift-corrects the engine clock to `buffer_time` - the OSC/NTP time
    /// (the 32.32 fixed-point value since 1900 that OSC bundles carry) at which `output`'s first
    /// frame is heard - so time-tagged commands land sample-accurately even as the audio device
    /// clock drifts against the host clock.
    ///
    /// Call this once per audio callback, on whole-block-multiple buffers, passing the buffer's host
    /// time mapped to OSC/NTP (e.g. from a `cpal` output timestamp). Hosts that do not schedule
    /// commands can keep using plain [`World::fill`], whose clock free-runs at the nominal rate.
    /// RT-safe.
    pub fn fill_at(&mut self, output: &mut [f32], out_channels: usize, buffer_time: u64) {
        self.fill_duplex_at(output, out_channels, &[], 0, buffer_time);
    }

    /// [`World::fill_duplex`] with the clock resync of [`World::fill_at`].
    pub fn fill_duplex_at(
        &mut self,
        output: &mut [f32],
        out_channels: usize,
        input: &[f32],
        in_channels: usize,
        buffer_time: u64,
    ) {
        let emitted = self.buf_counter.wrapping_mul(self.block_size as u64);
        self.clock.resync(buffer_time, emitted);
        self.fill_duplex(output, out_channels, input, in_channels);
    }

    /// Bytes of the real-time pool currently allocated to live synths' state (scsynth's `/status`
    /// RT-memory figure). With no live synths this is `0`. Walks the pool, so it is `O(chunks)` -
    /// diagnostics, not the hot path.
    pub fn rt_memory_used(&self) -> usize {
        self.pool.used_bytes()
    }

    /// Like [`World::fill`], but also feeds interleaved host `input` (`in_channels` wide) into the
    /// input bus region for `In.ar` to read.
    ///
    /// Input is deinterleaved one control block at a time, so for exact input/output alignment call
    /// this with `output`/`input` lengths that are whole multiples of the block size (and do not
    /// interleave it with plain [`World::fill`] on the same `World`); otherwise the tail of a block
    /// that straddles a buffer boundary reads as zero. RT-safe.
    pub fn fill_duplex(
        &mut self,
        output: &mut [f32],
        out_channels: usize,
        input: &[f32],
        in_channels: usize,
    ) {
        if out_channels == 0 {
            return;
        }
        let frames = output.len() / out_channels;
        let out_bus_channels = self.buses.output_channels();
        let mut frame = 0;
        while frame < frames {
            if self.block_frames_emitted >= self.block_size {
                if in_channels > 0 {
                    let avail = (frames - frame).min(self.block_size);
                    let block_in = &input[frame * in_channels..(frame + avail) * in_channels];
                    self.buses.write_input(block_in, in_channels);
                }
                self.run_one_block();
                self.block_frames_emitted = 0;
            }
            let avail = self.block_size - self.block_frames_emitted;
            let n = avail.min(frames - frame);
            let offset = self.block_frames_emitted;
            for c in 0..out_channels {
                if c < out_bus_channels {
                    let chan = self.buses.audio().channel(c);
                    for i in 0..n {
                        output[(frame + i) * out_channels + c] = chan[offset + i];
                    }
                } else {
                    for i in 0..n {
                        output[(frame + i) * out_channels + c] = 0.0;
                    }
                }
            }
            self.block_frames_emitted += n;
            frame += n;
        }
    }

    /// Compute one control block: drain commands, run the tree, apply done actions, silence
    /// untouched output channels.
    fn run_one_block(&mut self) {
        self.drain_commands();
        self.apply_due_scheduled();
        self.buf_counter += 1;
        self.done_nodes.clear();
        self.trigger_buf.clear();
        // Borrow the World's fields disjointly to assemble the per-block bundle: the pool, the shared
        // wire/output scratch, the buses/buffers, and the constants - all distinct fields, threaded
        // through the tree to each graph.
        let mut block = Block {
            audio: &self.audio,
            control: &self.control,
            wavetables: &self.wavetables,
            buses: &mut self.buses,
            buffers: &mut self.buffers,
            buf_counter: self.buf_counter,
            pool: &mut self.pool,
            wire_scratch: &mut self.wire_scratch[..],
            unit_scratch: &mut self.unit_scratch[..],
            triggers: &mut self.trigger_buf,
            trigger_cap: self.trigger_cap,
        };
        self.tree.process(&mut block, &mut self.done_nodes);
        self.buses.silence_untouched_outputs(self.buf_counter);
        self.apply_done_actions();
        self.drain_triggers();
        self.clock.advance();
    }

    /// Push this block's `SendTrig` triggers onto the trigger ring; drop any that do not fit (a `/tr`
    /// is best-effort, so there is no backlog - this keeps the lifecycle event ring untouched).
    fn drain_triggers(&mut self) {
        for trigger in self.trigger_buf.drain(..) {
            if self.triggers_tx.push(trigger).is_err() {
                break;
            }
        }
    }

    /// Apply every scheduled command due by the end of this control block, in time order, stamping
    /// each with its within-block sample offset (scsynth's per-event `mSampleOffset`). A late
    /// command (its time already past) applies at offset 0.
    fn apply_due_scheduled(&mut self) {
        let deadline = self.clock.block_end();
        while self
            .scheduler
            .next_time()
            .is_some_and(|time| time <= deadline)
        {
            let Some((time, command)) = self.scheduler.pop() else {
                break;
            };
            self.current_sample_offset = self.clock.sample_offset(time, self.block_size);
            self.apply(command);
        }
        self.current_sample_offset = 0;
    }

    /// Apply the done actions collected during the tree walk. Each may free its synth (possibly with
    /// neighbours or the enclosing group) and/or pause a node; the tree restructures into the
    /// pre-allocated `freed_nodes`/`paused_nodes` sinks, which then drain into `/n_end`/`/n_off`
    /// notifications off the audio thread.
    fn apply_done_actions(&mut self) {
        if self.done_nodes.is_empty() {
            return;
        }
        // Borrow the sinks out so the tree can take them by `&mut` alongside its own `&mut self`.
        let mut freed = core::mem::take(&mut self.freed_nodes);
        let mut paused = core::mem::take(&mut self.paused_nodes);
        for i in 0..self.done_nodes.len() {
            let (idx, action) = self.done_nodes[i];
            self.tree
                .apply_done_action(idx, action, &mut freed, &mut paused);
        }
        self.drain_freed(&mut freed);
        for id in paused.drain(..) {
            self.emit(Event::NodePaused { id });
        }
        self.freed_nodes = freed;
        self.paused_nodes = paused;
    }

    fn drain_commands(&mut self) {
        self.flush_pending_trash();
        self.flush_pending_events();
        self.flush_pending_replies();
        while let Ok(timed) = self.rx.pop() {
            match timed.time {
                CommandTime::Immediate => self.apply(timed.command),
                // Hold a future command in the scheduler until its block. If the scheduler is full,
                // apply it now rather than drop it on the audio thread - `apply` routes any owned
                // `Box` to the trash ring, so this never frees here; degraded timing, no lost
                // command, still RT-safe.
                CommandTime::At(time) => {
                    if let Err(command) = self.scheduler.push(time, timed.command) {
                        self.apply(command);
                    }
                }
            }
        }
    }

    fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::DefineGraphDef { def_id, def } => {
                if let Some(slot) = self.def_table.get_mut(def_id as usize) {
                    *slot = Some(def);
                }
            }
            Command::FreeGraphDef { def_id } => {
                if let Some(slot) = self.def_table.get_mut(def_id as usize) {
                    *slot = None;
                }
            }
            Command::AddSynth {
                id,
                def_id,
                target,
                action,
            } => self.add_synth(id, def_id, target, action),
            Command::AddGroup { id, target, action } => {
                if self.tree.add_group(id, target, action) {
                    self.emit(Event::NodeStarted { id });
                }
            }
            Command::SetControl { node, param, value } => {
                let World { tree, pool, .. } = self;
                if let Some(graph) = tree.synth_mut(node) {
                    graph.set_control(pool, param, value);
                }
            }
            Command::SetControlBus { bus, value } => {
                self.buses.control_mut().set(bus as usize, value);
            }
            Command::MapControl { node, param, bus } => {
                let World { tree, pool, .. } = self;
                if let Some(graph) = tree.synth_mut(node) {
                    graph.map_control(pool, param, bus);
                }
            }
            Command::SetBuffer { index, buffer } => {
                let old = self.buffers.set(index, buffer);
                self.trash_slot(old);
            }
            Command::CueStream { index, playback } => {
                let old = self.buffers.cue(index, playback);
                self.trash_slot(old);
            }
            Command::FreeBuffer { index } => {
                let old = self.buffers.free(index);
                self.trash_slot(old);
            }
            Command::SetBufferSample {
                index,
                sample,
                value,
            } => {
                if let Some(buffer) = self.buffers.get_mut(index) {
                    buffer.set_flat(sample, value);
                }
            }
            Command::FillBuffer {
                index,
                start,
                count,
                value,
            } => {
                if let Some(buffer) = self.buffers.get_mut(index) {
                    // Clamp to the buffer so a huge `count` cannot spin the audio thread; per-sample
                    // `set_flat` already ignores any stray out-of-range index.
                    let len = buffer.data().len();
                    let end = start.saturating_add(count).min(len);
                    for sample in start.min(len)..end {
                        buffer.set_flat(sample, value);
                    }
                }
            }
            Command::SetBufferSampleRate { index, sample_rate } => {
                if let Some(buffer) = self.buffers.get_mut(index) {
                    buffer.set_sample_rate(sample_rate);
                }
            }
            Command::CopyBufferRegion {
                dst,
                dst_start,
                src,
                src_start,
                count,
            } => self
                .buffers
                .copy_region(dst, dst_start, src, src_start, count),
            Command::FreeNode { node } => {
                let mut sink = core::mem::take(&mut self.freed_nodes);
                sink.clear();
                self.tree.free_node(node, &mut sink);
                self.drain_freed(&mut sink);
                self.freed_nodes = sink;
            }
            Command::MoveNode {
                node,
                target,
                action,
            } => {
                // scsynth broadcasts `/n_move` for a node it actually relinks; `move_node` returns
                // false for an invalid move (unknown/self/ancestor target), which emits nothing.
                if self.tree.move_node(node, target, action)
                    && let Some(info) = self.tree.node_info(node)
                {
                    self.emit(Event::NodeMoved {
                        node: info.node,
                        parent: info.parent,
                        prev: info.prev,
                        next: info.next,
                        is_group: info.is_group as i32,
                        head: info.head,
                        tail: info.tail,
                    });
                }
            }
            Command::FreeAll { group } => {
                let mut sink = core::mem::take(&mut self.freed_nodes);
                sink.clear();
                self.tree.free_all(group, &mut sink);
                self.drain_freed(&mut sink);
                self.freed_nodes = sink;
            }
            Command::DeepFree { group } => {
                let mut sink = core::mem::take(&mut self.freed_nodes);
                sink.clear();
                self.tree.deep_free(group, &mut sink);
                self.drain_freed(&mut sink);
                self.freed_nodes = sink;
            }
            Command::NodeRun { node, run } => {
                if let Some(id) = self.tree.set_run(node, run) {
                    let event = if run {
                        Event::NodeResumed { id }
                    } else {
                        Event::NodePaused { id }
                    };
                    self.emit(event);
                }
            }
            Command::ClearSched => {
                while let Some((_, command)) = self.scheduler.pop() {
                    self.trash_command(command);
                }
            }

            // --- Queries. Each reads live state and answers over the reply ring (FIFO). ---
            Command::QuerySync { id } => self.reply(Reply::Synced { id }),
            Command::QueryStatus => {
                let (num_synths, num_groups, num_ugens) = self.tree.counts();
                let num_synthdefs = self.def_table.iter().filter(|s| s.is_some()).count();
                let sr = self.audio.sample_rate;
                self.reply(Reply::Status {
                    num_ugens: num_ugens as i32,
                    num_synths: num_synths as i32,
                    num_groups: num_groups as i32,
                    num_synthdefs: num_synthdefs as i32,
                    avg_cpu: 0.0,
                    peak_cpu: 0.0,
                    nominal_sr: sr,
                    actual_sr: sr,
                });
            }
            Command::QueryRtMemory => {
                let total_free = self.pool.free_bytes() as i32;
                let largest_free = self.pool.largest_free_block() as i32;
                self.reply(Reply::RtMemoryStatus {
                    total_free,
                    largest_free,
                });
            }
            Command::QueryNode { node } => match self.tree.node_info(node) {
                Some(info) => self.reply(Reply::NodeInfo {
                    node: info.node,
                    parent: info.parent,
                    prev: info.prev,
                    next: info.next,
                    is_group: info.is_group as i32,
                    head: info.head,
                    tail: info.tail,
                }),
                None => self.reply(Reply::NodeNotFound { node }),
            },
            Command::QueryControlBus { bus } => {
                let value = self.buses.control().read(bus as usize);
                self.reply(Reply::ControlValue {
                    bus: bus as i32,
                    value,
                });
            }
            Command::QueryControlBusRange { start, count } => {
                let count = count.min(MAX_QUERY_RANGE as u32);
                self.reply(Reply::ControlRangeHeader {
                    start: start as i32,
                    count: count as i32,
                });
                for i in 0..count {
                    let value = self.buses.control().read((start + i) as usize);
                    self.reply(Reply::RangeValue { value });
                }
            }
            Command::QuerySynthControl { node, control } => {
                let value = {
                    let World { tree, pool, .. } = self;
                    tree.synth(node)
                        .map(|g| g.control_value(pool, control).unwrap_or(0.0))
                };
                match value {
                    Some(value) => self.reply(Reply::SGetValue {
                        node,
                        control: control as i32,
                        value,
                    }),
                    None => self.reply(Reply::SGetMissing { node }),
                }
            }
            Command::QuerySynthControlRange {
                node,
                control,
                count,
            } => {
                let count = count.min(MAX_QUERY_RANGE);
                let exists = {
                    let World { tree, .. } = self;
                    tree.synth(node).is_some()
                };
                if !exists {
                    self.reply(Reply::SGetMissing { node });
                } else {
                    self.reply(Reply::SGetRangeHeader {
                        node,
                        control: control as i32,
                        count: count as i32,
                    });
                    for i in 0..count {
                        let value = {
                            let World { tree, pool, .. } = self;
                            tree.synth(node)
                                .and_then(|g| g.control_value(pool, control + i))
                                .unwrap_or(0.0)
                        };
                        self.reply(Reply::RangeValue { value });
                    }
                }
            }
            Command::QueryBuffer { buf, index } => {
                let value = self
                    .buffers
                    .get(buf)
                    .and_then(|b| b.data().get(index).copied())
                    .unwrap_or(0.0);
                self.reply(Reply::BufferValue {
                    buf: buf as i32,
                    index: index as i32,
                    value,
                });
            }
            Command::QueryBufferRange { buf, index, count } => {
                let count = count.min(MAX_QUERY_RANGE);
                self.reply(Reply::BufferRangeHeader {
                    buf: buf as i32,
                    index: index as i32,
                    count: count as i32,
                });
                for i in 0..count {
                    let value = self
                        .buffers
                        .get(buf)
                        .and_then(|b| b.data().get(index + i).copied())
                        .unwrap_or(0.0);
                    self.reply(Reply::RangeValue { value });
                }
            }
            Command::QueryTree { group, flag } => self.query_tree(group, flag, false),
            Command::DumpTree { group, flag } => self.query_tree(group, flag, true),
        }
    }

    /// Stream the subtree under `group` over the reply ring (`/g_queryTree`/`/g_dumpTree`). Fills the
    /// pre-allocated `tree_scratch` (so the walk borrows the tree + pool while the reply ring is
    /// untouched), then drains it in order. A `dump` opens the stream with [`Reply::DumpTreeHeader`]
    /// so the dispatcher routes it to a text sink.
    fn query_tree(&mut self, group: i32, flag: bool, dump: bool) {
        let mut scratch = core::mem::take(&mut self.tree_scratch);
        scratch.clear();
        let header = if dump {
            Reply::DumpTreeHeader { flag: flag as i32 }
        } else {
            Reply::QueryTreeHeader { flag: flag as i32 }
        };
        scratch.push(header);
        {
            let World { tree, pool, .. } = self;
            tree.query_tree(group, flag, pool, &mut scratch);
        }
        // Always terminate the stream, even if the walk filled the scratch to capacity (overwrite the
        // last record rather than reallocate on the audio thread).
        if scratch.len() < scratch.capacity() {
            scratch.push(Reply::QueryTreeEnd);
        } else if let Some(last) = scratch.last_mut() {
            *last = Reply::QueryTreeEnd;
        }
        for &r in &scratch {
            self.reply(r);
        }
        self.tree_scratch = scratch;
    }

    /// Construct a synth from the resident def at `def_id` and link it into the tree. On a missing def
    /// or pool exhaustion, emits [`Event::SynthFailed`] and creates no node (scsynth's
    /// out-of-real-time-memory path).
    fn add_synth(&mut self, id: i32, def_id: u32, target: i32, action: AddAction) {
        let Some(def) = self.def_table.get(def_id as usize).cloned().flatten() else {
            self.emit(Event::SynthFailed { id });
            return;
        };
        let Some(graph) = self.build_graph(&def) else {
            self.emit(Event::SynthFailed { id });
            return;
        };
        match self.tree.add_synth(id, graph, target, action) {
            Ok(()) => self.emit(Event::NodeStarted { id }),
            Err(returned) => self.pool.dealloc(returned.into_block()),
        }
    }

    /// Allocate and initialise a synth's per-instance block from `def`: one pool allocation, then copy
    /// the state-arena image, seed the control wires from the defaults, set the param maps unmapped,
    /// and re-seed each unit's randomness for this instance. Returns `None` if the pool is exhausted.
    fn build_graph(&mut self, def: &Arc<GraphDef>) -> Option<Graph> {
        let layout = def.layout();
        let region = self.pool.alloc(layout.total)?;
        let seed = self.next_seed;
        self.next_seed = self.next_seed.wrapping_add(SEED_STEP);

        let buf = self.pool.slice_mut(&region);
        // Carve the block into its disjoint spans. The layout guarantees they are in-bounds and
        // non-overlapping, so this never fails.
        let [state_arena, demand_state, ctrl_bytes, pmap_bytes] = buf
            .get_disjoint_mut([
                layout.state.range(),
                layout.demand_state.range(),
                layout.control.range(),
                layout.pmaps.range(),
            ])
            .expect("graph block layout spans are disjoint by construction");
        state_arena.copy_from_slice(def.state_image());
        demand_state.copy_from_slice(def.demand_state_image());
        cast_slice_mut::<u8, f32>(ctrl_bytes).copy_from_slice(def.control_defaults());
        for m in cast_slice_mut::<u8, u32>(pmap_bytes) {
            *m = u32::MAX;
        }
        // Re-seed each unit's randomness for this instance (calc units, then demand units, on one
        // continuing index so two instances of a def decorrelate reproducibly).
        for (u, v) in def.units().iter().enumerate() {
            let slot = &mut state_arena[v.state_offset..v.state_offset + v.state_size];
            (v.reseed)(slot, seed.wrapping_add((u as u64).wrapping_mul(SEED_STEP)));
        }
        let calc_count = def.units().len() as u64;
        for (d, v) in def.demand_units().iter().enumerate() {
            let slot = &mut demand_state[v.state_offset..v.state_offset + v.state_size];
            (v.reseed)(
                slot,
                seed.wrapping_add((calc_count + d as u64).wrapping_mul(SEED_STEP)),
            );
        }

        Some(Graph::new(
            region,
            Arc::clone(def),
            self.current_sample_offset,
        ))
    }

    /// Route a freed `Box` back to the NRT side, retaining it for retry if the ring is full (never
    /// dropped here on the audio thread).
    fn trash(&mut self, item: Trash) {
        if let Err(PushError::Full(item)) = self.trash_tx.push(item) {
            self.pending_trash.push(item);
        }
    }

    /// Route any heap a discarded scheduled command owns to the trash ring before the command is
    /// dropped, so clearing the scheduler (`/clearSched`) never frees a `Box` on the audio thread.
    /// Only [`SetBuffer`](Command::SetBuffer) and [`CueStream`](Command::CueStream) own such a `Box`;
    /// every other command is flat or holds a non-final `Arc` (the Controller retains its own), so
    /// letting it drop here is RT-safe.
    fn trash_command(&mut self, command: Command) {
        match command {
            Command::SetBuffer { buffer, .. } => self.trash(Trash::Buffer(buffer)),
            Command::CueStream { playback, .. } => self.trash(Trash::Stream(playback)),
            _ => {}
        }
    }

    /// Route a displaced buffer-table slot to the trash ring (an empty slot needs no dropping).
    fn trash_slot(&mut self, slot: Option<BufferSlot>) {
        match slot {
            Some(BufferSlot::Loaded(buffer)) => self.trash(Trash::Buffer(buffer)),
            Some(BufferSlot::Stream(stream)) => self.trash(Trash::Stream(stream)),
            Some(BufferSlot::Empty) | None => {}
        }
    }

    /// Reclaim each freed graph's pool block (on the audio thread) and notify each freed node
    /// (`NodeEnded`).
    fn drain_freed(&mut self, sink: &mut Vec<FreedNode>) {
        for (id, graph) in sink.drain(..) {
            if let Some(graph) = graph {
                self.pool.dealloc(graph.into_block());
            }
            self.emit(Event::NodeEnded { id });
        }
    }

    /// Send a notification to the NRT side, retaining it for retry if the ring is full.
    fn emit(&mut self, event: Event) {
        if let Err(PushError::Full(event)) = self.events_tx.push(event) {
            self.pending_events.push(event);
        }
    }

    /// Send a query answer to the NRT side, preserving FIFO order (the dispatcher reassembles
    /// strictly in query order). Once a backlog exists, later answers queue behind it rather than
    /// jumping ahead into the ring; never dropped here on the audio thread.
    fn reply(&mut self, r: Reply) {
        if self.pending_replies.is_empty() {
            if let Err(PushError::Full(r)) = self.replies_tx.push(r) {
                self.pending_replies.push_back(r);
            }
        } else {
            self.pending_replies.push_back(r);
        }
    }

    fn flush_pending_trash(&mut self) {
        while let Some(item) = self.pending_trash.pop() {
            if let Err(PushError::Full(item)) = self.trash_tx.push(item) {
                self.pending_trash.push(item);
                break;
            }
        }
    }

    fn flush_pending_events(&mut self) {
        while let Some(event) = self.pending_events.pop() {
            if let Err(PushError::Full(event)) = self.events_tx.push(event) {
                self.pending_events.push(event);
                break;
            }
        }
    }

    fn flush_pending_replies(&mut self) {
        // Drain front-to-back so order is preserved; stop at the first that does not fit.
        while let Some(&r) = self.pending_replies.front() {
            if self.replies_tx.push(r).is_err() {
                break;
            }
            self.pending_replies.pop_front();
        }
    }
}
