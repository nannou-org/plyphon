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

use std::sync::Arc;

use bytemuck::cast_slice_mut;
use rtrb::{Consumer, Producer, PushError};

use crate::buffer::{BufferSlot, BufferTable};
use crate::bus::Buses;
use crate::command::{Command, Event, Trash};
use crate::engine::Options;
use crate::graph::{Block, Graph, Pool};
use crate::graphdef::GraphDef;
use crate::rate::RateInfo;
use crate::tree::{AddAction, FreedNode, NodeTree};
use crate::unit::DoneAction;
use crate::wavetable::Wavetables;

/// The seed the per-instance RNG counter starts from (any fixed non-zero value; keeps runs
/// deterministic while decorrelating distinct synth instances).
const SEED_INIT: u64 = 0x853c_49e6_748f_ea9b;

/// The golden-ratio odd constant used to spread per-instance and per-unit seeds.
const SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;

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
    rx: Consumer<Command>,
    trash_tx: Producer<Trash>,
    events_tx: Producer<Event>,
    /// Freed items awaiting space in the trash ring (pre-allocated; never reallocates at runtime).
    pending_trash: Vec<Trash>,
    /// Events awaiting space in the events ring (pre-allocated; never reallocates at runtime).
    pending_events: Vec<Event>,
    /// Scratch list of `(slot index, action)` for nodes whose units requested a done action.
    done_nodes: Vec<(u32, DoneAction)>,
    /// Scratch sink for nodes removed by a free, so freeing a whole group allocates nothing.
    freed_nodes: Vec<FreedNode>,
    buf_counter: u64,
    block_size: usize,
    /// How many frames of the current control block have already been emitted to the host.
    block_frames_emitted: usize,
}

impl World {
    pub(crate) fn new(
        options: &Options,
        audio: RateInfo,
        control: RateInfo,
        rx: Consumer<Command>,
        trash_tx: Producer<Trash>,
        events_tx: Producer<Event>,
    ) -> Self {
        let capacity = options.max_nodes.max(1);
        let bs = options.block_size;
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
            tree: NodeTree::new(options.max_nodes, crate::engine::ROOT_GROUP_ID),
            pool: Pool::with_capacity_bytes(options.pool_bytes),
            def_table: vec![None; options.max_synthdefs],
            wire_scratch: vec![0.0f32; options.max_wire_bufs * bs].into_boxed_slice(),
            unit_scratch: vec![0.0f32; options.max_unit_outputs * bs].into_boxed_slice(),
            next_seed: SEED_INIT,
            rx,
            trash_tx,
            events_tx,
            pending_trash: Vec::with_capacity(capacity),
            pending_events: Vec::with_capacity(capacity),
            done_nodes: Vec::with_capacity(capacity),
            freed_nodes: Vec::with_capacity(capacity),
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
        self.buf_counter += 1;
        self.done_nodes.clear();
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
        };
        self.tree.process(&mut block, &mut self.done_nodes);
        self.buses.silence_untouched_outputs(self.buf_counter);
        self.apply_done_actions();
    }

    /// Apply the done actions collected during the tree walk (free or pause the node).
    fn apply_done_actions(&mut self) {
        for i in 0..self.done_nodes.len() {
            let (idx, action) = self.done_nodes[i];
            match action {
                DoneAction::FreeSelf => {
                    if let Some((id, graph)) = self.tree.free_by_index(idx) {
                        self.pool.dealloc(graph.into_block());
                        self.emit(Event::NodeEnded { id });
                    }
                }
                DoneAction::Pause => {
                    if let Some(id) = self.tree.pause_by_index(idx) {
                        self.emit(Event::NodePaused { id });
                    }
                }
                DoneAction::Nothing => {}
            }
        }
    }

    fn drain_commands(&mut self) {
        self.flush_pending_trash();
        self.flush_pending_events();
        while let Ok(cmd) = self.rx.pop() {
            self.apply(cmd);
        }
    }

    fn apply(&mut self, cmd: Command) {
        match cmd {
            Command::DefineGraphDef { def_id, def } => {
                if let Some(slot) = self.def_table.get_mut(def_id as usize) {
                    *slot = Some(def);
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
                self.tree.move_node(node, target, action);
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
        }
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
        let layout = def.layout;
        let region = self.pool.alloc(layout.total)?;
        let seed = self.next_seed;
        self.next_seed = self.next_seed.wrapping_add(SEED_STEP);

        let buf = self.pool.slice_mut(&region);
        // Carve the block into its disjoint spans. The layout guarantees they are in-bounds and
        // non-overlapping, so this never fails.
        let [state_arena, ctrl_bytes, pmap_bytes] = buf
            .get_disjoint_mut([
                layout.state.range(),
                layout.control.range(),
                layout.pmaps.range(),
            ])
            .expect("graph block layout spans are disjoint by construction");
        state_arena.copy_from_slice(&def.state_image);
        cast_slice_mut::<u8, f32>(ctrl_bytes).copy_from_slice(&def.control_defaults);
        for m in cast_slice_mut::<u8, u32>(pmap_bytes) {
            *m = u32::MAX;
        }
        for (u, v) in def.units.iter().enumerate() {
            let slot = &mut state_arena[v.state_offset..v.state_offset + v.state_size];
            (v.reseed)(slot, seed.wrapping_add((u as u64).wrapping_mul(SEED_STEP)));
        }

        Some(Graph::new(region, Arc::clone(def)))
    }

    /// Route a freed `Box` back to the NRT side, retaining it for retry if the ring is full (never
    /// dropped here on the audio thread).
    fn trash(&mut self, item: Trash) {
        if let Err(PushError::Full(item)) = self.trash_tx.push(item) {
            self.pending_trash.push(item);
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
}
