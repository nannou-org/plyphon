//! The real-time side of the engine - plyphon's port of scsynth's `World`/`World_Run`.
//!
//! `World` owns the buses, node tree, and wavetables. The host's audio callback drives it via
//! [`World::fill`], which reblocks the engine's fixed control-block size to the host's arbitrary
//! buffer size. Every per-block step is O(1) link manipulation or a bounded loop over pre-allocated
//! buffers: no allocation, locks, or blocking on the audio thread.
//!
//! Two streams flow back to the NRT side (drained by the [`Nrt`](crate::nrt::Nrt)): freed synths go
//! to the trash ring rather than being dropped here, and node notifications go to the events ring.
//! Done actions (a UGen asking to free or pause its synth) are applied here after the tree runs.

use rtrb::{Consumer, Producer, PushError};

use crate::bus::Buses;
use crate::command::{Command, Event, Trash};
use crate::engine::Options;
use crate::rate::RateInfo;
use crate::tree::NodeTree;
use crate::ugen::{DoneAction, ProcessContext};
use crate::wavetable::Wavetables;

/// The real-time engine half.
pub struct World {
    audio: RateInfo,
    control: RateInfo,
    wavetables: Wavetables,
    buses: Buses,
    tree: NodeTree,
    rx: Consumer<Command>,
    trash_tx: Producer<Trash>,
    events_tx: Producer<Event>,
    /// Freed items awaiting space in the trash ring (pre-allocated; never reallocates at runtime).
    pending_trash: Vec<Trash>,
    /// Events awaiting space in the events ring (pre-allocated; never reallocates at runtime).
    pending_events: Vec<Event>,
    /// Scratch list of `(slot index, action)` for nodes whose UGens requested a done action.
    done_nodes: Vec<(u32, DoneAction)>,
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
        World {
            audio,
            control,
            wavetables: Wavetables::new(),
            buses: Buses::new(
                options.output_channels,
                options.input_channels,
                options.audio_bus_channels,
                options.control_bus_channels,
                options.block_size,
            ),
            tree: NodeTree::new(options.max_nodes, crate::engine::ROOT_GROUP_ID),
            rx,
            trash_tx,
            events_tx,
            pending_trash: Vec::with_capacity(capacity),
            pending_events: Vec::with_capacity(capacity),
            done_nodes: Vec::with_capacity(capacity),
            buf_counter: 0,
            block_size: options.block_size,
            // Force a fresh control block on the first fill.
            block_frames_emitted: options.block_size,
        }
    }

    /// Fill `output` (interleaved, `out_channels` wide) with synthesized audio.
    ///
    /// Reblocks the fixed control-block size to `output`'s arbitrary length. RT-safe.
    pub fn fill(&mut self, output: &mut [f32], out_channels: usize) {
        self.fill_duplex(output, out_channels, &[], 0);
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
        let buf_counter = self.buf_counter;
        let ctx = ProcessContext {
            audio: &self.audio,
            control: &self.control,
            buf_counter,
            wavetables: &self.wavetables,
        };
        self.done_nodes.clear();
        self.tree
            .process(&ctx, &mut self.buses, &mut self.done_nodes);
        self.buses.silence_untouched_outputs(buf_counter);
        self.apply_done_actions();
    }

    /// Apply the done actions collected during the tree walk (free or pause the node).
    fn apply_done_actions(&mut self) {
        for i in 0..self.done_nodes.len() {
            let (idx, action) = self.done_nodes[i];
            match action {
                DoneAction::FreeSelf => {
                    if let Some((id, synth)) = self.tree.free_by_index(idx) {
                        self.trash(Trash::Synth(synth));
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
            Command::AddSynth {
                id,
                synth,
                target,
                action,
            } => match self.tree.add_synth(id, synth, target, action) {
                Ok(()) => self.emit(Event::NodeStarted { id }),
                Err(returned) => self.trash(Trash::Synth(returned)),
            },
            Command::AddGroup { id, target, action } => {
                if self.tree.add_group(id, target, action) {
                    self.emit(Event::NodeStarted { id });
                }
            }
            Command::SetControl { node, param, value } => {
                if let Some(synth) = self.tree.synth_mut(node) {
                    synth.set_control(param, value);
                }
            }
            Command::SetControlBus { bus, value } => {
                self.buses.control_mut().set(bus as usize, value);
            }
            Command::MapControl { node, param, bus } => {
                if let Some(synth) = self.tree.synth_mut(node) {
                    synth.map_control(param, bus);
                }
            }
            Command::FreeNode { node } => {
                if let Some(synth) = self.tree.free_node(node) {
                    self.trash(Trash::Synth(synth));
                    self.emit(Event::NodeEnded { id: node });
                }
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

    /// Route a freed `Box` back to the NRT side, retaining it for retry if the ring is full (never
    /// dropped here on the audio thread).
    fn trash(&mut self, item: Trash) {
        if let Err(PushError::Full(item)) = self.trash_tx.push(item) {
            self.pending_trash.push(item);
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
