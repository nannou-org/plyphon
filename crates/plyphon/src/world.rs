//! The real-time side of the engine - plyphon's port of scsynth's `World`/`World_Run`.
//!
//! `World` owns the buses, node tree, and wavetables. The host's audio callback drives it via
//! [`World::fill`], which reblocks the engine's fixed control-block size to the host's arbitrary
//! buffer size. Every per-block step is O(1) link manipulation or a bounded loop over pre-allocated
//! buffers: no allocation, locks, or blocking on the audio thread. Freed synths are routed back to
//! the control side via the trash ring rather than dropped here.

use rtrb::{Consumer, Producer, PushError};

use crate::bus::AudioBus;
use crate::command::{Command, Trash};
use crate::engine::Options;
use crate::rate::RateInfo;
use crate::tree::NodeTree;
use crate::ugen::ProcessContext;
use crate::wavetable::Wavetables;

/// The real-time engine half.
pub struct World {
    audio: RateInfo,
    control: RateInfo,
    wavetables: Wavetables,
    out_bus: AudioBus,
    tree: NodeTree,
    rx: Consumer<Command>,
    trash_tx: Producer<Trash>,
    /// Freed items awaiting space in the trash ring (pre-allocated; never reallocates at runtime).
    pending_trash: Vec<Trash>,
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
    ) -> Self {
        World {
            audio,
            control,
            wavetables: Wavetables::new(),
            out_bus: AudioBus::new(options.output_channels, options.block_size),
            tree: NodeTree::new(options.max_nodes, crate::engine::ROOT_GROUP_ID),
            rx,
            trash_tx,
            pending_trash: Vec::with_capacity(options.max_nodes.max(1)),
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
        if out_channels == 0 {
            return;
        }
        let frames = output.len() / out_channels;
        let bus_channels = self.out_bus.num_channels();
        let mut frame = 0;
        while frame < frames {
            if self.block_frames_emitted >= self.block_size {
                self.run_one_block();
                self.block_frames_emitted = 0;
            }
            let avail = self.block_size - self.block_frames_emitted;
            let n = avail.min(frames - frame);
            let offset = self.block_frames_emitted;
            for c in 0..out_channels {
                if c < bus_channels {
                    let chan = self.out_bus.channel(c);
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

    /// Compute one control block: drain commands, run the tree, silence untouched output channels.
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
        self.tree.process(&ctx, &mut self.out_bus);
        self.out_bus.silence_untouched(buf_counter);
    }

    fn drain_commands(&mut self) {
        self.flush_pending_trash();
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
            } => {
                if let Err(returned) = self.tree.add_synth(id, synth, target, action) {
                    self.trash(Trash::Synth(returned));
                }
            }
            Command::AddGroup { id, target, action } => {
                let _ = self.tree.add_group(id, target, action);
            }
            Command::SetControl { node, param, value } => {
                if let Some(synth) = self.tree.synth_mut(node) {
                    synth.set_control(param, value);
                }
            }
            Command::FreeNode { node } => {
                if let Some(synth) = self.tree.free_node(node) {
                    self.trash(Trash::Synth(synth));
                }
            }
        }
    }

    /// Route a heap-owning item back to the control side, retaining it for retry if the ring is full
    /// (never dropped here on the audio thread).
    fn trash(&mut self, item: Trash) {
        if let Err(PushError::Full(item)) = self.trash_tx.push(item) {
            self.pending_trash.push(item);
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
}
