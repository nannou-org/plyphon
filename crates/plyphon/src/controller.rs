//! The control side of the engine - plyphon's port of the non-real-time half of scsynth's
//! command handling.
//!
//! The `Controller` owns the [`SynthDefLibrary`] and the [`UgenRegistry`]; the audio thread never
//! touches them. It instantiates synths (all allocation and UGen construction happens here) and
//! ships the finished `Box<Synth>` to the [`World`](crate::world::World) over the command ring.
//! Reactive NRT work - dropping freed synths and surfacing notifications - lives in the
//! [`Nrt`](crate::nrt::Nrt) instead.

use rtrb::Producer;
use thiserror::Error;

use crate::buffer::Buffer;
use crate::command::Command;
use crate::error::BuildError;
use crate::rate::RateInfo;
use crate::stream::{StreamProducer, cue};
use crate::synthdef::{SynthDef, SynthDefLibrary};
use crate::tree::AddAction;
use crate::ugen::registry::UgenRegistry;

/// The command ring was full; the command was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("command queue full")]
pub struct QueueFull;

/// Failure to create a new synth.
#[derive(Debug, Error)]
pub enum SynthNewError {
    /// No SynthDef registered under the given name.
    #[error("unknown synthdef: {0}")]
    UnknownDef(String),
    /// The SynthDef failed to instantiate.
    #[error(transparent)]
    Build(#[from] BuildError),
    /// The command ring was full.
    #[error("command queue full")]
    QueueFull,
}

/// The control side of the engine.
pub struct Controller {
    registry: UgenRegistry,
    defs: SynthDefLibrary,
    audio: RateInfo,
    control: RateInfo,
    tx: Producer<Command>,
    next_id: i32,
    next_seed: u64,
}

impl Controller {
    pub(crate) fn new(audio: RateInfo, control: RateInfo, tx: Producer<Command>) -> Self {
        Controller {
            registry: UgenRegistry::with_builtins(),
            defs: SynthDefLibrary::new(),
            audio,
            control,
            tx,
            // Client node ids start above the root group (id 0).
            next_id: 1000,
            next_seed: 0x123456789ABCDEF,
        }
    }

    /// Mutable access to the UGen registry, for registering custom UGens.
    pub fn registry_mut(&mut self) -> &mut UgenRegistry {
        &mut self.registry
    }

    /// Add (or replace) a synth definition.
    pub fn add_synthdef(&mut self, def: SynthDef) {
        self.defs.insert(def);
    }

    /// Look up a registered synth definition (e.g. to resolve a parameter index by name).
    pub fn synthdef(&self, name: &str) -> Option<&SynthDef> {
        self.defs.get(name)
    }

    /// Instantiate a synth from definition `def_name` and link it under group `target`.
    ///
    /// All allocation happens here, off the audio thread; the finished synth is shipped to the
    /// `World`. Returns the new synth's client id.
    pub fn synth_new(
        &mut self,
        def_name: &str,
        target: i32,
        action: AddAction,
    ) -> Result<i32, SynthNewError> {
        let id = self.next_id;
        self.synth_new_with_id(id, def_name, target, action)?;
        self.next_id += 1;
        Ok(id)
    }

    /// Instantiate a synth with a caller-chosen client id (e.g. an id from an OSC `/s_new`).
    pub fn synth_new_with_id(
        &mut self,
        id: i32,
        def_name: &str,
        target: i32,
        action: AddAction,
    ) -> Result<(), SynthNewError> {
        let seed = self.next_seed;
        let def = self
            .defs
            .get(def_name)
            .ok_or_else(|| SynthNewError::UnknownDef(def_name.to_string()))?;
        let synth = def.instantiate(&self.registry, &self.audio, &self.control, seed)?;
        self.tx
            .push(Command::AddSynth {
                id,
                synth,
                target,
                action,
            })
            .map_err(|_| SynthNewError::QueueFull)?;
        self.next_seed = self.next_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        Ok(())
    }

    /// Create an empty group under group `target`. Returns the new group's client id.
    pub fn new_group(&mut self, target: i32, action: AddAction) -> Result<i32, QueueFull> {
        let id = self.next_id;
        self.new_group_with_id(id, target, action)?;
        self.next_id += 1;
        Ok(id)
    }

    /// Create an empty group with a caller-chosen client id (e.g. an id from an OSC `/g_new`).
    pub fn new_group_with_id(
        &mut self,
        id: i32,
        target: i32,
        action: AddAction,
    ) -> Result<(), QueueFull> {
        self.tx
            .push(Command::AddGroup { id, target, action })
            .map_err(|_| QueueFull)
    }

    /// Set control parameter `param` of node `node` to `value`.
    pub fn set_control(&mut self, node: i32, param: usize, value: f32) -> Result<(), QueueFull> {
        self.tx
            .push(Command::SetControl { node, param, value })
            .map_err(|_| QueueFull)
    }

    /// Set control bus channel `bus` to `value` (scsynth's `/c_set`).
    pub fn set_control_bus(&mut self, bus: u32, value: f32) -> Result<(), QueueFull> {
        self.tx
            .push(Command::SetControlBus { bus, value })
            .map_err(|_| QueueFull)
    }

    /// Set consecutive control bus channels from `start` to `values` (scsynth's `/c_setn`).
    pub fn set_control_bus_n(&mut self, start: u32, values: &[f32]) -> Result<(), QueueFull> {
        for (i, &value) in values.iter().enumerate() {
            self.set_control_bus(start + i as u32, value)?;
        }
        Ok(())
    }

    /// Map control parameter `param` of node `node` to control `bus`, or unmap it with `None`
    /// (scsynth's `/n_map`). While mapped, the parameter reads the bus each block.
    pub fn map_control(
        &mut self,
        node: i32,
        param: usize,
        bus: Option<u32>,
    ) -> Result<(), QueueFull> {
        self.tx
            .push(Command::MapControl { node, param, bus })
            .map_err(|_| QueueFull)
    }

    /// Map `count` consecutive control parameters of `node` (from `param`) to consecutive control
    /// buses (from `bus`), or unmap them all with `None` (scsynth's `/n_mapn`).
    pub fn map_control_n(
        &mut self,
        node: i32,
        param: usize,
        bus: Option<u32>,
        count: usize,
    ) -> Result<(), QueueFull> {
        for i in 0..count {
            let mapped = bus.map(|b| b + i as u32);
            self.map_control(node, param + i, mapped)?;
        }
        Ok(())
    }

    /// Free node `node`.
    pub fn free(&mut self, node: i32) -> Result<(), QueueFull> {
        self.tx
            .push(Command::FreeNode { node })
            .map_err(|_| QueueFull)
    }

    /// Pause or resume node `node` (scsynth's `/n_run`).
    pub fn node_run(&mut self, node: i32, run: bool) -> Result<(), QueueFull> {
        self.tx
            .push(Command::NodeRun { node, run })
            .map_err(|_| QueueFull)
    }

    /// Install (or replace) the buffer at `index` with an already-built buffer.
    ///
    /// The buffer is built off the audio thread (this is where any allocation or sample loading
    /// happens); the `World` only does an O(1) swap and routes any previous buffer to the trash ring.
    pub fn buffer_set(&mut self, index: usize, buffer: Box<Buffer>) -> Result<(), QueueFull> {
        self.tx
            .push(Command::SetBuffer { index, buffer })
            .map_err(|_| QueueFull)
    }

    /// Allocate a zeroed buffer of `num_frames` x `num_channels` at `index` (scsynth's `/b_alloc`).
    pub fn buffer_alloc(
        &mut self,
        index: usize,
        num_frames: usize,
        num_channels: usize,
        sample_rate: f64,
    ) -> Result<(), QueueFull> {
        let buffer = Box::new(Buffer::zeroed(num_frames, num_channels, sample_rate));
        self.buffer_set(index, buffer)
    }

    /// Free the buffer at `index` (scsynth's `/b_free`).
    pub fn buffer_free(&mut self, index: usize) -> Result<(), QueueFull> {
        self.tx
            .push(Command::FreeBuffer { index })
            .map_err(|_| QueueFull)
    }

    /// Cue a disk-streaming buffer at `index` (scsynth's `Buffer.cueSoundFile`).
    ///
    /// Allocates a queue of `num_chunks` chunks of `chunk_frames` frames each (off the audio thread)
    /// and installs the RT-side playback endpoint. Returns the [`StreamProducer`] for a feeder to
    /// fill from a sound source; `DiskIn.ar(channels, index)` plays it.
    pub fn buffer_cue(
        &mut self,
        index: usize,
        channels: usize,
        sample_rate: f64,
        chunk_frames: usize,
        num_chunks: usize,
    ) -> Result<StreamProducer, QueueFull> {
        let (playback, producer) = cue(channels, sample_rate, chunk_frames, num_chunks);
        self.tx
            .push(Command::CueStream { index, playback })
            .map_err(|_| QueueFull)?;
        Ok(producer)
    }
}
