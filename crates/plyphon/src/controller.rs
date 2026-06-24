//! The control side of the engine - plyphon's port of the non-real-time half of scsynth's
//! command handling.
//!
//! The `Controller` owns the [`SynthDefLibrary`] and the [`UnitRegistry`]; the audio thread never
//! touches them. It *compiles* each def into an immutable [`GraphDef`] (all unit construction happens
//! here), installs it in the `World`'s resident def table once via [`Command::DefineGraphDef`], and
//! thereafter `s_new` ships only a `def_id` - the synth itself is built on the audio thread. Reactive
//! NRT work - dropping freed buffers/streams and surfacing notifications - lives in the
//! [`Nrt`](crate::nrt::Nrt) instead.
//!
//! The controller retains a strong `Arc` to every `GraphDef` it has ever compiled (current ones in
//! `compiled`, superseded ones in `graveyard`), for the engine's lifetime. That way an
//! `Arc<GraphDef>` dropped on the audio thread (a freed graph's, or a def-table slot replaced by a
//! redefinition) is never the final reference, so the heavy drop never lands on the audio thread.

use std::collections::HashMap;
use std::sync::Arc;

use rtrb::Producer;
use thiserror::Error;

use crate::buffer::Buffer;
use crate::command::Command;
use crate::engine::Options;
use crate::error::BuildError;
use crate::graphdef::GraphDef;
use crate::rate::RateInfo;
use crate::stream::{StreamProducer, cue};
use crate::synthdef::{SynthDef, SynthDefLibrary};
use crate::tree::AddAction;
use crate::unit::registry::UnitRegistry;

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
    /// The SynthDef failed to compile.
    #[error(transparent)]
    Build(#[from] BuildError),
    /// More distinct synthdefs were used than the engine's `max_synthdefs` allows.
    #[error("too many synthdefs (limit reached)")]
    TooManyDefs,
    /// The command ring was full.
    #[error("command queue full")]
    QueueFull,
}

/// The control side of the engine.
pub struct Controller {
    registry: UnitRegistry,
    defs: SynthDefLibrary,
    /// Current compiled def per name (also the controller's retained `Arc` for it).
    compiled: HashMap<String, Arc<GraphDef>>,
    /// Stable name -> `def_id`, assigned on first compile and reused across recompiles.
    def_ids: HashMap<String, u32>,
    /// Superseded compiled defs, retained for the engine's lifetime (see the module docs).
    graveyard: Vec<Arc<GraphDef>>,
    next_def_id: u32,
    audio: RateInfo,
    control: RateInfo,
    max_synthdefs: usize,
    max_wire_bufs: usize,
    max_unit_outputs: usize,
    tx: Producer<Command>,
    next_id: i32,
}

impl Controller {
    pub(crate) fn new(
        options: &Options,
        audio: RateInfo,
        control: RateInfo,
        tx: Producer<Command>,
    ) -> Self {
        Controller {
            registry: UnitRegistry::with_builtins(),
            defs: SynthDefLibrary::new(),
            compiled: HashMap::new(),
            def_ids: HashMap::new(),
            graveyard: Vec::new(),
            next_def_id: 0,
            audio,
            control,
            max_synthdefs: options.max_synthdefs,
            max_wire_bufs: options.max_wire_bufs,
            max_unit_outputs: options.max_unit_outputs,
            tx,
            // Client node ids start above the root group (id 0).
            next_id: 1000,
        }
    }

    /// Mutable access to the unit registry, for registering custom units.
    pub fn registry_mut(&mut self) -> &mut UnitRegistry {
        &mut self.registry
    }

    /// The engine's audio sample rate in Hz (e.g. to stamp a freshly allocated buffer).
    pub fn sample_rate(&self) -> f64 {
        self.audio.sample_rate
    }

    /// Add (or replace) a synth definition. Compilation is deferred to the first `synth_new` that
    /// uses it (so it can surface a [`BuildError`]); redefining a name retires any current compiled
    /// form to the graveyard and forces a recompile on next use.
    pub fn add_synthdef(&mut self, def: SynthDef) {
        if let Some(old) = self.compiled.remove(&def.name) {
            self.graveyard.push(old);
        }
        self.defs.insert(def);
    }

    /// Look up a registered synth definition (e.g. to resolve a parameter index by name).
    pub fn synthdef(&self, name: &str) -> Option<&SynthDef> {
        self.defs.get(name)
    }

    /// Create a synth from definition `def_name` and link it under group `target`.
    ///
    /// The def is compiled (and installed in the `World`'s def table) on first use; the synth itself
    /// is constructed on the audio thread. Returns the new synth's client id.
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

    /// Create a synth with a caller-chosen client id (e.g. an id from an OSC `/s_new`).
    pub fn synth_new_with_id(
        &mut self,
        id: i32,
        def_name: &str,
        target: i32,
        action: AddAction,
    ) -> Result<(), SynthNewError> {
        let def_id = self.ensure_compiled(def_name)?;
        self.tx
            .push(Command::AddSynth {
                id,
                def_id,
                target,
                action,
            })
            .map_err(|_| SynthNewError::QueueFull)?;
        Ok(())
    }

    /// Ensure `def_name` is compiled and resident in the `World`'s def table, returning its `def_id`.
    /// Compiles (and ships a [`Command::DefineGraphDef`]) on first use or after a redefinition.
    fn ensure_compiled(&mut self, def_name: &str) -> Result<u32, SynthNewError> {
        if self.compiled.contains_key(def_name) {
            return Ok(self.def_ids[def_name]);
        }
        // Compile the authored def (the only place unit construction / allocation happens).
        let graphdef = {
            let authored = self
                .defs
                .get(def_name)
                .ok_or_else(|| SynthNewError::UnknownDef(def_name.to_string()))?;
            authored.compile(
                &self.registry,
                &self.audio,
                &self.control,
                self.max_wire_bufs,
                self.max_unit_outputs,
            )?
        };
        // Assign a stable def_id (reused if this name was compiled before).
        let def_id = match self.def_ids.get(def_name).copied() {
            Some(id) => id,
            None => {
                let id = self.next_def_id;
                if id as usize >= self.max_synthdefs {
                    return Err(SynthNewError::TooManyDefs);
                }
                self.next_def_id += 1;
                self.def_ids.insert(def_name.to_string(), id);
                id
            }
        };
        let def = Arc::new(graphdef);
        self.tx
            .push(Command::DefineGraphDef {
                def_id,
                def: Arc::clone(&def),
            })
            .map_err(|_| SynthNewError::QueueFull)?;
        self.compiled.insert(def_name.to_string(), def);
        Ok(def_id)
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

    /// Free node `node` (deeply for a group: the group and its whole subtree).
    pub fn free(&mut self, node: i32) -> Result<(), QueueFull> {
        self.tx
            .push(Command::FreeNode { node })
            .map_err(|_| QueueFull)
    }

    /// Move node `node` to `target`/`action` (scsynth's `/g_head`, `/g_tail`, `/n_before`,
    /// `/n_after`).
    pub fn move_node(
        &mut self,
        node: i32,
        target: i32,
        action: AddAction,
    ) -> Result<(), QueueFull> {
        self.tx
            .push(Command::MoveNode {
                node,
                target,
                action,
            })
            .map_err(|_| QueueFull)
    }

    /// Free every node in group `group`, leaving it empty (scsynth's `/g_freeAll`).
    pub fn free_all(&mut self, group: i32) -> Result<(), QueueFull> {
        self.tx
            .push(Command::FreeAll { group })
            .map_err(|_| QueueFull)
    }

    /// Free every synth in group `group` and its subgroups, keeping the groups (`/g_deepFree`).
    pub fn deep_free(&mut self, group: i32) -> Result<(), QueueFull> {
        self.tx
            .push(Command::DeepFree { group })
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
