//! The control side of the engine - plyphon's port of the non-real-time half of scsynth's
//! command handling.
//!
//! The `Controller` owns the [`SynthDefLibrary`] and the [`UgenRegistry`]; the audio thread never
//! touches them. It instantiates synths (all allocation and UGen construction happens here) and
//! ships the finished `Box<Synth>` to the [`World`](crate::world::World) over the command ring, and
//! it drains the trash ring to drop freed synths off the audio thread.

use rtrb::{Consumer, Producer};

use crate::command::{Command, Trash};
use crate::error::BuildError;
use crate::rate::RateInfo;
use crate::synthdef::{SynthDef, SynthDefLibrary};
use crate::tree::AddAction;
use crate::ugen::registry::UgenRegistry;

/// The command ring was full; the command was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFull;

/// Failure to create a new synth.
#[derive(Debug)]
pub enum SynthNewError {
    /// No SynthDef registered under the given name.
    UnknownDef(String),
    /// The SynthDef failed to instantiate.
    Build(BuildError),
    /// The command ring was full.
    QueueFull,
}

impl From<BuildError> for SynthNewError {
    fn from(error: BuildError) -> Self {
        SynthNewError::Build(error)
    }
}

/// The control side of the engine.
pub struct Controller {
    registry: UgenRegistry,
    defs: SynthDefLibrary,
    audio: RateInfo,
    control: RateInfo,
    tx: Producer<Command>,
    trash_rx: Consumer<Trash>,
    next_id: i32,
}

impl Controller {
    pub(crate) fn new(
        audio: RateInfo,
        control: RateInfo,
        tx: Producer<Command>,
        trash_rx: Consumer<Trash>,
    ) -> Self {
        Controller {
            registry: UgenRegistry::with_builtins(),
            defs: SynthDefLibrary::new(),
            audio,
            control,
            tx,
            trash_rx,
            // Client node ids start above the root group (id 0).
            next_id: 1000,
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
        let def = self
            .defs
            .get(def_name)
            .ok_or_else(|| SynthNewError::UnknownDef(def_name.to_string()))?;
        let synth = def.instantiate(&self.registry, &self.audio, &self.control)?;
        self.tx
            .push(Command::AddSynth {
                id,
                synth,
                target,
                action,
            })
            .map_err(|_| SynthNewError::QueueFull)?;
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

    /// Free node `node`.
    pub fn free(&mut self, node: i32) -> Result<(), QueueFull> {
        self.tx
            .push(Command::FreeNode { node })
            .map_err(|_| QueueFull)
    }

    /// Drop any synths the audio thread has finished with. Call periodically from the control side.
    pub fn drain_trash(&mut self) {
        while self.trash_rx.pop().is_ok() {
            // Each popped `Trash` is dropped here, on the control thread.
        }
    }
}
