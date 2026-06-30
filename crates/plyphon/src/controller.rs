//! The control side of the engine - plyphon's port of the non-real-time half of scsynth's
//! command handling.
//!
//! The `Controller` owns the [`SynthDefLibrary`] and the [`UnitRegistry`]; the audio thread never
//! touches them. It *compiles* each def into an immutable [`GraphDef`] (all unit construction happens
//! here), installs it in the `World`'s resident def table once via [`Command::DefineGraphDef`], and
//! thereafter `s_new` ships only a `def_id` - the synth itself is built on the audio thread. Reactive
//! NRT work - dropping freed buffers/streams and surfacing notifications - lives in the
//! [`Nrt`](plyphon_rt::nrt::Nrt) instead.
//!
//! The controller is the sole owner of every compiled `GraphDef` (current ones in `compiled`,
//! retired ones in `retiring`). An `Arc<GraphDef>` dropped on the audio thread (a freed graph's
//! clone, or a def-table slot cleared or replaced) is therefore never the final reference, so the
//! heavy drop never lands on the audio thread.
//! [`reap_retired_defs`](Controller::reap_retired_defs) drops each retired def on the control thread
//! once `Arc::strong_count` shows the audio thread is done with it, keeping `retiring` bounded by
//! live def state rather than leaking for the engine's lifetime.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use hashbrown::HashMap;

use rtrb::Producer;
use thiserror::Error;

use crate::synthdef::{SynthDef, SynthDefLibrary};
use plyphon_dsp::buffer::Buffer;
use plyphon_dsp::rate::RateInfo;
use plyphon_dsp::stream::{StreamConsumer, StreamProducer, cue, cue_recording};
use plyphon_rt::Options;
use plyphon_rt::command::{Command, CommandTime, TimedCommand};
use plyphon_rt::tree::AddAction;
use plyphon_unit::error::BuildError;
use plyphon_unit::graphdef::GraphDef;
use plyphon_unit::unit::registry::UnitRegistry;

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
    /// Per-def graph-rate overrides keyed by def name, absent for an ordinary def: `(reblock block
    /// size, oversample factor)` - scsynth's `Reblock(n)` / `Resample(n)`. The authored `SynthDef`
    /// carries no such field (it is constructed by literal in many places), so the setting rides
    /// alongside here and is threaded into `compile`.
    graph_rate: HashMap<String, (Option<usize>, usize)>,
    /// Current compiled def per name (also the controller's retained `Arc` for it).
    compiled: HashMap<String, Arc<GraphDef>>,
    /// Stable name -> `def_id`, assigned on first compile and reused across recompiles.
    def_ids: HashMap<String, u32>,
    /// Retired compiled defs (superseded by a redefinition, or freed) awaiting their last
    /// audio-thread reference to drop; drained by [`reap_retired_defs`](Self::reap_retired_defs).
    retiring: Vec<Arc<GraphDef>>,
    next_def_id: u32,
    audio: RateInfo,
    control: RateInfo,
    max_synthdefs: usize,
    max_wire_bufs: usize,
    max_unit_outputs: usize,
    tx: Producer<TimedCommand>,
    /// The time tag applied to commands while a scheduling window is open (see
    /// [`Controller::begin_scheduled`]); [`CommandTime::Immediate`] otherwise.
    schedule: CommandTime,
    next_id: i32,
}

impl Controller {
    pub(crate) fn new(
        options: &Options,
        audio: RateInfo,
        control: RateInfo,
        tx: Producer<TimedCommand>,
    ) -> Self {
        Controller {
            registry: UnitRegistry::with_builtins(),
            defs: SynthDefLibrary::new(),
            graph_rate: HashMap::new(),
            compiled: HashMap::new(),
            def_ids: HashMap::new(),
            retiring: Vec::new(),
            next_def_id: 0,
            audio,
            control,
            max_synthdefs: options.max_synthdefs,
            max_wire_bufs: options.max_wire_bufs,
            max_unit_outputs: options.max_unit_outputs,
            tx,
            schedule: CommandTime::Immediate,
            // Client node ids start above the root group (id 0).
            next_id: 1000,
        }
    }

    /// Open a scheduling window: commands issued until [`end_scheduled`](Self::end_scheduled) take
    /// effect at `time` (an absolute OSC/NTP time) instead of immediately, letting the OSC
    /// front-end honour a bundle's time tag. Def installs are always immediate regardless, so a
    /// scheduled `synth_new`'s def is resident before it fires.
    ///
    /// Returns the previous schedule time, so a caller applying a nested time-tagged bundle can
    /// restore the enclosing window with `begin_scheduled(prev)`.
    pub fn begin_scheduled(&mut self, time: CommandTime) -> CommandTime {
        core::mem::replace(&mut self.schedule, time)
    }

    /// Close the scheduling window opened by [`begin_scheduled`](Self::begin_scheduled); subsequent
    /// commands are immediate again.
    pub fn end_scheduled(&mut self) {
        self.schedule = CommandTime::Immediate;
    }

    /// Push `command` to the RT ring with the controller's current schedule time.
    fn send(&mut self, command: Command) -> Result<(), QueueFull> {
        self.push(self.schedule, command)
    }

    /// Push `command` to the RT ring to take effect immediately, ignoring any open scheduling
    /// window (for control-side bookkeeping that must be resident before later commands fire).
    fn send_now(&mut self, command: Command) -> Result<(), QueueFull> {
        self.push(CommandTime::Immediate, command)
    }

    /// The single point at which a command crosses to the RT side, wrapped with its time tag.
    fn push(&mut self, time: CommandTime, command: Command) -> Result<(), QueueFull> {
        self.tx
            .push(TimedCommand { time, command })
            .map_err(|_| QueueFull)
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
    /// form (see [`reap_retired_defs`](Self::reap_retired_defs)) and forces a recompile on next use.
    pub fn add_synthdef(&mut self, def: SynthDef) {
        self.graph_rate.remove(&def.name);
        self.insert_def(def);
    }

    /// Add (or replace) a synth definition that runs *reblocked* - its graph at a smaller control
    /// block `block_size` (scsynth's `Reblock(n)`): finer envelope/trigger timing and lower-latency
    /// `LocalIn`/`LocalOut` feedback. `block_size` must be a power of two no larger than the World
    /// block, else the deferred compile fails with [`BuildError::InvalidReblock`]. The plyphon analogue
    /// of a SynthDef carrying a `Reblock`, for callers that author defs in memory rather than parsing
    /// scsynth's binary v3 format.
    pub fn add_synthdef_reblocked(&mut self, def: SynthDef, block_size: usize) {
        self.graph_rate
            .insert(def.name.clone(), (Some(block_size), 1));
        self.insert_def(def);
    }

    /// Add (or replace) a synth definition that runs *oversampled* by `factor` (scsynth's
    /// `Resample(n)`): its graph runs at `factor`x the World sample rate, reducing aliasing in
    /// nonlinear units. `factor` must be a power of two, else the deferred compile fails with
    /// [`BuildError::InvalidResample`].
    pub fn add_synthdef_resampled(&mut self, def: SynthDef, factor: usize) {
        self.graph_rate.insert(def.name.clone(), (None, factor));
        self.insert_def(def);
    }

    /// Shared body of [`add_synthdef`]/[`add_synthdef_reblocked`]: retire any compiled form and store.
    fn insert_def(&mut self, def: SynthDef) {
        if let Some(old) = self.compiled.remove(&def.name) {
            self.retiring.push(old);
        }
        self.defs.insert(def);
        self.reap_retired_defs();
    }

    /// Look up a registered synth definition (e.g. to resolve a parameter index by name).
    pub fn synthdef(&self, name: &str) -> Option<&SynthDef> {
        self.defs.get(name)
    }

    /// Free the synth definition named `name` (scsynth's `/d_free`): empty its resident `World`
    /// def-table slot and forget its authored and compiled forms, so a later [`synth_new`](Self::synth_new)
    /// with this name fails until the def is added again. Returns whether a def by that name existed.
    ///
    /// The compiled `Arc` is retired (see [`reap_retired_defs`](Self::reap_retired_defs)), so the
    /// def-table slot's drop on the audio thread is never the final reference; the `def_id` stays
    /// reserved, so re-adding the same name reuses its slot.
    pub fn free_def(&mut self, name: &str) -> Result<bool, QueueFull> {
        let known = self.defs.get(name).is_some() || self.compiled.contains_key(name);
        if !known {
            return Ok(false);
        }
        // Clear the resident slot first: the slot is itself a strong ref, so reaping relies on it
        // dropping for a retired def's count to fall to 1 (see `reap_retired_defs`).
        if let Some(&def_id) = self.def_ids.get(name) {
            self.send_now(Command::FreeGraphDef { def_id })?;
        }
        if let Some(old) = self.compiled.remove(name) {
            self.retiring.push(old);
        }
        self.defs.remove(name);
        self.graph_rate.remove(name);
        self.reap_retired_defs();
        Ok(true)
    }

    /// Free every registered synth definition (scsynth's `/d_freeAll`).
    pub fn free_all_defs(&mut self) -> Result<(), QueueFull> {
        // Collect names first: `free_def` mutates the library, so it can't borrow it for iteration.
        let names: Vec<String> = self.defs.names().map(ToString::to_string).collect();
        for name in names {
            self.free_def(&name)?;
        }
        Ok(())
    }

    /// Drop every retired `GraphDef` the audio thread has finished with, reclaiming its memory on
    /// the control thread. Cheap and allocation-free; [`add_synthdef`](Self::add_synthdef) and
    /// [`free_def`](Self::free_def) call it opportunistically, but a long-running host should also
    /// call it periodically (e.g. once per control tick) so a def pinned only by a since-freed synth
    /// is reclaimed promptly rather than waiting for the next def change.
    ///
    /// A retired def's `Arc::strong_count` reaching 1 means this `Vec` holds the only remaining
    /// strong reference: the `World` has cleared or replaced its def-table slot *and* every synth
    /// that used it has been freed. Acting on that is sound because the audio thread can only mint a
    /// new `Arc<GraphDef>` clone via a def-table slot that still points at the def, and that slot is
    /// itself a strong reference - so while any new clone is possible the count is already >= 2.
    /// `strong_count == 1` is therefore a stable terminal state (no later clone can occur), and the
    /// relaxed snapshot can only ever read stale-high (reaping one cycle late), never stale-low.
    pub fn reap_retired_defs(&mut self) {
        self.retiring.retain(|def| Arc::strong_count(def) > 1);
    }

    /// The number of retired `GraphDef`s still awaiting reclamation (not yet dropped by
    /// [`reap_retired_defs`](Self::reap_retired_defs) because the audio thread still references
    /// them). A diagnostic counterpart to [`World::rt_memory_used`](plyphon_rt::world::World::rt_memory_used);
    /// it should stay bounded by live def state, never growing without bound.
    pub fn retired_defs_len(&self) -> usize {
        self.retiring.len()
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
        self.send(Command::AddSynth {
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
        let (reblock, resample) = self.graph_rate.get(def_name).copied().unwrap_or((None, 1));
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
                reblock,
                resample,
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
        self.send_now(Command::DefineGraphDef {
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
        self.send(Command::AddGroup { id, target, action })
    }

    /// Set control parameter `param` of node `node` to `value`.
    pub fn set_control(&mut self, node: i32, param: usize, value: f32) -> Result<(), QueueFull> {
        self.send(Command::SetControl { node, param, value })
    }

    /// Set control bus channel `bus` to `value` (scsynth's `/c_set`).
    pub fn set_control_bus(&mut self, bus: u32, value: f32) -> Result<(), QueueFull> {
        self.send(Command::SetControlBus { bus, value })
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
        self.send(Command::MapControl { node, param, bus })
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

    /// Map audio-rate parameter `param` of node `node` to audio `bus`, or unmap it with `None`
    /// (scsynth's `/n_mapa`). While mapped, the parameter's audio wire takes the bus each block. Only
    /// meaningful for an `AudioControl` parameter.
    pub fn map_control_audio(
        &mut self,
        node: i32,
        param: usize,
        bus: Option<u32>,
    ) -> Result<(), QueueFull> {
        self.send(Command::MapControlAudio { node, param, bus })
    }

    /// Map `count` consecutive audio-rate parameters of `node` (from `param`) to consecutive audio
    /// buses (from `bus`), or unmap them all with `None` (scsynth's `/n_mapan`).
    pub fn map_control_audio_n(
        &mut self,
        node: i32,
        param: usize,
        bus: Option<u32>,
        count: usize,
    ) -> Result<(), QueueFull> {
        for i in 0..count {
            let mapped = bus.map(|b| b + i as u32);
            self.map_control_audio(node, param + i, mapped)?;
        }
        Ok(())
    }

    /// Free node `node` (deeply for a group: the group and its whole subtree).
    pub fn free(&mut self, node: i32) -> Result<(), QueueFull> {
        self.send(Command::FreeNode { node })
    }

    /// Move node `node` to `target`/`action` (scsynth's `/g_head`, `/g_tail`, `/n_before`,
    /// `/n_after`).
    pub fn move_node(
        &mut self,
        node: i32,
        target: i32,
        action: AddAction,
    ) -> Result<(), QueueFull> {
        self.send(Command::MoveNode {
            node,
            target,
            action,
        })
    }

    /// Free every node in group `group`, leaving it empty (scsynth's `/g_freeAll`).
    pub fn free_all(&mut self, group: i32) -> Result<(), QueueFull> {
        self.send(Command::FreeAll { group })
    }

    /// Free every synth in group `group` and its subgroups, keeping the groups (`/g_deepFree`).
    pub fn deep_free(&mut self, group: i32) -> Result<(), QueueFull> {
        self.send(Command::DeepFree { group })
    }

    /// Pause or resume node `node` (scsynth's `/n_run`).
    pub fn node_run(&mut self, node: i32, run: bool) -> Result<(), QueueFull> {
        self.send(Command::NodeRun { node, run })
    }

    /// Clear every command still pending in the `World`'s scheduler (scsynth's `/clearSched`). Sent
    /// immediately, ignoring any open scheduling window.
    pub fn clear_sched(&mut self) -> Result<(), QueueFull> {
        self.send_now(Command::ClearSched)
    }

    /// Install (or replace) the buffer at `index` with an already-built buffer.
    ///
    /// The buffer is built off the audio thread (this is where any allocation or sample loading
    /// happens); the `World` only does an O(1) swap and routes any previous buffer to the trash ring.
    pub fn buffer_set(&mut self, index: usize, buffer: Box<Buffer>) -> Result<(), QueueFull> {
        self.send(Command::SetBuffer { index, buffer })
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
        self.send(Command::FreeBuffer { index })
    }

    /// Overwrite one sample of the buffer at `index`, in place (scsynth's `/b_set`/`/b_setn`).
    /// `sample` is a flat interleaved index (`frame * num_channels + channel`).
    pub fn buffer_set_sample(
        &mut self,
        index: usize,
        sample: usize,
        value: f32,
    ) -> Result<(), QueueFull> {
        self.send(Command::SetBufferSample {
            index,
            sample,
            value,
        })
    }

    /// Fill `count` consecutive samples of the buffer at `index` with `value`, from flat index
    /// `start` (scsynth's `/b_fill`).
    pub fn buffer_fill(
        &mut self,
        index: usize,
        start: usize,
        count: usize,
        value: f32,
    ) -> Result<(), QueueFull> {
        self.send(Command::FillBuffer {
            index,
            start,
            count,
            value,
        })
    }

    /// Overwrite the sample-rate metadata of the buffer at `index` (scsynth's `/b_setSampleRate`).
    pub fn buffer_set_sample_rate(
        &mut self,
        index: usize,
        sample_rate: f64,
    ) -> Result<(), QueueFull> {
        self.send(Command::SetBufferSampleRate { index, sample_rate })
    }

    /// Copy `count` interleaved samples from buffer `src` (flat `src_start`) into buffer `dst` (flat
    /// `dst_start`), on the audio thread (`/b_gen "copy"`). Overlap-safe; clamped to both buffers.
    pub fn buffer_copy_region(
        &mut self,
        dst: usize,
        dst_start: usize,
        src: usize,
        src_start: usize,
        count: usize,
    ) -> Result<(), QueueFull> {
        self.send(Command::CopyBufferRegion {
            dst,
            dst_start,
            src,
            src_start,
            count,
        })
    }

    /// Splice `src` into the live buffer at `index`, starting at flat (interleaved) index `dst_start`,
    /// leaving the buffer's dimensions unchanged (scsynth's `/b_read` into an already-allocated
    /// buffer). The buffer stays in place; `src` is copied in (clamped to both buffers) then trashed
    /// off the audio thread. Build `src` off the audio thread (the file region, already decoded).
    pub fn buffer_write_region(
        &mut self,
        index: usize,
        dst_start: usize,
        src: Box<Buffer>,
    ) -> Result<(), QueueFull> {
        self.send(Command::WriteBufferRegion {
            index,
            dst_start,
            src,
        })
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
        self.send(Command::CueStream { index, playback })?;
        Ok(producer)
    }

    /// Cue a disk-streaming *recording* buffer at `index`, for `DiskOut`.
    ///
    /// Allocates a queue of `num_chunks` chunks of `chunk_frames` frames each (off the audio thread)
    /// and installs the RT-side recording endpoint. Returns the [`StreamConsumer`] for a drainer to
    /// write to a sink; `DiskOut.ar(index, channels)` fills it.
    pub fn buffer_cue_write(
        &mut self,
        index: usize,
        channels: usize,
        sample_rate: f64,
        chunk_frames: usize,
        num_chunks: usize,
    ) -> Result<StreamConsumer, QueueFull> {
        let (recording, consumer) = cue_recording(channels, sample_rate, chunk_frames, num_chunks);
        self.send(Command::CueRecording { index, recording })?;
        Ok(consumer)
    }

    /// Close a left-open `DiskOut` recording at `index` (`/b_close`): flush its final partial chunk to
    /// the consumer (mirroring scsynth's `DiskOut_Dtor`), then free the slot. The dropped recording
    /// abandons the consumer, so a drainer reaches [`StreamConsumer::is_finished`](plyphon_dsp::stream::StreamConsumer::is_finished)
    /// once the flushed tail is drained. Immediate (`send_now`): a host action, never time-tagged.
    pub fn close_recording(&mut self, index: usize) -> Result<(), QueueFull> {
        self.send_now(Command::CloseRecording { index })
    }

    /// Snapshot the in-memory buffer at `index` to a host sink (scsynth's `/b_write`, `leaveOpen=0`).
    ///
    /// Unlike [`buffer_cue_write`](Self::buffer_cue_write) - which installs a *recording* slot for
    /// `DiskOut` to fill with live audio - this leaves the buffer in place and has the engine copy its
    /// existing samples into a fresh recording stream over the following blocks (back-pressured by the
    /// returned [`StreamConsumer`]'s recycle ring), so RT readers are undisturbed. Allocates the chunk
    /// pool off the audio thread; drain the returned consumer to a sink and poll
    /// [`StreamConsumer::is_finished`](plyphon_dsp::stream::StreamConsumer::is_finished) (or
    /// `StreamDrainer::is_done` in `plyphon-buffers`) for when the copy is complete.
    /// `channels`/`sample_rate` should match the buffer (they head the written file).
    /// Immediate (`send_now`): the copy-out is a host action the dispatcher drives after queueing, so
    /// it never rides a bundle time tag.
    pub fn buffer_write_out(
        &mut self,
        index: usize,
        channels: usize,
        sample_rate: f64,
        chunk_frames: usize,
        num_chunks: usize,
    ) -> Result<StreamConsumer, QueueFull> {
        let (recording, consumer) = cue_recording(channels, sample_rate, chunk_frames, num_chunks);
        self.send_now(Command::WriteBuffer { index, recording })?;
        Ok(consumer)
    }

    // --- Queries (getters). Each pushes a query the World answers over the reply ring, drained on the
    // NRT side. They take effect immediately (`send_now`) except `/sync`, which honors a bundle time
    // tag (`send`) so a scheduled barrier fires in its block. ---

    /// `/sync`: a command-stream barrier answered with `/synced <id>` once every earlier command has
    /// been applied.
    pub fn query_sync(&mut self, id: i32) -> Result<(), QueueFull> {
        self.send(Command::QuerySync { id })
    }

    /// `/status`: query engine counts and sample rate.
    pub fn query_status(&mut self) -> Result<(), QueueFull> {
        self.send_now(Command::QueryStatus)
    }

    /// `/rtMemoryStatus`: query rt-pool free/largest-chunk bytes.
    pub fn query_rt_memory(&mut self) -> Result<(), QueueFull> {
        self.send_now(Command::QueryRtMemory)
    }

    /// `/n_query`: query node `node`'s tree position.
    pub fn query_node(&mut self, node: i32) -> Result<(), QueueFull> {
        self.send_now(Command::QueryNode { node })
    }

    /// `/c_get`: query control bus channel `bus`.
    pub fn query_control_bus(&mut self, bus: u32) -> Result<(), QueueFull> {
        self.send_now(Command::QueryControlBus { bus })
    }

    /// `/c_getn`: query a run of `count` control buses from `start`.
    pub fn query_control_bus_range(&mut self, start: u32, count: u32) -> Result<(), QueueFull> {
        self.send_now(Command::QueryControlBusRange { start, count })
    }

    /// `/s_get`: query control `control` of synth `node`.
    pub fn query_synth_control(&mut self, node: i32, control: usize) -> Result<(), QueueFull> {
        self.send_now(Command::QuerySynthControl { node, control })
    }

    /// `/s_getn`: query a run of `count` controls of synth `node` from `control`.
    pub fn query_synth_control_range(
        &mut self,
        node: i32,
        control: usize,
        count: usize,
    ) -> Result<(), QueueFull> {
        self.send_now(Command::QuerySynthControlRange {
            node,
            control,
            count,
        })
    }

    /// `/b_get`: query flat sample `index` of buffer `buf`.
    pub fn query_buffer(&mut self, buf: usize, index: usize) -> Result<(), QueueFull> {
        self.send_now(Command::QueryBuffer { buf, index })
    }

    /// `/b_getn`: query a run of `count` samples of buffer `buf` from `index`.
    pub fn query_buffer_range(
        &mut self,
        buf: usize,
        index: usize,
        count: usize,
    ) -> Result<(), QueueFull> {
        self.send_now(Command::QueryBufferRange { buf, index, count })
    }

    /// `/g_queryTree`: stream the subtree under `group` (with control values if `flag`).
    pub fn query_tree(&mut self, group: i32, flag: bool) -> Result<(), QueueFull> {
        self.send_now(Command::QueryTree { group, flag })
    }

    /// `/g_dumpTree`: like [`query_tree`](Self::query_tree) but routed to a text sink, not an OSC reply.
    pub fn dump_tree(&mut self, group: i32, flag: bool) -> Result<(), QueueFull> {
        self.send_now(Command::DumpTree { group, flag })
    }

    /// `/n_trace`: dump the synth at `node`'s per-unit inputs/outputs for one block (scsynth's
    /// `Graph_CalcTrace`). The dump streams back over the reply ring as a node-tagged `Trace*` sequence
    /// the dispatcher routes to a host text sink; a group or unknown id is a no-op.
    pub fn trace_node(&mut self, node: i32) -> Result<(), QueueFull> {
        self.send_now(Command::TraceNode { node })
    }
}
