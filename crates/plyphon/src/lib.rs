//! plyphon: a pure-Rust rewrite of SuperCollider's `scsynth` audio engine core.
//!
//! A host builds the engine with [`engine()`], giving three handles for three roles:
//!
//! - [`Controller`] (control side): owns the SynthDef library and unit registry, instantiates
//!   synths, and issues commands.
//! - [`World`] (real-time side): owns the buses and node tree, drained once per block by the audio
//!   callback via [`World::fill`].
//! - [`Nrt`] (non-real-time side): drops freed synths and surfaces [`Event`] notifications off the
//!   audio thread - the piece that *enables* the RT thread's guarantees.
//!
//! They communicate only through lock-free rings, so the audio thread never allocates, blocks, or
//! locks. plyphon is split into focused crates, whose public items are all re-exported here:
//!
//! - [`plyphon_dsp`] - the shared DSP primitives (rates, RNG, wavetables, buses, buffers, streams).
//! - [`plyphon_unit`] - the [`Unit`] abstraction, the built-in units, and the compiled [`GraphDef`].
//! - [`plyphon_rt`] - the real-time [`World`] engine, node tree, command protocol, and [`Nrt`].
//! - this crate - the control-side [`Controller`], the authored [`SynthDef`] and its compilation, and
//!   the [`engine()`] builder that wires them together.
//!
//! The whole stack uses no `unsafe` (the rt-pool and `bytemuck` keep theirs internal) and no global
//! mutable state, and compiles for native and `wasm32-unknown-unknown` alike.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[macro_use]
extern crate alloc;

pub mod controller;
pub mod engine;
pub mod render;
pub mod synthdef;

pub use plyphon_dsp::{Buffer, Chunk, Rate, RateInfo, StreamProducer};
pub use plyphon_rt::{
    AddAction, CommandTime, Event, Graph, Nrt, Options, ROOT_GROUP_ID, Reply, World,
};
pub use plyphon_unit::{
    BuildContext, BuildError, BuiltUnit, DoneAction, GraphDef, InitCtx, Inputs, Outputs,
    ProcessCtx, Unit, UnitDef, UnitRegistry, unit_spec,
};

pub use controller::{Controller, QueueFull, SynthNewError};
pub use engine::engine;
pub use render::{Render, RenderUntil};
pub use synthdef::{InputRef, Param, SynthDef, UnitSpec};

/// Anything that can fill an interleaved, `channels`-wide block of `f32` output samples.
///
/// This is the engine's host-facing interface: a host (e.g. a `cpal` callback) hands us an
/// interleaved output buffer to fill. [`World`] implements it by reblocking its fixed control-block
/// size to the host's buffer length.
pub trait Source {
    /// Fill `output` (interleaved, `channels` samples per frame) with the next block of audio.
    fn fill(&mut self, output: &mut [f32], channels: usize);
}

impl Source for World {
    fn fill(&mut self, output: &mut [f32], channels: usize) {
        World::fill(self, output, channels);
    }
}
