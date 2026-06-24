//! plyphon: a pure-Rust rewrite of SuperCollider's `scsynth` audio engine core.
//!
//! A host builds the engine with [`engine()`], giving three handles for three roles:
//!
//! - [`Controller`] (control side): owns the SynthDef library and unit registry, instantiates
//!   synths, and issues commands.
//! - [`World`] (real-time side): owns the buses and node tree, drained once per block by the audio
//!   callback via [`World::fill`].
//! - [`Nrt`] (non-real-time side): drops freed synths and surfaces [`Event`] notifications off the
//!   audio thread - the piece that *enables* the RT thread's guarantees. See the [`nrt`] module for
//!   the threading lifecycle.
//!
//! They communicate only through lock-free rings, so the audio thread never allocates, blocks, or
//! locks. The synthesis primitives live in [`rate`], [`wavetable`], [`bus`], [`buffer`], [`unit`](mod@unit)
//! (the [`Unit`] trait plus oscillators/filters/noise/ops, and [`DoneAction`]s), [`graph`],
//! [`graphdef`], and [`synthdef`].
//!
//! The crate uses no `unsafe` itself (the rt-pool and `bytemuck` keep theirs internal) and no global
//! mutable state - everything a unit needs is passed by argument - and compiles for native and
//! `wasm32-unknown-unknown` alike.

#![forbid(unsafe_code)]

pub mod buffer;
pub mod bus;
pub mod command;
pub mod controller;
pub mod engine;
pub mod error;
pub mod graph;
pub mod graphdef;
pub mod nrt;
pub mod rate;
pub mod rng;
pub mod stream;
pub mod synthdef;
pub mod tree;
pub mod unit;
pub mod wavetable;
pub mod world;

pub use buffer::Buffer;
pub use command::Event;
pub use controller::{Controller, SynthNewError};
pub use engine::{Options, ROOT_GROUP_ID, engine};
pub use error::BuildError;
pub use graph::Graph;
pub use graphdef::GraphDef;
pub use nrt::Nrt;
pub use rate::{Rate, RateInfo};
pub use stream::{Chunk, StreamProducer};
pub use synthdef::{InputRef, Param, SynthDef, UnitSpec};
pub use tree::AddAction;
pub use unit::{
    BuildContext, BuiltUnit, DoneAction, InitCtx, Inputs, Outputs, ProcessCtx, Unit, UnitDef,
    UnitRegistry, unit_spec,
};
pub use world::World;

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
