//! plyphon: a pure-Rust rewrite of SuperCollider's `scsynth` audio engine core.
//!
//! A host builds the engine with [`engine()`], giving three handles for three roles:
//!
//! - [`Controller`] (control side): owns the SynthDef library and UGen registry, instantiates
//!   synths, and issues commands.
//! - [`World`] (real-time side): owns the buses and node tree, drained once per block by the audio
//!   callback via [`World::fill`].
//! - [`Nrt`] (non-real-time side): drops freed synths and surfaces [`Event`] notifications off the
//!   audio thread - the piece that *enables* the RT thread's guarantees. See the [`nrt`] module for
//!   the threading lifecycle.
//!
//! They communicate only through lock-free rings, so the audio thread never allocates, blocks, or
//! locks. The synthesis primitives live in [`rate`], [`wavetable`], [`bus`], [`buffer`], [`ugen`]
//! (the [`Ugen`] trait plus oscillators/filters/noise/ops, and [`DoneAction`]s), [`synth`], and
//! [`synthdef`].
//!
//! The whole crate is `unsafe`-free and free of global mutable state - everything a UGen needs is
//! passed by argument - and compiles for native and `wasm32-unknown-unknown` alike.

#![forbid(unsafe_code)]

pub mod buffer;
pub mod bus;
pub mod command;
pub mod controller;
pub mod engine;
pub mod error;
pub mod io;
pub mod nrt;
pub mod rate;
pub mod rng;
pub mod stream;
pub mod synth;
pub mod synthdef;
pub mod tree;
pub mod ugen;
pub mod wavetable;
pub mod world;

pub use buffer::Buffer;
pub use command::Event;
pub use controller::Controller;
pub use engine::{Options, ROOT_GROUP_ID, engine};
pub use io::Io;
pub use nrt::Nrt;
pub use rate::{Rate, RateInfo};
pub use stream::{Chunk, StreamProducer};
pub use synth::Synth;
pub use synthdef::{InputRef, Param, SynthDef, UgenSpec};
pub use tree::AddAction;
pub use ugen::{DoneAction, Ugen, UgenRegistry};
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
