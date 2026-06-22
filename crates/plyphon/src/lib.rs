//! plyphon: a pure-Rust rewrite of SuperCollider's `scsynth` audio engine core.
//!
//! A host builds the engine with [`engine()`], giving a [`Controller`] (control side: owns the
//! SynthDef library and UGen registry, instantiates synths, issues commands) and a [`World`] (the
//! real-time side: owns the buses and node tree, drained once per block by the audio callback via
//! [`World::fill`]). They communicate only through lock-free rings, so the audio thread never
//! allocates, blocks, or locks. The synthesis primitives live in [`rate`], [`wavetable`], [`bus`],
//! [`ugen`] (the [`Ugen`] trait plus `SinOsc`/`Out`), [`synth`], and [`synthdef`].
//!
//! The whole crate is `unsafe`-free and free of global mutable state - everything a UGen needs is
//! passed by argument - and compiles for native and `wasm32-unknown-unknown` alike.

#![forbid(unsafe_code)]

pub mod bus;
pub mod command;
pub mod controller;
pub mod engine;
pub mod error;
pub mod rate;
pub mod synth;
pub mod synthdef;
pub mod tree;
pub mod ugen;
pub mod wavetable;
pub mod world;

pub use controller::Controller;
pub use engine::{Options, ROOT_GROUP_ID, engine};
pub use rate::{Rate, RateInfo};
pub use synth::Synth;
pub use synthdef::{InputRef, Param, SynthDef, UgenSpec};
pub use tree::AddAction;
pub use ugen::{Ugen, UgenRegistry};
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
