//! `plyphon-dsp`: the shared DSP primitives the plyphon engine is built from.
//!
//! These are the signal-storage and synthesis substrate that both the units (which read and write
//! them while processing) and the [`World`](../plyphon_rt/struct.World.html) (which owns them) need:
//!
//! - [`rate`] - calculation rates ([`Rate`]) and the derived per-block constants ([`RateInfo`]).
//! - [`rng`] - the per-unit Taus88 random number generator embedded in unit state.
//! - [`wavetable`] - engine-owned wavetables (the sine table) lent to oscillator units.
//! - [`bus`] - the shared audio and control bus banks `In`/`Out` units read and write.
//! - [`buffer`] - in-memory sample buffers and the buffer table.
//! - [`stream`] - disk-streaming playback over lock-free chunk rings.
//!
//! The crate uses no `unsafe` (`bytemuck` keeps its own internal) and no global mutable state -
//! every primitive is owned by the engine and passed by argument - so it compiles for native and
//! `wasm32-unknown-unknown` alike.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[macro_use]
extern crate alloc;

pub mod buffer;
pub mod bus;
pub mod fft;
pub mod interp;
pub mod math;
pub mod ops;
pub mod rate;
pub mod rng;
pub mod stream;
pub mod wavetable;

pub use buffer::{Buffer, SpectrumCoord};
pub use fft::{FftTables, WindowType};
pub use rate::{Rate, RateInfo};
pub use stream::{Chunk, StreamConsumer, StreamProducer, StreamRecording, cue_recording};
