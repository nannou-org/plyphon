//! `plyphon-rt`: the real-time half of the plyphon engine.
//!
//! [`World`] is the audio-thread engine: it owns the rt-pool, the resident def table, the buses, and
//! the node [`tree`], and is drained once per block by the audio callback via [`World::fill`]. It
//! never allocates, blocks, or locks. It draws on the modules here:
//!
//! - [`command`] - the messages crossing the control/RT boundary ([`Command`] in, [`Event`]/[`Trash`]
//!   out), all over lock-free rings.
//! - [`graph`] - a live synth instance, one rt-pool allocation processed once per control block.
//! - [`tree`] - the node tree (groups and synths) the engine walks in calc order.
//! - [`nrt`] - the non-real-time side ([`Nrt`]) that drops freed buffers and surfaces node events off
//!   the audio thread, the piece that *enables* the RT thread's guarantees.
//! - [`options`] - the engine configuration ([`Options`]) and the root group id.
//!
//! It builds on [`plyphon_unit`] (the units it runs and the [`GraphDef`](plyphon_unit::GraphDef)s it
//! instantiates) and [`plyphon_dsp`] (the buses, buffers, and wavetables it owns). The crate uses no
//! `unsafe` (the rt-pool and `bytemuck` keep theirs internal) and compiles for native and
//! `wasm32-unknown-unknown` alike.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[macro_use]
extern crate alloc;

pub mod command;
pub mod graph;
pub mod nrt;
pub mod options;
mod sched;
pub mod tree;
pub mod world;

pub use command::{Command, CommandTime, Event, TimedCommand, Trash};
pub use graph::Graph;
pub use nrt::Nrt;
pub use options::{Options, ROOT_GROUP_ID};
pub use tree::AddAction;
pub use world::World;
