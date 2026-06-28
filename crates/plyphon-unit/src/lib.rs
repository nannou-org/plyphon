//! `plyphon-unit`: the unit-generator abstraction the plyphon engine executes.
//!
//! A [`Unit`] is plyphon's port of scsynth's server-side `Unit`: a `Pod` state plus a
//! [`process`](Unit::process) calc function, built off the audio thread by a [`UnitDef`] (looked up
//! in a [`UnitRegistry`]) and run once per control block. The [`unit`](mod@unit) module holds the
//! trait, the per-block [`ProcessCtx`]/[`InitCtx`], and the built-in units (oscillators, filters,
//! noise, ops).
//!
//! A `SynthDef` compiles to a [`GraphDef`] (the [`graphdef`] module) - the immutable, shareable
//! template the real-time engine instantiates. [`BuildError`] (the [`error`] module) is what
//! compilation returns on failure.
//!
//! The units operate on the shared primitives in [`plyphon_dsp`] (buses, buffers, wavetables,
//! rates). The crate uses no `unsafe` and no global mutable state, and compiles for native and
//! `wasm32-unknown-unknown` alike.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod error;
pub mod graphdef;
pub mod unit;

pub use error::BuildError;
pub use graphdef::GraphDef;
pub use unit::{
    Aux, BuildContext, BuiltUnit, DoneAction, InitCtx, Inputs, NodeMsg, NodeMsgKind, NodeMsgSink,
    Outputs, ProcessCtx, Trigger, TriggerSink, Unit, UnitDef, UnitRegistry, unit_spec,
    unit_spec_aux,
};
