#![cfg_attr(not(feature = "std"), no_std)]
//! Fluent [`SynthDef`](plyphon::SynthDef) creation with sclang-style multichannel expansion.
//!
//! A [`SynthDefBuilder`] owns the units. Rate constructors (`SinOsc::ar(g)`, `SinOsc::kr(g)`) return
//! fluent *builders* with every input pre-set to its sclang default; named setters replace them
//! (`.freq(440.0)`), and the builder emits its unit(s) when it is *used* - passed as an input to
//! another Signal, combined with an operator, or finalized explicitly with [`UGenBuilder::signal`].
//!
//! An array input expands a UGen into parallel units (shorter arrays wrap, nested arrays nest),
//! multi-output units return their output proxies as a channel array, and the usual math
//! operators emit `BinaryOpUGen`/`UnaryOpUGen` - all exactly as in sclang.
//!
//! ```
//! use plyphon_synthdef::{SynthDefBuilder, UGenBuilder, Out, Pan2, Saw};
//!
//! let (def, ()) = SynthDefBuilder::build_with("detuned-saw", |g| {
//!     let freq = g.add_control_param("freq", 110.0);
//!     let saws = Saw::ar(g).freq([freq + 0.0, freq + 1.5]); // still a builder
//!     let sig = saws * 0.2;                                 // emits: 2 Saws, 2 muls
//!     Out::ar(g)
//!         .channels(Pan2::ar(g).input(sig).pos([-0.5, 0.5])) // 2 Pan2s, flat-spread
//!         .emit();
//! });
//! // 2 detune adds + 2 Saw + 2 level muls + 2 Pan2 + 1 Out
//! assert_eq!(def.units.len(), 9);
//! ```
//!
//! Builders are single-use; to feed one signal to several consumers, finalize it once
//! (`let sig = ...signal();`) and reuse the returned [`Signal`] handle (the compiler's
//! "use of moved value" error on a builder is the reminder).
//!
//! `AUTHORING.md` in this crate's directory is the authoring guide: what fails at compile time,
//! what panics at def-build time, and the silent footguns (chiefly: a sink like [`Out`] must
//! be finalized explicitly with `.emit()`, or it is never emitted).
//!
//! # Custom UGens
//!
//! Downstream crates can declare wrappers for units this crate hasn't wrapped yet (or override
//! a built-in one) with the same one-invocation [`ugen!`] macro used internally - see
//! `src/ugens.rs` for the built-in declarations' shape. Units with special input
//! shapes can be hand-written against the public core API: [`SynthDefBuilder::add`] (emission +
//! multichannel expansion), [`RateMode`], [`UGenInput`] (including [`UGenInput::flatten`] for
//! `Out`-like flat-spread sinks), the [`UGenBuilder`] trait for finalize-on-use, and
//! [`impl_builder_ops!`] for the math operators.

extern crate alloc;

mod builder;
mod macros;
mod ops;
mod ugens;

pub use builder::{Channel, RateMode, Signal, SynthDefBuilder, UGenInput};
pub use ops::UGenBuilder;
pub use ugens::*;

// Re-exported so downstream crates don't need a direct plyphon dependency just to declare params.
pub use plyphon::{Param, Rate};

// Implementation details the exported macros expand to. Not public API. The paths must be
// reachable from downstream expansions of `ugen!`/`impl_builder_ops!`.
#[doc(hidden)]
pub mod __private {
    pub use crate::ops::{binary_op, unary_op};
    pub use alloc::vec;
}

/// Build a multichannel [`UGenInput`] from mixed element types (an array literal requires one
/// type, so `[440.0, sig]` doesn't compile): `mce![440.0, sig]`.
#[macro_export]
macro_rules! mce {
    ($($x:expr),* $(,)?) => {
        $crate::UGenInput::Multi($crate::__private::vec![$($crate::UGenInput::from($x)),*])
    };
}
