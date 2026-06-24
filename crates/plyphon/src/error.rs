//! Control-side error types. These never surface on the audio thread.

use thiserror::Error;

/// Errors from instantiating a [`crate::synthdef::SynthDef`] into a [`crate::synth::Synth`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BuildError {
    /// The SynthDef references a UGen name not present in the registry.
    #[error("unknown ugen: {0}")]
    UnknownUgen(String),
    /// An input reference (parameter or UGen index) is out of range.
    #[error("input reference out of range")]
    BadInputRef,
    /// A UGen used a `special_index` operator that is not implemented.
    #[error("unsupported operator index: {0}")]
    UnsupportedOp(i16),
    /// A UGen was instantiated with the wrong number of inputs.
    #[error("wrong number of inputs for ugen")]
    WrongInputCount,
    /// The def needs more audio wire buffers than the engine's `max_wire_bufs` allows.
    #[error("def needs {needed} audio wires but the engine allows {limit}")]
    TooManyWires {
        /// Audio wires the def requires.
        needed: usize,
        /// The engine's `max_wire_bufs` limit.
        limit: usize,
    },
    /// A UGen has more outputs than the engine's `max_ugen_outputs` scratch allows.
    #[error("a ugen has {needed} outputs but the engine allows {limit}")]
    TooManyOutputs {
        /// Outputs the widest UGen requires.
        needed: usize,
        /// The engine's `max_ugen_outputs` limit.
        limit: usize,
    },
}
