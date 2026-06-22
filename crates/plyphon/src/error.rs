//! Control-side error types. These never surface on the audio thread.

use core::fmt;

/// Errors from instantiating a [`crate::synthdef::SynthDef`] into a [`crate::synth::Synth`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The SynthDef references a UGen name not present in the registry.
    UnknownUgen(String),
    /// An input reference (parameter or UGen index) is out of range.
    BadInputRef,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::UnknownUgen(name) => write!(f, "unknown ugen: {name}"),
            BuildError::BadInputRef => write!(f, "input reference out of range"),
        }
    }
}

impl std::error::Error for BuildError {}
