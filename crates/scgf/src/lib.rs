//! `scgf`: a parser and encoder for SuperCollider's binary SynthDef format (SCgf).
//!
//! This crate models the file format faithfully and knows nothing about any synthesis engine -
//! [`parse`] turns bytes into a [`SynthDefFile`] and [`encode`] turns one back into bytes. Both
//! format versions are supported on read (v1 uses `int16` counts/indices, v2 uses `int32`);
//! [`encode`] always writes v2.
//!
//! Consumers (such as `plyphon`) interpret the parsed graph - e.g. folding `Control` UGens into
//! their own parameter model.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

mod read;
mod write;

pub use read::parse;
pub use write::encode;

/// The calculation rate of a UGen or one of its outputs.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Rate {
    /// Computed once (a constant).
    Scalar,
    /// One value per control block.
    Control,
    /// One value per sample.
    Audio,
    /// Pulled on demand.
    Demand,
}

impl Rate {
    /// The SCgf rate code for this rate (`0`/`1`/`2`/`3`).
    pub fn code(self) -> i8 {
        match self {
            Rate::Scalar => 0,
            Rate::Control => 1,
            Rate::Audio => 2,
            Rate::Demand => 3,
        }
    }

    /// The rate for an SCgf rate code, or `None` if out of range.
    pub fn from_code(code: i8) -> Option<Rate> {
        match code {
            0 => Some(Rate::Scalar),
            1 => Some(Rate::Control),
            2 => Some(Rate::Audio),
            3 => Some(Rate::Demand),
            _ => None,
        }
    }
}

/// A parsed SCgf file: a format version and the definitions it contains.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthDefFile {
    /// File format version (1, 2, or 3).
    pub version: i32,
    /// The synth definitions in the file.
    pub defs: Vec<SynthDef>,
}

/// A single synth definition.
///
/// The reblock/resample fields are the raw version-3 trailing values (defaulting to "no reblock, no
/// oversample" for v1/v2). `block_size` is `0` for no reblock, `N > 0` for a fixed control block, or
/// `-1` for a control-driven block size (the value in `block_size_index`); `resample_factor` is `1.0`
/// (or `0.0`) for no oversample, `> 1.0` for a fixed factor, or `-1.0` for a control-driven factor
/// (`resample_index`). Interpreting these into an engine setting is the consumer's job.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthDef {
    /// Definition name.
    pub name: String,
    /// Constant values referenced by UGen inputs.
    pub constants: Vec<f32>,
    /// Initial control (parameter) values.
    pub param_values: Vec<f32>,
    /// Named parameters, each pointing at an index into `param_values`.
    pub param_names: Vec<ParamName>,
    /// UGens, in topological calc order.
    pub ugens: Vec<Ugen>,
    /// Variants (named alternative parameter sets).
    pub variants: Vec<Variant>,
    /// Reblock control block (scsynth's `Reblock`): `0` none, `N` fixed, `-1` control-driven.
    pub block_size: i32,
    /// Synth-control index for a control-driven `block_size` (`-1`); else `0`.
    pub block_size_index: u32,
    /// Resample factor (scsynth's `Resample`): `1.0`/`0.0` none, `> 1.0` fixed, `-1.0` control-driven.
    pub resample_factor: f32,
    /// Synth-control index for a control-driven `resample_factor` (`-1.0`); else `0`.
    pub resample_index: u32,
}

impl Default for SynthDef {
    /// An empty def with the "ordinary" rate fields - no reblock, no oversample (`resample_factor`
    /// `1.0`). Lets a caller build a def by literal and fill the reblock/resample tail with
    /// `..Default::default()`.
    fn default() -> Self {
        SynthDef {
            name: String::new(),
            constants: Vec::new(),
            param_values: Vec::new(),
            param_names: Vec::new(),
            ugens: Vec::new(),
            variants: Vec::new(),
            block_size: 0,
            block_size_index: 0,
            resample_factor: 1.0,
            resample_index: 0,
        }
    }
}

/// A named parameter: a name and its index into [`SynthDef::param_values`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamName {
    /// Parameter name.
    pub name: String,
    /// Index into the parameter value array.
    pub index: u32,
}

/// One UGen within a [`SynthDef`].
#[derive(Clone, Debug, PartialEq)]
pub struct Ugen {
    /// UGen class name (e.g. `"SinOsc"`).
    pub name: String,
    /// Calculation rate.
    pub rate: Rate,
    /// Class-specific selector (e.g. which binary op).
    pub special_index: i16,
    /// Inputs, in order.
    pub inputs: Vec<Input>,
    /// Output rates, one per output.
    pub outputs: Vec<Rate>,
}

/// A UGen input: either a constant or another UGen's output.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Input {
    /// A constant, referenced by index into [`SynthDef::constants`].
    Constant {
        /// Index into the constants array.
        index: u32,
    },
    /// The output of an earlier UGen.
    Ugen {
        /// Index of the source UGen.
        ugen: u32,
        /// Which output of that UGen.
        output: u32,
    },
}

/// A named variant: an alternative set of parameter values.
#[derive(Clone, Debug, PartialEq)]
pub struct Variant {
    /// Variant name.
    pub name: String,
    /// One value per parameter.
    pub values: Vec<f32>,
}

/// An error parsing or encoding an SCgf buffer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The buffer ended before a field could be read.
    #[error("unexpected end of SCgf buffer")]
    Truncated,
    /// The buffer does not start with the `SCgf` magic.
    #[error("not an SCgf buffer (bad magic)")]
    BadMagic,
    /// The format version is not one of 1, 2, or 3.
    #[error("unsupported SCgf version: {0}")]
    UnsupportedVersion(i32),
    /// A calculation-rate byte was out of range.
    #[error("invalid calc-rate code: {0}")]
    BadRate(i8),
    /// A string was longer than the 255-byte SCgf limit (encoding only).
    #[error("string exceeds the 255-byte SCgf limit")]
    NameTooLong,
}
