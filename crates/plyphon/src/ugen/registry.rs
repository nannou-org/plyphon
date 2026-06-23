//! The UGen registry: maps UGen names to constructors for SynthDef instantiation.
//!
//! This is plyphon's instance-based replacement for scsynth's global `gUnitDefLib`. A
//! [`UgenRegistry`] is owned by the control-side [`crate::controller::Controller`]; the audio thread
//! never touches it.

use std::collections::HashMap;

use crate::error::BuildError;
use crate::rate::{Rate, RateInfo};
use crate::ugen::Ugen;
use crate::ugen::binary_op::BinaryOpCtor;
use crate::ugen::filter::{ButterCtor, Kind};
use crate::ugen::input::InCtor;
use crate::ugen::line::LineCtor;
use crate::ugen::noise::WhiteNoiseCtor;
use crate::ugen::out::OutCtor;
use crate::ugen::sin_osc::SinOscCtor;
use crate::ugen::unary_op::UnaryOpCtor;

/// Build-time context for constructing a UGen. Runs off the audio thread, so allocation is fine.
pub struct BuildContext<'a> {
    /// The resolved calc rate of each input, in order - drives input rate specialization.
    pub input_rates: &'a [Rate],
    /// The UGen's own calculation rate (so it can specialize its output: a block vs one value).
    pub rate: Rate,
    /// Number of outputs the SynthDef assigns this UGen (e.g. how many channels `In` reads).
    pub num_outputs: usize,
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// scsynth's `mSpecialIndex` (e.g. which binary/unary operator).
    pub special_index: i16,
    /// A seed for this UGen's random number generator (distinct per UGen and per synth instance).
    pub seed: u64,
}

/// Constructs a [`Ugen`] from a [`BuildContext`] during SynthDef instantiation.
pub trait UgenCtor: Send + Sync {
    /// Build a UGen instance, or fail (e.g. an unsupported operator or bad input count).
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError>;
}

/// Maps UGen names to their constructors.
pub struct UgenRegistry {
    map: HashMap<String, Box<dyn UgenCtor>>,
}

impl UgenRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        UgenRegistry {
            map: HashMap::new(),
        }
    }

    /// A registry pre-populated with the built-in UGens.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register("SinOsc", Box::new(SinOscCtor));
        registry.register("Out", Box::new(OutCtor));
        registry.register("In", Box::new(InCtor));
        registry.register("BinaryOpUGen", Box::new(BinaryOpCtor));
        registry.register("UnaryOpUGen", Box::new(UnaryOpCtor));
        registry.register("Line", Box::new(LineCtor));
        registry.register("LPF", Box::new(ButterCtor(Kind::LowPass)));
        registry.register("HPF", Box::new(ButterCtor(Kind::HighPass)));
        registry.register("WhiteNoise", Box::new(WhiteNoiseCtor));
        registry
    }

    /// Register `ctor` under `name`, replacing any existing entry.
    pub fn register(&mut self, name: &str, ctor: Box<dyn UgenCtor>) {
        self.map.insert(name.to_string(), ctor);
    }

    /// Look up a constructor by name.
    pub fn get(&self, name: &str) -> Option<&dyn UgenCtor> {
        self.map.get(name).map(|boxed| boxed.as_ref())
    }
}

impl Default for UgenRegistry {
    fn default() -> Self {
        Self::new()
    }
}
