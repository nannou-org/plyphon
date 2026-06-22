//! The UGen registry: maps UGen names to constructors for SynthDef instantiation.
//!
//! This is plyphon's instance-based replacement for scsynth's global `gUnitDefLib`. A
//! [`UgenRegistry`] is owned by the control-side [`crate::controller::Controller`]; the audio thread
//! never touches it.

use std::collections::HashMap;

use crate::rate::{Rate, RateInfo};
use crate::ugen::Ugen;
use crate::ugen::out::OutCtor;
use crate::ugen::sin_osc::SinOscCtor;

/// Build-time context for constructing a UGen. Runs off the audio thread, so allocation is fine.
pub struct BuildContext<'a> {
    /// The resolved calc rate of each input, in order - drives rate specialization.
    pub input_rates: &'a [Rate],
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// scsynth's `mSpecialIndex` (e.g. which binary op). Unused by the current UGens.
    pub special_index: i16,
}

/// Constructs a [`Ugen`] from a [`BuildContext`] during SynthDef instantiation.
pub trait UgenCtor: Send + Sync {
    /// Build a UGen instance.
    fn build(&self, ctx: &BuildContext<'_>) -> Box<dyn Ugen>;
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
