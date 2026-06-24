//! The UGen registry: maps UGen names to their definitions for SynthDef compilation.
//!
//! This is plyphon's instance-based replacement for scsynth's global `gUnitDefLib` (a [`UgenDef`] is
//! plyphon's `UnitDef`). A [`UgenRegistry`] is owned by the control-side
//! [`crate::controller::Controller`]; the audio thread never touches it.

use std::collections::HashMap;

use crate::error::BuildError;
use crate::rate::{Rate, RateInfo};
use crate::ugen::BuiltUgen;
use crate::ugen::band_limited::{PulseCtor, SawCtor};
use crate::ugen::binary_op::BinaryOpCtor;
use crate::ugen::disk_in::DiskInCtor;
use crate::ugen::env::EnvGenCtor;
use crate::ugen::filter::{ButterCtor, Kind};
use crate::ugen::input::InCtor;
use crate::ugen::lf::{ImpulseCtor, LFPulseCtor, LFSawCtor};
use crate::ugen::line::LineCtor;
use crate::ugen::noise::WhiteNoiseCtor;
use crate::ugen::out::OutCtor;
use crate::ugen::pan::Pan2Ctor;
use crate::ugen::play_buf::PlayBufCtor;
use crate::ugen::sin_osc::SinOscCtor;
use crate::ugen::unary_op::UnaryOpCtor;
use crate::ugen::util::{AmplitudeCtor, LagCtor, MulAddCtor};

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

/// A UGen definition - scsynth's `UnitDef`. Builds a [`BuiltUgen`] (vtable + initial state image)
/// from a [`BuildContext`] when a SynthDef is compiled, off the audio thread.
pub trait UgenDef: Send + Sync {
    /// Build a UGen, or fail (e.g. an unsupported operator or bad input count).
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError>;
}

/// Maps UGen names to their definitions.
pub struct UgenRegistry {
    map: HashMap<String, Box<dyn UgenDef>>,
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
        registry.register("PlayBuf", Box::new(PlayBufCtor));
        registry.register("DiskIn", Box::new(DiskInCtor));
        registry.register("LFSaw", Box::new(LFSawCtor));
        registry.register("LFPulse", Box::new(LFPulseCtor));
        registry.register("Impulse", Box::new(ImpulseCtor));
        registry.register("Saw", Box::new(SawCtor));
        registry.register("Pulse", Box::new(PulseCtor));
        registry.register("Pan2", Box::new(Pan2Ctor));
        registry.register("MulAdd", Box::new(MulAddCtor));
        registry.register("Lag", Box::new(LagCtor));
        registry.register("Amplitude", Box::new(AmplitudeCtor));
        registry.register("EnvGen", Box::new(EnvGenCtor));
        registry
    }

    /// Register `def` under `name`, replacing any existing entry.
    pub fn register(&mut self, name: &str, def: Box<dyn UgenDef>) {
        self.map.insert(name.to_string(), def);
    }

    /// Look up a definition by name.
    pub fn get(&self, name: &str) -> Option<&dyn UgenDef> {
        self.map.get(name).map(|boxed| boxed.as_ref())
    }
}

impl Default for UgenRegistry {
    fn default() -> Self {
        Self::new()
    }
}
