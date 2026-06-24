//! The unit registry: maps unit names to their definitions for SynthDef compilation.
//!
//! This is plyphon's instance-based replacement for scsynth's global `gUnitDefLib` (a [`UnitDef`] is
//! plyphon's `UnitDef`). A [`UnitRegistry`] is owned by the control-side `Controller`; the audio
//! thread never touches it.

use std::collections::HashMap;

use crate::error::BuildError;
use crate::unit::BuiltUnit;
use crate::unit::band_limited::{PulseCtor, SawCtor};
use crate::unit::binary_op::BinaryOpCtor;
use crate::unit::disk_in::DiskInCtor;
use crate::unit::env::EnvGenCtor;
use crate::unit::filter::{ButterCtor, Kind};
use crate::unit::input::InCtor;
use crate::unit::lf::{ImpulseCtor, LFPulseCtor, LFSawCtor};
use crate::unit::line::LineCtor;
use crate::unit::noise::WhiteNoiseCtor;
use crate::unit::out::OutCtor;
use crate::unit::pan::Pan2Ctor;
use crate::unit::play_buf::PlayBufCtor;
use crate::unit::sin_osc::SinOscCtor;
use crate::unit::unary_op::UnaryOpCtor;
use crate::unit::util::{AmplitudeCtor, LagCtor, MulAddCtor};
use plyphon_dsp::rate::{Rate, RateInfo};

/// Build-time context for constructing a unit. Runs off the audio thread, so allocation is fine.
pub struct BuildContext<'a> {
    /// The resolved calc rate of each input, in order - drives input rate specialization.
    pub input_rates: &'a [Rate],
    /// The unit's own calculation rate (so it can specialize its output: a block vs one value).
    pub rate: Rate,
    /// Number of outputs the SynthDef assigns this unit (e.g. how many channels `In` reads).
    pub num_outputs: usize,
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// scsynth's `mSpecialIndex` (e.g. which binary/unary operator).
    pub special_index: i16,
    /// A seed for this unit's random number generator (distinct per unit and per synth instance).
    pub seed: u64,
}

/// A unit definition - scsynth's `UnitDef`. Builds a [`BuiltUnit`] (vtable + initial state image)
/// from a [`BuildContext`] when a SynthDef is compiled, off the audio thread.
pub trait UnitDef: Send + Sync {
    /// Build a unit, or fail (e.g. an unsupported operator or bad input count).
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError>;
}

/// Maps unit names to their definitions.
pub struct UnitRegistry {
    map: HashMap<String, Box<dyn UnitDef>>,
}

impl UnitRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        UnitRegistry {
            map: HashMap::new(),
        }
    }

    /// A registry pre-populated with the built-in units.
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
    pub fn register(&mut self, name: &str, def: Box<dyn UnitDef>) {
        self.map.insert(name.to_string(), def);
    }

    /// Look up a definition by name.
    pub fn get(&self, name: &str) -> Option<&dyn UnitDef> {
        self.map.get(name).map(|boxed| boxed.as_ref())
    }
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::new()
    }
}
