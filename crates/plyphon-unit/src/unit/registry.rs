//! The unit registry: maps unit names to their definitions for SynthDef compilation.
//!
//! This is plyphon's instance-based replacement for scsynth's global `gUnitDefLib` (a [`UnitDef`] is
//! plyphon's `UnitDef`). A [`UnitRegistry`] is owned by the control-side `Controller`; the audio
//! thread never touches it.

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use hashbrown::HashMap;

use crate::error::BuildError;
use crate::unit::BuiltUnit;
use crate::unit::band_limited::{PulseCtor, SawCtor};
use crate::unit::binary_op::BinaryOpCtor;
use crate::unit::demand::BuiltDemandUnit;
use crate::unit::demand::demand_ugen::DemandCtor;
use crate::unit::demand::dseq::DseqCtor;
use crate::unit::demand::dseries::DseriesCtor;
use crate::unit::demand::duty::DutyCtor;
use crate::unit::demand::dwhite::DwhiteCtor;
use crate::unit::disk_in::DiskInCtor;
use crate::unit::env::EnvGenCtor;
use crate::unit::filter::{ButterCtor, Kind};
use crate::unit::info::{BufInfoCtor, BufInfoKind, InfoCtor, InfoKind};
use crate::unit::input::InCtor;
use crate::unit::lf::{ImpulseCtor, LFPulseCtor, LFSawCtor};
use crate::unit::line::LineCtor;
use crate::unit::local_io::{LocalInCtor, LocalOutCtor};
use crate::unit::node_ctl::{
    DoneCtor, FreeCtor, FreeSelfCtor, FreeSelfWhenDoneCtor, PauseCtor, PauseSelfCtor,
    PauseSelfWhenDoneCtor,
};
use crate::unit::noise::WhiteNoiseCtor;
use crate::unit::out::{OffsetOutCtor, OutCtor};
use crate::unit::pan::Pan2Ctor;
use crate::unit::play_buf::PlayBufCtor;
use crate::unit::rate_conv::{A2KCtor, DcCtor, K2ACtor, T2ACtor};
use crate::unit::send_trig::SendTrigCtor;
use crate::unit::sin_osc::SinOscCtor;
use crate::unit::unary_op::UnaryOpCtor;
use crate::unit::util::{AmplitudeCtor, LagCtor, MulAddCtor};
use plyphon_dsp::rate::{Rate, RateInfo};

/// Build-time context for constructing a unit. Runs off the audio thread, so allocation is fine.
pub struct BuildContext<'a> {
    /// The resolved calc rate of each input, in order - drives input rate specialization.
    pub input_rates: &'a [Rate],
    /// For each input, the calc-unit index of the unit that produces it (if it comes from a calc
    /// unit), else `None` - plyphon's analogue of scsynth's `mInput[i]->mFromUnit`. A done-watching
    /// unit (`Done`/`FreeSelfWhenDone`/`PauseSelfWhenDone`) reads input 0's entry to find the unit
    /// whose done flag it observes.
    pub input_units: &'a [Option<u32>],
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

/// A demand-rate unit definition - the demand-plan analogue of [`UnitDef`]. Builds a
/// [`BuiltDemandUnit`] (pull/reset/seed vtable + initial state image) when a SynthDef is compiled.
pub trait DemandUnitDef: Send + Sync {
    /// Build a demand unit, or fail.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError>;
}

/// Maps unit names to their definitions. Demand-rate units live in a separate map: a SynthDef unit at
/// [`Rate::Demand`] is looked up there, every other rate in `map`.
pub struct UnitRegistry {
    map: HashMap<String, Box<dyn UnitDef>>,
    demand_map: HashMap<String, Box<dyn DemandUnitDef>>,
}

impl UnitRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        UnitRegistry {
            map: HashMap::new(),
            demand_map: HashMap::new(),
        }
    }

    /// A registry pre-populated with the built-in units.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register("SinOsc", Box::new(SinOscCtor));
        registry.register("Out", Box::new(OutCtor));
        registry.register("OffsetOut", Box::new(OffsetOutCtor));
        registry.register("In", Box::new(InCtor));
        // `InFeedback` reads a global audio bus tolerating a later writer; plyphon's `In` already
        // reads current bus contents with no "written-this-block" check, so it is the same unit.
        registry.register("InFeedback", Box::new(InCtor));
        registry.register("LocalIn", Box::new(LocalInCtor));
        registry.register("LocalOut", Box::new(LocalOutCtor));
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
        registry.register("SendTrig", Box::new(SendTrigCtor));
        // Info: engine constants and per-buffer info.
        registry.register("SampleRate", Box::new(InfoCtor(InfoKind::SampleRate)));
        registry.register("SampleDur", Box::new(InfoCtor(InfoKind::SampleDur)));
        registry.register(
            "RadiansPerSample",
            Box::new(InfoCtor(InfoKind::RadiansPerSample)),
        );
        registry.register("ControlRate", Box::new(InfoCtor(InfoKind::ControlRate)));
        registry.register("ControlDur", Box::new(InfoCtor(InfoKind::ControlDur)));
        registry.register(
            "NumOutputBuses",
            Box::new(InfoCtor(InfoKind::NumOutputBuses)),
        );
        registry.register("NumInputBuses", Box::new(InfoCtor(InfoKind::NumInputBuses)));
        registry.register("NumAudioBuses", Box::new(InfoCtor(InfoKind::NumAudioBuses)));
        registry.register(
            "NumControlBuses",
            Box::new(InfoCtor(InfoKind::NumControlBuses)),
        );
        registry.register("BufFrames", Box::new(BufInfoCtor(BufInfoKind::Frames)));
        registry.register("BufChannels", Box::new(BufInfoCtor(BufInfoKind::Channels)));
        registry.register("BufSamples", Box::new(BufInfoCtor(BufInfoKind::Samples)));
        registry.register(
            "BufSampleRate",
            Box::new(BufInfoCtor(BufInfoKind::SampleRate)),
        );
        registry.register(
            "BufRateScale",
            Box::new(BufInfoCtor(BufInfoKind::RateScale)),
        );
        registry.register("BufDur", Box::new(BufInfoCtor(BufInfoKind::Dur)));
        // In-graph node control.
        registry.register("FreeSelf", Box::new(FreeSelfCtor));
        registry.register("PauseSelf", Box::new(PauseSelfCtor));
        registry.register("Done", Box::new(DoneCtor));
        registry.register("FreeSelfWhenDone", Box::new(FreeSelfWhenDoneCtor));
        registry.register("PauseSelfWhenDone", Box::new(PauseSelfWhenDoneCtor));
        registry.register("Free", Box::new(FreeCtor));
        registry.register("Pause", Box::new(PauseCtor));
        // Rate conversion.
        registry.register("DC", Box::new(DcCtor));
        registry.register("K2A", Box::new(K2ACtor));
        registry.register("A2K", Box::new(A2KCtor));
        registry.register("T2A", Box::new(T2ACtor));
        // Demand-rate consumers (normal calc-rate units that pull from the demand plan).
        registry.register("Duty", Box::new(DutyCtor));
        registry.register("Demand", Box::new(DemandCtor));
        // Demand-rate sources (the demand plan).
        registry.register_demand("Dseq", Box::new(DseqCtor));
        registry.register_demand("Dseries", Box::new(DseriesCtor));
        registry.register_demand("Dwhite", Box::new(DwhiteCtor));
        registry
    }

    /// Register `def` under `name`, replacing any existing entry.
    pub fn register(&mut self, name: &str, def: Box<dyn UnitDef>) {
        self.map.insert(name.to_string(), def);
    }

    /// Register a demand-rate `def` under `name`, replacing any existing entry.
    pub fn register_demand(&mut self, name: &str, def: Box<dyn DemandUnitDef>) {
        self.demand_map.insert(name.to_string(), def);
    }

    /// Look up a definition by name.
    pub fn get(&self, name: &str) -> Option<&dyn UnitDef> {
        self.map.get(name).map(|boxed| boxed.as_ref())
    }

    /// Look up a demand-rate definition by name.
    pub fn get_demand(&self, name: &str) -> Option<&dyn DemandUnitDef> {
        self.demand_map.get(name).map(|boxed| boxed.as_ref())
    }
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::new()
    }
}
