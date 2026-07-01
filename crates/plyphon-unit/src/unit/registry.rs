//! The unit registry: maps unit names to their definitions for SynthDef compilation.
//!
//! This is plyphon's instance-based replacement for scsynth's global `gUnitDefLib` (a [`UnitDef`] is
//! plyphon's `UnitDef`). A [`UnitRegistry`] is owned by the control-side `Controller`; the audio
//! thread never touches it.

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use hashbrown::HashMap;

use crate::error::BuildError;
use crate::unit::band_limited::{PulseCtor, SawCtor};
use crate::unit::binary_op::BinaryOpCtor;
use crate::unit::buf_wr::BufWrCtor;
use crate::unit::chaos::{
    CuspNCtor, GbmanNCtor, LatoocarfianNCtor, LinCongNCtor, QuadNCtor, StandardNCtor,
};
use crate::unit::decay::{Decay2Ctor, DecayCtor};
use crate::unit::delay::{DelayCtor, FeedbackDelayCtor, Interp};
use crate::unit::demand::BuiltDemandUnit;
use crate::unit::demand::dbrown::DbrownCtor;
use crate::unit::demand::dbufrd::DbufrdCtor;
use crate::unit::demand::dbufwr::DbufwrCtor;
use crate::unit::demand::demand_ugen::DemandCtor;
use crate::unit::demand::dgeom::DgeomCtor;
use crate::unit::demand::dibrown::DibrownCtor;
use crate::unit::demand::diwhite::DiwhiteCtor;
use crate::unit::demand::dpoll::DpollCtor;
use crate::unit::demand::dseq::DseqCtor;
use crate::unit::demand::dseries::DseriesCtor;
use crate::unit::demand::duty::DutyCtor;
use crate::unit::demand::dwhite::DwhiteCtor;
use crate::unit::disk_in::DiskInCtor;
use crate::unit::disk_out::DiskOutCtor;
use crate::unit::dynamics::{CompanderCtor, DetectSilenceCtor};
use crate::unit::env::EnvGenCtor;
#[cfg(feature = "fft")]
use crate::unit::fft::{FftCtor, IfftCtor};
use crate::unit::filter::{ButterCtor, Kind};
use crate::unit::filter_simple::{
    APFCtor, BPZ2Ctor, BRZ2Ctor, Delay1Ctor, Delay2Ctor, HPZ1Ctor, HPZ2Ctor, LPZ1Ctor, LPZ2Ctor,
    SlewCtor, SlopeCtor,
};
use crate::unit::info::{BufInfoCtor, BufInfoKind, InfoCtor, InfoKind};
use crate::unit::input::InCtor;
use crate::unit::lf::{
    ImpulseCtor, LFCubCtor, LFParCtor, LFPulseCtor, LFSawCtor, LFTriCtor, SyncSawCtor, VarSawCtor,
};
use crate::unit::lf_noise::{
    LFClipNoiseCtor, LFDClipNoiseCtor, LFDNoise0Ctor, LFDNoise1Ctor, LFDNoise3Ctor, LFNoise0Ctor,
    LFNoise1Ctor, LFNoise2Ctor,
};
use crate::unit::line::{LineCtor, XLineCtor};
use crate::unit::local_io::{LocalInCtor, LocalOutCtor};
use crate::unit::node_ctl::{
    DoneCtor, FreeCtor, FreeSelfCtor, FreeSelfWhenDoneCtor, PauseCtor, PauseSelfCtor,
    PauseSelfWhenDoneCtor,
};
use crate::unit::noise::{
    BrownNoiseCtor, ClipNoiseCtor, Dust2Ctor, DustCtor, GrayNoiseCtor, PinkNoiseCtor,
    WhiteNoiseCtor,
};
use crate::unit::one_pole::{IntegratorCtor, LeakDCCtor, OnePoleCtor, OneZeroCtor};
use crate::unit::out::{OffsetOutCtor, OutCtor, ReplaceOutCtor};
use crate::unit::pan::{
    Balance2Ctor, LinPan2Ctor, LinXFade2Ctor, Pan2Ctor, Rotate2Ctor, XFade2Ctor,
};
use crate::unit::physical::{BallCtor, SpringCtor, TBallCtor};
use crate::unit::play_buf::PlayBufCtor;
#[cfg(feature = "fft")]
use crate::unit::pv_combine::{
    ComplexKind, PolarKind, PvComplexCtor, PvCopyCtor, PvCopyPhaseCtor, PvPolarCtor,
};
#[cfg(feature = "fft")]
use crate::unit::pv_mag_mul::PvMagMulCtor;
#[cfg(feature = "fft")]
use crate::unit::pv_mag_squared::PvMagSquaredCtor;
#[cfg(feature = "fft")]
use crate::unit::pv_ops::{
    MagKind, PvBrickWallCtor, PvConjCtor, PvLocalMaxCtor, PvMagThreshCtor, PvPhaseQuarterCtor,
};
use crate::unit::rate_conv::{A2KCtor, DcCtor, K2ACtor, T2ACtor, T2KCtor};
use crate::unit::record_buf::RecordBufCtor;
use crate::unit::resonant::{BPFCtor, BRFCtor, RHPFCtor, RLPFCtor, ResonzCtor, RingzCtor};
use crate::unit::send_reply::SendReplyCtor;
use crate::unit::send_trig::SendTrigCtor;
use crate::unit::shape::{
    InRangeCtor, InRectCtor, LinExpCtor, RangeKind, RangeShaperCtor, UnwrapCtor,
};
use crate::unit::sin_osc::{FSinOscCtor, SinOscCtor};
use crate::unit::test::{CheckBadValuesCtor, SanitizeCtor};
use crate::unit::timing::{
    PhasorCtor, PulseCountCtor, PulseDividerCtor, StepperCtor, SweepCtor, TimerCtor,
    ZeroCrossingCtor,
};
use crate::unit::trigger::{
    GateCtor, LatchCtor, SchmidtCtor, SetResetFFCtor, TDelayCtor, ToggleFFCtor, Trig1Ctor, TrigCtor,
};
use crate::unit::two_pole::{TwoPoleCtor, TwoZeroCtor};
use crate::unit::unary_op::UnaryOpCtor;
use crate::unit::util::{AmplitudeCtor, LagCtor, MulAddCtor};
use crate::unit::{BuiltUnit, InputSource};
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
    /// Each input's resolved source, in order (the same data `input_rates`/`input_units` are derived
    /// from). A unit that must size auxiliary memory at compile time reads a scalar input's value
    /// here via [`BuildContext::const_input`] - e.g. `DelayN`'s `maxdelaytime`.
    pub input_sources: &'a [InputSource],
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

impl BuildContext<'_> {
    /// The compile-time value of input `i` if it is a baked constant, else `None`. A unit that sizes
    /// auxiliary memory from a scalar input requires a constant here (it must reject a non-constant,
    /// e.g. with [`BuildError::AuxRequiresConstant`]), mirroring scsynth, where a delay's
    /// `maxdelaytime` is read once at ctor and can never change.
    pub fn const_input(&self, i: usize) -> Option<f32> {
        match self.input_sources.get(i) {
            Some(InputSource::Constant(v)) => Some(*v),
            _ => None,
        }
    }
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
        registry.register("ReplaceOut", Box::new(ReplaceOutCtor));
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
        registry.register("XLine", Box::new(XLineCtor));
        registry.register("Clip", Box::new(RangeShaperCtor(RangeKind::Clip)));
        registry.register("Wrap", Box::new(RangeShaperCtor(RangeKind::Wrap)));
        registry.register("Fold", Box::new(RangeShaperCtor(RangeKind::Fold)));
        registry.register("ModDif", Box::new(RangeShaperCtor(RangeKind::ModDif)));
        registry.register("InRange", Box::new(InRangeCtor));
        registry.register("InRect", Box::new(InRectCtor));
        registry.register("LinExp", Box::new(LinExpCtor));
        registry.register("Unwrap", Box::new(UnwrapCtor));
        registry.register("LPF", Box::new(ButterCtor(Kind::LowPass)));
        registry.register("HPF", Box::new(ButterCtor(Kind::HighPass)));
        registry.register("OnePole", Box::new(OnePoleCtor));
        registry.register("OneZero", Box::new(OneZeroCtor));
        registry.register("Integrator", Box::new(IntegratorCtor));
        registry.register("LeakDC", Box::new(LeakDCCtor));
        registry.register("TwoPole", Box::new(TwoPoleCtor));
        registry.register("TwoZero", Box::new(TwoZeroCtor));
        registry.register("Decay", Box::new(DecayCtor));
        registry.register("Decay2", Box::new(Decay2Ctor));
        registry.register("RLPF", Box::new(RLPFCtor));
        registry.register("RHPF", Box::new(RHPFCtor));
        registry.register("BPF", Box::new(BPFCtor));
        registry.register("BRF", Box::new(BRFCtor));
        registry.register("Resonz", Box::new(ResonzCtor));
        registry.register("Ringz", Box::new(RingzCtor));
        registry.register("LPZ1", Box::new(LPZ1Ctor));
        registry.register("HPZ1", Box::new(HPZ1Ctor));
        registry.register("LPZ2", Box::new(LPZ2Ctor));
        registry.register("HPZ2", Box::new(HPZ2Ctor));
        registry.register("BPZ2", Box::new(BPZ2Ctor));
        registry.register("BRZ2", Box::new(BRZ2Ctor));
        registry.register("Delay1", Box::new(Delay1Ctor));
        registry.register("Delay2", Box::new(Delay2Ctor));
        registry.register("Slope", Box::new(SlopeCtor));
        registry.register("Slew", Box::new(SlewCtor));
        registry.register("APF", Box::new(APFCtor));
        // Delay lines: plain (Delay*) and recirculating (Comb*/Allpass*), sharing one read kernel.
        registry.register("DelayN", Box::new(DelayCtor(Interp::None)));
        registry.register("DelayL", Box::new(DelayCtor(Interp::Lin)));
        registry.register("DelayC", Box::new(DelayCtor(Interp::Cubic)));
        registry.register(
            "CombN",
            Box::new(FeedbackDelayCtor {
                interp: Interp::None,
                allpass: false,
            }),
        );
        registry.register(
            "CombL",
            Box::new(FeedbackDelayCtor {
                interp: Interp::Lin,
                allpass: false,
            }),
        );
        registry.register(
            "CombC",
            Box::new(FeedbackDelayCtor {
                interp: Interp::Cubic,
                allpass: false,
            }),
        );
        registry.register(
            "AllpassN",
            Box::new(FeedbackDelayCtor {
                interp: Interp::None,
                allpass: true,
            }),
        );
        registry.register(
            "AllpassL",
            Box::new(FeedbackDelayCtor {
                interp: Interp::Lin,
                allpass: true,
            }),
        );
        registry.register(
            "AllpassC",
            Box::new(FeedbackDelayCtor {
                interp: Interp::Cubic,
                allpass: true,
            }),
        );
        registry.register("WhiteNoise", Box::new(WhiteNoiseCtor));
        registry.register("ClipNoise", Box::new(ClipNoiseCtor));
        registry.register("GrayNoise", Box::new(GrayNoiseCtor));
        registry.register("PinkNoise", Box::new(PinkNoiseCtor));
        registry.register("BrownNoise", Box::new(BrownNoiseCtor));
        registry.register("Dust", Box::new(DustCtor));
        registry.register("Dust2", Box::new(Dust2Ctor));
        // Low-frequency / dynamic noise (a new random value at an average `freq`).
        registry.register("LFNoise0", Box::new(LFNoise0Ctor));
        registry.register("LFNoise1", Box::new(LFNoise1Ctor));
        registry.register("LFNoise2", Box::new(LFNoise2Ctor));
        registry.register("LFClipNoise", Box::new(LFClipNoiseCtor));
        registry.register("LFDNoise0", Box::new(LFDNoise0Ctor));
        registry.register("LFDNoise1", Box::new(LFDNoise1Ctor));
        registry.register("LFDNoise3", Box::new(LFDNoise3Ctor));
        registry.register("LFDClipNoise", Box::new(LFDClipNoiseCtor));
        registry.register("CuspN", Box::new(CuspNCtor));
        registry.register("QuadN", Box::new(QuadNCtor));
        registry.register("LinCongN", Box::new(LinCongNCtor));
        registry.register("GbmanN", Box::new(GbmanNCtor));
        registry.register("StandardN", Box::new(StandardNCtor));
        registry.register("LatoocarfianN", Box::new(LatoocarfianNCtor));
        registry.register("PlayBuf", Box::new(PlayBufCtor));
        registry.register("DiskIn", Box::new(DiskInCtor));
        registry.register("DiskOut", Box::new(DiskOutCtor));
        registry.register("RecordBuf", Box::new(RecordBufCtor));
        registry.register("BufWr", Box::new(BufWrCtor));
        registry.register("LFSaw", Box::new(LFSawCtor));
        registry.register("LFPulse", Box::new(LFPulseCtor));
        registry.register("Impulse", Box::new(ImpulseCtor));
        registry.register("LFTri", Box::new(LFTriCtor));
        registry.register("LFPar", Box::new(LFParCtor));
        registry.register("LFCub", Box::new(LFCubCtor));
        registry.register("VarSaw", Box::new(VarSawCtor));
        registry.register("SyncSaw", Box::new(SyncSawCtor));
        registry.register("FSinOsc", Box::new(FSinOscCtor));
        registry.register("Saw", Box::new(SawCtor));
        registry.register("Pulse", Box::new(PulseCtor));
        registry.register("Pan2", Box::new(Pan2Ctor));
        registry.register("LinPan2", Box::new(LinPan2Ctor));
        registry.register("Balance2", Box::new(Balance2Ctor));
        registry.register("XFade2", Box::new(XFade2Ctor));
        registry.register("LinXFade2", Box::new(LinXFade2Ctor));
        registry.register("Rotate2", Box::new(Rotate2Ctor));
        registry.register("Spring", Box::new(SpringCtor));
        registry.register("Ball", Box::new(BallCtor));
        registry.register("TBall", Box::new(TBallCtor));
        registry.register("MulAdd", Box::new(MulAddCtor));
        registry.register("Lag", Box::new(LagCtor));
        registry.register("Amplitude", Box::new(AmplitudeCtor));
        registry.register("Compander", Box::new(CompanderCtor));
        registry.register("DetectSilence", Box::new(DetectSilenceCtor));
        registry.register("EnvGen", Box::new(EnvGenCtor));
        registry.register("SendTrig", Box::new(SendTrigCtor));
        registry.register("Trig", Box::new(TrigCtor));
        registry.register("Trig1", Box::new(Trig1Ctor));
        registry.register("TDelay", Box::new(TDelayCtor));
        registry.register("ToggleFF", Box::new(ToggleFFCtor));
        registry.register("SetResetFF", Box::new(SetResetFFCtor));
        registry.register("Latch", Box::new(LatchCtor));
        registry.register("Gate", Box::new(GateCtor));
        registry.register("Schmidt", Box::new(SchmidtCtor));
        registry.register("PulseCount", Box::new(PulseCountCtor));
        registry.register("PulseDivider", Box::new(PulseDividerCtor));
        registry.register("Stepper", Box::new(StepperCtor));
        registry.register("ZeroCrossing", Box::new(ZeroCrossingCtor));
        registry.register("Timer", Box::new(TimerCtor));
        registry.register("Sweep", Box::new(SweepCtor));
        registry.register("Phasor", Box::new(PhasorCtor));
        registry.register("SendReply", Box::new(SendReplyCtor));
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
        registry.register(
            "NumRunningSynths",
            Box::new(InfoCtor(InfoKind::NumRunningSynths)),
        );
        registry.register("NumBuffers", Box::new(InfoCtor(InfoKind::NumBuffers)));
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
        registry.register("T2K", Box::new(T2KCtor));
        // Diagnostic guards (NaN/inf/subnormal detection).
        registry.register("CheckBadValues", Box::new(CheckBadValuesCtor));
        registry.register("Sanitize", Box::new(SanitizeCtor));
        // Demand-rate consumers (normal calc-rate units that pull from the demand plan).
        registry.register("Duty", Box::new(DutyCtor));
        registry.register("Demand", Box::new(DemandCtor));
        // Demand-rate sources (the demand plan).
        registry.register_demand("Dseq", Box::new(DseqCtor));
        registry.register_demand("Dseries", Box::new(DseriesCtor));
        registry.register_demand("Dgeom", Box::new(DgeomCtor));
        registry.register_demand("Dwhite", Box::new(DwhiteCtor));
        registry.register_demand("Diwhite", Box::new(DiwhiteCtor));
        registry.register_demand("Dbrown", Box::new(DbrownCtor));
        registry.register_demand("Dibrown", Box::new(DibrownCtor));
        registry.register_demand("Dbufrd", Box::new(DbufrdCtor));
        registry.register_demand("Dbufwr", Box::new(DbufwrCtor));
        registry.register_demand("Dpoll", Box::new(DpollCtor));
        // FFT / spectral - only when built with the `fft` feature.
        #[cfg(feature = "fft")]
        {
            registry.register("FFT", Box::new(FftCtor));
            registry.register("IFFT", Box::new(IfftCtor));
            registry.register("PV_MagMul", Box::new(PvMagMulCtor));
            registry.register("PV_MagSquared", Box::new(PvMagSquaredCtor));
            registry.register("PV_MagAbove", Box::new(PvMagThreshCtor(MagKind::Above)));
            registry.register("PV_MagBelow", Box::new(PvMagThreshCtor(MagKind::Below)));
            registry.register("PV_MagClip", Box::new(PvMagThreshCtor(MagKind::Clip)));
            registry.register("PV_LocalMax", Box::new(PvLocalMaxCtor));
            registry.register("PV_PhaseShift90", Box::new(PvPhaseQuarterCtor(false)));
            registry.register("PV_PhaseShift270", Box::new(PvPhaseQuarterCtor(true)));
            registry.register("PV_BrickWall", Box::new(PvBrickWallCtor));
            registry.register("PV_Conj", Box::new(PvConjCtor));
            registry.register("PV_Add", Box::new(PvComplexCtor(ComplexKind::Add)));
            registry.register("PV_Mul", Box::new(PvComplexCtor(ComplexKind::Mul)));
            registry.register("PV_Div", Box::new(PvComplexCtor(ComplexKind::Div)));
            registry.register("PV_Max", Box::new(PvPolarCtor(PolarKind::Max)));
            registry.register("PV_Min", Box::new(PvPolarCtor(PolarKind::Min)));
            registry.register("PV_CopyPhase", Box::new(PvCopyPhaseCtor));
            registry.register("PV_Copy", Box::new(PvCopyCtor));
        }
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
