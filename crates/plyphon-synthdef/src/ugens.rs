//! UGen wrappers, fluent-builder style: `SinOsc::kr(g).freq(440.0).phase(1.0)`.
//!
//! A rate constructor returns a *builder* holding the builder reference and the input trees,
//! pre-filled with the sclang defaults. Nothing is emitted into the def until the builder is
//! finalized - by being used as an input to another UGen (or an operator), or explicitly with
//! [`UGenBuilder::signal`]. Deferring emission is what lets a later setter change the channel count
//! (`.freq(vec![440.0, 443.0])`) without splitting an already-emitted unit; expansion happens
//! once, at finalize, through [`SynthDefBuilder::add`].
//!
//! Builders are single-use (finalize-on-use consumes them); to feed one signal to several
//! consumers, finalize first (`let sig = ...signal();`) and reuse the [`Signal`] handle.
//!
//! Most wrappers are one [`ugen!`](crate::ugen) invocation (the macro is exported, so downstream
//! crates can declare custom UGens the same way); units with special input shapes (`In`, `Out`)
//! are hand-written.
//!
//! Multi-output units still to be wrapped (fixed count): `LinPan2`, `Balance2`, `Rotate2`,
//! `Hilbert`, `FreeVerb2`, `GVerb` (2), `PanB2`, `BiPanB2` (3), `Pan4`, `PanB` (4). Variable count
//! (N chosen at def time, like [`In`]): `LocalIn`, `DiskIn`, `VDiskIn`, `PlayBuf`, `BufRd`,
//! `DecodeB2`, `PanAz`, `GrainSin`, `GrainFM`, `GrainIn`, `GrainBuf`, `TGrains`, `Warp1`, `Demand`.

use plyphon::Rate;

use crate::builder::{RateMode, Signal, SynthDefBuilder, UGenInput};
use crate::ops::UGenBuilder;
use crate::{impl_builder_ops, ugen};

ugen!(
    /// Band-limited sawtooth.
    "Saw" => Saw [ar: Rate::Audio, kr: Rate::Control](
        /// Frequency in Hz (default 440).
        freq = 440.0,
    ) -> 1
);

ugen!(
    /// Sine oscillator.
    "SinOsc" => SinOsc [ar: Rate::Audio, kr: Rate::Control](
        /// Frequency in Hz (default 440).
        freq = 440.0,
        /// Phase offset in radians (default 0).
        phase = 0.0,
    ) -> 1
);

ugen!(
    /// Second-order Butterworth low-pass.
    "LPF" => LPF [ar: Rate::Audio, kr: Rate::Control](
        /// The signal to filter (default 0).
        input = 0.0,
        /// Cutoff frequency in Hz (default 440).
        freq = 440.0,
    ) -> 1
);

ugen!(
    /// Resonant two-pole low-pass.
    "RLPF" => RLPF [ar: Rate::Audio, kr: Rate::Control](
        /// The signal to filter (default 0).
        input = 0.0,
        /// Cutoff frequency in Hz (default 440).
        freq = 440.0,
        /// Reciprocal of Q: bandwidth / cutoff; smaller is more resonant (default 1).
        rq = 1.0,
    ) -> 1
);

ugen!(
    /// Equal-power stereo panner: 2 outputs.
    "Pan2" => Pan2 [ar: Rate::Audio, kr: Rate::Control](
        /// The signal to pan (default 0).
        input = 0.0,
        /// Pan position, -1 (left) to 1 (right) (default 0).
        pos = 0.0,
        /// Level scale (default 1).
        level = 1.0,
    ) -> 2
);

ugen!(
    /// Demand-rate sequence: yields `list` items in order, looping `repeats` times.
    "Dseq" => DSeq [new: Rate::Demand](
        /// Number of repeats (default 1).
        repeats = 1.0,
        /// Sequence items.
        list = [],
    ) -> 1
);

/// Bus input: `num_channels` outputs read from consecutive buses starting at `bus` (default 0).
/// The channel count is structural (it fixes the unit's output count at def time), so it is a
/// constructor argument rather than a setter.
pub struct In;

impl In {
    pub fn ar(g: &SynthDefBuilder, num_channels: usize) -> InBuilder<'_> {
        InBuilder {
            builder: g,
            rate: Rate::Audio,
            bus: UGenInput::Constant(0.0),
            num_channels,
        }
    }

    pub fn kr(g: &SynthDefBuilder, num_channels: usize) -> InBuilder<'_> {
        InBuilder {
            builder: g,
            rate: Rate::Control,
            bus: UGenInput::Constant(0.0),
            num_channels,
        }
    }
}

#[must_use = "a UGen builder emits nothing until it is used as an input or finalized with .signal()"]
pub struct InBuilder<'g> {
    builder: &'g SynthDefBuilder,
    rate: Rate,
    bus: UGenInput<'g>,
    num_channels: usize,
}

impl<'g> InBuilder<'g> {
    /// Starting bus index (default 0).
    pub fn bus(mut self, value: impl Into<UGenInput<'g>>) -> Self {
        self.bus = value.into();
        self
    }
}

impl<'g> UGenBuilder<'g> for InBuilder<'g> {
    fn signal(self) -> Signal<'g> {
        let InBuilder {
            builder,
            rate,
            bus,
            num_channels,
        } = self;
        builder.add("In", RateMode::Fixed(rate), &[bus], num_channels, 0)
    }
}

impl<'g> From<InBuilder<'g>> for UGenInput<'g> {
    fn from(b: InBuilder<'g>) -> Self {
        b.signal().into()
    }
}

impl_builder_ops!(InBuilder);

/// Bus output (a sink: 0 outputs). Input 0 is the starting bus; `channels` is flat-spread into
/// consecutive inputs, one per bus channel from there - a multichannel `channels` value does NOT
/// expand `Out` itself (sclang semantics). An *array* `bus` does expand into one `Out` per bus.
///
/// A sink can't be used as an input, so the only finalizer is the inherent [`OutBuilder::emit`] - don't
/// forget it, or the unit is never emitted (`#[must_use]` warns).
pub struct Out;

impl Out {
    pub fn ar(g: &SynthDefBuilder) -> OutBuilder<'_> {
        OutBuilder {
            builder: g,
            rate: Rate::Audio,
            bus: UGenInput::Constant(0.0),
            channels: UGenInput::Constant(0.0),
        }
    }

    pub fn kr(g: &SynthDefBuilder) -> OutBuilder<'_> {
        OutBuilder {
            builder: g,
            rate: Rate::Control,
            bus: UGenInput::Constant(0.0),
            channels: UGenInput::Constant(0.0),
        }
    }
}

#[must_use = "an Out builder emits nothing until it is finalized with .emit()"]
pub struct OutBuilder<'g> {
    builder: &'g SynthDefBuilder,
    rate: Rate,
    bus: UGenInput<'g>,
    channels: UGenInput<'g>,
}

impl<'g> OutBuilder<'g> {
    /// Starting bus index (default 0).
    pub fn bus(mut self, value: impl Into<UGenInput<'g>>) -> Self {
        self.bus = value.into();
        self
    }

    /// The signal(s) to write; a channel array is flat-spread to consecutive buses (default: a
    /// constant 0, i.e. silence on one channel).
    pub fn channels(mut self, value: impl Into<UGenInput<'g>>) -> Self {
        self.channels = value.into();
        self
    }

    /// Emit the `Out` unit(s) into the def. A sink has no output, so there is nothing to return -
    /// and nothing that could consume the builder as an input, which is why this explicit terminal
    /// is the only finalizer (`#[must_use]` warns if it is forgotten).
    pub fn emit(self) {
        let mut inputs = alloc::vec![self.bus];
        self.channels.flatten(&mut inputs);
        self.builder
            .add("Out", RateMode::Fixed(self.rate), &inputs, 0, 0);
    }
}
