//! Info units - plyphon's ports of scsynth's info UGens.
//!
//! These surface engine-level constants to the graph: the audio sample rate and its reciprocal,
//! `RadiansPerSample`, the control rate/duration, and the bus counts ([`Info`]); plus per-buffer
//! info - frame/channel/sample counts, sample rate, rate scale, and duration ([`BufInfo`]). They
//! hold no per-instance state. Each writes a single value, broadcast across the block every block,
//! so a `BufInfo` tracks a buffer that is reallocated under it (scsynth re-reads the buffer each
//! calc too).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// Which engine constant an [`Info`] unit reports. The build-time domain; stored in [`Info`] as a
/// `u32` tag (via [`InfoKind::to_tag`]) so the state is [`Pod`] and lives in the rt-pool.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InfoKind {
    /// Audio sample rate in Hz (`SampleRate`).
    SampleRate,
    /// Seconds per sample (`SampleDur`).
    SampleDur,
    /// Radians per sample at 1 Hz (`RadiansPerSample`).
    RadiansPerSample,
    /// Control blocks per second (`ControlRate`).
    ControlRate,
    /// Seconds per control block (`ControlDur`).
    ControlDur,
    /// Number of hardware output bus channels (`NumOutputBuses`).
    NumOutputBuses,
    /// Number of hardware input bus channels (`NumInputBuses`).
    NumInputBuses,
    /// Total audio bus channels - output, input, and private (`NumAudioBuses`).
    NumAudioBuses,
    /// Total control bus channels (`NumControlBuses`).
    NumControlBuses,
}

impl InfoKind {
    /// Encode as the `u32` tag stored in [`Info`].
    fn to_tag(self) -> u32 {
        match self {
            InfoKind::SampleRate => 0,
            InfoKind::SampleDur => 1,
            InfoKind::RadiansPerSample => 2,
            InfoKind::ControlRate => 3,
            InfoKind::ControlDur => 4,
            InfoKind::NumOutputBuses => 5,
            InfoKind::NumInputBuses => 6,
            InfoKind::NumAudioBuses => 7,
            InfoKind::NumControlBuses => 8,
        }
    }

    /// Decode the `u32` tag stored in [`Info`] (any out-of-range tag is `SampleRate`).
    fn from_tag(tag: u32) -> InfoKind {
        match tag {
            1 => InfoKind::SampleDur,
            2 => InfoKind::RadiansPerSample,
            3 => InfoKind::ControlRate,
            4 => InfoKind::ControlDur,
            5 => InfoKind::NumOutputBuses,
            6 => InfoKind::NumInputBuses,
            7 => InfoKind::NumAudioBuses,
            8 => InfoKind::NumControlBuses,
            _ => InfoKind::SampleRate,
        }
    }

    /// The constant's value for this block. `ControlRate`/`ControlDur` derive from the audio rate
    /// (`buf_rate`/`buf_dur` are `sample_rate / block_size` and its reciprocal).
    fn value(self, ctx: &ProcessCtx<'_>) -> f32 {
        match self {
            InfoKind::SampleRate => ctx.audio.sample_rate as f32,
            InfoKind::SampleDur => ctx.audio.sample_dur as f32,
            InfoKind::RadiansPerSample => ctx.audio.radians_per_sample as f32,
            InfoKind::ControlRate => ctx.audio.buf_rate as f32,
            InfoKind::ControlDur => ctx.audio.buf_dur as f32,
            InfoKind::NumOutputBuses => unit::num_output_buses(ctx.buses) as f32,
            InfoKind::NumInputBuses => unit::num_input_buses(ctx.buses) as f32,
            InfoKind::NumAudioBuses => unit::num_audio_buses(ctx.buses) as f32,
            InfoKind::NumControlBuses => unit::num_control_buses(ctx.buses) as f32,
        }
    }
}

/// An engine-constant info unit (`SampleRate`, `ControlDur`, `NumOutputBuses`, ...). Takes no inputs
/// and writes one value, the same across the block.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Info {
    /// The [`InfoKind`] tag.
    kind: u32,
}

impl Unit for Info {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let value = InfoKind::from_tag(self.kind).value(ctx);
        ctx.outs.audio(0).fill(value);
        DoneAction::Nothing
    }
}

/// Constructor for [`Info`], parameterized by [`InfoKind`].
pub struct InfoCtor(pub InfoKind);

impl UnitDef for InfoCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Info {
            kind: self.0.to_tag(),
        }))
    }
}

/// Which per-buffer quantity a [`BufInfo`] unit reports.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BufInfoKind {
    /// Number of frames, i.e. samples per channel (`BufFrames`).
    Frames,
    /// Number of channels (`BufChannels`).
    Channels,
    /// Total samples, `frames * channels` (`BufSamples`).
    Samples,
    /// The buffer's own sample rate in Hz (`BufSampleRate`).
    SampleRate,
    /// Buffer sample rate divided by the engine sample rate (`BufRateScale`).
    RateScale,
    /// Buffer duration in seconds, `frames / sampleRate` (`BufDur`).
    Dur,
}

impl BufInfoKind {
    /// Encode as the `u32` tag stored in [`BufInfo`].
    fn to_tag(self) -> u32 {
        match self {
            BufInfoKind::Frames => 0,
            BufInfoKind::Channels => 1,
            BufInfoKind::Samples => 2,
            BufInfoKind::SampleRate => 3,
            BufInfoKind::RateScale => 4,
            BufInfoKind::Dur => 5,
        }
    }

    /// Decode the `u32` tag stored in [`BufInfo`] (any out-of-range tag is `Frames`).
    fn from_tag(tag: u32) -> BufInfoKind {
        match tag {
            1 => BufInfoKind::Channels,
            2 => BufInfoKind::Samples,
            3 => BufInfoKind::SampleRate,
            4 => BufInfoKind::RateScale,
            5 => BufInfoKind::Dur,
            _ => BufInfoKind::Frames,
        }
    }
}

/// A per-buffer info unit (`BufFrames`, `BufDur`, `BufRateScale`, ...). Input 0 is the buffer index;
/// it writes one value, re-read from the buffer table every block. A missing buffer reports `0`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BufInfo {
    /// The [`BufInfoKind`] tag.
    kind: u32,
}

impl BufInfo {
    const BUF: usize = 0;
}

impl Unit for BufInfo {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let kind = BufInfoKind::from_tag(self.kind);
        let index = ctx.ins.control(Self::BUF).max(0.0) as usize;
        let engine_sr = ctx.audio.sample_rate;
        let value = match unit::buffer_at(ctx.buffers, index) {
            Some(buf) => {
                let frames = buf.num_frames();
                let channels = buf.num_channels();
                let buf_sr = buf.sample_rate();
                match kind {
                    BufInfoKind::Frames => frames as f32,
                    BufInfoKind::Channels => channels as f32,
                    BufInfoKind::Samples => (frames * channels) as f32,
                    BufInfoKind::SampleRate => buf_sr as f32,
                    BufInfoKind::RateScale => {
                        if engine_sr > 0.0 {
                            (buf_sr / engine_sr) as f32
                        } else {
                            0.0
                        }
                    }
                    BufInfoKind::Dur => {
                        if buf_sr > 0.0 {
                            (frames as f64 / buf_sr) as f32
                        } else {
                            0.0
                        }
                    }
                }
            }
            None => 0.0,
        };
        ctx.outs.audio(0).fill(value);
        DoneAction::Nothing
    }
}

/// Constructor for [`BufInfo`], parameterized by [`BufInfoKind`].
pub struct BufInfoCtor(pub BufInfoKind);

impl UnitDef for BufInfoCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(BufInfo {
            kind: self.0.to_tag(),
        }))
    }
}
