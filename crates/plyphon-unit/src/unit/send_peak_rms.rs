//! `SendPeakRMS` - periodically reports the peak and RMS level of its input channels as an OSC
//! message, plyphon's port of scsynth's `SendPeakRMS` (`TriggerUGens.cpp`).
//!
//! Like [`SendReply`](crate::unit::SendReply) it emits a `/<cmdName> [nodeID, replyID, values...]`
//! message through the bounded inline [`NodeMsg`] carrier, with the path decoded from its constant
//! character inputs at build time.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::decay::LOG001;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    BuiltUnit, DoneAction, Inputs, MAX_LABEL, MAX_VALUES, NodeMsg, NodeMsgKind, ProcessCtx, Unit,
    unit_spec,
};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// The most channels one `SendPeakRMS` reports: two values each must fit the carrier.
const MAX_CHANNELS: usize = MAX_VALUES / 2;

/// `std::max(a, b)`: `b` if `a < b`, else `a`.
fn std_max(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// One lane of a vector `max` (NEON's `vmaxq_f32`): `NaN` if either operand is `NaN`.
fn lane_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if a < b {
        b
    } else {
        a
    }
}

/// nova-simd's scalar `peak_rms_vec`: raise `peak` to each sample's magnitude and add each
/// sample's square to `squared_sum`, in order.
fn peak_rms(samples: &[f32], peak: &mut f32, squared_sum: &mut f32) {
    let mut local_peak = *peak;
    let mut local_squared_sum = *squared_sum;
    for &x in samples {
        local_peak = std_max(local_peak, x.abs());
        local_squared_sum += x * x;
    }
    *peak = local_peak;
    *squared_sum = local_squared_sum;
}

/// nova-simd's `peak_rms_vec_simd` over four-lane vectors (NEON, or SSE with SSE3), for a whole
/// number of 16-sample chunks: `peak` and `squared_sum` start in lane 0; each chunk adds
/// `((x0² + x1²) + x2²) + x3²` to each lane from its four 4-sample vectors; the lanes are then summed
/// as `(l0 + l1) + (l2 + l3)`. The different summation order makes the result differ, in the last
/// bits, from [`peak_rms`].
fn peak_rms_simd(samples: &[f32], peak: &mut f32, squared_sum: &mut f32) {
    let mut maximum = [*peak, 0.0, 0.0, 0.0];
    let mut sum = [*squared_sum, 0.0, 0.0, 0.0];
    for chunk in samples.chunks_exact(16) {
        for lane in 0..4 {
            let [a, b, c, d] = [
                chunk[lane],
                chunk[4 + lane],
                chunk[8 + lane],
                chunk[12 + lane],
            ];
            let local_max = lane_max(lane_max(a.abs(), b.abs()), lane_max(c.abs(), d.abs()));
            maximum[lane] = lane_max(maximum[lane], local_max);
            sum[lane] += a * a + b * b + c * c + d * d;
        }
    }
    *peak = std_max(
        lane_max(maximum[0], maximum[2]),
        lane_max(maximum[1], maximum[3]),
    );
    *squared_sum = (sum[0] + sum[1]) + (sum[2] + sum[3]);
}

/// `SendPeakRMS.kr/ar(sig, replyRate, peakLag, cmdName, replyID)`: measures the peak and the RMS of
/// each channel of `sig` and, `replyRate` times a second, emits `/<cmdName> [nodeID, replyID,
/// peak0, rms0, peak1, rms1, ...]`. The peak falls back from its maximum with a -60 dB lag of
/// `peakLag` seconds (it rises at once); the RMS covers the samples since the last report.
///
/// Inputs follow scsynth's layout: `[replyRate, peakLag, replyID, numChannels, channels...,
/// cmdNameLen, cmdNameChars...]`. The channel count and path are fixed at build time; at most 16
/// channels (half of [`MAX_VALUES`]).
///
/// A direct port of scsynth's `SendPeakRMS`: the reply interval is counted in samples at audio rate
/// (`FULLRATE / replyRate`) and in blocks at control rate (`BUFRATE / replyRate`), an audio-rate
/// unit reports mid-block where the interval ends, and the sums use nova-simd's four-lane
/// order when the graph's block size is a multiple of 16, as scsynth's `perform_*<true>` do.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SendPeakRMS {
    /// Each channel's peak since the last report.
    level: [f32; MAX_CHANNELS],
    /// Each channel's sum of squares since the last report.
    squared_sum: [f32; MAX_CHANNELS],
    /// Each channel's lagged peak, as last reported.
    lagged: [f32; MAX_CHANNELS],
    /// The OSC path bytes, baked from the constant char inputs.
    label: [u8; MAX_LABEL],
    /// Valid byte length of `label`.
    label_len: u32,
    /// Number of signal channels (`<= MAX_CHANNELS`).
    channel_count: u32,
    /// The peak lag coefficient (scsynth's `mB1`).
    b1: f32,
    /// Samples between reports at audio rate (scsynth's `mAudioSamplesPerTick`).
    audio_samples_per_tick: i32,
    /// Blocks between reports at control rate (scsynth's `mControlSamplesPerTick`).
    control_samples_per_tick: i32,
    /// Samples (or blocks) left until the next report (scsynth's `mPhaseRemain`).
    phase_remain: i32,
    /// `0`/`1`: whether this is an audio-rate unit.
    audio: u32,
    /// `0`/`1`: whether the graph's block size is a multiple of 16 (the `simd` calc variants).
    simd: u32,
}

impl SendPeakRMS {
    const REPLY_RATE: usize = 0;
    const LEVEL_LAG: usize = 1;
    const REPLY_ID: usize = 2;
    const CHANNEL_COUNT: usize = 3;
    const SIGNAL_START: usize = 4;

    /// `analyzeFullBlock`: fold each channel's whole input buffer - a block for an audio-rate
    /// input, one value otherwise - into its peak and sum of squares.
    fn analyze_full_block(&mut self, ins: &Inputs<'_>) {
        for i in 0..self.channel_count as usize {
            let input = Self::SIGNAL_START + i;
            let (level, squared_sum) = (&mut self.level[i], &mut self.squared_sum[i]);
            if ins.rate(input) == Rate::Audio {
                let samples = ins.audio(input);
                if self.simd != 0 {
                    peak_rms_simd(samples, level, squared_sum);
                } else {
                    peak_rms(samples, level, squared_sum);
                }
            } else {
                peak_rms(&[ins.control(input)], level, squared_sum);
            }
        }
    }

    /// `analyzePartialBlock`: fold `count` samples from `first` of each audio-rate channel in, and
    /// a non-audio channel's one value only for the part starting at sample 0. A part of a whole
    /// number of 16-sample chunks, starting at a multiple of 4, takes the four-lane sum.
    fn analyze_partial_block(&mut self, ins: &Inputs<'_>, first: usize, count: usize) {
        for i in 0..self.channel_count as usize {
            let input = Self::SIGNAL_START + i;
            let (level, squared_sum) = (&mut self.level[i], &mut self.squared_sum[i]);
            if ins.rate(input) == Rate::Audio {
                let samples = &ins.audio(input)[first..first + count];
                if count & 15 == 0 && first & 3 == 0 {
                    peak_rms_simd(samples, level, squared_sum);
                } else {
                    peak_rms(samples, level, squared_sum);
                }
            } else if first == 0 {
                peak_rms(&[ins.control(input)], level, squared_sum);
            }
        }
    }

    /// `sendReply`: lag each channel's peak, divide its sum of squares by the reply interval of the
    /// channel's own rate, emit `[peak, rms]` per channel, and start the next interval.
    fn send_reply(&mut self, ctx: &mut ProcessCtx<'_>) {
        let mut values = [0.0f32; MAX_VALUES];
        for i in 0..self.channel_count as usize {
            let y0 = self.level[i];
            let y1 = &mut self.lagged[i];
            *y1 = if y0 >= *y1 {
                y0
            } else {
                y0 + self.b1 * (*y1 - y0)
            };
            let ticks = if ctx.ins.rate(Self::SIGNAL_START + i) == Rate::Audio {
                self.audio_samples_per_tick
            } else {
                self.control_samples_per_tick
            };
            values[2 * i] = *y1;
            values[2 * i + 1] = math::sqrt(self.squared_sum[i] / ticks as f32);
        }
        ctx.node_msgs.push(NodeMsg {
            node: ctx.node_id,
            reply_id: ctx.ins.control(Self::REPLY_ID) as i32,
            kind: NodeMsgKind::Reply,
            label: self.label,
            label_len: self.label_len,
            values,
            num_values: 2 * self.channel_count,
        });
        self.level = [0.0; MAX_CHANNELS];
        self.squared_sum = [0.0; MAX_CHANNELS];
    }

    /// `next_k`: count down one block, reporting (before this block's analysis) when the interval
    /// runs out.
    fn next_k(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.phase_remain = self.phase_remain.wrapping_sub(1);
        if self.phase_remain <= 0 {
            self.phase_remain = self
                .phase_remain
                .wrapping_add(self.control_samples_per_tick);
            self.send_reply(ctx);
        }
        let ins = ctx.ins;
        self.analyze_full_block(&ins);
    }

    /// `next_a`: analyze the block in parts, reporting wherever an interval runs out.
    fn next_a(&mut self, ctx: &mut ProcessCtx<'_>) {
        let ins = ctx.ins;
        let num_samples = ctx.audio.block_size as i32;
        if self.phase_remain >= num_samples {
            self.phase_remain -= num_samples;
            self.analyze_full_block(&ins);
            return;
        }
        if self.phase_remain == 0 {
            self.send_reply(ctx);
            self.phase_remain = self.audio_samples_per_tick;
        }
        let mut start = 0;
        let mut to_analyze = self.phase_remain.min(num_samples);
        let mut remain = num_samples;
        loop {
            // A reply interval of no samples (a `replyRate` at or above the sample rate, or not
            // positive) makes scsynth analyze a zero or negative count, which runs its loop out
            // of bounds; plyphon ends the block there instead.
            if to_analyze <= 0 {
                break;
            }
            self.analyze_partial_block(&ins, start as usize, to_analyze as usize);
            start += to_analyze;
            self.phase_remain -= to_analyze;
            if self.phase_remain == 0 {
                self.send_reply(ctx);
                self.phase_remain = self.audio_samples_per_tick;
            }
            remain -= to_analyze;
            to_analyze = remain.min(self.phase_remain);
            if remain == 0 {
                break;
            }
        }
    }
}

impl Unit for SendPeakRMS {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor runs no calc: it zeroes the channel state (as built) and sets the reply
        // interval and the peak lag from the first input values.
        let reply_rate = ctx.ins.control(Self::REPLY_RATE);
        self.audio_samples_per_tick = (ctx.audio.sample_rate / reply_rate as f64) as i32;
        self.control_samples_per_tick = (ctx.own.buf_rate / reply_rate as f64) as i32;
        self.phase_remain = if self.audio != 0 {
            self.audio_samples_per_tick
        } else {
            self.control_samples_per_tick
        };
        let lag = ctx.ins.control(Self::LEVEL_LAG);
        self.b1 = if lag != 0.0 {
            math::exp(LOG001 / (lag * reply_rate) as f64) as f32
        } else {
            0.0
        };
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.audio != 0 {
            self.next_a(ctx);
        } else {
            self.next_k(ctx);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`SendPeakRMS`] - decodes the channel count and OSC path from their constant
/// inputs and validates them against the inline carrier's bounds. Declares no outputs.
pub struct SendPeakRMSCtor;

impl UnitDef for SendPeakRMSCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        if count <= SendPeakRMS::CHANNEL_COUNT {
            return Err(BuildError::WrongInputCount);
        }
        // `(unsigned int)IN0(channelCountIndex)`.
        let channels = ctx
            .const_input(SendPeakRMS::CHANNEL_COUNT)
            .ok_or(BuildError::EmitBadLabel)? as u32 as usize;
        if channels > MAX_CHANNELS {
            return Err(BuildError::EmitTooManyValues {
                count: 2 * channels,
                limit: MAX_VALUES,
            });
        }
        let len_index = SendPeakRMS::SIGNAL_START + channels;
        let len = ctx.const_input(len_index).ok_or(BuildError::EmitBadLabel)? as usize;
        if len > MAX_LABEL {
            return Err(BuildError::EmitLabelTooLong {
                len,
                limit: MAX_LABEL,
            });
        }
        let mut label = [0u8; MAX_LABEL];
        for (i, b) in label.iter_mut().take(len).enumerate() {
            *b = ctx
                .const_input(len_index + 1 + i)
                .ok_or(BuildError::EmitBadLabel)? as u8;
        }
        Ok(unit_spec(SendPeakRMS {
            level: [0.0; MAX_CHANNELS],
            squared_sum: [0.0; MAX_CHANNELS],
            lagged: [0.0; MAX_CHANNELS],
            label,
            label_len: len as u32,
            channel_count: channels as u32,
            b1: 0.0,
            audio_samples_per_tick: 0,
            control_samples_per_tick: 0,
            phase_remain: 0,
            audio: (ctx.rate == Rate::Audio) as u32,
            // `(FULLBUFLENGTH & 15) == 0`.
            simd: (ctx.audio.block_size & 15 == 0) as u32,
        }))
    }
}
