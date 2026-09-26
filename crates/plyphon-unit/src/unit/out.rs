//! `Out`/`OffsetOut` - write signals to audio or control bus channels, plyphon's ports of scsynth's
//! `Out` and `OffsetOut`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec, unit_spec_aux};
use plyphon_dsp::rate::Rate;

/// `Out.ar(bus, signals)` / `Out.kr(bus, signals)`: writes each signal input to a consecutive bus
/// channel starting at `bus`, summing with anything already written to that channel this block.
/// `Out.ar` targets the audio bus bank, `Out.kr` the control bus bank, chosen by the unit's rate.
///
/// As in scsynth, the first write to a channel in a block copies the signal over it; only later
/// writes sum (a reblocked graph's `Out.ar` clears the channel and sums instead, as
/// `Out_next_a_reblock` does). A reblocked graph writes a control bus on its first tick only.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Out {
    /// `0`/`1`: whether this writes the audio (`Out.ar`) or control (`Out.kr`) bus bank.
    audio: u32,
}

impl Unit for Out {
    fn init(&mut self, _ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor runs no calc; the output starts at zero.
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.ins.is_empty() {
            return DoneAction::Nothing;
        }
        // Input 0 is the starting bus channel; the rest are signals to write.
        let base = ctx.ins.control(0) as usize;
        if self.audio != 0 {
            let factor = ctx.resample_factor;
            for k in 1..ctx.ins.len() {
                let signal = ctx.ins.audio(k);
                // A reblocked/resampled graph writes each sub-block tick into its own slice of the
                // World-block channel, decimating its `factor`x-oversampled samples down to the World
                // rate. `tick` 0, `factor` 1 makes this a plain `Out`.
                let out_samples = signal.len() / factor;
                let offset = ctx.tick * out_samples;
                unit::audio_out_decimated(
                    ctx.buses,
                    ctx.buf_counter,
                    base + (k - 1),
                    offset,
                    signal,
                    factor,
                );
            }
        } else if ctx.tick == 0 {
            // A reblocked graph writes a control bus on its first tick only (`Out_next_k_reblock`).
            for k in 1..ctx.ins.len() {
                unit::control_out(
                    ctx.buses,
                    ctx.buf_counter,
                    base + (k - 1),
                    ctx.ins.control(k),
                );
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Out`].
pub struct OutCtor;

impl UnitDef for OutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Out {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `ReplaceOut.ar(bus, signals)` / `ReplaceOut.kr(bus, signals)`: like [`Out`], but *overwrites* each
/// bus channel with its signal instead of summing onto whatever earlier units wrote this block.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ReplaceOut {
    /// `0`/`1`: whether this writes the audio or control bus bank.
    audio: u32,
}

impl Unit for ReplaceOut {
    fn init(&mut self, _ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor runs no calc; the output starts at zero.
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.ins.is_empty() {
            return DoneAction::Nothing;
        }
        let base = ctx.ins.control(0) as usize;
        if self.audio != 0 {
            let factor = ctx.resample_factor;
            for k in 1..ctx.ins.len() {
                let signal = ctx.ins.audio(k);
                let out_samples = signal.len() / factor;
                let offset = ctx.tick * out_samples;
                unit::audio_replace_decimated(
                    ctx.buses,
                    ctx.buf_counter,
                    base + (k - 1),
                    offset,
                    signal,
                    factor,
                );
            }
        } else if ctx.tick == 0 {
            // A reblocked graph writes a control bus on its first tick only
            // (`ReplaceOut_next_k_reblock`).
            for k in 1..ctx.ins.len() {
                unit::control_replace(
                    ctx.buses,
                    ctx.buf_counter,
                    base + (k - 1),
                    ctx.ins.control(k),
                );
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`ReplaceOut`].
pub struct ReplaceOutCtor;

impl UnitDef for ReplaceOutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(ReplaceOut {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `XOut.ar(bus, xfade, signals)` / `XOut.kr(...)`: crossfades each signal into a consecutive bus
/// channel starting at `bus`, against whatever earlier units wrote there this block. `xfade = 0`
/// leaves the bus unchanged and `xfade = 1` replaces it.
///
/// A direct port of scsynth's `XOut`. At audio rate the unit keeps the previous block's `xfade`
/// (scsynth's `m_xfade`, set from the first `xfade` in the constructor) and picks a branch from it:
///
/// - `xfade` changed: ramp from the old value to the new one across the block (`CALCSLOPE`).
/// - old `xfade` is 1: copy the signal over the channel.
/// - old `xfade` is 0: leave the channel alone, without marking it written.
/// - otherwise: crossfade at the old `xfade`.
///
/// A channel already written this block is crossfaded; an untouched one takes `signal * xfade`.
/// With a block size that is a multiple of 16, scsynth uses nova-simd's kernels (`XOut_next_a_nova`),
/// which crossfade as `bus * (1 - xfade) + signal * xfade` and build a ramp four lanes at a time
/// (as nova-simd's NEON `set_slope` does); other block sizes use `XOut_next_a`'s
/// `bus + xfade * (signal - bus)` with a per-sample ramp. A reblocked or resampled graph follows
/// `XOut_next_a_reblock`. At control rate the crossfade is `XOut_next_k`'s, with no branches, and a
/// reblocked graph writes on its first tick only.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct XOut {
    /// `0`/`1`: whether this crossfades the audio or control bus bank.
    audio: u32,
    /// `0`/`1`: whether the graph's block size is a multiple of 16 (scsynth's nova-simd variant).
    nova: u32,
    /// The previous block's `xfade` (scsynth's `m_xfade`).
    xfade: f32,
}

impl XOut {
    const BUS: usize = 0;
    const XFADE: usize = 1;
    const SIGNAL_START: usize = 2;

    /// `XOut_next_a` and `XOut_next_a_nova`: one whole block of an ordinary graph.
    fn next_a(&mut self, ctx: &mut ProcessCtx<'_>, base: usize) {
        let next_xfade = ctx.ins.control(Self::XFADE);
        let xfade0 = self.xfade;
        let nova = self.nova != 0;
        let ins = ctx.ins;
        for k in Self::SIGNAL_START..ins.len() {
            let ch = base + (k - Self::SIGNAL_START);
            let signal = ins.audio(k);
            let touched = unit::audio_in_touched(ctx.buses, ch, ctx.buf_counter);
            let Some(out) = unit::audio_channel_mut(ctx.buses, ch) else {
                continue;
            };
            let n = out.len().min(signal.len());
            let (out, signal) = (&mut out[..n], &signal[..n]);
            if xfade0 != next_xfade {
                let slope = (next_xfade - xfade0) * ctx.own.slope_factor as f32;
                if nova {
                    let mut to = NeonRamp::new(xfade0, slope);
                    if touched {
                        let mut from = NeonRamp::new(1.0 - xfade0, -slope);
                        for (j, (o, &x)) in out.iter_mut().zip(signal).enumerate() {
                            *o = *o * from.at(j) + x * to.at(j);
                        }
                    } else {
                        for (j, (o, &x)) in out.iter_mut().zip(signal).enumerate() {
                            *o = x * to.at(j);
                        }
                    }
                } else {
                    let mut xfade = xfade0;
                    for (o, &x) in out.iter_mut().zip(signal) {
                        *o = if touched {
                            *o + xfade * (x - *o)
                        } else {
                            x * xfade
                        };
                        xfade += slope;
                    }
                }
            } else if xfade0 == 1.0 {
                out.copy_from_slice(signal);
            } else if xfade0 == 0.0 {
                continue;
            } else if touched {
                for (o, &x) in out.iter_mut().zip(signal) {
                    *o = if nova {
                        *o * (1.0 - xfade0) + x * xfade0
                    } else {
                        *o + xfade0 * (x - *o)
                    };
                }
            } else {
                for (o, &x) in out.iter_mut().zip(signal) {
                    *o = x * xfade0;
                }
            }
            unit::audio_touch(ctx.buses, ch, ctx.buf_counter);
        }
        self.xfade = next_xfade;
    }

    /// `XOut_next_a_reblock`: one tick of a reblocked or resampled graph, writing its slice of the
    /// World-block channel, one sample in every `factor`.
    fn next_a_reblock(&mut self, ctx: &mut ProcessCtx<'_>, base: usize) {
        let factor = ctx.resample_factor.max(1);
        let ins = ctx.ins;
        // scsynth's `inNumSamples`: the graph's block.
        let num_samples = ctx.audio.block_size;
        let input_offset = ctx.tick * num_samples;
        // All of this tick's samples fall between the World-rate samples.
        if input_offset & (factor - 1) != 0 {
            return;
        }
        let next_xfade = ins.control(Self::XFADE);
        let xfade0 = self.xfade;
        let shift = factor.trailing_zeros();
        let out_samples = (num_samples >> shift).max(1);
        let out_offset = input_offset >> shift;
        let first_tick = ctx.tick == 0;
        for k in Self::SIGNAL_START..ins.len() {
            let ch = base + (k - Self::SIGNAL_START);
            let signal = ins.audio(k);
            let untouched = !unit::audio_in_touched(ctx.buses, ch, ctx.buf_counter);
            let Some(channel) = unit::audio_channel_mut(ctx.buses, ch) else {
                continue;
            };
            let end = (out_offset + out_samples).min(channel.len());
            let at = |j: usize| signal.get(j << shift).copied().unwrap_or(0.0);
            if xfade0 != next_xfade {
                let slope = (next_xfade - xfade0) * ctx.own.slope_factor as f32 * factor as f32;
                let touch = first_tick && untouched;
                if touch {
                    // The first tick clears an untouched channel whole, so every tick sums into it.
                    channel.fill(0.0);
                }
                let mut xfade = xfade0;
                for (j, o) in channel[out_offset.min(end)..end].iter_mut().enumerate() {
                    *o += xfade * (at(j) - *o);
                    xfade += slope;
                }
                if touch {
                    unit::audio_touch(ctx.buses, ch, ctx.buf_counter);
                }
            } else if xfade0 == 1.0 {
                // scsynth sums here, onto whatever the channel holds.
                for (j, o) in channel[out_offset.min(end)..end].iter_mut().enumerate() {
                    *o += at(j);
                }
                unit::audio_touch(ctx.buses, ch, ctx.buf_counter);
            } else if xfade0 == 0.0 {
                continue;
            } else {
                let touch = first_tick && untouched;
                if touch {
                    // The first tick clears an untouched channel whole, so every tick sums into it.
                    channel.fill(0.0);
                }
                for (j, o) in channel[out_offset.min(end)..end].iter_mut().enumerate() {
                    *o += xfade0 * (at(j) - *o);
                }
                if touch {
                    unit::audio_touch(ctx.buses, ch, ctx.buf_counter);
                }
            }
        }
        self.xfade = next_xfade;
    }
}

/// One of nova-simd's `slope_argument` ramps as its NEON vectors hold it: four lanes starting at
/// `start`, `start + slope`, `start + slope + slope` and `start + slope + slope + slope`, each
/// stepped by `slope + slope + slope + slope` for every four samples.
struct NeonRamp {
    lanes: [f32; 4],
    step: f32,
    /// The index of the first sample `lanes` holds.
    first: usize,
}

impl NeonRamp {
    fn new(start: f32, slope: f32) -> Self {
        let s1 = start + slope;
        let s2 = s1 + slope;
        NeonRamp {
            lanes: [start, s1, s2, s2 + slope],
            step: slope + slope + slope + slope,
            first: 0,
        }
    }

    /// The ramp's value at sample `j`; samples are read in order.
    fn at(&mut self, j: usize) -> f32 {
        while j >= self.first + 4 {
            for lane in &mut self.lanes {
                *lane += self.step;
            }
            self.first += 4;
        }
        self.lanes[j - self.first]
    }
}

impl Unit for XOut {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor keeps the first `xfade` and runs no calc.
        if ctx.ins.len() > Self::XFADE {
            self.xfade = ctx.ins.control(Self::XFADE);
        }
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // Needs at least `bus` and `xfade`.
        if ctx.ins.len() < Self::SIGNAL_START {
            return DoneAction::Nothing;
        }
        let base = ctx.ins.control(Self::BUS) as usize;
        if self.audio != 0 {
            let reblocked =
                ctx.resample_factor != 1 || ctx.audio.block_size != ctx.buses.audio().block_size();
            if reblocked {
                self.next_a_reblock(ctx, base);
            } else {
                self.next_a(ctx, base);
            }
        } else if ctx.tick == 0 {
            // `XOut_next_k`; a reblocked graph writes on its first tick only
            // (`XOut_next_k_reblock`).
            let xfade = ctx.ins.control(Self::XFADE);
            for k in Self::SIGNAL_START..ctx.ins.len() {
                unit::control_crossfade(
                    ctx.buses,
                    ctx.buf_counter,
                    base + (k - Self::SIGNAL_START),
                    ctx.ins.control(k),
                    xfade,
                );
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`XOut`].
pub struct XOutCtor;

impl UnitDef for XOutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(XOut {
            audio: (ctx.rate == Rate::Audio) as u32,
            // `boost::alignment::is_aligned(BUFLENGTH, 16)`.
            nova: ctx.audio.block_size.is_multiple_of(16) as u32,
            xfade: 0.0,
        }))
    }
}

/// `OffsetOut.ar(bus, signals)`: like [`Out`], but a synth created partway into a control block - by
/// a scheduled, time-tagged command - has its whole output delayed by the creation offset, so it
/// becomes audible at exactly the scheduled sample. plyphon's port of scsynth's `OffsetOut`.
///
/// The offset is the within-block sample at which the synth was created (scsynth's `mSampleOffset`),
/// delivered as [`ProcessCtx::sample_offset`] (non-zero only on the synth's first block, so it is
/// captured and held). Each block emits `[carry, signal[..bs - offset]]` and saves this block's last
/// `offset` samples into the per-channel `carry` (`aux`) for the next block's front - scsynth's
/// `OffsetOut_next` delay-and-carry, which shifts every sample forward by `offset` for the synth's
/// life. On the first block the carry holds nothing, so the leading `offset` samples are silence
/// (scsynth's `m_empty`).
///
/// One divergence from scsynth: when the synth is freed, the final `offset` samples still in the
/// carry are not flushed to the bus - scsynth does this in `OffsetOut_Dtor`, but plyphon units have
/// no destructor. For a voice ending in silence (the usual enveloped case) those samples are ~0.
/// `OffsetOut.kr` ignores the offset, since a control block is a single value.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OffsetOut {
    /// `0`/`1`: whether this writes the audio (`OffsetOut.ar`) or control (`OffsetOut.kr`) bus bank.
    audio: u32,
    /// The audio block size, baked at build - the per-channel stride into the carry buffer.
    block_size: u32,
    /// The synth's creation offset within its first block (scsynth's `mSampleOffset`), captured on
    /// the first block and held for the synth's life.
    offset: u32,
    /// `0` until the first block has run; until then the carry holds no samples, so the leading
    /// `offset` output samples are silence (scsynth's `m_empty`).
    warmed: u32,
}

impl Unit for OffsetOut {
    fn init(&mut self, _ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor runs no calc; the output starts at zero.
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.ins.is_empty() {
            return DoneAction::Nothing;
        }
        let base = ctx.ins.control(0) as usize;
        if self.audio == 0 {
            // Control rate: a block is a single value, so the offset is meaningless - like `Out.kr`.
            for k in 1..ctx.ins.len() {
                unit::control_out(
                    ctx.buses,
                    ctx.buf_counter,
                    base + (k - 1),
                    ctx.ins.control(k),
                );
            }
            return DoneAction::Nothing;
        }
        // The offset arrives only on the first block; capture and hold it for the synth's life.
        if self.warmed == 0 {
            self.offset = ctx.sample_offset as u32;
        }
        let stride = self.block_size as usize;
        let offset = self.offset as usize;
        let first = self.warmed == 0;
        let carries = ctx.aux.f32_mut();
        for k in 1..ctx.ins.len() {
            let signal = ctx.ins.audio(k);
            let bs = signal.len();
            let channel = k - 1;
            // This channel's `offset`-sample carry within its `stride`-wide region of `aux`.
            let carry = &mut carries[channel * stride..channel * stride + offset.min(stride)];
            // Stage the delayed block in channel-0 scratch, then accumulate it onto this tick's slice
            // of the bus channel, decimated to the World rate (offset 0, factor 1 for an ordinary
            // def). Under reblock/resample the onset offset spans the World block while the carry runs
            // per tick - a documented coarsening, see the type doc.
            let factor = ctx.resample_factor;
            let bus_offset = ctx.tick * (bs / factor);
            let staged = ctx.outs.audio(0);
            shift_and_carry(&mut staged[..bs], signal, carry, offset, first);
            // On the first block of an ordinary graph, a channel another writer has already
            // touched keeps its leading `offset` samples as they are ("just keep the existing bus
            // content") and only the signal is summed in after them; an untouched one is copied
            // whole, the leading samples cleared (`OffsetOut_next_a`).
            let whole_block = factor == 1 && bs == ctx.buses.audio().block_size();
            let keep = if first
                && whole_block
                && unit::audio_in_touched(ctx.buses, base + channel, ctx.buf_counter)
            {
                offset.min(bs)
            } else {
                0
            };
            unit::audio_out_decimated(
                ctx.buses,
                ctx.buf_counter,
                base + channel,
                bus_offset + keep,
                &staged[keep..bs],
                factor,
            );
        }
        self.warmed = 1;
        DoneAction::Nothing
    }
}

/// Stage one channel's block delayed by `offset` samples (scsynth's `OffsetOut_next` per channel):
/// emit `[carry (or silence on the `first` block), signal[..bs - offset]]` into `staged`, and save
/// `signal[bs - offset..]` into `carry` for the next block's front. Every length is clamped so a bad
/// caller can never panic on the audio thread.
fn shift_and_carry(
    staged: &mut [f32],
    signal: &[f32],
    carry: &mut [f32],
    offset: usize,
    first: bool,
) {
    let bs = staged.len().min(signal.len());
    let offset = offset.min(bs).min(carry.len());
    let remain = bs - offset;
    if first {
        staged[..offset].fill(0.0);
    } else {
        staged[..offset].copy_from_slice(&carry[..offset]);
    }
    staged[offset..bs].copy_from_slice(&signal[..remain]);
    carry[..offset].copy_from_slice(&signal[remain..bs]);
}

/// Constructor for [`OffsetOut`]. The audio form reserves a per-channel `offset`-sample carry buffer
/// (sized for the worst-case offset, one block) folded into the synth's `aux`.
pub struct OffsetOutCtor;

impl UnitDef for OffsetOutCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let audio = ctx.rate == Rate::Audio;
        let block_size = ctx.audio.block_size;
        let state = OffsetOut {
            audio: audio as u32,
            block_size: block_size as u32,
            offset: 0,
            warmed: 0,
        };
        // Signal channels are the inputs after the bus index. The control form needs no carry.
        let channels = ctx.input_rates.len().saturating_sub(1);
        if !audio || channels == 0 {
            return Ok(unit_spec(state));
        }
        Ok(unit_spec_aux(
            state,
            channels * block_size * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::shift_and_carry;
    use alloc::vec;

    #[test]
    fn offset_out_delays_the_signal_by_the_offset() {
        // A ramp input (sample n holds value n) makes the delay visible: with offset 3, every input
        // sample must reappear three samples later - not be dropped (which is what the old gate did).
        let (bs, offset) = (8usize, 3usize);
        let in0: vec::Vec<f32> = (0..bs).map(|n| n as f32).collect();
        let in1: vec::Vec<f32> = (bs..2 * bs).map(|n| n as f32).collect();
        let mut carry = vec![0.0f32; offset];
        let mut staged = vec![0.0f32; bs];

        // First block: leading `offset` silent (scsynth's `m_empty`), then signal[..bs-offset].
        shift_and_carry(&mut staged, &in0, &mut carry, offset, true);
        assert_eq!(staged, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(carry, vec![5.0, 6.0, 7.0], "block 0's tail is carried");

        // Second block: the carried tail of block 0, then signal[..bs-offset].
        shift_and_carry(&mut staged, &in1, &mut carry, offset, false);
        assert_eq!(staged, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(carry, vec![13.0, 14.0, 15.0]);
        // Concatenated, the output is the input delayed by 3 (3 leading zeros, then 0,1,2,...,12) -
        // every sample preserved and shifted, the delay-and-carry, not the truncating gate.
    }

    #[test]
    fn offset_zero_is_a_plain_copy() {
        // No offset: the block passes straight through (OffsetOut == Out), no carry.
        let signal = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut carry: vec::Vec<f32> = vec![];
        let mut staged = vec![0.0f32; 4];
        shift_and_carry(&mut staged, &signal, &mut carry, 0, true);
        assert_eq!(staged, signal);
    }
}
