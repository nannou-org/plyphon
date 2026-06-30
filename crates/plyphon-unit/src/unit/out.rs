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
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Out {
    /// `0`/`1`: whether this writes the audio (`Out.ar`) or control (`Out.kr`) bus bank.
    audio: u32,
}

impl Unit for Out {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if ctx.ins.is_empty() {
            return DoneAction::Nothing;
        }
        // Input 0 is the starting bus channel; the rest are signals to write.
        let base = ctx.ins.control(0) as usize;
        if self.audio != 0 {
            for k in 1..ctx.ins.len() {
                let signal = ctx.ins.audio(k);
                // A reblocked graph writes each sub-block tick into its own slice of the World-block
                // channel; `tick` is 0 (offset 0) for an ordinary def, so this is a plain `Out`.
                let offset = ctx.tick * signal.len();
                unit::audio_out_at(ctx.buses, ctx.buf_counter, base + (k - 1), offset, signal);
            }
        } else {
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
            // of the bus channel (offset 0 for an ordinary def). Under reblock the onset offset spans
            // the World block while the carry runs per tick - a documented coarsening, see the type doc.
            let bus_offset = ctx.tick * bs;
            let staged = ctx.outs.audio(0);
            shift_and_carry(&mut staged[..bs], signal, carry, offset, first);
            unit::audio_out_at(
                ctx.buses,
                ctx.buf_counter,
                base + channel,
                bus_offset,
                &staged[..bs],
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
