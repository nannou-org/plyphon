//! `RunningSum` - plyphon's port of scsynth's `RunningSum` (`FeatureDetection.cpp`), a running sum
//! over a fixed window, the time-domain building block for onset detection and RMS.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::rate::Rate;

/// `RunningSum(in, numsamp = 40)`: the sum of the last `numsamp` input samples.
///
/// The window is a ring of the last `numsamp` inputs; each sample subtracts the value leaving the
/// window and adds the one entering it. A second sum restarts at zero every time the ring wraps and
/// replaces the running sum then, so rounding error cannot accumulate beyond one window (scsynth's
/// `RunningSum_next_k`, used at both rates). `numsamp` is read once, when the synth starts, and the
/// ring is allocated from the engine's pool then; the window starts full of zeros.
///
/// A negative `numsamp` fails the allocation as in scsynth, leaving the unit silent and done. A
/// `numsamp` of zero leaves the output at zero: the reference loops forever there.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RunningSum {
    /// The window length in samples, `(int)numsamp` when the synth started.
    samp: i32,
    /// The ring's write position.
    count: i32,
    /// The running sum.
    sum: f32,
    /// The sum since the ring last wrapped, which replaces [`sum`](Self::sum) at the next wrap.
    sum2: f32,
    /// `1` when `numsamp` was negative, so the allocation failed.
    failed: u32,
}

impl RunningSum {
    const IN: usize = 0;
    const NUMSAMP: usize = 1;
}

impl Unit for RunningSum {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // scsynth's `RunningSum_Ctor`: `msamp = (int)ZIN0(1)`, then a zeroed ring of that many
        // floats. A negative count wraps to a size no allocator can satisfy.
        self.samp = ctx.ins.control(Self::NUMSAMP) as i32;
        let Ok(samp) = usize::try_from(self.samp) else {
            self.failed = 1;
            return;
        };
        if aux.alloc(samp * core::mem::size_of::<f32>()) {
            aux.f32_mut().fill(0.0);
        }
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor writes a zero output without running the calc; a failed allocation
        // instead clears the unit and marks it done (`ClearUnitIfMemFailed`).
        if self.failed != 0 {
            ctx.done.mark_done();
        }
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let out = ctx.outs.audio(0);
        if self.failed != 0 || self.samp == 0 {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        let samp = self.samp as usize;
        let data = &mut ctx.aux.f32_mut()[..samp];
        let audio_in = ins.rate(Self::IN) == Rate::Audio;
        let mut count = self.count as usize;
        let mut sum = self.sum;
        let mut sum2 = self.sum2;
        for (i, o) in out.iter_mut().enumerate() {
            let next = if audio_in {
                ins.audio(Self::IN)[i]
            } else {
                ins.control(Self::IN)
            };
            sum -= data[count];
            data[count] = next;
            sum += next;
            sum2 += next;
            *o = sum;
            count += 1;
            if count == samp {
                count = 0;
                sum = sum2;
                sum2 = 0.0;
            }
        }
        self.count = count as i32;
        self.sum = sum;
        self.sum2 = sum2;
        DoneAction::Nothing
    }
}

/// Constructor for [`RunningSum`]: the unit sizes its window when the synth starts.
pub struct RunningSumCtor;

impl UnitDef for RunningSumCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= RunningSum::NUMSAMP {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(RunningSum::zeroed()))
    }
}
