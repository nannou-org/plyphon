//! `FFTTrigger` - plyphon's port of scsynth's `FFTTrigger` (`FFT_UGens.cpp`): an FFT chain source
//! with no analysis. Compiled only with the `fft` feature.
//!
//! Where `FFT` fills its chain buffer with a transform, `FFTTrigger` only announces the buffer:
//! every `hop * frames` samples it outputs the buffer's number (a ready frame), and `-1` on the
//! blocks between, so `PV_*` units can edit a spectrum written by other means (`PackFFT`, `BufWr`,
//! `/b_setn`). When the synth starts it tags the buffer's spectrum as polar or Cartesian, as its
//! `polar` input asks, which tells the chain how to read the bins.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::buffer::SpectrumCoord;

/// `FFTTrigger(buffer, hop = 0.5, polar = 0)`: output `buffer` once every `hop * frames` samples
/// and `-1` otherwise, without transforming anything. Control rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FftTrigger {
    /// The buffer number a ready frame outputs (scsynth's `m_fftbufnum`).
    bufnum: u32,
    /// Blocks between two ready frames (scsynth's `m_numPeriods`).
    num_periods: i32,
    /// Blocks left before the next ready frame (scsynth's `m_periodsRemain`).
    periods_remain: i32,
}

impl FftTrigger {
    const BUFFER: usize = 0;
    const HOP: usize = 1;
    const POLAR: usize = 2;
}

impl Unit for FftTrigger {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `FFTTrigger_Ctor` (`FFT_UGens.cpp:335`). The buffer number is `(uint32)IN0(0)`.
        // A graph-local number past the synth's local buffers falls back to world buffer 0, which
        // then also becomes the number the unit outputs. scsynth's bound is `localBufNum <=
        // localMaxBufNum`, so the number one past the last local buffer indexes past its array
        // (undefined behaviour); plyphon keeps that number and finds no buffer behind it.
        let input = ctx.ins.control(Self::BUFFER);
        let mut bufnum = input as u32;
        let capacity = unit::num_buffers(ctx.buffers);
        if let Some(local) = (bufnum as usize).checked_sub(capacity)
            && local > ctx.local_bufs.len()
        {
            bufnum = 0;
        }
        let polar = ctx.ins.control(Self::POLAR) == 1.0;
        // The buffer's sample count (`buf->samples`); a buffer with no storage has none. The
        // constructor also sets the buffer's coordinate form from `polar`.
        let samples = match unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum as usize) {
            Some(mut buffer) => {
                buffer.set_coord(if polar {
                    SpectrumCoord::Polar
                } else {
                    SpectrumCoord::Complex
                });
                buffer.data().len()
            }
            None => 0,
        };
        // `(int)(((float)m_fullbufsize * dataHopSize) / numSamples) - 1`, where `numSamples` is
        // `FULLBUFLENGTH`, the graph's audio block size.
        let hop = ctx.ins.control(Self::HOP);
        let periods = (samples as f32 * hop) / ctx.audio.block_size as f32;
        self.bufnum = bufnum;
        self.num_periods = (periods as i32).saturating_sub(1);
        self.periods_remain = self.num_periods;
        // The constructor outputs the raw input (`OUT0(0) = IN0(0)`) and runs no calc.
        *ctx.outs.control(0) = input;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `FFTTrigger_next` (`FFT_UGens.cpp:378`).
        *ctx.outs.control(0) = if self.periods_remain > 0 {
            self.periods_remain -= 1;
            -1.0
        } else {
            self.periods_remain = self.num_periods;
            self.bufnum as f32
        };
        DoneAction::Nothing
    }
}

/// Constructor for [`FftTrigger`].
pub struct FftTriggerCtor;

impl UnitDef for FftTriggerCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= FftTrigger::POLAR {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(FftTrigger::zeroed()))
    }
}
