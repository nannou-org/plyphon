//! `Demand` - a trigger-driven demand consumer, plyphon's port of scsynth's `Demand`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{demand_next, demand_reset};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// The most demand inputs (hence outputs) a single `Demand` may have. Its held values live in fixed
/// `Pod` state, so the count is capped (scsynth allocates this dynamically; plyphon bounds it). A def
/// exceeding this is rejected at compile time.
pub const MAX_DEMAND_OUTPUTS: usize = 8;

/// `Demand.kr/ar(trig, reset, demandUGens)`: on each rising edge of `trig`, demands the next value
/// from each demand source and outputs it, holding it until the next trigger. A rising `reset` resets
/// the sources. An exhausted source (`NaN`) holds its previous value. Inputs are
/// `[trig, reset, source0, source1, ...]`, with one output per source.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Demand {
    /// The last value emitted on each output, held between triggers.
    prev_out: [f32; MAX_DEMAND_OUTPUTS],
    /// Previous `trig` value, for rising-edge detection.
    prev_trig: f32,
    /// Previous `reset` value, for rising-edge detection.
    prev_reset: f32,
    /// Number of demand inputs / outputs (`<= MAX_DEMAND_OUTPUTS`).
    num_outputs: u32,
    /// `0`/`1`: control-rate (one value per block) vs audio-rate (a full block).
    audio: u32,
}

impl Demand {
    const TRIG: usize = 0;
    const RESET: usize = 1;
    /// The first demand-source input; sources occupy `2..`.
    const FIRST_SOURCE: usize = 2;

    /// Read trigger/reset input `input` at sample `i` - the audio slice if it is audio-rate, else the
    /// (broadcast) control value.
    fn trig_at(ctx: &ProcessCtx<'_>, input: usize, i: usize) -> f32 {
        if ctx.ins.rate(input) == Rate::Audio {
            ctx.ins.audio(input)[i]
        } else {
            ctx.ins.control(input)
        }
    }

    /// Reset every demand source (rising `reset`).
    fn reset_sources(&mut self, ctx: &mut ProcessCtx<'_>) {
        for k in 0..self.num_outputs as usize {
            demand_reset(&ctx.ins, &mut ctx.demand, Self::FIRST_SOURCE + k);
        }
    }

    /// Demand the next value from each source into `prev_out` (rising `trig`). An exhausted source
    /// (`NaN`) keeps its previous value.
    fn pull_all(&mut self, ctx: &mut ProcessCtx<'_>) {
        for k in 0..self.num_outputs as usize {
            let x = demand_next(&ctx.ins, &mut ctx.demand, Self::FIRST_SOURCE + k);
            if !x.is_nan() {
                self.prev_out[k] = x;
            }
        }
    }
}

impl Unit for Demand {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let n = self.num_outputs as usize;
        if self.audio != 0 {
            let bs = ctx.audio.block_size;
            for i in 0..bs {
                let trig = Self::trig_at(ctx, Self::TRIG, i);
                let reset = Self::trig_at(ctx, Self::RESET, i);
                if reset > 0.0 && self.prev_reset <= 0.0 {
                    self.reset_sources(ctx);
                }
                if trig > 0.0 && self.prev_trig <= 0.0 {
                    self.pull_all(ctx);
                }
                for k in 0..n {
                    ctx.outs.audio(k)[i] = self.prev_out[k];
                }
                self.prev_trig = trig;
                self.prev_reset = reset;
            }
        } else {
            let trig = ctx.ins.control(Self::TRIG);
            let reset = ctx.ins.control(Self::RESET);
            if reset > 0.0 && self.prev_reset <= 0.0 {
                self.reset_sources(ctx);
            }
            if trig > 0.0 && self.prev_trig <= 0.0 {
                self.pull_all(ctx);
            }
            for k in 0..n {
                *ctx.outs.control(k) = self.prev_out[k];
            }
            self.prev_trig = trig;
            self.prev_reset = reset;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Demand`].
pub struct DemandCtor;

impl UnitDef for DemandCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // Inputs are [trig, reset, source0, ...]; one output per source.
        let num_sources = ctx.input_rates.len().saturating_sub(Demand::FIRST_SOURCE);
        if num_sources == 0 || ctx.num_outputs != num_sources {
            return Err(BuildError::WrongInputCount);
        }
        if num_sources > MAX_DEMAND_OUTPUTS {
            return Err(BuildError::TooManyOutputs {
                needed: num_sources,
                limit: MAX_DEMAND_OUTPUTS,
            });
        }
        Ok(unit_spec(Demand {
            prev_out: [0.0; MAX_DEMAND_OUTPUTS],
            prev_trig: 0.0,
            prev_reset: 0.0,
            num_outputs: num_sources as u32,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
