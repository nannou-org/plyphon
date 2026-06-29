//! `Dbufwr` - a demand-rate buffer writer, plyphon's port of scsynth's `Dbufwr`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::buffer::sc_loop;

/// `Dbufwr(input, bufnum, phase, loop)`: on each demand, writes `input` into buffer `bufnum` at the
/// (truncated) flat index `phase` (wrapping into the buffer, or clamping to the ends when `loop` is 0),
/// then returns `input` - so it passes the value through, like scsynth. An exhausted `input` or `phase`
/// (`NaN`) writes nothing and returns `NaN`; a missing or empty buffer writes nothing. Like scsynth it
/// indexes the interleaved sample array flat (the mono case).
///
/// Stateless: the `Pod` state is just padding for a non-zero slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dbufwr {
    _pad: u32,
}

impl Dbufwr {
    const INPUT: usize = 0;
    const BUFNUM: usize = 1;
    const PHASE: usize = 2;
    const LOOP: usize = 3;
}

impl DemandUnit for Dbufwr {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        ctx.reset(Self::INPUT);
        ctx.reset(Self::BUFNUM);
        ctx.reset(Self::PHASE);
        ctx.reset(Self::LOOP);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let value = ctx.demand(Self::INPUT);
        let bufnum = ctx.demand(Self::BUFNUM);
        let phase = ctx.demand(Self::PHASE);
        let looping = ctx.demand(Self::LOOP) != 0.0;
        // An exhausted input or phase terminates the writer (and writes nothing this demand).
        if value.is_nan() || phase.is_nan() {
            return f32::NAN;
        }
        let bufnum = bufnum.max(0.0) as usize;
        if let Some(buffer) = ctx.buffer_mut(bufnum).filter(|b| b.num_frames() > 0) {
            // `loopMax`: whole buffer when looping, last frame when not (scsynth's bound).
            let loop_max = (buffer.num_frames() - usize::from(!looping)) as f64;
            let (index, _) = sc_loop(phase as f64, loop_max, looping);
            buffer.set_flat(index as usize, value);
        }
        value
    }
}

/// Constructor for [`Dbufwr`].
pub struct DbufwrCtor;

impl DemandUnitDef for DbufwrCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() != 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(demand_unit_spec(Dbufwr { _pad: 0 }))
    }
}
