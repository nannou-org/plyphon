//! `Dbufrd` - a demand-rate buffer reader, plyphon's port of scsynth's `Dbufrd`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::buffer::sc_loop;

/// `Dbufrd(bufnum, phase, loop)`: on each demand, reads one sample from buffer `bufnum` at the
/// (truncated) flat index `phase`, wrapping into the buffer (or clamping to the ends when `loop` is 0)
/// with scsynth's [`sc_loop`]. `phase` is typically itself a demand source (e.g. `Dseries`); when it is
/// exhausted (`NaN`) the read returns `NaN` too. A missing or empty buffer yields `0`. Like scsynth it
/// indexes the interleaved sample array flat, so it is a single-value reader (the mono case).
///
/// Stateless: each read is computed fresh, so the `Pod` state is just padding for a non-zero slot.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dbufrd {
    _pad: u32,
}

impl Dbufrd {
    const BUFNUM: usize = 0;
    const PHASE: usize = 1;
    const LOOP: usize = 2;
}

impl DemandUnit for Dbufrd {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        ctx.reset(Self::BUFNUM);
        ctx.reset(Self::PHASE);
        ctx.reset(Self::LOOP);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let bufnum = ctx.demand(Self::BUFNUM);
        let phase = ctx.demand(Self::PHASE);
        if phase.is_nan() {
            return f32::NAN;
        }
        let looping = ctx.demand(Self::LOOP) != 0.0;
        let bufnum = bufnum.max(0.0) as usize;
        match ctx.buffer(bufnum) {
            Some(buffer) if buffer.num_frames() > 0 => {
                // `loopMax`: whole buffer when looping, last frame when not (scsynth's bound).
                let loop_max = (buffer.num_frames() - usize::from(!looping)) as f64;
                let (index, _) = sc_loop(phase as f64, loop_max, looping);
                buffer.data().get(index as usize).copied().unwrap_or(0.0)
            }
            _ => 0.0,
        }
    }
}

/// Constructor for [`Dbufrd`].
pub struct DbufrdCtor;

impl DemandUnitDef for DbufrdCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() != 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(demand_unit_spec(Dbufrd { _pad: 0 }))
    }
}
