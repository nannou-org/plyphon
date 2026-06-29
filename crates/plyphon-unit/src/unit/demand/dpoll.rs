//! `Dpoll` - a demand-rate poll/post, plyphon's port of scsynth's `Dpoll`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::MAX_LABEL;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};

/// `Dpoll(in, label, run, trigid)`: on each demand, pulls `in`, posts `label: value` to the host when
/// `run` is non-zero (and the value is not the exhaustion `NaN`), then returns the value - a
/// pass-through. The post has no OSC form; the server prints it to the console (see
/// [`NodeMsgKind::Poll`](crate::unit::NodeMsgKind::Poll)). The label is baked at compile time from
/// constant char inputs, exactly as [`SendReply`](crate::unit::SendReply) encodes its path.
///
/// Inputs follow scsynth's server-side layout: `[in, trigid, run, labelLen, labelChars...]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dpoll {
    /// The label bytes (UTF-8), baked from the constant char inputs.
    label: [u8; MAX_LABEL],
    /// Valid byte length of `label`.
    label_len: u32,
    /// The trigger id echoed with the post (scsynth's `trigid`).
    trigid: i32,
}

impl Dpoll {
    const IN: usize = 0;
    const TRIGID: usize = 1;
    const RUN: usize = 2;
    const LABEL_LEN: usize = 3;
    /// First label-char input index.
    const FIRST_CHAR: usize = 4;
}

impl DemandUnit for Dpoll {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        ctx.reset(Self::IN);
        ctx.reset(Self::RUN);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let value = ctx.demand(Self::IN);
        let run = ctx.demand(Self::RUN);
        if run > 0.0 && !value.is_nan() {
            ctx.post(&self.label, self.label_len, self.trigid, value);
        }
        value
    }
}

/// Constructor for [`Dpoll`] - bakes the label from the constant char inputs and reads the trigger id.
pub struct DpollCtor;

impl DemandUnitDef for DpollCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        let count = ctx.input_rates.len();
        // Needs at least `in`, `trigid`, `run`, `labelLen`.
        if count < Dpoll::FIRST_CHAR {
            return Err(BuildError::WrongInputCount);
        }
        let trigid = ctx.const_input(Dpoll::TRIGID).unwrap_or(-1.0) as i32;
        let len = ctx
            .const_input(Dpoll::LABEL_LEN)
            .ok_or(BuildError::EmitBadLabel)? as usize;
        if len > MAX_LABEL {
            return Err(BuildError::EmitLabelTooLong {
                len,
                limit: MAX_LABEL,
            });
        }
        // The label's chars must all be present.
        if count < Dpoll::FIRST_CHAR + len {
            return Err(BuildError::EmitBadLabel);
        }
        let mut label = [0u8; MAX_LABEL];
        for (i, b) in label.iter_mut().take(len).enumerate() {
            *b = ctx
                .const_input(Dpoll::FIRST_CHAR + i)
                .ok_or(BuildError::EmitBadLabel)? as u8;
        }
        Ok(demand_unit_spec(Dpoll {
            label,
            label_len: len as u32,
            trigid,
        }))
    }
}
