//! `SendReply` - emits an OSC message `/<path> [nodeID, replyID, values...]` on each rising edge.
//!
//! plyphon's port of scsynth's `SendReply`. The OSC path is encoded as constant float inputs (one
//! char each), matching scsynth so SCgf-compiled defs load; the path is decoded once in `build()` and
//! baked into the unit's `Pod` state, so the audio thread only copies it into the bounded inline
//! [`NodeMsg`] carrier - no allocation. Like [`SendTrig`](crate::unit::SendTrig) it has no output and
//! fires on every rising edge (per sample at audio rate, once per block at control rate).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    BuiltUnit, DoneAction, Inputs, MAX_LABEL, MAX_VALUES, NodeMsg, NodeMsgKind, ProcessCtx, Unit,
    unit_spec,
};
use plyphon_dsp::rate::Rate;

/// `SendReply.kr(trig, cmdName, values, replyID)` (or `.ar`): on a rising edge of `trig`, emits
/// `/<cmdName> [nodeID, replyID, values...]`. The path and value count are fixed at compile time.
///
/// Inputs follow scsynth's layout: `[trig, replyID, cmdNameLen, cmdNameChars..., values...]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SendReply {
    /// Previous trigger sample, for rising-edge detection across blocks.
    prev_trig: f32,
    /// The OSC path bytes, baked from the constant char inputs (a byte array is `Pod`).
    label: [u8; MAX_LABEL],
    /// Valid byte length of `label`.
    label_len: u32,
    /// First value input index (`3 + label_len`).
    values_start: u32,
    /// Number of value inputs (`<= MAX_VALUES`).
    num_values: u32,
}

impl SendReply {
    const IN: usize = 0;
    const REPLY_ID: usize = 1;
    const LABEL_LEN: usize = 2;

    /// Build the [`NodeMsg`] for one firing, sampling each value input. `sample` is the within-block
    /// index for an audio-rate trigger (so audio-rate values are sampled at the edge); `None` at
    /// control rate.
    fn message(
        &self,
        ins: &Inputs<'_>,
        node: i32,
        reply_id: i32,
        sample: Option<usize>,
    ) -> NodeMsg {
        let mut values = [0.0f32; MAX_VALUES];
        for (j, v) in values.iter_mut().take(self.num_values as usize).enumerate() {
            let idx = self.values_start as usize + j;
            *v = match sample {
                Some(i) if ins.rate(idx) == Rate::Audio => ins.audio(idx)[i],
                _ => ins.control(idx),
            };
        }
        NodeMsg {
            node,
            reply_id,
            kind: NodeMsgKind::Reply,
            label: self.label,
            label_len: self.label_len,
            values,
            num_values: self.num_values,
        }
    }
}

impl Unit for SendReply {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins; // `Inputs` is `Copy` and borrows the wires, not `ctx` - so we can also
        let node = ctx.node_id; // push to the disjoint `ctx.node_msgs` field below.
        let reply_id = ins.control(Self::REPLY_ID) as i32;
        if ins.rate(Self::IN) == Rate::Audio {
            // Audio-rate trigger: scan the block, firing on every rising edge.
            let trig = ins.audio(Self::IN);
            let mut prev = self.prev_trig;
            for (i, &cur) in trig.iter().enumerate() {
                if prev <= 0.0 && cur > 0.0 {
                    let msg = self.message(&ins, node, reply_id, Some(i));
                    ctx.node_msgs.push(msg);
                }
                prev = cur;
            }
            self.prev_trig = prev;
        } else {
            let cur = ins.control(Self::IN);
            if self.prev_trig <= 0.0 && cur > 0.0 {
                let msg = self.message(&ins, node, reply_id, None);
                ctx.node_msgs.push(msg);
            }
            self.prev_trig = cur;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`SendReply`] - decodes the OSC path from the constant char inputs and validates
/// the path length and value count against the inline carrier's bounds. Declares no outputs.
pub struct SendReplyCtor;

impl UnitDef for SendReplyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        // Needs at least `trig`, `replyID`, `cmdNameLen`.
        if count < 3 {
            return Err(BuildError::WrongInputCount);
        }
        let len = ctx
            .const_input(SendReply::LABEL_LEN)
            .ok_or(BuildError::EmitBadLabel)? as usize;
        if len > MAX_LABEL {
            return Err(BuildError::EmitLabelTooLong {
                len,
                limit: MAX_LABEL,
            });
        }
        let values_start = 3 + len;
        // The label's chars must all be present.
        if count < values_start {
            return Err(BuildError::EmitBadLabel);
        }
        let num_values = count - values_start;
        if num_values > MAX_VALUES {
            return Err(BuildError::EmitTooManyValues {
                count: num_values,
                limit: MAX_VALUES,
            });
        }
        let mut label = [0u8; MAX_LABEL];
        for (i, b) in label.iter_mut().take(len).enumerate() {
            *b = ctx.const_input(3 + i).ok_or(BuildError::EmitBadLabel)? as u8;
        }
        Ok(unit_spec(SendReply {
            prev_trig: 0.0,
            label,
            label_len: len as u32,
            values_start: values_start as u32,
            num_values: num_values as u32,
        }))
    }
}
