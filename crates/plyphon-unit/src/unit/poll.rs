//! `Poll` - posts a running value to the host, plyphon's port of scsynth's `Poll`
//! (`TriggerUGens.cpp`).
//!
//! On each rising edge of `trig`, `Poll` posts `label: value` to the host (the same `NodeMsg`/`Poll`
//! carrier [`Dpoll`](crate::unit::demand) uses, printed by the CLI) and, when `trigid >= 0`, also sends
//! a `/tr [nodeID, trigid, value]` (the same [`Trigger`] path [`SendTrig`](crate::unit::SendTrig) uses).
//! `value` is the polled `in` sample at the edge. Unlike `SendTrig`/`SendReply`, `Poll` has an output:
//! it passes `in` straight through, so it can be spliced inline into a signal chain to watch it.
//!
//! No engine change is needed - `Poll` reuses the emission sinks already on [`ProcessCtx`]. The label
//! is baked from the constant char inputs at build (like `SendReply`); the audio thread only copies it
//! into the bounded inline carrier.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    BuiltUnit, DoneAction, MAX_LABEL, MAX_VALUES, NodeMsg, NodeMsgKind, ProcessCtx, Trigger, Unit,
    unit_spec,
};
use plyphon_dsp::rate::Rate;

/// `Poll.ar/kr(trig, in, trigid, label)`: post `label: in` to the host on each rising `trig`, pass
/// `in` through. Inputs follow scsynth's layout: `[trig, in, trigid, labelLen, labelChars...]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Poll {
    /// Previous trigger sample, for rising-edge detection across blocks.
    prev_trig: f32,
    /// The label bytes, baked from the constant char inputs.
    label: [u8; MAX_LABEL],
    /// Valid byte length of `label`.
    label_len: u32,
    /// `0`/`1`: control-rate (one value) vs audio-rate (a full block) output.
    audio: u32,
}

impl Poll {
    const TRIG: usize = 0;
    const IN: usize = 1;
    const TRIGID: usize = 2;
    const LABEL_LEN: usize = 3;
    const FIRST_CHAR: usize = 4;

    /// The console-post carrier for one firing (`kind = Poll`, a single value).
    fn message(&self, node: i32, trigid: i32, value: f32) -> NodeMsg {
        let mut values = [0.0f32; MAX_VALUES];
        values[0] = value;
        NodeMsg {
            node,
            reply_id: trigid,
            kind: NodeMsgKind::Poll,
            label: self.label,
            label_len: self.label_len,
            values,
            num_values: 1,
        }
    }
}

impl Unit for Poll {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins; // `Inputs` is `Copy` (borrows the wires, not `ctx`).
        let node = ctx.node_id;
        let trigid = ins.control(Self::TRIGID) as i32;
        let in_audio = (ins.rate(Self::IN) == Rate::Audio).then(|| ins.audio(Self::IN));
        let in_ctrl = ins.control(Self::IN);

        // Fire on each rising edge, sampling the polled `in` value at the edge.
        let fire = |ctx: &mut ProcessCtx<'_>, value: f32| {
            ctx.node_msgs.push(self.message(node, trigid, value));
            if trigid >= 0 {
                ctx.triggers.push(Trigger {
                    node,
                    id: trigid,
                    value,
                });
            }
        };
        if ins.rate(Self::TRIG) == Rate::Audio {
            let trig = ins.audio(Self::TRIG);
            let mut prev = self.prev_trig;
            for (i, &cur) in trig.iter().enumerate() {
                if prev <= 0.0 && cur > 0.0 {
                    fire(ctx, in_audio.map_or(in_ctrl, |s| s[i]));
                }
                prev = cur;
            }
            self.prev_trig = prev;
        } else {
            let cur = ins.control(Self::TRIG);
            if self.prev_trig <= 0.0 && cur > 0.0 {
                fire(ctx, in_ctrl);
            }
            self.prev_trig = cur;
        }

        // Pass `in` through to the output.
        if self.audio != 0 {
            for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
                *o = in_audio.map_or(in_ctrl, |s| s[i]);
            }
        } else {
            *ctx.outs.control(0) = in_ctrl;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Poll`] - bakes the label from the constant char inputs.
pub struct PollCtor;

impl UnitDef for PollCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let count = ctx.input_rates.len();
        // Needs at least `trig`, `in`, `trigid`, `labelLen`.
        if count < Poll::FIRST_CHAR {
            return Err(BuildError::WrongInputCount);
        }
        let len = ctx
            .const_input(Poll::LABEL_LEN)
            .ok_or(BuildError::EmitBadLabel)? as usize;
        if len > MAX_LABEL {
            return Err(BuildError::EmitLabelTooLong {
                len,
                limit: MAX_LABEL,
            });
        }
        if count < Poll::FIRST_CHAR + len {
            return Err(BuildError::EmitBadLabel);
        }
        let mut label = [0u8; MAX_LABEL];
        for (i, b) in label.iter_mut().take(len).enumerate() {
            *b = ctx
                .const_input(Poll::FIRST_CHAR + i)
                .ok_or(BuildError::EmitBadLabel)? as u8;
        }
        Ok(unit_spec(Poll {
            prev_trig: 0.0,
            label,
            label_len: len as u32,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
