//! `SendTrig` - sends a `/tr` message to clients on each rising edge of its trigger input.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Trigger, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// `SendTrig.kr(in, id, value)` (or `.ar`): whenever `in` crosses from `<= 0` to `> 0`, sends a
/// `/tr [nodeID, id, value]` message to notified clients, with `value` sampled at the trigger. It has
/// no signal output - it runs purely for the side effect. At audio rate it tests every sample and can
/// fire several times per block, as scsynth does; at control rate it tests once per block.
///
/// `Pod` state for the rt-pool: just the previous trigger sample, for rising-edge detection across
/// block boundaries.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SendTrig {
    prev_trig: f32,
}

impl SendTrig {
    const IN: usize = 0;
    const ID: usize = 1;
    const VALUE: usize = 2;
}

impl Unit for SendTrig {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let id = ctx.ins.control(Self::ID) as i32;
        let node = ctx.node_id;
        if ctx.ins.rate(Self::IN) == Rate::Audio {
            // Audio-rate trigger: scan the block, firing on every rising edge. A per-sample `value`
            // (audio rate) is sampled at the edge; otherwise the block's single control value is used.
            let trig = ctx.ins.audio(Self::IN);
            let value_audio =
                (ctx.ins.rate(Self::VALUE) == Rate::Audio).then(|| ctx.ins.audio(Self::VALUE));
            let value_ctrl = ctx.ins.control(Self::VALUE);
            let mut prev = self.prev_trig;
            for (i, &cur) in trig.iter().enumerate() {
                if prev <= 0.0 && cur > 0.0 {
                    let value = value_audio.map_or(value_ctrl, |v| v[i]);
                    ctx.triggers.push(Trigger { node, id, value });
                }
                prev = cur;
            }
            self.prev_trig = prev;
        } else {
            let cur = ctx.ins.control(Self::IN);
            if self.prev_trig <= 0.0 && cur > 0.0 {
                let value = ctx.ins.control(Self::VALUE);
                ctx.triggers.push(Trigger { node, id, value });
            }
            self.prev_trig = cur;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`SendTrig`] - it declares no outputs (it is purely side-effecting).
pub struct SendTrigCtor;

impl UnitDef for SendTrigCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(SendTrig { prev_trig: 0.0 }))
    }
}
