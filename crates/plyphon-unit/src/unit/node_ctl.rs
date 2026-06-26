//! In-graph node-control units - plyphon's ports of scsynth's `FreeSelf`/`PauseSelf` (and, later,
//! the `*WhenDone` and by-id variants).
//!
//! These act on the *enclosing synth* from inside the graph by returning a [`DoneAction`] the engine
//! applies after the block: `FreeSelf` frees the synth on a rising trigger, `PauseSelf` pauses it.
//! As in scsynth they pass their trigger input through unchanged (sclang's `^in`), so the value can
//! still feed the rest of the graph.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// `FreeSelf.kr(in)` / `PauseSelf.kr(in)`: on a rising edge of `in` (crossing above zero), return a
/// [`DoneAction`] - free or pause the enclosing synth - applied by the engine after the block. The
/// trigger is passed through to the output (when the unit has one), as scsynth does.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SelfTrig {
    /// Previous-block trigger value, for rising-edge detection.
    prev: f32,
    /// `0`/`1`: whether the unit has an output wire to pass the trigger through to.
    has_output: u32,
    /// The [`DoneAction`] tag to return on a rising edge (`FreeSelf` or `PauseSelf`).
    action: u32,
}

impl SelfTrig {
    const IN: usize = 0;
}

impl Unit for SelfTrig {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let trig = ctx.ins.control(Self::IN);
        if self.has_output != 0 {
            ctx.outs.audio(0).fill(trig);
        }
        let fired = self.prev <= 0.0 && trig > 0.0;
        self.prev = trig;
        if fired {
            DoneAction::from_tag(self.action)
        } else {
            DoneAction::Nothing
        }
    }
}

/// Build a [`SelfTrig`] firing `action` on a rising edge.
fn build_self_trig(ctx: &BuildContext<'_>, action: DoneAction) -> Result<BuiltUnit, BuildError> {
    Ok(unit_spec(SelfTrig {
        prev: 0.0,
        has_output: (ctx.num_outputs > 0) as u32,
        action: action.to_tag(),
    }))
}

/// Constructor for `FreeSelf`.
pub struct FreeSelfCtor;

impl UnitDef for FreeSelfCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        build_self_trig(ctx, DoneAction::FreeSelf)
    }
}

/// Constructor for `PauseSelf`.
pub struct PauseSelfCtor;

impl UnitDef for PauseSelfCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        build_self_trig(ctx, DoneAction::PauseSelf)
    }
}
