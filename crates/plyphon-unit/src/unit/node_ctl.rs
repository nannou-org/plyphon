//! In-graph node-control units - plyphon's ports of scsynth's `FreeSelf`/`PauseSelf`, `Done`,
//! `FreeSelfWhenDone`/`PauseSelfWhenDone`, and the by-id `Free`/`Pause`.
//!
//! Most act on the *enclosing synth* by returning a [`DoneAction`] the engine applies after the
//! block: `FreeSelf`/`PauseSelf` fire on a rising trigger; `FreeSelfWhenDone`/`PauseSelfWhenDone`
//! fire once a watched source unit finishes (reading its done flag - scsynth's `mDone`). `Done` is a
//! pure observer that outputs `1` once its source is done. `Free`/`Pause` instead act on *another*
//! node by id, emitting a deferred node op the engine applies after the tree walk. The trigger/source
//! signal is passed through to the output (sclang's `^in`), so it can still feed the rest of the
//! graph.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, NodeOp, NodeOpKind, ProcessCtx, Unit, unit_spec};

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

/// The source unit a `Done`/`*WhenDone` watcher observes: the calc-unit index of input 0's producer
/// (scsynth's `mInput[0]->mFromUnit`), captured at build, or `u32::MAX` if input 0 is not a calc unit
/// (a constant or parameter) - in which case the source is treated as never done.
fn source_unit(ctx: &BuildContext<'_>) -> u32 {
    ctx.input_units
        .first()
        .copied()
        .flatten()
        .unwrap_or(u32::MAX)
}

/// Whether the watched source unit (calc index `src`) has finished. `u32::MAX` (no calc source) is
/// never done, matching scsynth's `if (src) … else 0`.
fn source_done(src: u32, ctx: &ProcessCtx<'_>) -> bool {
    src != u32::MAX && ctx.done.is_done(src as usize)
}

/// `Done.kr(src)`: outputs `1` once the source unit (e.g. an `EnvGen`/`Line`/`PlayBuf`) has finished,
/// `0` until then. A pure observer - it takes no node action.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Done {
    /// Calc-unit index of the watched source, or `u32::MAX` for "no source".
    src: u32,
    /// `0`/`1`: whether the unit has an output wire to write the done flag to.
    has_output: u32,
}

impl Unit for Done {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.has_output != 0 {
            let value = if source_done(self.src, ctx) { 1.0 } else { 0.0 };
            ctx.outs.audio(0).fill(value);
        }
        DoneAction::Nothing
    }
}

/// Constructor for `Done`.
pub struct DoneCtor;

impl UnitDef for DoneCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Done {
            src: source_unit(ctx),
            has_output: (ctx.num_outputs > 0) as u32,
        }))
    }
}

/// `FreeSelfWhenDone.kr(src)` / `PauseSelfWhenDone.kr(src)`: when the source unit finishes, return a
/// [`DoneAction`] - free or pause the enclosing synth. The source's signal is passed through to the
/// output (as scsynth does), so the unit can be inserted transparently in a chain.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct WhenDone {
    /// Calc-unit index of the watched source, or `u32::MAX` for "no source".
    src: u32,
    /// `0`/`1`: whether the unit has an output wire to pass the source signal through to.
    has_output: u32,
    /// The [`DoneAction`] tag to return once the source is done (`FreeSelf` or `PauseSelf`).
    action: u32,
}

impl WhenDone {
    const IN: usize = 0;
}

impl Unit for WhenDone {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let done = source_done(self.src, ctx);
        if self.has_output != 0 {
            let passthrough = ctx.ins.control(Self::IN);
            ctx.outs.audio(0).fill(passthrough);
        }
        if done {
            DoneAction::from_tag(self.action)
        } else {
            DoneAction::Nothing
        }
    }
}

/// Build a [`WhenDone`] firing `action` once its source is done.
fn build_when_done(ctx: &BuildContext<'_>, action: DoneAction) -> Result<BuiltUnit, BuildError> {
    Ok(unit_spec(WhenDone {
        src: source_unit(ctx),
        has_output: (ctx.num_outputs > 0) as u32,
        action: action.to_tag(),
    }))
}

/// Constructor for `FreeSelfWhenDone`.
pub struct FreeSelfWhenDoneCtor;

impl UnitDef for FreeSelfWhenDoneCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        build_when_done(ctx, DoneAction::FreeSelf)
    }
}

/// Constructor for `PauseSelfWhenDone`.
pub struct PauseSelfWhenDoneCtor;

impl UnitDef for PauseSelfWhenDoneCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        build_when_done(ctx, DoneAction::PauseSelf)
    }
}

/// `Free.kr(trig, id)`: on a rising edge of `trig`, free the node with client id `id` (any node, not
/// just the enclosing synth). The engine applies the free after the block. The trigger is passed
/// through to the output.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Free {
    /// Previous-block trigger value, for rising-edge detection.
    prev: f32,
    /// `0`/`1`: whether the unit has an output wire to pass the trigger through to.
    has_output: u32,
}

impl Free {
    const TRIG: usize = 0;
    const ID: usize = 1;
}

impl Unit for Free {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let trig = ctx.ins.control(Self::TRIG);
        let id = ctx.ins.control(Self::ID) as i32;
        if self.prev <= 0.0 && trig > 0.0 {
            ctx.node_ops.push(NodeOp {
                node: id,
                kind: NodeOpKind::Free,
            });
        }
        self.prev = trig;
        if self.has_output != 0 {
            ctx.outs.audio(0).fill(trig);
        }
        DoneAction::Nothing
    }
}

/// Constructor for `Free`.
pub struct FreeCtor;

impl UnitDef for FreeCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Free {
            prev: 0.0,
            has_output: (ctx.num_outputs > 0) as u32,
        }))
    }
}

/// `Pause.kr(gate, id)`: pause (gate `0`) or resume (gate non-zero) the node with client id `id`
/// whenever the gate's state *changes*, applied after the block. Starts assuming the target runs
/// (scsynth's `m_state = 1`), so a gate that begins low pauses it. The gate is passed through.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Pause {
    /// Last gate state (`0` = pause requested, `1` = run requested). Starts at `1`, as scsynth does.
    state: u32,
    /// `0`/`1`: whether the unit has an output wire to pass the gate through to.
    has_output: u32,
}

impl Pause {
    const GATE: usize = 0;
    const ID: usize = 1;
}

impl Unit for Pause {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let gate = ctx.ins.control(Self::GATE);
        let id = ctx.ins.control(Self::ID) as i32;
        let new_state = (gate != 0.0) as u32;
        if new_state != self.state {
            self.state = new_state;
            ctx.node_ops.push(NodeOp {
                node: id,
                kind: NodeOpKind::Run(new_state != 0),
            });
        }
        if self.has_output != 0 {
            ctx.outs.audio(0).fill(gate);
        }
        DoneAction::Nothing
    }
}

/// Constructor for `Pause`.
pub struct PauseCtor;

impl UnitDef for PauseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Pause {
            state: 1,
            has_output: (ctx.num_outputs > 0) as u32,
        }))
    }
}
