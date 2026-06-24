//! `Line` - a line generator that ramps from a start to an end value over a duration.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::rate::Rate;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};

/// `Line.ar/kr(start, end, dur, doneAction)`: ramps linearly from `start` to `end` over `dur`
/// seconds, then holds at `end`. The arguments are latched on the first block (as in SuperCollider).
/// At control rate it advances once per block (producing one output value); at audio rate, once per
/// sample. When the ramp completes it requests its `doneAction` once (e.g. free the enclosing synth).
///
/// `Pod` state for the rt-pool: `f64`s first, then `0`/`1` flags and the [`DoneAction`] tag, with an
/// explicit pad word so the `repr(C)` layout has no implicit padding.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Line {
    value: f64,
    end: f64,
    slope: f64,
    remaining: f64,
    /// `0`/`1`: audio rate (advance per sample) vs control rate (per block).
    audio: u32,
    /// `0`/`1`: whether the done action has already fired.
    done: u32,
    /// The latched done action, as a [`DoneAction::to_tag`] value.
    done_action: u32,
    _pad: u32,
}

impl Line {
    const START: usize = 0;
    const END: usize = 1;
    const DUR: usize = 2;
    const DONE: usize = 3;

    fn advance(&mut self) {
        if self.remaining > 0.0 {
            self.value += self.slope;
            self.remaining -= 1.0;
        } else {
            self.value = self.end;
        }
    }
}

impl Unit for Line {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Latch the ramp arguments from the (now live) inputs, as SuperCollider does at first calc.
        let start = ctx.ins.control(Self::START) as f64;
        let end = ctx.ins.control(Self::END) as f64;
        let dur = (ctx.ins.control(Self::DUR) as f64).max(0.0);
        // Frames to ramp over: samples at audio rate, control blocks at control rate.
        let rate = if self.audio != 0 {
            ctx.audio.sample_rate
        } else {
            ctx.control.sample_rate
        };
        let frames = (dur * rate).max(1.0);
        self.value = start;
        self.end = end;
        self.slope = (end - start) / frames;
        self.remaining = frames;
        let done_action = if ctx.ins.len() > Self::DONE {
            DoneAction::from_code(ctx.ins.control(Self::DONE))
        } else {
            DoneAction::Nothing
        };
        self.done_action = done_action.to_tag();
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.audio != 0 {
            let out = ctx.outs.audio(0);
            for o in out.iter_mut() {
                *o = self.value as f32;
                self.advance();
            }
        } else {
            *ctx.outs.control(0) = self.value as f32;
            self.advance();
        }
        // Signal the done action exactly once, on the block the ramp completes.
        if self.remaining <= 0.0 && self.done == 0 {
            self.done = 1;
            DoneAction::from_tag(self.done_action)
        } else {
            DoneAction::Nothing
        }
    }
}

/// Constructor for [`Line`].
pub struct LineCtor;

impl UnitDef for LineCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // start, end, dur, and an optional trailing doneAction input.
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Line {
            value: 0.0,
            end: 0.0,
            slope: 0.0,
            remaining: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
            done: 0,
            done_action: DoneAction::Nothing.to_tag(),
            _pad: 0,
        }))
    }
}
