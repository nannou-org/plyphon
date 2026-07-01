//! First-order filters - plyphon's ports of scsynth's `OnePole`, `OneZero`, `Integrator`, `LeakDC`.
//!
//! Each carries a single sample of history and one coefficient read from an input. Following the
//! same convention as [`Butter`](crate::unit::filter::Butter), the coefficient is read once per
//! block (control rate) rather than slope-interpolated per sample as scsynth does - so the transfer
//! function matches scsynth for a steady coefficient, without the per-sample de-zippering. State is
//! kept in `f64` and flushed with `zap` (scsynth's `zapgremlins`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};

/// `OnePole.ar(in, coef)`: a one-pole filter, `out(i) = (1 - |coef|) * in(i) + coef * out(i-1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OnePole {
    y1: f64,
}

impl Unit for OnePole {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let b1 = ctx.ins.control(1) as f64;
        let mut y1 = self.y1;
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        // scsynth splits on the sign of the coefficient so a negative pole stays stable; both branches
        // reduce to `(1 - |b1|) * y0 + b1 * y1`.
        if b1 >= 0.0 {
            for (o, &x) in out.iter_mut().zip(input) {
                let y0 = x as f64;
                y1 = y0 + b1 * (y1 - y0);
                *o = y1 as f32;
            }
        } else {
            for (o, &x) in out.iter_mut().zip(input) {
                let y0 = x as f64;
                y1 = y0 + b1 * (y1 + y0);
                *o = y1 as f32;
            }
        }
        self.y1 = zap(y1);
        DoneAction::Nothing
    }
}

/// Constructor for [`OnePole`].
pub struct OnePoleCtor;

impl UnitDef for OnePoleCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(OnePole { y1: 0.0 }))
    }
}

/// `OneZero.ar(in, coef)`: a one-zero filter, `out(i) = (1 - |coef|) * in(i) + coef * in(i-1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct OneZero {
    x1: f64,
}

impl Unit for OneZero {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Seed the input history with the current input (scsynth's `m_x1 = ZIN0(0)`).
        self.x1 = ctx.ins.control(0) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let b1 = ctx.ins.control(1) as f64;
        let mut x1 = self.x1;
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        if b1 >= 0.0 {
            for (o, &x) in out.iter_mut().zip(input) {
                let x0 = x as f64;
                *o = (x0 + b1 * (x1 - x0)) as f32;
                x1 = x0;
            }
        } else {
            for (o, &x) in out.iter_mut().zip(input) {
                let x0 = x as f64;
                *o = (x0 + b1 * (x1 + x0)) as f32;
                x1 = x0;
            }
        }
        self.x1 = x1;
        DoneAction::Nothing
    }
}

/// Constructor for [`OneZero`].
pub struct OneZeroCtor;

impl UnitDef for OneZeroCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(OneZero { x1: 0.0 }))
    }
}

/// `Integrator.ar(in, coef)`: a leaky integrator, `out(i) = in(i) + coef * out(i-1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Integrator {
    y1: f64,
}

impl Unit for Integrator {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let b1 = ctx.ins.control(1) as f64;
        let mut y1 = self.y1;
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            y1 = x as f64 + b1 * y1;
            *o = y1 as f32;
        }
        self.y1 = zap(y1);
        DoneAction::Nothing
    }
}

/// Constructor for [`Integrator`].
pub struct IntegratorCtor;

impl UnitDef for IntegratorCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Integrator { y1: 0.0 }))
    }
}

/// `LeakDC.ar(in, coef)`: a DC-blocking filter, `out(i) = in(i) - in(i-1) + coef * out(i-1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LeakDC {
    x1: f64,
    y1: f64,
}

impl Unit for LeakDC {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Seed the input history (scsynth's `m_x1 = ZIN0(0)`), so the first sample does not step.
        self.x1 = ctx.ins.control(0) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let b1 = ctx.ins.control(1) as f64;
        let mut x1 = self.x1;
        let mut y1 = self.y1;
        let input = ctx.ins.audio(0);
        let out = ctx.outs.audio(0);
        for (o, &x) in out.iter_mut().zip(input) {
            let x0 = x as f64;
            y1 = x0 - x1 + b1 * y1;
            x1 = x0;
            *o = y1 as f32;
        }
        self.x1 = x1;
        self.y1 = zap(y1);
        DoneAction::Nothing
    }
}

/// Constructor for [`LeakDC`].
pub struct LeakDCCtor;

impl UnitDef for LeakDCCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(LeakDC { x1: 0.0, y1: 0.0 }))
    }
}
