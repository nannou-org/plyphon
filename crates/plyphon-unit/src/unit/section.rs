//! Explicit-coefficient filter sections - plyphon's ports of scsynth's `FOS` and `SOS`.
//!
//! Unlike the resonant/biquad units (which derive their coefficients from a `freq`/`rq`), these take
//! the raw difference-equation coefficients as inputs, so any filter shape can be built on top of them.
//! In particular scsynth's whole `B*` EQ suite (`BLowPass`/`BHiPass`/`BPeakEQ`/…) is a *language-side*
//! macro that computes RBJ biquad coefficients and feeds this [`SOS`]; a compiled SynthDef contains the
//! coefficient math plus an `SOS`, so porting `SOS` covers the entire B-series.
//!
//! Both are direct-form-II sections (the feedback `b*` terms are *added*, matching scsynth's sign
//! convention). Coefficients are read once per block (plyphon's block-rate convention; scsynth
//! additionally `CALCSLOPE`-interpolates control-rate coefficients across the block). The `f64`
//! feedback state is flushed with the shared `zap` helper.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// `FOS.ar(in, a0, a1, b1)`: a first-order section `y(i) = a0*x(i) + a1*x(i-1) + b1*y(i-1)`,
/// implemented in direct form II (`w = x + b1*w1; y = a0*w + a1*w1`). Sweeping the coefficients morphs
/// between first-order low-pass, high-pass and all-pass responses.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FOS {
    y1: f64,
}

impl FOS {
    const IN: usize = 0;
    const A0: usize = 1;
    const A1: usize = 2;
    const B1: usize = 3;
}

impl Unit for FOS {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a0 = ctx.ins.control(Self::A0) as f64;
        let a1 = ctx.ins.control(Self::A1) as f64;
        let b1 = ctx.ins.control(Self::B1) as f64;
        let mut y1 = self.y1;
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let y0 = x as f64 + b1 * y1;
            *o = (a0 * y0 + a1 * y1) as f32;
            y1 = y0;
        }
        self.y1 = zap(y1);
        DoneAction::Nothing
    }
}

/// Constructor for [`FOS`].
pub struct FOSCtor;

impl UnitDef for FOSCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(FOS { y1: 0.0 }))
    }
}

/// `SOS.ar(in, a0, a1, a2, b1, b2)`: a second-order section (biquad)
/// `y(i) = a0*x(i) + a1*x(i-1) + a2*x(i-2) + b1*y(i-1) + b2*y(i-2)`, in direct form II
/// (`w = x + b1*w1 + b2*w2; y = a0*w + a1*w1 + a2*w2`). scsynth's `B*` EQ macros feed their computed
/// RBJ coefficients into this.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SOS {
    y1: f64,
    y2: f64,
}

impl SOS {
    const IN: usize = 0;
    const A0: usize = 1;
    const A1: usize = 2;
    const A2: usize = 3;
    const B1: usize = 4;
    const B2: usize = 5;
}

impl Unit for SOS {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a0 = ctx.ins.control(Self::A0) as f64;
        let a1 = ctx.ins.control(Self::A1) as f64;
        let a2 = ctx.ins.control(Self::A2) as f64;
        let b1 = ctx.ins.control(Self::B1) as f64;
        let b2 = ctx.ins.control(Self::B2) as f64;
        let (mut y1, mut y2) = (self.y1, self.y2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let y0 = x as f64 + b1 * y1 + b2 * y2;
            *o = (a0 * y0 + a1 * y1 + a2 * y2) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// Constructor for [`SOS`].
pub struct SOSCtor;

impl UnitDef for SOSCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 6 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(SOS { y1: 0.0, y2: 0.0 }))
    }
}
