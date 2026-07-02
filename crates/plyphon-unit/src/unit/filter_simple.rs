//! Fixed-coefficient and delay filters - plyphon's ports of scsynth's `LPZ1`, `HPZ1`, `LPZ2`,
//! `HPZ2`, `BPZ2`, `BRZ2`, `Delay1`, `Delay2`, `Slope`, `Slew` and `APF`.
//!
//! These carry only a sample or two of history and either fixed coefficients (the `*Z*` FIR
//! filters), none at all (the unit delays and `Slope`), or a two-pole allpass (`APF`). The FIR
//! filters and `Slope`/`Slew` seed their history from the current input in [`Unit::init`], matching
//! scsynth's constructors. Feedback state (`APF`) is `f64`, flushed with `zap`.

use core::f64::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `LPZ1.ar(in)`: a two-point averaging low-pass, `out = 0.5 * (in(i) + in(i-1))`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LPZ1 {
    x1: f64,
}

impl Unit for LPZ1 {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.x1 = ctx.ins.control(0) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut x1 = self.x1;
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let x0 = x as f64;
            *o = (0.5 * (x0 + x1)) as f32;
            x1 = x0;
        }
        self.x1 = x1;
        DoneAction::Nothing
    }
}

/// `HPZ1.ar(in)`: a two-point differencing high-pass, `out = 0.5 * (in(i) - in(i-1))`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HPZ1 {
    x1: f64,
}

impl Unit for HPZ1 {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.x1 = ctx.ins.control(0) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut x1 = self.x1;
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let x0 = x as f64;
            *o = (0.5 * (x0 - x1)) as f32;
            x1 = x0;
        }
        self.x1 = x1;
        DoneAction::Nothing
    }
}

/// `Slope.ar(in)`: the slope of the signal, `out = sampleRate * (in(i) - in(i-1))`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Slope {
    x1: f64,
}

impl Unit for Slope {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.x1 = ctx.ins.control(0) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.own.sample_rate;
        let mut x1 = self.x1;
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let x0 = x as f64;
            *o = (sr * (x0 - x1)) as f32;
            x1 = x0;
        }
        self.x1 = x1;
        DoneAction::Nothing
    }
}

/// A one-input FIR filter with two samples of input history and fixed coefficients.
macro_rules! fir2 {
    ($name:ident, $ctor:ident, $doc:expr, |$x0:ident, $x1:ident, $x2:ident| $body:expr) => {
        #[doc = $doc]
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        pub struct $name {
            x1: f64,
            x2: f64,
        }

        impl Unit for $name {
            fn init(&mut self, ctx: &InitCtx<'_>) {
                self.x1 = ctx.ins.control(0) as f64;
                self.x2 = self.x1;
            }

            fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
                let (mut $x1, mut $x2) = (self.x1, self.x2);
                for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
                    let $x0 = x as f64;
                    *o = ($body) as f32;
                    $x2 = $x1;
                    $x1 = $x0;
                }
                self.x1 = $x1;
                self.x2 = $x2;
                DoneAction::Nothing
            }
        }

        #[doc = concat!("Constructor for [`", stringify!($name), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.is_empty() {
                    return Err(BuildError::WrongInputCount);
                }
                Ok(unit_spec($name { x1: 0.0, x2: 0.0 }))
            }
        }
    };
}

fir2!(
    LPZ2,
    LPZ2Ctor,
    "`LPZ2.ar(in)`: a two-zero low-pass, `out = 0.25 * (in(i) + 2*in(i-1) + in(i-2))`.",
    |x0, x1, x2| (x0 + 2.0 * x1 + x2) * 0.25
);
fir2!(
    HPZ2,
    HPZ2Ctor,
    "`HPZ2.ar(in)`: a two-zero high-pass, `out = 0.25 * (in(i) - 2*in(i-1) + in(i-2))`.",
    |x0, x1, x2| (x0 - 2.0 * x1 + x2) * 0.25
);
fir2!(
    BPZ2,
    BPZ2Ctor,
    "`BPZ2.ar(in)`: a two-zero band-pass, `out = 0.5 * (in(i) - in(i-2))`.",
    |x0, x1, x2| {
        let _ = x1;
        (x0 - x2) * 0.5
    }
);
fir2!(
    BRZ2,
    BRZ2Ctor,
    "`BRZ2.ar(in)`: a two-zero band-reject, `out = 0.5 * (in(i) + in(i-2))`.",
    |x0, x1, x2| {
        let _ = x1;
        (x0 + x2) * 0.5
    }
);

/// The `LPZ1`/`HPZ1`/`Slope` constructors, whose single-sample history seeds from the input.
macro_rules! fir1_ctor {
    ($ctor:ident, $name:ident) => {
        #[doc = concat!("Constructor for [`", stringify!($name), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.is_empty() {
                    return Err(BuildError::WrongInputCount);
                }
                Ok(unit_spec($name { x1: 0.0 }))
            }
        }
    };
}

fir1_ctor!(LPZ1Ctor, LPZ1);
fir1_ctor!(HPZ1Ctor, HPZ1);
fir1_ctor!(SlopeCtor, Slope);

/// `Delay1.ar(in)`: a one-sample delay, `out(i) = in(i-1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Delay1 {
    x1: f64,
}

impl Unit for Delay1 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut x1 = self.x1;
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            *o = x1 as f32;
            x1 = x as f64;
        }
        self.x1 = x1;
        DoneAction::Nothing
    }
}

/// Constructor for [`Delay1`].
pub struct Delay1Ctor;

impl UnitDef for Delay1Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Delay1 { x1: 0.0 }))
    }
}

/// `Delay2.ar(in)`: a two-sample delay, `out(i) = in(i-2)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Delay2 {
    x1: f64,
    x2: f64,
}

impl Unit for Delay2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (mut x1, mut x2) = (self.x1, self.x2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            *o = x2 as f32;
            x2 = x1;
            x1 = x as f64;
        }
        self.x1 = x1;
        self.x2 = x2;
        DoneAction::Nothing
    }
}

/// Constructor for [`Delay2`].
pub struct Delay2Ctor;

impl UnitDef for Delay2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Delay2 { x1: 0.0, x2: 0.0 }))
    }
}

/// `Slew.ar(in, up, dn)`: a slew-rate limiter clamping the change per second to `up` (rising) and
/// `dn` (falling).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Slew {
    level: f64,
}

impl Slew {
    const IN: usize = 0;
    const UP: usize = 1;
    const DN: usize = 2;
}

impl Unit for Slew {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.level = ctx.ins.control(Self::IN) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sample_dur = 1.0 / ctx.own.sample_rate;
        let upf = ctx.ins.control(Self::UP) as f64 * sample_dur;
        let dnf = -(ctx.ins.control(Self::DN) as f64) * sample_dur;
        let mut level = self.level;
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let slope = x as f64 - level;
            level += slope.min(upf).max(dnf); // clip(slope, dnf, upf)
            *o = level as f32;
        }
        self.level = level;
        DoneAction::Nothing
    }
}

/// Constructor for [`Slew`].
pub struct SlewCtor;

impl UnitDef for SlewCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Slew { level: 0.0 }))
    }
}

/// `APF.ar(in, freq, radius)`: a two-pole all-pass, passing all frequencies with a
/// frequency-dependent phase shift.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct APF {
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    x1: f64,
    x2: f64,
    freq: f32,
    reson: f32,
}

impl Unit for APF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(1);
        let reson = ctx.ins.control(2);
        if freq != self.freq || reson != self.reson {
            let w = freq as f64 * TAU / ctx.own.sample_rate;
            self.b1 = 2.0 * reson as f64 * math::cos(w);
            self.b2 = -(reson as f64 * reson as f64);
            self.freq = freq;
            self.reson = reson;
        }
        let (b1, b2) = (self.b1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        let (mut x1, mut x2) = (self.x1, self.x2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let x0 = x as f64;
            let y0 = x0 + b1 * (y1 - x1) + b2 * (y2 - x2);
            *o = y0 as f32;
            y2 = y1;
            y1 = y0;
            x2 = x1;
            x1 = x0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        self.x1 = x1;
        self.x2 = x2;
        DoneAction::Nothing
    }
}

/// Constructor for [`APF`].
pub struct APFCtor;

impl UnitDef for APFCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(APF {
            b1: 0.0,
            b2: 0.0,
            y1: 0.0,
            y2: 0.0,
            x1: 0.0,
            x2: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            reson: f32::NAN,
        }))
    }
}
