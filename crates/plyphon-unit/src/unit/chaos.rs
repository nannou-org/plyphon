//! Chaotic map generators - plyphon's ports of scsynth's `CuspN`, `QuadN`, `GbmanN`, `LinCongN`,
//! `StandardN`, `LatoocarfianN` (`ChaosUGens.cpp`).
//!
//! Each iterates a chaotic map at a `freq` rate and holds the value between iterations (the `*N`,
//! non-interpolating, sample-and-hold form). Maps and their internal state are computed in `f64`; the
//! `freq` and map coefficients are read once per block. The initial state is seeded from the init
//! inputs (re-seeding on a runtime change of the init inputs is not implemented - the common case
//! uses constants).

use core::f64::consts::PI;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

const TWO_PI: f64 = 2.0 * PI;
const REC_PI: f64 = 1.0 / PI;

/// The hold length in samples for a map running at `freq` Hz (scsynth's `samplesPerCycle`).
fn samples_per_cycle(freq: f32, sr: f32) -> f32 {
    if freq < sr { sr / freq.max(0.001) } else { 1.0 }
}

/// Drive a one-variable map: iterate `map` every `samples_per_cycle` samples, holding between, and
/// write `out(value)` each sample. Returns the final map value.
fn chaos1(
    ctx: &mut ProcessCtx<'_>,
    counter: &mut f32,
    xn: f64,
    mut map: impl FnMut(f64) -> f64,
    out: impl Fn(f64) -> f64,
) -> f64 {
    let spc = samples_per_cycle(ctx.ins.control(0), ctx.own.sample_rate as f32);
    let mut x = xn;
    for o in ctx.outs.audio(0).iter_mut() {
        if *counter >= spc {
            *counter -= spc;
            x = map(x);
        }
        *counter += 1.0;
        *o = out(x) as f32;
    }
    x
}

/// Drive a two-variable map, holding `x` (the output variable) between iterations.
fn chaos2(
    ctx: &mut ProcessCtx<'_>,
    counter: &mut f32,
    xn: f64,
    yn: f64,
    mut map: impl FnMut(f64, f64) -> (f64, f64),
    out: impl Fn(f64) -> f64,
) -> (f64, f64) {
    let spc = samples_per_cycle(ctx.ins.control(0), ctx.own.sample_rate as f32);
    let (mut x, mut y) = (xn, yn);
    for o in ctx.outs.audio(0).iter_mut() {
        if *counter >= spc {
            *counter -= spc;
            let (nx, ny) = map(x, y);
            x = nx;
            y = ny;
        }
        *counter += 1.0;
        *o = out(x) as f32;
    }
    (x, y)
}

/// `CuspN.ar(freq, a, b, xi)`: the cusp map `x = a - b*sqrt(|x|)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CuspN {
    xn: f64,
    counter: f32,
    _pad: u32,
}

impl Unit for CuspN {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.xn = ctx.ins.control(3) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = ctx.ins.control(1) as f64;
        let b = ctx.ins.control(2) as f64;
        self.xn = chaos1(
            ctx,
            &mut self.counter,
            self.xn,
            |x| a - b * math::sqrt(x.abs()),
            |x| x,
        );
        DoneAction::Nothing
    }
}

/// `QuadN.ar(freq, a, b, c, xi)`: the quadratic map `x = a*x^2 + b*x + c`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuadN {
    xn: f64,
    counter: f32,
    _pad: u32,
}

impl Unit for QuadN {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.xn = ctx.ins.control(4) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = ctx.ins.control(1) as f64;
        let b = ctx.ins.control(2) as f64;
        let c = ctx.ins.control(3) as f64;
        self.xn = chaos1(
            ctx,
            &mut self.counter,
            self.xn,
            |x| a * x * x + b * x + c,
            |x| x,
        );
        DoneAction::Nothing
    }
}

/// `LinCongN.ar(freq, a, c, m, xi)`: a linear-congruential generator, scaled to `[-1, 1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LinCongN {
    xn: f64,
    counter: f32,
    _pad: u32,
}

impl Unit for LinCongN {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.xn = ctx.ins.control(4) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = ctx.ins.control(1) as f64;
        let c = ctx.ins.control(2) as f64;
        let m = (ctx.ins.control(3).max(0.001)) as f64;
        self.xn = chaos1(
            ctx,
            &mut self.counter,
            self.xn,
            |x| math::rem_euclid(x * a + c, m),
            |x| x * (2.0 / m) - 1.0,
        );
        DoneAction::Nothing
    }
}

/// `GbmanN.ar(freq, xi, yi)`: the Gingerbreadman map.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GbmanN {
    xn: f64,
    yn: f64,
    counter: f32,
    _pad: u32,
}

impl Unit for GbmanN {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.xn = ctx.ins.control(1) as f64;
        self.yn = ctx.ins.control(2) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (x, y) = chaos2(
            ctx,
            &mut self.counter,
            self.xn,
            self.yn,
            |x, y| {
                let nx = if x < 0.0 { 1.0 - y - x } else { 1.0 - y + x };
                (nx, x)
            },
            |x| x,
        );
        self.xn = x;
        self.yn = y;
        DoneAction::Nothing
    }
}

/// `StandardN.ar(freq, k, xi, yi)`: the standard (kicked-rotor) map, scaled to `[-1, 1)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct StandardN {
    xn: f64,
    yn: f64,
    counter: f32,
    _pad: u32,
}

impl Unit for StandardN {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.xn = ctx.ins.control(2) as f64;
        self.yn = ctx.ins.control(3) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let k = ctx.ins.control(1) as f64;
        let (x, y) = chaos2(
            ctx,
            &mut self.counter,
            self.xn,
            self.yn,
            |x, y| {
                let ny = math::rem_euclid(y + k * math::sin(x), TWO_PI);
                let nx = math::rem_euclid(x + ny, TWO_PI);
                (nx, ny)
            },
            |x| (x - PI) * REC_PI,
        );
        self.xn = x;
        self.yn = y;
        DoneAction::Nothing
    }
}

/// `LatoocarfianN.ar(freq, a, b, c, d, xi, yi)`: the Latoocarfian map.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LatoocarfianN {
    xn: f64,
    yn: f64,
    counter: f32,
    _pad: u32,
}

impl Unit for LatoocarfianN {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.xn = ctx.ins.control(5) as f64;
        self.yn = ctx.ins.control(6) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = ctx.ins.control(1) as f64;
        let b = ctx.ins.control(2) as f64;
        let c = ctx.ins.control(3) as f64;
        let d = ctx.ins.control(4) as f64;
        let (x, y) = chaos2(
            ctx,
            &mut self.counter,
            self.xn,
            self.yn,
            |x, y| {
                let nx = math::sin(y * b) + c * math::sin(x * b);
                let ny = math::sin(x * a) + d * math::sin(y * a);
                (nx, ny)
            },
            |x| x,
        );
        self.xn = x;
        self.yn = y;
        DoneAction::Nothing
    }
}

/// Build a chaos generator with zeroed state and the given minimum input count.
macro_rules! chaos_ctor {
    ($ctor:ident, $unit:ident, $min_inputs:expr, { $($field:ident: $init:expr),* $(,)? }) => {
        #[doc = concat!("Constructor for [`", stringify!($unit), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.len() < $min_inputs {
                    return Err(BuildError::WrongInputCount);
                }
                Ok(unit_spec($unit { $($field: $init,)* counter: 0.0, _pad: 0 }))
            }
        }
    };
}

chaos_ctor!(CuspNCtor, CuspN, 4, { xn: 0.0 });
chaos_ctor!(QuadNCtor, QuadN, 5, { xn: 0.0 });
chaos_ctor!(LinCongNCtor, LinCongN, 5, { xn: 0.0 });
chaos_ctor!(GbmanNCtor, GbmanN, 3, { xn: 0.0, yn: 0.0 });
chaos_ctor!(StandardNCtor, StandardN, 4, { xn: 0.0, yn: 0.0 });
chaos_ctor!(LatoocarfianNCtor, LatoocarfianN, 7, { xn: 0.0, yn: 0.0 });
