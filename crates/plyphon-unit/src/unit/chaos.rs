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

/// Validate an exact fixed-shape chaos unit before it reaches the audio thread.
fn validate_exact_chaos(
    ctx: &BuildContext<'_>,
    inputs: usize,
    outputs: usize,
    node_rates: &[plyphon_dsp::rate::Rate],
) -> Result<(), BuildError> {
    use plyphon_dsp::rate::Rate;

    if ctx.input_rates.len() != inputs {
        return Err(BuildError::WrongInputCount);
    }
    if ctx.num_outputs != outputs {
        return Err(BuildError::WrongOutputCount {
            expected: outputs,
            actual: ctx.num_outputs,
        });
    }
    if ctx.special_index != 0 {
        return Err(BuildError::UnsupportedOp(ctx.special_index));
    }
    if !node_rates.contains(&ctx.rate)
        || ctx.input_rates.iter().any(|rate| {
            *rate == Rate::Demand || (ctx.rate == Rate::Control && *rate == Rate::Audio)
        })
    {
        return Err(BuildError::UnsupportedUnitRate);
    }
    Ok(())
}

/// Read one coordinate sample, expanding scalar/control values and sanitizing non-finite input.
fn coordinate(ctx: &ProcessCtx<'_>, inlet: usize, sample: usize) -> f32 {
    use plyphon_dsp::rate::Rate;

    let value = if ctx.ins.rate(inlet) == Rate::Audio {
        ctx.ins.audio(inlet).get(sample).copied().unwrap_or(0.0)
    } else {
        ctx.ins.control(inlet)
    };
    if value.is_finite() { value } else { 0.0 }
}

/// Ken Perlin's improved three-dimensional gradient-noise function.
fn improved_perlin3(x: f32, y: f32, z: f32) -> f32 {
    let xf = math::floor(x);
    let yf = math::floor(y);
    let zf = math::floor(z);
    let xi = xf as i64 as usize & 255;
    let yi = yf as i64 as usize & 255;
    let zi = zf as i64 as usize & 255;
    let x = x - xf;
    let y = y - yf;
    let z = z - zf;
    let u = perlin_fade(x);
    let v = perlin_fade(y);
    let w = perlin_fade(z);

    let a = PERLIN_PERMUTATION[xi] as usize + yi;
    let aa = PERLIN_PERMUTATION[a & 255] as usize + zi;
    let ab = PERLIN_PERMUTATION[(a + 1) & 255] as usize + zi;
    let b = PERLIN_PERMUTATION[(xi + 1) & 255] as usize + yi;
    let ba = PERLIN_PERMUTATION[b & 255] as usize + zi;
    let bb = PERLIN_PERMUTATION[(b + 1) & 255] as usize + zi;

    let x1 = perlin_lerp(
        u,
        perlin_gradient(PERLIN_PERMUTATION[aa & 255], x, y, z),
        perlin_gradient(PERLIN_PERMUTATION[ba & 255], x - 1.0, y, z),
    );
    let x2 = perlin_lerp(
        u,
        perlin_gradient(PERLIN_PERMUTATION[ab & 255], x, y - 1.0, z),
        perlin_gradient(PERLIN_PERMUTATION[bb & 255], x - 1.0, y - 1.0, z),
    );
    let y1 = perlin_lerp(v, x1, x2);

    let x1 = perlin_lerp(
        u,
        perlin_gradient(PERLIN_PERMUTATION[(aa + 1) & 255], x, y, z - 1.0),
        perlin_gradient(PERLIN_PERMUTATION[(ba + 1) & 255], x - 1.0, y, z - 1.0),
    );
    let x2 = perlin_lerp(
        u,
        perlin_gradient(PERLIN_PERMUTATION[(ab + 1) & 255], x, y - 1.0, z - 1.0),
        perlin_gradient(
            PERLIN_PERMUTATION[(bb + 1) & 255],
            x - 1.0,
            y - 1.0,
            z - 1.0,
        ),
    );
    perlin_lerp(w, y1, perlin_lerp(v, x1, x2))
}

/// Improved-noise quintic interpolation weight.
fn perlin_fade(value: f32) -> f32 {
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation in the published improved-noise arithmetic order.
fn perlin_lerp(weight: f32, a: f32, b: f32) -> f32 {
    a + weight * (b - a)
}

/// Select one of the twelve improved-noise cube gradients.
fn perlin_gradient(hash: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { -u }) + if h & 2 == 0 { v } else { -v }
}

/// `Perlin3(x, y, z)`: stateless improved gradient noise at audio or control rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Perlin3;

impl Unit for Perlin3 {
    /// Evaluates gradient noise for every requested audio sample.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        for sample in 0..ctx.outs.audio(0).len() {
            let value = improved_perlin3(
                coordinate(ctx, 0, sample),
                coordinate(ctx, 1, sample),
                coordinate(ctx, 2, sample),
            );
            ctx.outs.audio(0)[sample] = if value.is_finite() { value } else { 0.0 };
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Perlin3`].
pub struct Perlin3Ctor;

impl UnitDef for Perlin3Ctor {
    /// Validates the fixed ABI and constructs the stateless noise unit.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        use plyphon_dsp::rate::Rate;

        validate_exact_chaos(ctx, 3, 1, &[Rate::Audio, Rate::Control])?;
        Ok(unit_spec(Perlin3))
    }
}

/// One retained Rössler state machine with linearly interpolated audio-rate output.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RosslerL {
    xn: f64,
    yn: f64,
    zn: f64,
    xnm1: f64,
    ynm1: f64,
    znm1: f64,
    counter: f64,
    frac: f64,
    frequency: f32,
    a: f32,
    b: f32,
    c: f32,
    h: f32,
    xi: f32,
    yi: f32,
    zi: f32,
    restart_pending: u32,
    _pad: u32,
}

impl RosslerL {
    /// Remember one finite callback control, retaining the prior value for NaN or infinity.
    fn control(input: f32, remembered: &mut f32) -> f32 {
        if input.is_finite() {
            *remembered = input;
        }
        *remembered
    }

    /// Apply one textbook fourth-order Runge-Kutta step to the Rössler equations.
    fn rk4(x: f64, y: f64, z: f64, a: f64, b: f64, c: f64, h: f64) -> (f64, f64, f64) {
        let (k1x, k1y, k1z) = rossler_derivative(x, y, z, a, b, c);
        let (k2x, k2y, k2z) = rossler_derivative(
            x + h * k1x * 0.5,
            y + h * k1y * 0.5,
            z + h * k1z * 0.5,
            a,
            b,
            c,
        );
        let (k3x, k3y, k3z) = rossler_derivative(
            x + h * k2x * 0.5,
            y + h * k2y * 0.5,
            z + h * k2z * 0.5,
            a,
            b,
            c,
        );
        let (k4x, k4y, k4z) = rossler_derivative(x + h * k3x, y + h * k3y, z + h * k3z, a, b, c);
        let sixth_h = h / 6.0;
        (
            x + sixth_h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
            y + sixth_h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            z + sixth_h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
        )
    }
}

impl Unit for RosslerL {
    /// Initializes retained coordinates and integration cadence from scalar controls.
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.frequency = finite_or(ctx.ins.control(0), 22_050.0);
        self.a = finite_or(ctx.ins.control(1), 0.2);
        self.b = finite_or(ctx.ins.control(2), 0.2);
        self.c = finite_or(ctx.ins.control(3), 5.7);
        self.h = finite_or(ctx.ins.control(4), 0.05);
        self.xi = finite_or(ctx.ins.control(5), 0.1);
        self.yi = finite_or(ctx.ins.control(6), 0.0);
        self.zi = finite_or(ctx.ins.control(7), 0.0);
        self.xn = self.xi as f64;
        self.yn = self.yi as f64;
        self.zn = self.zi as f64;
        self.xnm1 = self.xn;
        self.ynm1 = self.yn;
        self.znm1 = self.zn;

        let samples = rossler_samples_per_cycle(self.frequency, ctx.own.sample_rate as f32);
        self.counter = 1.0;
        self.frac = 1.0 / samples;
    }

    /// Advances and interpolates the Rössler trajectory for one callback.
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let frequency = Self::control(ctx.ins.control(0), &mut self.frequency);
        let a = Self::control(ctx.ins.control(1), &mut self.a) as f64;
        let b = Self::control(ctx.ins.control(2), &mut self.b) as f64;
        let c = Self::control(ctx.ins.control(3), &mut self.c) as f64;
        let h = Self::control(ctx.ins.control(4), &mut self.h) as f64;

        let prior_xi = self.xi;
        let prior_yi = self.yi;
        let prior_zi = self.zi;
        let xi = Self::control(ctx.ins.control(5), &mut self.xi);
        let yi = Self::control(ctx.ins.control(6), &mut self.yi);
        let zi = Self::control(ctx.ins.control(7), &mut self.zi);
        if xi != prior_xi || yi != prior_yi || zi != prior_zi {
            self.xnm1 = self.xn;
            self.ynm1 = self.yn;
            self.znm1 = self.zn;
            self.xn = xi as f64;
            self.yn = yi as f64;
            self.zn = zi as f64;
        }

        let samples = rossler_samples_per_cycle(frequency, ctx.own.sample_rate as f32);
        let slope = 1.0 / samples;
        let output_len = ctx.outs.audio(0).len();
        for sample in 0..output_len {
            if self.counter >= samples {
                self.counter -= samples;
                self.frac = 0.0;
                self.xnm1 = self.xn;
                self.ynm1 = self.yn;
                self.znm1 = self.zn;

                if self.restart_pending != 0 {
                    self.xn = self.xi as f64;
                    self.yn = self.yi as f64;
                    self.zn = self.zi as f64;
                    self.xnm1 = self.xn;
                    self.ynm1 = self.yn;
                    self.znm1 = self.zn;
                    self.restart_pending = 0;
                }

                let next = Self::rk4(self.xn, self.yn, self.zn, a, b, c, h);
                if next.0.is_finite() && next.1.is_finite() && next.2.is_finite() {
                    self.xn = next.0;
                    self.yn = next.1;
                    self.zn = next.2;
                } else {
                    self.xn = self.xnm1;
                    self.yn = self.ynm1;
                    self.zn = self.znm1;
                    self.restart_pending = 1;
                }
            }

            self.counter += 1.0;
            let x = self.xnm1 + (self.xn - self.xnm1) * self.frac;
            let y = self.ynm1 + (self.yn - self.ynm1) * self.frac;
            let z = self.znm1 + (self.zn - self.znm1) * self.frac;
            ctx.outs.audio(0)[sample] = finite_scaled(x, 0.5);
            ctx.outs.audio(1)[sample] = finite_scaled(y, 0.5);
            ctx.outs.audio(2)[sample] = finite_scaled(z, 1.0);
            self.frac += slope;
        }
        DoneAction::Nothing
    }
}

/// Use `fallback` when an initial runtime control is not finite.
fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

/// Derivatives of the standard three-coordinate Rössler system.
fn rossler_derivative(x: f64, y: f64, z: f64, a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    (-y - z, x + a * y, b + z * (x - c))
}

/// Integration cadence for `RosslerL`, in output samples per RK4 step.
fn rossler_samples_per_cycle(frequency: f32, sample_rate: f32) -> f64 {
    if frequency < sample_rate {
        (sample_rate / frequency.max(0.001)) as f64
    } else {
        1.0
    }
}

/// Scale one finite interpolated coordinate, returning zero only as a final safety guard.
fn finite_scaled(value: f64, scale: f64) -> f32 {
    let output = (value * scale) as f32;
    if output.is_finite() { output } else { 0.0 }
}

/// Constructor for [`RosslerL`].
pub struct RosslerLCtor;

impl UnitDef for RosslerLCtor {
    /// Validates the fixed ABI and constructs a retained Rössler state machine.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        use plyphon_dsp::rate::Rate;

        validate_exact_chaos(ctx, 8, 3, &[Rate::Audio])?;
        Ok(unit_spec(RosslerL {
            xn: 0.1,
            yn: 0.0,
            zn: 0.0,
            xnm1: 0.1,
            ynm1: 0.0,
            znm1: 0.0,
            counter: 0.0,
            frac: 0.0,
            frequency: 22_050.0,
            a: 0.2,
            b: 0.2,
            c: 5.7,
            h: 0.05,
            xi: 0.1,
            yi: 0.0,
            zi: 0.0,
            restart_pending: 0,
            _pad: 0,
        }))
    }
}

/// Ken Perlin's published 256-entry improved-noise permutation.
#[rustfmt::skip]
const PERLIN_PERMUTATION: [u8; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];
