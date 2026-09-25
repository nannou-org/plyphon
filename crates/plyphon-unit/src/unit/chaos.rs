//! Chaotic map generators - plyphon's ports of scsynth's `CuspN`, `QuadN`, `GbmanN`, `LinCongN`,
//! `StandardN`, `LatoocarfianN`, `FBSineN`, `HenonN`, `CuspL`, `QuadL`, `HenonL`, `LorenzL`,
//! `StandardL`, `FBSineL`, `LatoocarfianL`, `LinCongL`, `FBSineC`, `HenonC`, `LatoocarfianC` and
//! `LinCongC` (`ChaosUGens.cpp`).
//!
//! Each unit iterates a chaotic map - or, for `LorenzL`, integrates a system of ODEs - at a `freq`
//! rate. The `*N` (non-interpolating) forms hold the iterate between iterations; the `*L` (linearly
//! interpolating) forms ramp from the previous iterate to the current one across the hold; the `*C`
//! (cubically interpolating) forms play the cubic through the last four iterates, which runs one
//! iteration behind the `*L` ramp. Maps and their internal state are computed in `f64`; `freq` and
//! the map coefficients are read once per block.
//!
//! The units differ in three ways:
//!
//! - `FBSineN`, `HenonN` and the `*L` and `*C` units other than `LinCongL` and `LinCongC` re-seed
//!   their state when an init input changes at run time (the Hénon units only once their stability
//!   latch has tripped), while the other units seed once and ignore later changes;
//! - the hold length of the `*L` and `*C` units, `FBSineN` and `HenonN` divides in `f64` and
//!   narrows to `f32`, while that of the other `*N` units divides in pure `f32`;
//! - `StandardL` wraps its phase with a C-style truncating remainder, while `StandardN` wraps with a
//!   Euclidean one, so the two disagree for phases far outside `[0, 2π)`.

use core::f64::consts::PI;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

const TWO_PI: f64 = 2.0 * PI;
const REC_PI: f64 = 1.0 / PI;
/// The reciprocal of 2π used by [`mod2pi`], written as scsynth's rounded decimal rather than
/// `1.0 / TWO_PI`: the two are different doubles, and the truncating branch is sensitive to which.
const REC_TWO_PI: f64 = 0.1591549430918953;
/// The Runge-Kutta weight in [`LorenzL`]'s integrator. As with [`REC_TWO_PI`] this is scsynth's
/// rounded decimal, which is not the nearest double to `1/6`.
const ONE_SIXTH: f64 = 0.1666666666666667;
/// [`LorenzL`]'s output scale. scsynth writes it as a `float` literal widened to `double`, which
/// lands a hair below `0.04`.
const LORENZ_OUT_SCALE: f64 = 0.04f32 as f64;

/// The hold length in samples for a map running at `freq` Hz (scsynth's `samplesPerCycle`).
fn samples_per_cycle(freq: f32, sr: f32) -> f32 {
    if freq < sr { sr / freq.max(0.001) } else { 1.0 }
}

/// The hold length in samples and the per-sample interpolation slope for an interpolating map
/// running at `freq` Hz (scsynth's `samplesPerCycle`/`slope` pair).
///
/// The hold length divides in `f64` and narrows to `f32`, and the slope is then the `f64` widening
/// of the `f32` reciprocal of that hold length - the mixed-precision arithmetic of the `*L`
/// prologues, which the interpolated output is sensitive to. The sample-and-hold
/// [`samples_per_cycle`] divides in pure `f32` instead. The hold length is never below one sample,
/// so a map can iterate at most once per output sample.
fn samples_per_cycle_slope(freq: f32, sample_rate: f64) -> (f32, f64) {
    let spc = hold_length(freq, sample_rate);
    (spc, f64::from(1.0 / spc))
}

/// The hold length in samples for a map running at `freq` Hz, as scsynth's prologues compute it:
/// the `f64` sample rate divided by the clamped `f32` frequency, narrowed to `f32`, and one sample
/// once `freq` reaches the sample rate.
fn hold_length(freq: f32, sample_rate: f64) -> f32 {
    if f64::from(freq) < sample_rate {
        (sample_rate / f64::from(freq.max(0.001))) as f32
    } else {
        1.0
    }
}

/// scsynth's `ipol3Coef`: the coefficients `[c0, c1, c2, c3]` of the cubic through four successive
/// iterates, which runs from `xnm2` at phase 0 to `xnm1` at phase 1.
///
/// The reference writes the weights as `float` literals, all exact in `f64`, so this is plain `f64`
/// arithmetic in the reference's evaluation order. It is not `plyphon_dsp::interp::cubicinterp`,
/// which evaluates a different arrangement of the same polynomial in `f32`.
fn ipol3_coefs(xnm3: f64, xnm2: f64, xnm1: f64, xn: f64) -> [f64; 4] {
    [
        xnm2,
        0.5 * (xnm1 - xnm3),
        xnm3 - (2.5 * xnm2) + xnm1 + xnm1 - 0.5 * xn,
        0.5 * (xn - xnm3) + 1.5 * (xnm2 - xnm1),
    ]
}

/// scsynth's `ipol3`: the cubic `coefs` evaluated by Horner's rule at the phase `frac`, which the
/// reference's `float` parameter narrows to `f32` before the `f64` evaluation.
fn ipol3(frac: f64, coefs: &[f64; 4]) -> f64 {
    let frac = f64::from(frac as f32);
    ((coefs[3] * frac + coefs[2]) * frac + coefs[1]) * frac + coefs[0]
}

/// scsynth's quick 2π modulo: a fast path over `[-2π, 4π)` and a truncating fallback outside it.
///
/// The fallback subtracts whole turns using a truncating 32-bit integer cast, so it is a C-style
/// remainder rather than a Euclidean one and returns negative results for inputs below `-2π`. The
/// cast saturates at the `i32` bounds and maps NaN to zero, giving the out-of-range inputs a defined
/// result. `StandardN` wraps with a Euclidean remainder instead, so the two families disagree
/// outside the fast path.
fn mod2pi(mut x: f64) -> f64 {
    if x >= TWO_PI {
        x -= TWO_PI;
        if x < TWO_PI {
            return x;
        }
    } else if x < 0.0 {
        x += TWO_PI;
        if x >= 0.0 {
            return x;
        }
    } else {
        return x;
    }
    x - TWO_PI * f64::from((x * REC_TWO_PI) as i32)
}

/// The cusp map `x = a - b*sqrt(|x|)`, shared by `CuspN` and `CuspL`.
fn cusp_map(a: f64, b: f64, x: f64) -> f64 {
    a - (b * math::sqrt(x.abs()))
}

/// The quadratic map `x = a*x^2 + b*x + c`, shared by `QuadN` and `QuadL`.
fn quad_map(a: f64, b: f64, c: f64, x: f64) -> f64 {
    a * x * x + b * x + c
}

/// The Latoocarfian map `x = sin(b*y) + c*sin(b*x)`, `y = sin(a*x) + d*sin(a*y)`, shared by
/// `LatoocarfianN`, `LatoocarfianL` and `LatoocarfianC`. Returns the new `(x, y)`.
fn latoocarfian_map(a: f64, b: f64, c: f64, d: f64, x: f64, y: f64) -> (f64, f64) {
    (
        math::sin(y * b) + c * math::sin(x * b),
        math::sin(x * a) + d * math::sin(y * a),
    )
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

/// Drive a linearly interpolating map: iterate `map` every `samples_per_cycle` samples and write
/// `out` of the value ramping from the previous iterate to the current one. Returns the final
/// `(xn, xnm1)` pair.
///
/// Only the interpolated variable is threaded through `map`; a multi-variable map keeps its other
/// variables in the closure, so one driver serves the one-, two- and three-variable `*L` units. `dx`
/// is recomputed on entry because a re-seed between blocks can move `xn` and `xnm1` apart.
fn chaos_interp(
    ctx: &mut ProcessCtx<'_>,
    counter: &mut f32,
    frac: &mut f64,
    xn: f64,
    xnm1: f64,
    mut map: impl FnMut(f64) -> f64,
    out: impl Fn(f64) -> f64,
) -> (f64, f64) {
    let (spc, slope) = samples_per_cycle_slope(ctx.ins.control(0), ctx.own.sample_rate);
    let (mut x, mut xm1) = (xn, xnm1);
    let mut dx = x - xm1;
    for o in ctx.outs.audio(0).iter_mut() {
        if *counter >= spc {
            *counter -= spc;
            *frac = 0.0;
            xm1 = x;
            x = map(x);
            dx = x - xm1;
        }
        *counter += 1.0;
        *o = out(xm1 + dx * *frac) as f32;
        *frac += slope;
    }
    (x, xm1)
}

/// Drive a cubically interpolating map: iterate `map` every `samples_per_cycle` samples, shift the
/// newest point into `history` (`[xnm1, xnm2, xnm3]`), refit `coefs` through the four points and
/// write `out` of the cubic at the current phase. Returns the newest point.
///
/// As with [`chaos_interp`], only the interpolated variable is threaded through `map`. The cubic
/// runs from `xnm2` to `xnm1`, one iteration behind the newest point, and `coefs` persist across
/// blocks rather than being refitted on entry: a re-seed shifts `history` but leaves the current
/// cubic playing until the next iteration.
fn chaos_cubic(
    ctx: &mut ProcessCtx<'_>,
    counter: &mut f32,
    frac: &mut f64,
    history: &mut [f64; 3],
    coefs: &mut [f64; 4],
    xn: f64,
    mut map: impl FnMut(f64) -> f64,
) -> f64 {
    let (spc, slope) = samples_per_cycle_slope(ctx.ins.control(0), ctx.own.sample_rate);
    let mut x = xn;
    for o in ctx.outs.audio(0).iter_mut() {
        if *counter >= spc {
            *counter -= spc;
            *frac = 0.0;
            *history = [x, history[0], history[1]];
            x = map(x);
            *coefs = ipol3_coefs(history[2], history[1], history[0], x);
        }
        *counter += 1.0;
        *o = ipol3(*frac, coefs) as f32;
        *frac += slope;
    }
    x
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
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.xn = ctx.ins.control(3) as f64;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = ctx.ins.control(1) as f64;
        let b = ctx.ins.control(2) as f64;
        self.xn = chaos1(
            ctx,
            &mut self.counter,
            self.xn,
            |x| cusp_map(a, b, x),
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
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.xn = ctx.ins.control(4) as f64;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = ctx.ins.control(1) as f64;
        let b = ctx.ins.control(2) as f64;
        let c = ctx.ins.control(3) as f64;
        self.xn = chaos1(
            ctx,
            &mut self.counter,
            self.xn,
            |x| quad_map(a, b, c, x),
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
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.xn = ctx.ins.control(4) as f64;
        self.process(ctx)
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
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.xn = ctx.ins.control(1) as f64;
        self.yn = ctx.ins.control(2) as f64;
        self.process(ctx)
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
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.xn = ctx.ins.control(2) as f64;
        self.yn = ctx.ins.control(3) as f64;
        self.process(ctx)
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
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.xn = ctx.ins.control(5) as f64;
        self.yn = ctx.ins.control(6) as f64;
        self.process(ctx)
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
            |x, y| latoocarfian_map(a, b, c, d, x, y),
            |x| x,
        );
        self.xn = x;
        self.yn = y;
        DoneAction::Nothing
    }
}

/// `CuspL.ar(freq, a, b, xi)`: the cusp map `x = a - b*sqrt(|x|)`, linearly interpolated.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CuspL {
    /// The iterate the current hold ramps towards.
    xn: f64,
    /// The iterate the current hold ramps from.
    xnm1: f64,
    /// The `xi` input the state was last seeded from; a change re-seeds the map.
    x0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for CuspL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(3));
        self.xn = self.x0;
        self.xnm1 = self.x0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = f64::from(ctx.ins.control(1));
        let b = f64::from(ctx.ins.control(2));
        let xi = f64::from(ctx.ins.control(3));
        if self.x0 != xi {
            self.xnm1 = self.xn;
            self.x0 = xi;
            self.xn = xi;
        }
        let (xn, xnm1) = chaos_interp(
            ctx,
            &mut self.counter,
            &mut self.frac,
            self.xn,
            self.xnm1,
            |x| cusp_map(a, b, x),
            |x| x,
        );
        self.xn = xn;
        self.xnm1 = xnm1;
        DoneAction::Nothing
    }
}

/// `QuadL.ar(freq, a, b, c, xi)`: the quadratic map `x = a*x^2 + b*x + c`, linearly interpolated.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuadL {
    /// The iterate the current hold ramps towards.
    xn: f64,
    /// The iterate the current hold ramps from.
    xnm1: f64,
    /// The `xi` input the state was last seeded from; a change re-seeds the map.
    x0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for QuadL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(4));
        self.xn = self.x0;
        self.xnm1 = self.x0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = f64::from(ctx.ins.control(1));
        let b = f64::from(ctx.ins.control(2));
        let c = f64::from(ctx.ins.control(3));
        let xi = f64::from(ctx.ins.control(4));
        if self.x0 != xi {
            self.xnm1 = self.xn;
            self.x0 = xi;
            self.xn = xi;
        }
        let (xn, xnm1) = chaos_interp(
            ctx,
            &mut self.counter,
            &mut self.frac,
            self.xn,
            self.xnm1,
            |x| quad_map(a, b, c, x),
            |x| x,
        );
        self.xn = xn;
        self.xnm1 = xnm1;
        DoneAction::Nothing
    }
}

/// `HenonL.ar(freq, a, b, x0, x1)`: the Hénon map `x = 1 - a*x'^2 + b*x''`, linearly interpolated.
///
/// The map diverges for many coefficient pairs, so it carries a stability latch: an iterate leaving
/// `[-1.5, 1.5]` zeroes the history and silences the unit until an init input changes. Because the
/// coefficients themselves take part in that comparison, a change while the map is still stable only
/// refreshes the cached values and lets the running iterates continue - re-seeding on every
/// coefficient change would restart the map whenever `a` or `b` is modulated.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HenonL {
    /// The iterate the current hold ramps towards.
    xnm1: f64,
    /// The iterate the current hold ramps from.
    xnm2: f64,
    /// The `a` input the state was last compared against.
    a: f64,
    /// The `b` input the state was last compared against.
    b: f64,
    /// The `x0` input the state was last compared against.
    x0: f64,
    /// The `x1` input the state was last compared against.
    x1: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    /// Nonzero while the map is iterating; zeroed once an iterate has escaped `[-1.5, 1.5]`.
    stable: u32,
}

impl Unit for HenonL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.a = f64::from(ctx.ins.control(1));
        self.b = f64::from(ctx.ins.control(2));
        self.x0 = f64::from(ctx.ins.control(3));
        self.x1 = f64::from(ctx.ins.control(4));
        // The seed is deliberately asymmetric: the first hold ramps from `x1` to `x0`.
        self.xnm1 = self.x0;
        self.xnm2 = self.x1;
        self.stable = 1;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = f64::from(ctx.ins.control(1));
        let b = f64::from(ctx.ins.control(2));
        let x0 = f64::from(ctx.ins.control(3));
        let x1 = f64::from(ctx.ins.control(4));
        let mut stable = self.stable != 0;
        if self.a != a || self.b != b || self.x0 != x0 || self.x1 != x1 {
            if !stable {
                // The reference also parks `x1` in its newest-iterate member here; that slot
                // is recomputed before every read, so the port keeps it as a loop local.
                self.xnm2 = x0;
                self.xnm1 = x0;
            }
            stable = true;
            self.a = a;
            self.b = b;
            self.x0 = x0;
            self.x1 = x1;
        }

        let (spc, slope) = samples_per_cycle_slope(ctx.ins.control(0), ctx.own.sample_rate);
        let (mut xnm1, mut xnm2) = (self.xnm1, self.xnm2);
        let (mut counter, mut frac) = (self.counter, self.frac);
        let mut diff = xnm1 - xnm2;
        for o in ctx.outs.audio(0).iter_mut() {
            if counter >= spc {
                counter -= spc;
                if stable {
                    let xn = 1.0 - (a * xnm1 * xnm1) + (b * xnm2);
                    // Two comparisons rather than a range test: both are false for a NaN iterate, so
                    // NaN leaves the latch untripped and keeps iterating, as the reference does.
                    #[allow(clippy::manual_range_contains)]
                    if xn > 1.5 || xn < -1.5 {
                        stable = false;
                        diff = 0.0;
                        xnm1 = 0.0;
                        xnm2 = 0.0;
                    } else {
                        xnm2 = xnm1;
                        xnm1 = xn;
                        diff = xnm1 - xnm2;
                    }
                    // Reached on the escaping iteration too, but on no boundary after it: a latched
                    // unit skips this branch entirely and leaves the phase free-running.
                    frac = 0.0;
                }
            }
            counter += 1.0;
            *o = (xnm2 + (diff * frac)) as f32;
            frac += slope;
        }

        self.xnm1 = xnm1;
        self.xnm2 = xnm2;
        self.counter = counter;
        self.frac = frac;
        self.stable = u32::from(stable);
        DoneAction::Nothing
    }
}

/// `LorenzL.ar(freq, s, r, b, h, xi, yi, zi)`: the Lorenz attractor integrated with 4th-order
/// Runge-Kutta over a step of `h`, its `x` component linearly interpolated and scaled to audio
/// range.
///
/// The integrator is unconditionally stepped rather than adaptive, so it diverges for large `h*s`
/// exactly as the reference does; the interesting coefficient range keeps the product small.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LorenzL {
    /// The `x` the current hold ramps towards.
    xn: f64,
    /// The current `y`.
    yn: f64,
    /// The current `z`.
    zn: f64,
    /// The `x` the current hold ramps from.
    xnm1: f64,
    /// The `xi` input the state was last seeded from; a change re-seeds the system.
    x0: f64,
    /// The `yi` input the state was last seeded from; a change re-seeds the system.
    y0: f64,
    /// The `zi` input the state was last seeded from; a change re-seeds the system.
    z0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last step.
    counter: f32,
    _pad: u32,
}

impl Unit for LorenzL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(5));
        self.y0 = f64::from(ctx.ins.control(6));
        self.z0 = f64::from(ctx.ins.control(7));
        self.xn = self.x0;
        self.yn = self.y0;
        self.zn = self.z0;
        self.xnm1 = self.x0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let s = f64::from(ctx.ins.control(1));
        let r = f64::from(ctx.ins.control(2));
        let b = f64::from(ctx.ins.control(3));
        let h = f64::from(ctx.ins.control(4));
        let xi = f64::from(ctx.ins.control(5));
        let yi = f64::from(ctx.ins.control(6));
        let zi = f64::from(ctx.ins.control(7));
        if self.x0 != xi || self.y0 != yi || self.z0 != zi {
            // The reference also shifts its `ynm1`/`znm1` members here, but they are
            // overwritten before every read; the port keeps them as per-step locals.
            self.xnm1 = self.xn;
            self.x0 = xi;
            self.xn = xi;
            self.y0 = yi;
            self.yn = yi;
            self.z0 = zi;
            self.zn = zi;
        }

        let (mut yn, mut zn) = (self.yn, self.zn);
        let (xn, xnm1) = chaos_interp(
            ctx,
            &mut self.counter,
            &mut self.frac,
            self.xn,
            self.xnm1,
            |xnm1| {
                let ynm1 = yn;
                let znm1 = zn;
                let h_times_s = h * s;

                let k1x = h_times_s * (ynm1 - xnm1);
                let k1y = h * (xnm1 * (r - znm1) - ynm1);
                let k1z = h * (xnm1 * ynm1 - b * znm1);
                let (mut kx_half, mut ky_half, mut kz_half) = (k1x * 0.5, k1y * 0.5, k1z * 0.5);

                let k2x = h_times_s * (ynm1 + ky_half - xnm1 - kx_half);
                let k2y = h * ((xnm1 + kx_half) * (r - znm1 - kz_half) - (ynm1 + ky_half));
                let k2z = h * ((xnm1 + kx_half) * (ynm1 + ky_half) - b * (znm1 + kz_half));
                kx_half = k2x * 0.5;
                ky_half = k2y * 0.5;
                kz_half = k2z * 0.5;

                let k3x = h_times_s * (ynm1 + ky_half - xnm1 - kx_half);
                let k3y = h * ((xnm1 + kx_half) * (r - znm1 - kz_half) - (ynm1 + ky_half));
                let k3z = h * ((xnm1 + kx_half) * (ynm1 + ky_half) - b * (znm1 + kz_half));

                let k4x = h_times_s * (ynm1 + k3y - xnm1 - k3x);
                let k4y = h * ((xnm1 + k3x) * (r - znm1 - k3z) - (ynm1 + k3y));
                let k4z = h * ((xnm1 + k3x) * (ynm1 + k3y) - b * (znm1 + k3z));

                yn += (k1y + 2.0 * (k2y + k3y) + k4y) * ONE_SIXTH;
                zn += (k1z + 2.0 * (k2z + k3z) + k4z) * ONE_SIXTH;
                xnm1 + (k1x + 2.0 * (k2x + k3x) + k4x) * ONE_SIXTH
            },
            |x| x * LORENZ_OUT_SCALE,
        );
        self.xn = xn;
        self.xnm1 = xnm1;
        self.yn = yn;
        self.zn = zn;
        DoneAction::Nothing
    }
}

/// `StandardL.ar(freq, k, xi, yi)`: the standard (kicked-rotor) map, linearly interpolated and
/// scaled to `[-1, 1)`.
///
/// The phase wraps through `mod2pi`, so a phase driven far outside `[0, 2π)` can wrap negative -
/// unlike `StandardN`, which wraps Euclidean. A re-seed assigns `xi` unwrapped, exactly as the
/// reference does, so the first hold after a re-seed can leave the nominal output range.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct StandardL {
    /// The phase the current hold ramps towards.
    xn: f64,
    /// The current angular momentum.
    yn: f64,
    /// The phase the current hold ramps from.
    xnm1: f64,
    /// The `xi` input the state was last seeded from; a change re-seeds the map.
    x0: f64,
    /// The `yi` input the state was last seeded from; a change re-seeds the map.
    y0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for StandardL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(2));
        self.y0 = f64::from(ctx.ins.control(3));
        self.xn = self.x0;
        self.yn = self.y0;
        self.xnm1 = self.x0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let k = f64::from(ctx.ins.control(1));
        let xi = f64::from(ctx.ins.control(2));
        let yi = f64::from(ctx.ins.control(3));
        if self.x0 != xi || self.y0 != yi {
            self.xnm1 = self.xn;
            self.x0 = xi;
            self.xn = xi;
            self.y0 = yi;
            self.yn = yi;
        }

        let mut yn = self.yn;
        let (xn, xnm1) = chaos_interp(
            ctx,
            &mut self.counter,
            &mut self.frac,
            self.xn,
            self.xnm1,
            |x| {
                yn = mod2pi(yn + k * math::sin(x));
                mod2pi(x + yn)
            },
            |x| (x - PI) * REC_PI,
        );
        self.xn = xn;
        self.xnm1 = xnm1;
        self.yn = yn;
        DoneAction::Nothing
    }
}

/// The feedback sine map `x = sin(im*y + fb*x)`, `y = (a*y + c) mod 2π`, shared by `FBSineN`,
/// `FBSineL` and `FBSineC`. Returns the new `(x, y)`; `y` wraps through [`mod2pi`].
fn fb_sine_map(im: f64, fb: f64, a: f64, c: f64, x: f64, y: f64) -> (f64, f64) {
    (math::sin(im * y + fb * x), mod2pi(a * y + c))
}

/// `FBSineN.ar(freq, im, fb, a, c, xi, yi)`: the feedback sine map
/// `x = sin(im*y + fb*x)`, `y = (a*y + c) mod 2π`.
///
/// A run-time change of `xi` or `yi` re-seeds both variables.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FBSineN {
    /// The current iterate, which the unit holds.
    xn: f64,
    /// The current phase.
    yn: f64,
    /// The `xi` input the state was last seeded from.
    x0: f64,
    /// The `yi` input the state was last seeded from.
    y0: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for FBSineN {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(5));
        self.y0 = f64::from(ctx.ins.control(6));
        self.xn = self.x0;
        self.yn = self.y0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let spc = hold_length(ctx.ins.control(0), ctx.own.sample_rate);
        let im = f64::from(ctx.ins.control(1));
        let fb = f64::from(ctx.ins.control(2));
        let a = f64::from(ctx.ins.control(3));
        let c = f64::from(ctx.ins.control(4));
        let xi = f64::from(ctx.ins.control(5));
        let yi = f64::from(ctx.ins.control(6));
        if self.x0 != xi || self.y0 != yi {
            self.x0 = xi;
            self.xn = xi;
            self.y0 = yi;
            self.yn = yi;
        }

        let (mut x, mut y) = (self.xn, self.yn);
        for o in ctx.outs.audio(0).iter_mut() {
            if self.counter >= spc {
                self.counter -= spc;
                (x, y) = fb_sine_map(im, fb, a, c, x, y);
            }
            self.counter += 1.0;
            *o = x as f32;
        }
        self.xn = x;
        self.yn = y;
        DoneAction::Nothing
    }
}

/// `FBSineL.ar(freq, im, fb, a, c, xi, yi)`: the feedback sine map, linearly interpolated.
///
/// A run-time change of `xi` or `yi` re-seeds both variables, shifting the running iterate into the
/// history so the output ramps to the new seed.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FBSineL {
    /// The iterate the current hold ramps towards.
    xn: f64,
    /// The current phase.
    yn: f64,
    /// The iterate the current hold ramps from.
    xnm1: f64,
    /// The `xi` input the state was last seeded from.
    x0: f64,
    /// The `yi` input the state was last seeded from.
    y0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for FBSineL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(5));
        self.y0 = f64::from(ctx.ins.control(6));
        self.xn = self.x0;
        self.yn = self.y0;
        self.xnm1 = self.x0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let im = f64::from(ctx.ins.control(1));
        let fb = f64::from(ctx.ins.control(2));
        let a = f64::from(ctx.ins.control(3));
        let c = f64::from(ctx.ins.control(4));
        let xi = f64::from(ctx.ins.control(5));
        let yi = f64::from(ctx.ins.control(6));
        if self.x0 != xi || self.y0 != yi {
            self.xnm1 = self.xn;
            self.x0 = xi;
            self.xn = xi;
            self.y0 = yi;
            self.yn = yi;
        }

        let mut yn = self.yn;
        let (xn, xnm1) = chaos_interp(
            ctx,
            &mut self.counter,
            &mut self.frac,
            self.xn,
            self.xnm1,
            |x| {
                let (nx, ny) = fb_sine_map(im, fb, a, c, x, yn);
                yn = ny;
                nx
            },
            |x| x,
        );
        self.xn = xn;
        self.xnm1 = xnm1;
        self.yn = yn;
        DoneAction::Nothing
    }
}

/// `FBSineC.ar(freq, im, fb, a, c, xi, yi)`: the feedback sine map, cubically interpolated.
///
/// A run-time change of `xi` or `yi` assigns `xi` to the newest iterate and shifts it into the
/// history; the phase `y` is not re-seeded - `yi` is only cached for the comparison, exactly as the
/// reference does. Until the first iteration the unit plays its zeroed cubic, so it emits silence
/// for the first hold.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FBSineC {
    /// The newest iterate.
    xn: f64,
    /// The current phase.
    yn: f64,
    /// The previous three iterates, newest first (`xnm1`, `xnm2`, `xnm3`).
    history: [f64; 3],
    /// The coefficients of the cubic the current hold plays.
    coefs: [f64; 4],
    /// The `xi` input the state was last seeded from.
    x0: f64,
    /// The `yi` input last seen.
    y0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for FBSineC {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(5));
        self.y0 = f64::from(ctx.ins.control(6));
        self.xn = self.x0;
        self.yn = self.y0;
        self.history = [self.x0; 3];
        self.coefs = [0.0; 4];
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let im = f64::from(ctx.ins.control(1));
        let fb = f64::from(ctx.ins.control(2));
        let a = f64::from(ctx.ins.control(3));
        let c = f64::from(ctx.ins.control(4));
        let xi = f64::from(ctx.ins.control(5));
        let yi = f64::from(ctx.ins.control(6));
        if self.x0 != xi || self.y0 != yi {
            // The newest iterate takes the seed before the shift, so the seed also lands in `xnm1`.
            self.x0 = xi;
            self.xn = xi;
            self.y0 = yi;
            self.history = [xi, self.history[0], self.history[1]];
        }

        let mut yn = self.yn;
        self.xn = chaos_cubic(
            ctx,
            &mut self.counter,
            &mut self.frac,
            &mut self.history,
            &mut self.coefs,
            self.xn,
            |x| {
                let (nx, ny) = fb_sine_map(im, fb, a, c, x, yn);
                yn = ny;
                nx
            },
        );
        self.yn = yn;
        DoneAction::Nothing
    }
}

/// `HenonN.ar(freq, a, b, x0, x1)`: the Hénon map `x = 1 - a*x'^2 + b*x''`, held.
///
/// The unit emits the older of its two history terms, so its output runs two iterations behind the
/// map. It carries the same stability latch as [`HenonL`], but an escaping iterate re-seeds the
/// history from the current `x0` and `x1` inputs rather than zeroing it, so a latched unit holds
/// `x0` rather than falling silent. While the map is stable, a change of an input only refreshes
/// the cached values.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HenonN {
    /// The newer history term.
    xnm1: f64,
    /// The older history term, which the unit holds.
    xnm2: f64,
    /// The `a` input the state was last compared against.
    a: f64,
    /// The `b` input the state was last compared against.
    b: f64,
    /// The `x0` input the state was last compared against.
    x0: f64,
    /// The `x1` input the state was last compared against.
    x1: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    /// Nonzero while the map is iterating; zeroed once an iterate has escaped `[-1.5, 1.5]`.
    stable: u32,
}

impl Unit for HenonN {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(3));
        self.x1 = f64::from(ctx.ins.control(4));
        self.xnm1 = self.x0;
        self.xnm2 = self.x1;
        self.a = f64::from(ctx.ins.control(1));
        self.b = f64::from(ctx.ins.control(2));
        self.stable = 1;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let spc = hold_length(ctx.ins.control(0), ctx.own.sample_rate);
        let a = f64::from(ctx.ins.control(1));
        let b = f64::from(ctx.ins.control(2));
        let x0 = f64::from(ctx.ins.control(3));
        let x1 = f64::from(ctx.ins.control(4));
        let mut stable = self.stable != 0;
        if self.a != a || self.b != b || self.x0 != x0 || self.x1 != x1 {
            if !stable {
                // The reference also parks `x1` in its newest-iterate member here; that member is
                // written before every read, so the port keeps the iterate as a loop local.
                self.xnm2 = x0;
                self.xnm1 = x0;
            }
            stable = true;
            self.a = a;
            self.b = b;
            self.x0 = x0;
            self.x1 = x1;
        }

        let (mut xnm1, mut xnm2) = (self.xnm1, self.xnm2);
        let mut counter = self.counter;
        for o in ctx.outs.audio(0).iter_mut() {
            if counter >= spc {
                counter -= spc;
                if stable {
                    let xn = 1.0 - (a * xnm1 * xnm1) + (b * xnm2);
                    // Two comparisons rather than a range test: both are false for a NaN iterate, so
                    // NaN leaves the latch untripped and keeps iterating, as the reference does.
                    #[allow(clippy::manual_range_contains)]
                    if xn > 1.5 || xn < -1.5 {
                        stable = false;
                        xnm2 = x0;
                        xnm1 = x1;
                    } else {
                        xnm2 = xnm1;
                        xnm1 = xn;
                    }
                }
            }
            counter += 1.0;
            *o = xnm2 as f32;
        }

        self.xnm1 = xnm1;
        self.xnm2 = xnm2;
        self.counter = counter;
        self.stable = u32::from(stable);
        DoneAction::Nothing
    }
}

/// `HenonC.ar(freq, a, b, x0, x1)`: the Hénon map `x = 1 - a*x'^2 + b*x''`, cubically interpolated.
///
/// It carries the same stability latch as [`HenonL`], with two differences taken from the
/// reference. An escaping iterate leaves the history at `(0, 0, 0)` and the newest iterate at `1`,
/// and the cubic fitted through those points keeps playing - the phase still resets at every hold
/// boundary - so a latched unit repeats a small cubic arc rather than falling silent. And until the
/// first iteration the unit plays its zeroed cubic, so it emits silence for the first hold.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HenonC {
    /// The newest iterate.
    xn: f64,
    /// The previous three iterates, newest first (`xnm1`, `xnm2`, `xnm3`).
    history: [f64; 3],
    /// The coefficients of the cubic the current hold plays.
    coefs: [f64; 4],
    /// The `a` input the state was last compared against.
    a: f64,
    /// The `b` input the state was last compared against.
    b: f64,
    /// The `x0` input the state was last compared against.
    x0: f64,
    /// The `x1` input the state was last compared against.
    x1: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    /// Nonzero while the map is iterating; zeroed once an iterate has escaped `[-1.5, 1.5]`.
    stable: u32,
}

impl Unit for HenonC {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(3));
        self.x1 = f64::from(ctx.ins.control(4));
        self.xn = self.x1;
        self.history = [self.x0, self.x1, self.x1];
        self.a = f64::from(ctx.ins.control(1));
        self.b = f64::from(ctx.ins.control(2));
        self.stable = 1;
        self.coefs = [0.0; 4];
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (spc, slope) = samples_per_cycle_slope(ctx.ins.control(0), ctx.own.sample_rate);
        let a = f64::from(ctx.ins.control(1));
        let b = f64::from(ctx.ins.control(2));
        let x0 = f64::from(ctx.ins.control(3));
        let x1 = f64::from(ctx.ins.control(4));
        let mut stable = self.stable != 0;
        if self.a != a || self.b != b || self.x0 != x0 || self.x1 != x1 {
            if !stable {
                self.history = [x0, x0, self.history[1]];
                self.xn = x1;
            }
            stable = true;
            self.a = a;
            self.b = b;
            self.x0 = x0;
            self.x1 = x1;
        }

        let mut xn = self.xn;
        let [mut xnm1, mut xnm2, mut xnm3] = self.history;
        let (mut counter, mut frac, mut coefs) = (self.counter, self.frac, self.coefs);
        for o in ctx.outs.audio(0).iter_mut() {
            if counter >= spc {
                counter -= spc;
                frac = 0.0;
                if stable {
                    xnm3 = xnm2;
                    xnm2 = xnm1;
                    xnm1 = xn;
                    xn = 1.0 - (a * xnm1 * xnm1) + (b * xnm2);
                    // Two comparisons rather than a range test: both are false for a NaN iterate, so
                    // NaN leaves the latch untripped and keeps iterating, as the reference does.
                    #[allow(clippy::manual_range_contains)]
                    if xn > 1.5 || xn < -1.5 {
                        stable = false;
                        xn = 1.0;
                        xnm1 = 0.0;
                        xnm2 = 0.0;
                        xnm3 = 0.0;
                    }
                    coefs = ipol3_coefs(xnm3, xnm2, xnm1, xn);
                }
            }
            counter += 1.0;
            *o = ipol3(frac, &coefs) as f32;
            frac += slope;
        }

        self.xn = xn;
        self.history = [xnm1, xnm2, xnm3];
        self.counter = counter;
        self.frac = frac;
        self.coefs = coefs;
        self.stable = u32::from(stable);
        DoneAction::Nothing
    }
}

/// `LatoocarfianL.ar(freq, a, b, c, d, xi, yi)`: the Latoocarfian map, linearly interpolated.
///
/// A run-time change of `xi` or `yi` re-seeds both variables, shifting the running iterate into the
/// history so the output ramps to the new seed.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LatoocarfianL {
    /// The iterate the current hold ramps towards.
    xn: f64,
    /// The current second variable.
    yn: f64,
    /// The iterate the current hold ramps from.
    xnm1: f64,
    /// The `xi` input the state was last seeded from.
    x0: f64,
    /// The `yi` input the state was last seeded from.
    y0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for LatoocarfianL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(5));
        self.y0 = f64::from(ctx.ins.control(6));
        self.xn = self.x0;
        self.yn = self.y0;
        self.xnm1 = self.x0;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = f64::from(ctx.ins.control(1));
        let b = f64::from(ctx.ins.control(2));
        let c = f64::from(ctx.ins.control(3));
        let d = f64::from(ctx.ins.control(4));
        let xi = f64::from(ctx.ins.control(5));
        let yi = f64::from(ctx.ins.control(6));
        if self.x0 != xi || self.y0 != yi {
            self.xnm1 = self.xn;
            self.x0 = xi;
            self.xn = xi;
            self.y0 = yi;
            self.yn = yi;
        }

        let mut yn = self.yn;
        let (xn, xnm1) = chaos_interp(
            ctx,
            &mut self.counter,
            &mut self.frac,
            self.xn,
            self.xnm1,
            |x| {
                let (nx, ny) = latoocarfian_map(a, b, c, d, x, yn);
                yn = ny;
                nx
            },
            |x| x,
        );
        self.xn = xn;
        self.xnm1 = xnm1;
        self.yn = yn;
        DoneAction::Nothing
    }
}

/// `LatoocarfianC.ar(freq, a, b, c, d, xi, yi)`: the Latoocarfian map, cubically interpolated.
///
/// A run-time change of `xi` or `yi` shifts the running iterate into the history and re-seeds both
/// variables. The constructor fills the history and all four cubic coefficients with `xi`, exactly
/// as the reference does, so the first hold plays the cubic `xi*(1 + t + t^2 + t^3)` rather than
/// holding `xi`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LatoocarfianC {
    /// The newest iterate.
    xn: f64,
    /// The current second variable.
    yn: f64,
    /// The previous three iterates, newest first (`xnm1`, `xnm2`, `xnm3`).
    history: [f64; 3],
    /// The coefficients of the cubic the current hold plays.
    coefs: [f64; 4],
    /// The `xi` input the state was last seeded from.
    x0: f64,
    /// The `yi` input the state was last seeded from.
    y0: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for LatoocarfianC {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.x0 = f64::from(ctx.ins.control(5));
        self.y0 = f64::from(ctx.ins.control(6));
        self.xn = self.x0;
        self.yn = self.y0;
        self.history = [self.x0; 3];
        self.coefs = [self.x0; 4];
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = f64::from(ctx.ins.control(1));
        let b = f64::from(ctx.ins.control(2));
        let c = f64::from(ctx.ins.control(3));
        let d = f64::from(ctx.ins.control(4));
        let xi = f64::from(ctx.ins.control(5));
        let yi = f64::from(ctx.ins.control(6));
        if self.x0 != xi || self.y0 != yi {
            self.history = [self.xn, self.history[0], self.history[1]];
            self.x0 = xi;
            self.xn = xi;
            self.y0 = yi;
            self.yn = yi;
        }

        let mut yn = self.yn;
        self.xn = chaos_cubic(
            ctx,
            &mut self.counter,
            &mut self.frac,
            &mut self.history,
            &mut self.coefs,
            self.xn,
            |x| {
                let (nx, ny) = latoocarfian_map(a, b, c, d, x, yn);
                yn = ny;
                nx
            },
        );
        self.yn = yn;
        DoneAction::Nothing
    }
}

/// scsynth's `sc_mod` for doubles: a floored modulo with a fast path over `[-hi, 2*hi)` and a
/// `hi == 0 -> 0` guard.
///
/// Outside the fast path this subtracts `hi * floor(x / hi)`, which rounds differently from the
/// exact remainder `rem_euclid` computes.
fn sc_mod(mut x: f64, hi: f64) -> f64 {
    if x >= hi {
        x -= hi;
        if x < hi {
            return x;
        }
    } else if x < 0.0 {
        x += hi;
        if x >= 0.0 {
            return x;
        }
    } else {
        return x;
    }
    if hi == 0.0 {
        return 0.0;
    }
    x - hi * math::floor(x / hi)
}

/// `LinCongL.ar(freq, a, c, m, xi)`: a linear-congruential generator, scaled to `[-1, 1)` and
/// linearly interpolated.
///
/// The constructor seeds the ramp's start with the unscaled `xi` while every later start is a
/// scaled iterate, so the first hold ramps from `xi` itself, exactly as the reference does. `xi` is
/// read only by the constructor: a run-time change does not re-seed.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LinCongL {
    /// The current unscaled iterate.
    xn: f64,
    /// The scaled iterate the current hold ramps from.
    xnm1: f64,
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for LinCongL {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.xn = f64::from(ctx.ins.control(4));
        self.xnm1 = self.xn;
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = f64::from(ctx.ins.control(1));
        let c = f64::from(ctx.ins.control(2));
        let m = f64::from(ctx.ins.control(3).max(0.001));
        let scale = 2.0 / m;
        let mut xn = self.xn;
        let (_, xnm1) = chaos_interp(
            ctx,
            &mut self.counter,
            &mut self.frac,
            xn * scale - 1.0,
            self.xnm1,
            |_| {
                xn = sc_mod(xn * a + c, m);
                xn * scale - 1.0
            },
            |x| x,
        );
        self.xn = xn;
        self.xnm1 = xnm1;
        DoneAction::Nothing
    }
}

/// `LinCongC.ar(freq, a, c, m, xi)`: a linear-congruential generator, scaled to `[-1, 1)` and
/// cubically interpolated.
///
/// The constructor fills the history and all four cubic coefficients with the unscaled `xi`, exactly
/// as the reference does, so the first hold plays `xi*(1 + t + t^2 + t^3)` and the next few
/// cubics are fitted through the unscaled `xi`. `xi` is read only by the constructor: a run-time
/// change does not re-seed.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LinCongC {
    /// The current unscaled iterate.
    xn: f64,
    /// The previous three scaled iterates, newest first (`xnm1`, `xnm2`, `xnm3`).
    history: [f64; 3],
    /// The coefficients of the cubic the current hold plays.
    coefs: [f64; 4],
    /// How far the current hold has advanced, in units of one hold length.
    frac: f64,
    /// Samples emitted since the last iteration.
    counter: f32,
    _pad: u32,
}

impl Unit for LinCongC {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let xi = f64::from(ctx.ins.control(4));
        self.xn = xi;
        self.history = [xi; 3];
        self.coefs = [xi; 4];
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let a = f64::from(ctx.ins.control(1));
        let c = f64::from(ctx.ins.control(2));
        let m = f64::from(ctx.ins.control(3).max(0.001));
        let scale = 2.0 / m;
        let mut xn = self.xn;
        chaos_cubic(
            ctx,
            &mut self.counter,
            &mut self.frac,
            &mut self.history,
            &mut self.coefs,
            xn * scale - 1.0,
            |_| {
                xn = sc_mod(xn * a + c, m);
                xn * scale - 1.0
            },
        );
        self.xn = xn;
        DoneAction::Nothing
    }
}

/// Build a chaos generator with zeroed state and the given minimum input count. Every unit here
/// seeds its own state in [`Unit::init`], on the audio thread, where the init inputs are readable,
/// so the constructor never carries a meaningful value.
macro_rules! chaos_ctor {
    ($ctor:ident, $unit:ident, $min_inputs:expr) => {
        #[doc = concat!("Constructor for [`", stringify!($unit), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.len() < $min_inputs {
                    return Err(BuildError::WrongInputCount);
                }
                Ok(unit_spec($unit::zeroed()))
            }
        }
    };
}

chaos_ctor!(CuspNCtor, CuspN, 4);
chaos_ctor!(QuadNCtor, QuadN, 5);
chaos_ctor!(LinCongNCtor, LinCongN, 5);
chaos_ctor!(GbmanNCtor, GbmanN, 3);
chaos_ctor!(StandardNCtor, StandardN, 4);
chaos_ctor!(LatoocarfianNCtor, LatoocarfianN, 7);
chaos_ctor!(CuspLCtor, CuspL, 4);
chaos_ctor!(QuadLCtor, QuadL, 5);
chaos_ctor!(HenonLCtor, HenonL, 5);
chaos_ctor!(LorenzLCtor, LorenzL, 8);
chaos_ctor!(StandardLCtor, StandardL, 4);
chaos_ctor!(FBSineNCtor, FBSineN, 7);
chaos_ctor!(FBSineLCtor, FBSineL, 7);
chaos_ctor!(FBSineCCtor, FBSineC, 7);
chaos_ctor!(HenonNCtor, HenonN, 5);
chaos_ctor!(HenonCCtor, HenonC, 5);
chaos_ctor!(LatoocarfianLCtor, LatoocarfianL, 7);
chaos_ctor!(LatoocarfianCCtor, LatoocarfianC, 7);
chaos_ctor!(LinCongLCtor, LinCongL, 5);
chaos_ctor!(LinCongCCtor, LinCongC, 5);
