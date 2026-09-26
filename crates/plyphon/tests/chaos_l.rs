//! Exercise the linearly interpolating chaos generators (`CuspL`, `QuadL`, `HenonL`, `LorenzL`,
//! `StandardL`) against in-test replicas of the recurrences they port from scsynth's
//! `ChaosUGens.cpp`.
//!
//! Each replica reproduces the whole schedule rather than just the map: the constructor's
//! one-sample prime, the mixed-precision hold length and slope, the per-sample interpolation and
//! the unit's own output expression. The comparisons are therefore bit-exact, which is what a
//! chaotic map needs - a one-ULP difference in the seed grows to O(1) within a hundred iterations,
//! so a tolerance-based comparison over a long horizon would be either red or vacuous. Pointwise
//! anchors at hand-picked coefficients, asserted on exactly identified samples, pin the parts a
//! replica shares with the implementation and so could not catch on its own: which map is iterated,
//! how the output is scaled, and where the very first iterate lands.

use plyphon::{
    AddAction, BuildError, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, SynthNewError,
    UnitSpec, World, engine,
};
use plyphon_dsp::math;

const SR: f64 = 48_000.0;
/// The World control block: every render advances in whole blocks of this size.
const BLOCK: usize = 64;
/// A `freq` whose hold length is exactly 48 samples at [`SR`], so hold boundaries land on known
/// sample indices.
const HOLD_FREQ: f32 = 1_000.0;
/// The hold length in samples at [`HOLD_FREQ`].
const HOLD: usize = 48;
/// A `freq` whose hold length is exactly two samples, giving the shortest schedule in which every
/// distinct role - prime, first iteration, interpolated midpoint, second iteration - lands on its
/// own sample.
const HALF_SR_FREQ: f32 = 24_000.0;
/// Map iterations per replica comparison. A chaotic map only becomes discriminating over a long
/// horizon.
const ITERATIONS: usize = 300;

const PI: f64 = core::f64::consts::PI;
const TWO_PI: f64 = 2.0 * PI;
/// scsynth's `RECTWOPI`, which is not the nearest double to `1/(2π)`; the truncating branch of
/// [`mod2pi`] is sensitive to the difference.
const REC_TWO_PI: f64 = 0.1591549430918953;
/// scsynth's `ONESIXTH`, which is not the nearest double to `1/6`.
const ONE_SIXTH: f64 = 0.1666666666666667;
/// `LorenzL`'s output scale: a `float` literal widened to `double`, a hair below `0.04`.
const LORENZ_OUT_SCALE: f64 = 0.04f32 as f64;

// ---------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------

fn options() -> Options {
    Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 1,
        ..Options::default()
    }
}

/// `name(inputs) -> Out.ar(0)`, with `params` as the def's control parameters.
fn chaos_def(name: &str, inputs: Vec<InputRef>, params: Vec<Param>) -> SynthDef {
    SynthDef {
        name: "c".to_string(),
        params,
        units: vec![
            UnitSpec::new(name, Rate::Audio, inputs, 1),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

fn constant_inputs(consts: &[f32]) -> Vec<InputRef> {
    consts.iter().map(|&c| InputRef::Constant(c)).collect()
}

/// Pull `frames` samples out of `world` in `host`-sample host buffers.
fn drain(world: &mut World, frames: usize, host: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + host);
    let mut buf = vec![0.0f32; host];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// Render `frames` samples of `name(consts)` in `host`-sample host buffers, optionally reblocking
/// the graph to `reblock` samples per tick.
fn render_inner(
    name: &str,
    consts: &[f32],
    frames: usize,
    host: usize,
    reblock: Option<usize>,
) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(options());
    let def = chaos_def(name, constant_inputs(consts), vec![]);
    match reblock {
        Some(block) => controller.add_synthdef_reblocked(def, block),
        None => controller.add_synthdef(def),
    }
    controller
        .synth_new("c", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    drain(&mut world, frames, host)
}

/// Render `frames` samples of `name(consts)` in one-World-block host buffers.
fn render(name: &str, consts: &[f32], frames: usize) -> Vec<f32> {
    render_inner(name, consts, frames, BLOCK, None)
}

/// Render `blocks` World blocks of `name(consts)` with input `param_index` driven by a control
/// parameter that starts at `consts[param_index]`, applying each `(block, value)` change at the
/// start of the named block.
fn render_with_param(
    name: &str,
    consts: &[f32],
    param_index: usize,
    blocks: usize,
    changes: &[(usize, f32)],
) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(options());
    let mut inputs = constant_inputs(consts);
    inputs[param_index] = InputRef::Param(0);
    controller.add_synthdef(chaos_def(
        name,
        inputs,
        vec![Param::control("p", consts[param_index])],
    ));
    let node = controller
        .synth_new("c", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for block in 0..blocks {
        for &(at, value) in changes {
            if at == block {
                controller.set_control(node, 0, value).expect("set_control");
            }
        }
        out.extend_from_slice(&drain(&mut world, BLOCK, BLOCK));
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Replicas of the pinned recurrences
// ---------------------------------------------------------------------------------------------

/// The hold length and per-sample interpolation slope, in the mixed precision the `*L` prologues
/// use: an `f64` division narrowed to `f32`, then the `f64` widening of the `f32` reciprocal.
fn hold_and_slope(freq: f32) -> (f32, f64) {
    if f64::from(freq) < SR {
        let spc = (SR / f64::from(freq.max(0.001))) as f32;
        (spc, f64::from(1.0 / spc))
    } else {
        (1.0, 1.0)
    }
}

/// scsynth's `mod2pi`: a fast path over `[-2π, 4π)` and a truncating (non-Euclidean) fallback.
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

/// A Euclidean wrap, for contrast: never negative.
fn rem_euclid_2pi(x: f64) -> f64 {
    x.rem_euclid(TWO_PI)
}

/// Replay the `*L` schedule for a map whose interpolated variable is seeded from `xi`.
///
/// The constructor seeds `xn == xnm1 == xi` and then runs one sample, which advances the counter to
/// 1 and the phase to one slope without iterating the map. From there the map iterates whenever the
/// counter reaches the hold length. `dx` is carried rather than recomputed per block because with
/// constant controls the unit's per-block recomputation reproduces exactly the carried value.
fn replay(
    freq: f32,
    xi: f64,
    frames: usize,
    mut map: impl FnMut(f64) -> f64,
    out: impl Fn(f64) -> f64,
) -> Vec<f32> {
    let (spc, slope) = hold_and_slope(freq);
    let (mut xn, mut xnm1) = (xi, xi);
    let (mut counter, mut frac) = (1.0f32, slope);
    let mut dx = 0.0;
    let mut samples = Vec::with_capacity(frames);
    for _ in 0..frames {
        if counter >= spc {
            counter -= spc;
            frac = 0.0;
            xnm1 = xn;
            xn = map(xn);
            dx = xn - xnm1;
        }
        counter += 1.0;
        samples.push(out(xnm1 + dx * frac) as f32);
        frac += slope;
    }
    samples
}

/// Replay `HenonL`: two history terms, an asymmetric constructor seed (`xnm1 = x0, xnm2 = x1`) and
/// the stability latch.
///
/// The newest iterate is scoped to the iteration in both the replica and the unit: the reference
/// parks it in a struct member, but every read is preceded by a write. Recovery from the latch is
/// covered through the engine instead, since this replica drives constant inputs.
fn replay_henon(freq: f32, a: f64, b: f64, x0: f64, x1: f64, frames: usize) -> Vec<f32> {
    let (spc, slope) = hold_and_slope(freq);
    let (mut xnm1, mut xnm2) = (x0, x1);
    let (mut counter, mut frac) = (1.0f32, slope);
    let mut stable = true;
    let mut diff = xnm1 - xnm2;
    let mut samples = Vec::with_capacity(frames);
    for _ in 0..frames {
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
                frac = 0.0;
            }
        }
        counter += 1.0;
        samples.push((xnm2 + (diff * frac)) as f32);
        frac += slope;
    }
    samples
}

/// The Lorenz system advanced by one 4th-order Runge-Kutta step of `h`.
fn lorenz_step(x: f64, y: f64, z: f64, s: f64, r: f64, b: f64, h: f64) -> (f64, f64, f64) {
    let h_times_s = h * s;
    let k1x = h_times_s * (y - x);
    let k1y = h * (x * (r - z) - y);
    let k1z = h * (x * y - b * z);
    let (mut kx, mut ky, mut kz) = (k1x * 0.5, k1y * 0.5, k1z * 0.5);

    let k2x = h_times_s * (y + ky - x - kx);
    let k2y = h * ((x + kx) * (r - z - kz) - (y + ky));
    let k2z = h * ((x + kx) * (y + ky) - b * (z + kz));
    kx = k2x * 0.5;
    ky = k2y * 0.5;
    kz = k2z * 0.5;

    let k3x = h_times_s * (y + ky - x - kx);
    let k3y = h * ((x + kx) * (r - z - kz) - (y + ky));
    let k3z = h * ((x + kx) * (y + ky) - b * (z + kz));

    let k4x = h_times_s * (y + k3y - x - k3x);
    let k4y = h * ((x + k3x) * (r - z - k3z) - (y + k3y));
    let k4z = h * ((x + k3x) * (y + k3y) - b * (z + k3z));

    (
        x + (k1x + 2.0 * (k2x + k3x) + k4x) * ONE_SIXTH,
        y + (k1y + 2.0 * (k2y + k3y) + k4y) * ONE_SIXTH,
        z + (k1z + 2.0 * (k2z + k3z) + k4z) * ONE_SIXTH,
    )
}

// ---------------------------------------------------------------------------------------------
// Shared assertions
// ---------------------------------------------------------------------------------------------

fn assert_bit_exact(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: sample {i} was {g}, the pinned recurrence gives {w}"
        );
    }
}

fn assert_finite(out: &[f32], what: &str) {
    assert!(
        out.iter().all(|s| s.is_finite()),
        "{what}: output was not finite"
    );
}

/// Assert `out` is a chain of linear ramps whose breakpoints all sit `hold` samples apart.
///
/// Within a hold the first difference is constant, and at a hold boundary it jumps to the next
/// iterate's slope. An iteration whose slope barely changes leaves no detectable jump, so this
/// requires most - not all - of the expected breakpoints, and insists that every breakpoint it does
/// find falls on the same phase.
fn assert_hold_cadence(out: &[f32], hold: usize, what: &str) {
    let diffs: Vec<f32> = out.windows(2).map(|w| w[1] - w[0]).collect();
    // Two ULP of an O(1) sample is ~2.4e-7, and a genuine iteration changes the ramp slope by
    // orders of magnitude more than that.
    let breaks: Vec<usize> = (1..diffs.len())
        .filter(|&i| (diffs[i] - diffs[i - 1]).abs() > 1e-5)
        .collect();
    let expected = out.len() / hold;
    assert!(
        breaks.len() * 10 >= expected * 9,
        "{what}: found {} hold boundaries, expected about {expected}",
        breaks.len()
    );
    let phase = breaks[0] % hold;
    for &i in &breaks {
        assert_eq!(
            i % hold,
            phase,
            "{what}: hold boundary at sample {i} is off the {hold}-sample grid"
        );
    }
}

/// Assert `got` is within one part in `1e6` of a hand-computed anchor.
fn assert_close(got: f32, want: f32, what: &str) {
    let tol = 1e-6 * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, expected {want}"
    );
}

// ---------------------------------------------------------------------------------------------
// Per-unit recurrence tests
// ---------------------------------------------------------------------------------------------

/// `CuspL` reproduces scsynth's `CuspL_next`, whose emitted sample is
/// `ZXP(out) = xnm1 + dx * frac;` over the map `xn = a - (b * sqrt(sc_abs(xn)))`.
#[test]
fn cusp_l_matches_pinned_recurrence() {
    let (a, b, xi) = (1.0f32, 1.9f32, 0.25f32);
    let frames = ITERATIONS * HOLD;
    let got = render("CuspL", &[HOLD_FREQ, a, b, xi], frames);
    let want = replay(
        HOLD_FREQ,
        f64::from(xi),
        frames,
        |x| f64::from(a) - (f64::from(b) * math::sqrt(x.abs())),
        |v| v,
    );
    assert_bit_exact(&got, &want, "CuspL");
    assert_finite(&got, "CuspL");
    assert_hold_cadence(&got, HOLD, "CuspL");

    // At two samples per hold the schedule is exactly identified: the constructor's prime carries
    // samples 0 and 1, the first iteration fires on sample 1 (emitting the seed, since the phase
    // resets with it), sample 2 is the midpoint of the ramp and sample 3 is the first iterate. With
    // `xi = 0.25` the closed form is `1 - 1.9*sqrt(0.25) = 0.05`, so the midpoint is
    // `(0.25 + 0.05)/2 = 0.15`. No quadratic map at these coefficients passes through both 0.05
    // and 0.15 from a 0.25 seed, so a swapped map cannot satisfy the pair.
    let anchor = render("CuspL", &[HALF_SR_FREQ, a, b, xi], 4);
    assert_close(anchor[0], 0.25, "CuspL seed");
    assert_close(anchor[2], 0.15, "CuspL ramp midpoint");
    assert_close(anchor[3], 0.05, "CuspL first iterate");
}

/// `QuadL` reproduces scsynth's `QuadL_next`, whose emitted sample is
/// `ZXP(out) = xnm1 + dx * frac;` over the map `xn = a * xn * xn + b * xn + c`.
#[test]
fn quad_l_matches_pinned_recurrence() {
    let (a, b, c, xi) = (1.0f32, -1.0f32, -0.75f32, 0.5f32);
    let frames = ITERATIONS * HOLD;
    let got = render("QuadL", &[HOLD_FREQ, a, b, c, xi], frames);
    let want = replay(
        HOLD_FREQ,
        f64::from(xi),
        frames,
        |x| f64::from(a) * x * x + f64::from(b) * x + f64::from(c),
        |v| v,
    );
    assert_bit_exact(&got, &want, "QuadL");
    assert_finite(&got, "QuadL");
    assert_hold_cadence(&got, HOLD, "QuadL");

    // Same two-samples-per-hold schedule as `CuspL`. With `xi = 0.5` the closed form is
    // `0.25 - 0.5 - 0.75 = -1.0`, so the ramp midpoint is `(0.5 - 1.0)/2 = -0.25`. The cusp map at
    // these coefficients would give `1 - (-1)*sqrt(0.5) = 1.707`, so a swapped map cannot pass.
    let anchor = render("QuadL", &[HALF_SR_FREQ, a, b, c, xi], 4);
    assert_close(anchor[0], 0.5, "QuadL seed");
    assert_close(anchor[2], -0.25, "QuadL ramp midpoint");
    assert_close(anchor[3], -1.0, "QuadL first iterate");
}

/// `HenonL` reproduces scsynth's `HenonL_next`, whose emitted sample is
/// `ZXP(out) = xnm2 + (diff * frac);` - it interpolates between the *previous two* iterates of
/// `xn = 1.f - (a * xnm1 * xnm1) + (b * xnm2)`.
#[test]
fn henon_l_matches_pinned_recurrence() {
    let (a, b, x0, x1) = (1.4f32, 0.3f32, 0.3f32, 0.5f32);
    let frames = ITERATIONS * HOLD;
    let got = render("HenonL", &[HOLD_FREQ, a, b, x0, x1], frames);
    let want = replay_henon(
        HOLD_FREQ,
        f64::from(a),
        f64::from(b),
        f64::from(x0),
        f64::from(x1),
        frames,
    );
    assert_bit_exact(&got, &want, "HenonL");
    assert_finite(&got, "HenonL");
    assert_hold_cadence(&got, HOLD, "HenonL");

    // The constructor seed is asymmetric (`xn = x1, xnm1 = x0, xnm2 = x1`), so the primed hold ramps
    // from `x1` towards `x0`: at two samples per hold, sample 0 is the midpoint `(0.5 + 0.3)/2` and
    // sample 1 - the first iteration - emits `x0`. Sample 3 is the first computed iterate,
    // `1 - 1.4*0.3^2 + 0.3*0.5 = 1.024`, and sample 2 the midpoint towards it.
    let anchor = render("HenonL", &[HALF_SR_FREQ, a, b, x0, x1], 4);
    assert_close(anchor[0], 0.4, "HenonL primed midpoint");
    assert_close(anchor[1], 0.3, "HenonL x0");
    assert_close(anchor[2], 0.662, "HenonL ramp midpoint");
    assert_close(anchor[3], 1.024, "HenonL first iterate");
}

/// `LorenzL` reproduces scsynth's `LorenzL_next`, whose emitted sample is
/// `ZXP(out) = (xnm1 + dx * frac) * 0.04f;` over a 4th-order Runge-Kutta step (the Euler block
/// commented out beneath it in the reference is not the implementation).
#[test]
fn lorenz_l_matches_pinned_recurrence() {
    let (s, r, b, h) = (10.0f32, 28.0f32, 2.667f32, 0.05f32);
    let (xi, yi, zi) = (0.1f32, 0.0f32, 0.0f32);
    let frames = ITERATIONS * HOLD;
    let consts = [HOLD_FREQ, s, r, b, h, xi, yi, zi];
    let got = render("LorenzL", &consts, frames);
    let (mut y, mut z) = (f64::from(yi), f64::from(zi));
    let want = replay(
        HOLD_FREQ,
        f64::from(xi),
        frames,
        |x| {
            let (nx, ny, nz) = lorenz_step(
                x,
                y,
                z,
                f64::from(s),
                f64::from(r),
                f64::from(b),
                f64::from(h),
            );
            y = ny;
            z = nz;
            nx
        },
        |v| v * LORENZ_OUT_SCALE,
    );
    assert_bit_exact(&got, &want, "LorenzL");
    assert_hold_cadence(&got, HOLD, "LorenzL");
    // `h*s = 0.5` here; the unconditional integrator legitimately diverges for much larger products,
    // so finiteness is asserted at this coefficient set rather than in general.
    assert_finite(&got, "LorenzL");

    // At two samples per hold, sample 3 is the first integrated state and sample 2 the midpoint of
    // the ramp towards it. The anchor runs on coefficients that are exact in both widths, so `X1` -
    // one Runge-Kutta step from `(0.25, 0, 0)` at `s = 10, r = 28, b = 2.5, h = 0.0625`, computed
    // independently of this crate - is unambiguous. The pair pins the integrator and the 0.04
    // output scale: a Euler step would give `0.25 + 0.0625*10*(0 - 0.25) = 0.09375`.
    const X0: f64 = 0.25;
    const X1: f64 = 0.237_644_714_322_717_25;
    let anchor = render(
        "LorenzL",
        &[HALF_SR_FREQ, 10.0, 28.0, 2.5, 0.0625, X0 as f32, 0.0, 0.0],
        4,
    );
    assert_close(
        anchor[0],
        (LORENZ_OUT_SCALE * X0) as f32,
        "LorenzL scaled seed",
    );
    assert_close(
        anchor[2],
        (LORENZ_OUT_SCALE * (X0 + (X1 - X0) * 0.5)) as f32,
        "LorenzL ramp midpoint",
    );
    assert_close(
        anchor[3],
        (LORENZ_OUT_SCALE * X1) as f32,
        "LorenzL first integrated state",
    );
}

/// `StandardL` reproduces scsynth's `StandardL_next`, whose emitted sample is
/// `ZXP(out) = (xnm1 + dx * frac - PI) * RECPI;` over the kicked-rotor map wrapped by `mod2pi`.
#[test]
fn standard_l_matches_pinned_recurrence() {
    let (k, xi, yi) = (1.0f32, 0.5f32, 0.0f32);
    let frames = ITERATIONS * HOLD;
    let got = render("StandardL", &[HOLD_FREQ, k, xi, yi], frames);
    let mut y = f64::from(yi);
    let want = replay(
        HOLD_FREQ,
        f64::from(xi),
        frames,
        |x| {
            y = mod2pi(y + f64::from(k) * math::sin(x));
            mod2pi(x + y)
        },
        |v| (v - PI) * (1.0 / PI),
    );
    assert_bit_exact(&got, &want, "StandardL");
    assert_finite(&got, "StandardL");
    assert_hold_cadence(&got, HOLD, "StandardL");

    // Sample 0 is the seed run through the affine output, `(xi - π)/π`, in both hold-length
    // branches: with a hold longer than a sample nothing has iterated yet, and with a one-sample
    // hold the iteration on sample 0 resets the phase to zero before the sample is emitted. That
    // pins the output's offset and scale independently of the map.
    let seed = ((f64::from(xi) - PI) * (1.0 / PI)) as f32;
    for freq in [HOLD_FREQ, SR as f32] {
        let first = render("StandardL", &[freq, k, xi, yi], 1);
        assert_close(first[0], seed, "StandardL seed sample");
    }
}

// ---------------------------------------------------------------------------------------------
// Re-seeding, latching, wrapping, reblocking, arity
// ---------------------------------------------------------------------------------------------

/// A run-time change of an init input re-seeds the map, and the re-seed shifts the running iterate
/// into the history rather than snapping the output to the new seed.
///
/// At `freq = 375` the hold is 128 samples, so nothing has iterated by the end of the first
/// 64-sample block and the interpolation phase stands at exactly `65/128`. A re-seed at the start
/// of the second block must therefore emit `old + (new - old) * 65/128`: an implementation that set
/// both the iterate and its history to the new seed would emit `new` instead. `HenonL` asserts the
/// negative - while it is stable, an init change only refreshes its cached comparison values.
#[test]
fn chaos_l_reseeds_on_init_change() {
    const RESEED_FREQ: f32 = 375.0;
    const FRAC: f64 = 65.0 / 128.0;
    let ramp = |old: f32, new: f32| f64::from(old) + (f64::from(new) - f64::from(old)) * FRAC;

    // CuspL re-seeds on `xi` (input 3).
    let out = render_with_param("CuspL", &[RESEED_FREQ, 1.0, 1.9, 0.25], 3, 2, &[(1, 0.75)]);
    assert_eq!(out[BLOCK].to_bits(), (ramp(0.25, 0.75) as f32).to_bits());

    // QuadL re-seeds on `xi` (input 4).
    let out = render_with_param(
        "QuadL",
        &[RESEED_FREQ, 1.0, -1.0, -0.75, 0.25],
        4,
        2,
        &[(1, 0.75)],
    );
    assert_eq!(out[BLOCK].to_bits(), (ramp(0.25, 0.75) as f32).to_bits());

    // LorenzL re-seeds on `xi` (input 5), and its output carries the 0.04 scale.
    let lorenz = [RESEED_FREQ, 10.0, 28.0, 2.667, 0.05, 0.25, 0.0, 0.0];
    let out = render_with_param("LorenzL", &lorenz, 5, 2, &[(1, 0.75)]);
    assert_eq!(
        out[BLOCK].to_bits(),
        ((ramp(0.25, 0.75) * LORENZ_OUT_SCALE) as f32).to_bits()
    );

    // StandardL re-seeds on `xi` (input 2), assigning it unwrapped through the affine output.
    let standard = [RESEED_FREQ, 1.0, 0.25, 0.0];
    let out = render_with_param("StandardL", &standard, 2, 2, &[(1, 0.75)]);
    assert_eq!(
        out[BLOCK].to_bits(),
        (((ramp(0.25, 0.75) - PI) * (1.0 / PI)) as f32).to_bits()
    );

    // StandardL's `yi` (input 3) and LorenzL's `yi`/`zi` (inputs 6, 7) are in the comparison set
    // too. They do not move the interpolated variable, so they show up as a divergence from the
    // untouched run once the map next iterates.
    for (name, consts, index) in [
        ("StandardL", &standard[..], 3usize),
        ("LorenzL", &lorenz[..], 6),
        ("LorenzL", &lorenz[..], 7),
    ] {
        let plain = render_with_param(name, consts, index, 8, &[]);
        let changed = render_with_param(name, consts, index, 8, &[(1, 0.5)]);
        let divergence = plain
            .iter()
            .zip(&changed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            divergence > 1e-6,
            "{name}: changing input {index} at run time had no effect"
        );
    }

    // HenonL, while stable, only refreshes its cached values: the running iterates are untouched, so
    // changing `x1` (input 4) leaves the render bit-identical.
    let henon = [HOLD_FREQ, 1.4, 0.3, 0.3, 0.5];
    let plain = render_with_param("HenonL", &henon, 4, 8, &[]);
    let changed = render_with_param("HenonL", &henon, 4, 8, &[(1, -0.4)]);
    assert_bit_exact(&changed, &plain, "HenonL stable re-seed");
}

/// An iterate leaving `[-1.5, 1.5]` latches `HenonL` silent until an init input changes, and the
/// recovery re-seeds from `(x0, x0, x1)` - not from the constructor's asymmetric seed.
#[test]
fn henon_l_escape_latch_and_recovery() {
    // `a = 20` sends the first iterate to `1 - 20*0.5^2 + 0.3*0.8 = -3.76`, well outside the stable
    // band. Holds are 48 samples, so iterations fire at samples 47, 95, 143 and 191.
    let (a, b, x0, x1) = (20.0f32, 0.3f32, 0.5f32, 0.8f32);
    let out = render_with_param("HenonL", &[HOLD_FREQ, a, b, x0, x1], 1, 3, &[(2, 1.4)]);

    // Before the escape the primed hold ramps from `x1` down towards `x0`; after it the unit is
    // silent, and stays silent across the whole of the next block.
    assert_close(out[0], x1 + (x0 - x1) / 48.0, "pre-escape ramp start");
    assert!(
        out[..47].iter().all(|&s| s > x0),
        "the unit went quiet before the first iteration"
    );
    assert!(
        out[47..2 * BLOCK].iter().all(|&s| s == 0.0),
        "a latched HenonL must be silent"
    );

    // `a = 1.4` at the start of block 2 recovers the unit. The recovery re-seed is
    // `xnm2 = xnm1 = x0, xn = x1`, so the difference driving the ramp is zero and the output holds
    // flat at `x0` up to and including the next iteration at sample 143 - it would hold at `x1`
    // under the constructor's asymmetric seed. That iteration computes
    // `1 - 1.4*0.5^2 + 0.3*0.5 = 0.8`, which the following hold ramps towards on the recovered
    // 48-sample cadence, reaching it at sample 191.
    assert!(
        out[2 * BLOCK..=143].iter().all(|&s| s == x0),
        "recovery must resume from the re-seeded x0"
    );
    assert_close(
        out[190],
        x0 + (0.8 - x0) * 47.0 / 48.0,
        "recovered ramp end",
    );
    assert_close(out[191], 0.8, "second recovered iterate");
}

/// `StandardL` wraps its phase with scsynth's truncating `mod2pi`, which returns negative values
/// below `-2π` - unlike a Euclidean wrap.
#[test]
fn standard_l_wraps_like_mod2pi() {
    // `k = 100` with a negative seed drives the momentum to about `-84`, far below the `[-2π, 4π)`
    // fast path, so every iteration exercises the truncating fallback.
    let (k, xi, yi) = (100.0f32, -1.0f32, 0.0f32);
    let frames = ITERATIONS * HOLD;
    let got = render("StandardL", &[HOLD_FREQ, k, xi, yi], frames);

    let standard = |wrap: fn(f64) -> f64| {
        let mut y = f64::from(yi);
        replay(
            HOLD_FREQ,
            f64::from(xi),
            frames,
            move |x| {
                y = wrap(y + f64::from(k) * math::sin(x));
                wrap(x + y)
            },
            |v| (v - PI) * (1.0 / PI),
        )
    };
    assert_bit_exact(&got, &standard(mod2pi), "StandardL truncating wrap");

    let euclidean = standard(rem_euclid_2pi);
    let divergence = got
        .iter()
        .zip(&euclidean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        divergence > 0.1,
        "the Euclidean wrap must be distinguishable here (max difference {divergence})"
    );
}

/// With constant controls the `*L` units are reblock-invariant on both axes: how the host chops up
/// the World's output, and how finely the graph itself is ticked.
#[test]
fn chaos_l_reblock_invariant_with_constant_controls() {
    let frames = 8 * HOLD;
    for (name, consts) in units() {
        let plain = render_inner(name, consts, frames, BLOCK, None);
        for host in [128usize, 480, 512] {
            let other = render_inner(name, consts, frames, host, None);
            assert_bit_exact(&other, &plain, &format!("{name} at host buffer {host}"));
        }
        for block in [8usize, 16, 32] {
            let other = render_inner(name, consts, frames, BLOCK, Some(block));
            assert_bit_exact(&other, &plain, &format!("{name} reblocked to {block}"));
        }
    }
}

/// The chaos constructors reject an under-supplied input list through the existing error taxonomy.
/// The `*L` chaos arms live here; `tests/pack_fft.rs` and `tests/convolution.rs` assert the
/// spectral, demand-operator, and convolution arities beside their own families.
#[test]
fn new_units_reject_wrong_input_counts() {
    for (name, consts) in units() {
        // One input short of the full list - the only discriminating arity for a
        // single minimum-count check.
        let short = consts.len() - 1;
        let (mut controller, _nrt, _world) = engine(options());
        controller.add_synthdef(chaos_def(name, constant_inputs(&consts[..short]), vec![]));
        let err = controller
            .synth_new("c", ROOT_GROUP_ID, AddAction::Tail)
            .expect_err("an under-supplied chaos unit must be rejected");
        assert!(
            matches!(err, SynthNewError::Build(BuildError::WrongInputCount)),
            "{name} with {short} inputs: expected WrongInputCount, got {err:?}"
        );
    }
}

/// The five interpolating chaos units at coefficient sets that stay bounded.
fn units() -> [(&'static str, &'static [f32]); 5] {
    [
        ("CuspL", &[HOLD_FREQ, 1.0, 1.9, 0.25]),
        ("QuadL", &[HOLD_FREQ, 1.0, -1.0, -0.75, 0.5]),
        ("HenonL", &[HOLD_FREQ, 1.4, 0.3, 0.3, 0.5]),
        (
            "LorenzL",
            &[HOLD_FREQ, 10.0, 28.0, 2.667, 0.05, 0.1, 0.0, 0.0],
        ),
        ("StandardL", &[HOLD_FREQ, 1.0, 0.5, 0.0]),
    ]
}
