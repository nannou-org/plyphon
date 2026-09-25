//! Pin the chaos generators ported from scsynth's `ChaosUGens.cpp` against scsynth's own output.
//!
//! Every expected value below is a float bit pattern produced by scsynth's `*_Ctor` and `*_next`
//! functions, compiled unmodified from `ChaosUGens.cpp` against the plugin headers and driven the
//! way the server drives them: the constructor at 48 kHz (which runs the calc for one sample), then
//! 64-sample blocks, with any input change applied at the start of a block. The source was compiled
//! with floating-point contraction disabled, so each expression is evaluated as written.
//!
//! Each unit is pinned at two rates: [`SLOW`], whose hold of `48000/7000` samples is fractional so
//! the map iterates less than once per sample on an uneven cadence, and [`FAST`], where the hold
//! clamps to one sample and the map iterates every sample. Each case pins the whole first block and
//! a handful of samples up to 4096 in, where a chaotic map has long since amplified any one-ULP
//! difference.

use plyphon::{
    AddAction, BuildError, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, SynthNewError,
    UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// The World control block: every render advances in whole blocks of this size.
const BLOCK: usize = 64;
/// A `freq` below the sample rate whose hold, `48000/7000` samples, is not a whole number.
const SLOW: f32 = 7_000.0;
/// A `freq` at the sample rate: the hold clamps to one sample.
const FAST: f32 = 48_000.0;
/// Samples rendered for each rate case; the `*_LATER` pins reach its last sample.
const FRAMES: usize = 4_096;

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

/// Pull `blocks` World blocks out of `world`.
fn drain(world: &mut World, blocks: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; blocks * BLOCK];
    for chunk in out.chunks_mut(BLOCK) {
        world.fill(chunk, 1);
    }
    out
}

/// Render `frames` samples of `name(consts)`.
fn render(name: &str, consts: &[f32], frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(options());
    controller.add_synthdef(chaos_def(name, constant_inputs(consts), vec![]));
    controller
        .synth_new("c", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    drain(&mut world, frames.div_ceil(BLOCK))
}

/// Render `blocks` World blocks of `name(consts)` with input `index` driven by a control parameter
/// that starts at `consts[index]` and is set to `value` at the start of block `at`.
fn render_with_change(
    name: &str,
    consts: &[f32],
    blocks: usize,
    (at, index, value): (usize, usize, f32),
) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(options());
    let mut inputs = constant_inputs(consts);
    inputs[index] = InputRef::Param(0);
    controller.add_synthdef(chaos_def(
        name,
        inputs,
        vec![Param::control("p", consts[index])],
    ));
    let node = controller
        .synth_new("c", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for block in 0..blocks {
        if block == at {
            controller.set_control(node, 0, value).expect("set_control");
        }
        out.extend(drain(&mut world, 1));
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------------------------

fn assert_bits(got: &[f32], want: &[u32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w,
            "{what}: sample {i} was {g}, scsynth gives {}",
            f32::from_bits(w)
        );
    }
}

fn assert_points(got: &[f32], want: &[(usize, u32)], what: &str) {
    for &(i, w) in want {
        assert_eq!(
            got[i].to_bits(),
            w,
            "{what}: sample {i} was {}, scsynth gives {}",
            got[i],
            f32::from_bits(w)
        );
    }
}

/// Assert `name` renders scsynth's pinned output at both [`SLOW`] and [`FAST`]. `consts` supplies
/// every input but `freq`.
fn assert_pinned(
    name: &str,
    consts: &[f32],
    slow: (&[u32; BLOCK], &[(usize, u32)]),
    fast: (&[u32; BLOCK], &[(usize, u32)]),
) {
    for (freq, (block, later), rate) in [(SLOW, slow, "slow"), (FAST, fast, "fast")] {
        let mut inputs = vec![freq];
        inputs.extend_from_slice(consts);
        let out = render(name, &inputs, FRAMES);
        let what = format!("{name} ({rate})");
        assert_bits(&out[..BLOCK], block, &what);
        assert_points(&out, later, &what);
    }
}

/// Assert `name(consts)` with input `index` changed to `value` at the start of the second block
/// renders scsynth's pinned second block.
fn assert_reseed(name: &str, consts: &[f32], index: usize, value: f32, want: &[u32; BLOCK]) {
    let out = render_with_change(name, consts, 2, (1, index, value));
    assert_bits(
        &out[BLOCK..],
        want,
        &format!("{name} after input {index} changed"),
    );
}

/// Assert a run-time change of input `index` leaves `name(consts)` bit-identical to an untouched
/// render over eight blocks.
fn assert_change_ignored(name: &str, consts: &[f32], index: usize, value: f32) {
    let plain = render_with_change(name, consts, 8, (1, index, consts[index]));
    let changed = render_with_change(name, consts, 8, (1, index, value));
    let want: Vec<u32> = plain.iter().map(|s| s.to_bits()).collect();
    assert_bits(
        &changed,
        &want,
        &format!("{name} after input {index} changed"),
    );
}

/// Every chaos unit at its pinned inputs (`freq` first).
const UNITS: &[(&str, &[f32])] = &[
    ("FBSineN", FB_SINE),
    ("FBSineL", FB_SINE),
    ("FBSineC", FB_SINE),
    ("HenonN", HENON),
    ("HenonC", HENON),
    ("LatoocarfianL", LATOOCARFIAN),
    ("LatoocarfianC", LATOOCARFIAN),
    ("LinCongL", LIN_CONG),
    ("LinCongC", LIN_CONG),
    ("GbmanL", GBMAN),
    ("QuadC", QUAD),
    ("CuspN", CUSP),
    ("QuadN", QUAD),
    ("LinCongN", LIN_CONG),
    ("GbmanN", GBMAN),
    ("StandardN", STANDARD),
    ("LatoocarfianN", LATOOCARFIAN),
    ("CuspL", CUSP),
    ("QuadL", QUAD),
    ("HenonL", HENON),
    ("LorenzL", LORENZ),
    ("StandardL", STANDARD),
];

// ---------------------------------------------------------------------------------------------
// FBSine
// ---------------------------------------------------------------------------------------------

/// `FBSine*(freq, im=1, fb=0.1, a=1.1, c=0.5, xi=0.1, yi=0.1)`, the reference's defaults bar `freq`.
/// The phase grows by `a = 1.1` per iteration, so `mod2pi` wraps it regularly.
const FB_SINE: &[f32] = &[SLOW, 1.0, 0.1, 1.1, 0.5, 0.1, 0.1];

#[test]
fn fb_sine_n_matches_scsynth() {
    assert_pinned(
        "FBSineN",
        &FB_SINE[1..],
        (&FB_SINE_N_SLOW_BLOCK, &FB_SINE_N_SLOW_LATER),
        (&FB_SINE_N_FAST_BLOCK, &FB_SINE_N_FAST_LATER),
    );
}

#[test]
fn fb_sine_l_matches_scsynth() {
    assert_pinned(
        "FBSineL",
        &FB_SINE[1..],
        (&FB_SINE_L_SLOW_BLOCK, &FB_SINE_L_SLOW_LATER),
        (&FB_SINE_L_FAST_BLOCK, &FB_SINE_L_FAST_LATER),
    );
}

/// `FBSineC` starts from a zeroed cubic, so its first hold is silent.
#[test]
fn fb_sine_c_matches_scsynth() {
    assert_pinned(
        "FBSineC",
        &FB_SINE[1..],
        (&FB_SINE_C_SLOW_BLOCK, &FB_SINE_C_SLOW_LATER),
        (&FB_SINE_C_FAST_BLOCK, &FB_SINE_C_FAST_LATER),
    );
}

/// A run-time change of `xi` re-seeds all three forms; a change of `yi` re-seeds both variables of
/// `FBSineN` but only the iterate of `FBSineC`, whose phase runs on.
#[test]
fn fb_sine_reseeds_like_scsynth() {
    assert_reseed("FBSineN", FB_SINE, 5, 0.6, &FB_SINE_N_RESEED);
    assert_reseed("FBSineL", FB_SINE, 5, 0.6, &FB_SINE_L_RESEED);
    assert_reseed("FBSineC", FB_SINE, 5, 0.6, &FB_SINE_C_RESEED);
    assert_reseed("FBSineN", FB_SINE, 6, 0.6, &FB_SINE_N_RESEED_Y);
    assert_reseed("FBSineC", FB_SINE, 6, 0.6, &FB_SINE_C_RESEED_Y);
}

// ---------------------------------------------------------------------------------------------
// Henon
// ---------------------------------------------------------------------------------------------

/// `Henon*(freq, a=1.4, b=0.3, x0=0.3, x1=0.5)`: the classic stable coefficients.
const HENON: &[f32] = &[SLOW, 1.4, 0.3, 0.3, 0.5];
/// `a = 20` sends the first iterate to `1 - 20*0.5^2 + 0.3*0.8 = -3.76`, outside the stable band,
/// at a hold of exactly 48 samples.
const HENON_ESCAPE: &[f32] = &[1_000.0, 20.0, 0.3, 0.5, 0.8];

/// `HenonN` emits the older history term, so its first hold is `x1`.
#[test]
fn henon_n_matches_scsynth() {
    assert_pinned(
        "HenonN",
        &HENON[1..],
        (&HENON_N_SLOW_BLOCK, &HENON_N_SLOW_LATER),
        (&HENON_N_FAST_BLOCK, &HENON_N_FAST_LATER),
    );
}

/// `HenonC` starts from a zeroed cubic, so its first hold is silent.
#[test]
fn henon_c_matches_scsynth() {
    assert_pinned(
        "HenonC",
        &HENON[1..],
        (&HENON_C_SLOW_BLOCK, &HENON_C_SLOW_LATER),
        (&HENON_C_FAST_BLOCK, &HENON_C_FAST_LATER),
    );
}

/// An escaping iterate latches both units until an input changes, and the change re-seeds them.
///
/// The iterate escapes at sample 47. `HenonN` then holds `x0` rather than falling silent, and
/// `HenonC` keeps replaying the small cubic arc through `(0, 0, 0, 1)` on every hold. Setting `a`
/// back to 1.4 at sample 128 re-seeds the history from `x0`, and the map runs again from the next
/// hold boundary.
#[test]
fn henon_latches_and_recovers_like_scsynth() {
    for (name, want) in [("HenonN", &HENON_N_LATCH), ("HenonC", &HENON_C_LATCH)] {
        let out = render_with_change(name, HENON_ESCAPE, 3, (2, 1, 1.4));
        assert_bits(&out, want, &format!("{name} latch and recovery"));
    }
}

/// While the map is stable, an input change only refreshes the cached comparison values: the
/// running iterates are untouched.
#[test]
fn henon_ignores_seed_changes_while_stable() {
    assert_change_ignored("HenonN", HENON, 4, -0.4);
    assert_change_ignored("HenonC", HENON, 4, -0.4);
}

// ---------------------------------------------------------------------------------------------
// Latoocarfian
// ---------------------------------------------------------------------------------------------

/// `Latoocarfian*(freq, a=1, b=3, c=0.5, d=0.5, xi=0.5, yi=0.5)`, the reference's defaults bar
/// `freq`.
const LATOOCARFIAN: &[f32] = &[SLOW, 1.0, 3.0, 0.5, 0.5, 0.5, 0.5];

#[test]
fn latoocarfian_l_matches_scsynth() {
    assert_pinned(
        "LatoocarfianL",
        &LATOOCARFIAN[1..],
        (&LATOOCARFIAN_L_SLOW_BLOCK, &LATOOCARFIAN_L_SLOW_LATER),
        (&LATOOCARFIAN_L_FAST_BLOCK, &LATOOCARFIAN_L_FAST_LATER),
    );
}

/// `LatoocarfianC`'s constructor sets every cubic coefficient to `xi`, so its first hold plays
/// `xi*(1 + t + t^2 + t^3)` rather than holding `xi`.
#[test]
fn latoocarfian_c_matches_scsynth() {
    assert_pinned(
        "LatoocarfianC",
        &LATOOCARFIAN[1..],
        (&LATOOCARFIAN_C_SLOW_BLOCK, &LATOOCARFIAN_C_SLOW_LATER),
        (&LATOOCARFIAN_C_FAST_BLOCK, &LATOOCARFIAN_C_FAST_LATER),
    );
}

/// A run-time change of `xi` shifts the running iterate into the history and re-seeds both
/// variables.
#[test]
fn latoocarfian_reseeds_like_scsynth() {
    assert_reseed(
        "LatoocarfianL",
        LATOOCARFIAN,
        5,
        0.2,
        &LATOOCARFIAN_L_RESEED,
    );
    assert_reseed(
        "LatoocarfianC",
        LATOOCARFIAN,
        5,
        0.2,
        &LATOOCARFIAN_C_RESEED,
    );
}

// ---------------------------------------------------------------------------------------------
// LinCong
// ---------------------------------------------------------------------------------------------

/// `LinCong*(freq, a=1.1, c=0.13, m=1, xi=0.25)`: the reference's defaults bar `freq` and a nonzero
/// seed, so the unscaled seed the constructors leave in the history is distinguishable from its
/// scaled value.
const LIN_CONG: &[f32] = &[SLOW, 1.1, 0.13, 1.0, 0.25];
/// A multiplier of `2^32` puts `(x*a + c) / m` above `2^29`, where `sc_mod`'s floored branch
/// `x - m*floor(x/m)` rounds and so parts from the exact remainder.
const LIN_CONG_WIDE: &[f32] = &[SLOW, 4_294_967_296.0, 0.13, 0.7, 0.25];

/// `LinCongL`'s first hold ramps from the unscaled `xi` to the first scaled iterate.
#[test]
fn lin_cong_l_matches_scsynth() {
    assert_pinned(
        "LinCongL",
        &LIN_CONG[1..],
        (&LIN_CONG_L_SLOW_BLOCK, &LIN_CONG_L_SLOW_LATER),
        (&LIN_CONG_L_FAST_BLOCK, &LIN_CONG_L_FAST_LATER),
    );
    assert_bits(
        &render("LinCongL", LIN_CONG_WIDE, BLOCK),
        &LIN_CONG_L_WIDE,
        "LinCongL (wide)",
    );
}

/// `LinCongC`'s constructor sets every cubic coefficient to the unscaled `xi`.
#[test]
fn lin_cong_c_matches_scsynth() {
    assert_pinned(
        "LinCongC",
        &LIN_CONG[1..],
        (&LIN_CONG_C_SLOW_BLOCK, &LIN_CONG_C_SLOW_LATER),
        (&LIN_CONG_C_FAST_BLOCK, &LIN_CONG_C_FAST_LATER),
    );
    assert_bits(
        &render("LinCongC", LIN_CONG_WIDE, BLOCK),
        &LIN_CONG_C_WIDE,
        "LinCongC (wide)",
    );
}

/// Only the constructors read `xi`, so a run-time change is ignored.
#[test]
fn lin_cong_ignores_seed_changes() {
    assert_change_ignored("LinCongL", LIN_CONG, 4, 0.75);
    assert_change_ignored("LinCongC", LIN_CONG, 4, 0.75);
}

// ---------------------------------------------------------------------------------------------
// GbmanL and QuadC
// ---------------------------------------------------------------------------------------------

/// `GbmanL(freq, xi=1.2, yi=2.1)`, the reference's defaults bar `freq`.
const GBMAN: &[f32] = &[SLOW, 1.2, 2.1];
/// `QuadC(freq, a=1, b=-1, c=-0.75, xi=0)`, the reference's defaults bar `freq`.
const QUAD: &[f32] = &[SLOW, 1.0, -1.0, -0.75, 0.0];

/// `GbmanL`'s first hold ramps from `yi` towards `xi`.
#[test]
fn gbman_l_matches_scsynth() {
    assert_pinned(
        "GbmanL",
        &GBMAN[1..],
        (&GBMAN_L_SLOW_BLOCK, &GBMAN_L_SLOW_LATER),
        (&GBMAN_L_FAST_BLOCK, &GBMAN_L_FAST_LATER),
    );
}

/// Only the constructor reads `xi` and `yi`, so a run-time change is ignored.
#[test]
fn gbman_l_ignores_seed_changes() {
    assert_change_ignored("GbmanL", GBMAN, 1, 0.5);
    assert_change_ignored("GbmanL", GBMAN, 2, 0.5);
}

/// `QuadC`'s constructor sets every cubic coefficient to `xi`; at `xi = 0` its first hold is silent.
#[test]
fn quad_c_matches_scsynth() {
    assert_pinned(
        "QuadC",
        &QUAD[1..],
        (&QUAD_C_SLOW_BLOCK, &QUAD_C_SLOW_LATER),
        (&QUAD_C_FAST_BLOCK, &QUAD_C_FAST_LATER),
    );
}

/// A run-time change of `xi` shifts the running iterate into the history and re-seeds the map.
#[test]
fn quad_c_reseeds_like_scsynth() {
    assert_reseed("QuadC", QUAD, 4, 0.5, &QUAD_C_RESEED);
}

// ---------------------------------------------------------------------------------------------
// The earlier ports
// ---------------------------------------------------------------------------------------------

/// `Cusp*(freq, a=1, b=1.9, xi=0.25)`.
const CUSP: &[f32] = &[SLOW, 1.0, 1.9, 0.25];
/// `Standard*(freq, k=1, xi=0.5, yi=0)`.
const STANDARD: &[f32] = &[SLOW, 1.0, 0.5, 0.0];
/// `LorenzL(freq, s=10, r=28, b=2.667, h=0.05, xi=0.1, yi=0, zi=0)`.
const LORENZ: &[f32] = &[SLOW, 10.0, 28.0, 2.667, 0.05, 0.1, 0.0, 0.0];

/// A unit's pinned renders at [`SLOW`] and [`FAST`]: its first block and later samples at each.
type Pins = (
    (&'static [u32; BLOCK], &'static [(usize, u32)]),
    (&'static [u32; BLOCK], &'static [(usize, u32)]),
);

/// The chaos generators plyphon ported before the ones above render scsynth's output.
#[test]
fn earlier_ports_match_scsynth() {
    let cases: [(&str, &[f32], Pins); 11] = [
        (
            "CuspN",
            CUSP,
            (
                (&CUSP_N_SLOW_BLOCK, &CUSP_N_SLOW_LATER),
                (&CUSP_N_FAST_BLOCK, &CUSP_N_FAST_LATER),
            ),
        ),
        (
            "QuadN",
            QUAD,
            (
                (&QUAD_N_SLOW_BLOCK, &QUAD_N_SLOW_LATER),
                (&QUAD_N_FAST_BLOCK, &QUAD_N_FAST_LATER),
            ),
        ),
        (
            "LinCongN",
            LIN_CONG,
            (
                (&LIN_CONG_N_SLOW_BLOCK, &LIN_CONG_N_SLOW_LATER),
                (&LIN_CONG_N_FAST_BLOCK, &LIN_CONG_N_FAST_LATER),
            ),
        ),
        (
            "GbmanN",
            GBMAN,
            (
                (&GBMAN_N_SLOW_BLOCK, &GBMAN_N_SLOW_LATER),
                (&GBMAN_N_FAST_BLOCK, &GBMAN_N_FAST_LATER),
            ),
        ),
        (
            "StandardN",
            STANDARD,
            (
                (&STANDARD_N_SLOW_BLOCK, &STANDARD_N_SLOW_LATER),
                (&STANDARD_N_FAST_BLOCK, &STANDARD_N_FAST_LATER),
            ),
        ),
        (
            "LatoocarfianN",
            LATOOCARFIAN,
            (
                (&LATOOCARFIAN_N_SLOW_BLOCK, &LATOOCARFIAN_N_SLOW_LATER),
                (&LATOOCARFIAN_N_FAST_BLOCK, &LATOOCARFIAN_N_FAST_LATER),
            ),
        ),
        (
            "CuspL",
            CUSP,
            (
                (&CUSP_L_SLOW_BLOCK, &CUSP_L_SLOW_LATER),
                (&CUSP_L_FAST_BLOCK, &CUSP_L_FAST_LATER),
            ),
        ),
        (
            "QuadL",
            QUAD,
            (
                (&QUAD_L_SLOW_BLOCK, &QUAD_L_SLOW_LATER),
                (&QUAD_L_FAST_BLOCK, &QUAD_L_FAST_LATER),
            ),
        ),
        (
            "HenonL",
            HENON,
            (
                (&HENON_L_SLOW_BLOCK, &HENON_L_SLOW_LATER),
                (&HENON_L_FAST_BLOCK, &HENON_L_FAST_LATER),
            ),
        ),
        (
            "LorenzL",
            LORENZ,
            (
                (&LORENZ_L_SLOW_BLOCK, &LORENZ_L_SLOW_LATER),
                (&LORENZ_L_FAST_BLOCK, &LORENZ_L_FAST_LATER),
            ),
        ),
        (
            "StandardL",
            STANDARD,
            (
                (&STANDARD_L_SLOW_BLOCK, &STANDARD_L_SLOW_LATER),
                (&STANDARD_L_FAST_BLOCK, &STANDARD_L_FAST_LATER),
            ),
        ),
    ];
    for (name, consts, (slow, fast)) in cases {
        assert_pinned(name, &consts[1..], slow, fast);
    }
}

/// The earlier interpolating ports re-seed on a run-time change of `xi` as scsynth does.
#[test]
fn earlier_interpolating_ports_reseed_like_scsynth() {
    assert_reseed("CuspL", CUSP, 3, 0.75, &CUSP_L_RESEED);
    assert_reseed("QuadL", QUAD, 4, 0.5, &QUAD_L_RESEED);
    assert_reseed("StandardL", STANDARD, 2, 2.0, &STANDARD_L_RESEED);
    assert_reseed("LorenzL", LORENZ, 5, 0.75, &LORENZ_L_RESEED);
}

/// The earlier sample-and-hold ports re-seed on a run-time change of their init inputs as scsynth
/// does. `StandardN` keeps holding the output of its old phase until the next iteration, because
/// the reference computes the held value before the re-seed.
#[test]
fn earlier_sample_and_hold_ports_reseed_like_scsynth() {
    assert_reseed("CuspN", CUSP, 3, 0.75, &CUSP_N_RESEED);
    assert_reseed("QuadN", QUAD, 4, 0.5, &QUAD_N_RESEED);
    assert_reseed(
        "LatoocarfianN",
        LATOOCARFIAN,
        5,
        0.2,
        &LATOOCARFIAN_N_RESEED,
    );
    assert_reseed(
        "LatoocarfianN",
        LATOOCARFIAN,
        6,
        0.2,
        &LATOOCARFIAN_N_RESEED_Y,
    );
    assert_reseed("StandardN", STANDARD, 2, 2.0, &STANDARD_N_RESEED);
    assert_reseed("StandardN", STANDARD, 3, 0.3, &STANDARD_N_RESEED_Y);
}

/// `GbmanN` and `LinCongN` read their init inputs only in the constructor, so a run-time change is
/// ignored.
#[test]
fn earlier_ports_without_reseed_ignore_seed_changes() {
    assert_change_ignored("GbmanN", GBMAN, 1, 0.5);
    assert_change_ignored("GbmanN", GBMAN, 2, 0.5);
    assert_change_ignored("LinCongN", LIN_CONG, 4, 0.75);
}

/// `StandardN(freq, k=100, xi=-1, yi=0)`: the kick drives the momentum to about `-84`, far below
/// `mod2pi`'s `[-2π, 4π)` fast path, so its truncating branch wraps both variables - negative,
/// where a Euclidean wrap would not be.
#[test]
fn standard_n_wraps_like_mod2pi() {
    assert_pinned(
        "StandardN",
        &[100.0, -1.0, 0.0],
        (&STANDARD_N_WIDE_SLOW_BLOCK, &STANDARD_N_WIDE_SLOW_LATER),
        (&STANDARD_N_WIDE_FAST_BLOCK, &STANDARD_N_WIDE_FAST_LATER),
    );
}

// ---------------------------------------------------------------------------------------------
// Reblocking and arity
// ---------------------------------------------------------------------------------------------

/// With constant controls every unit renders the same samples however finely the graph is ticked.
#[test]
fn units_are_reblock_invariant_with_constant_controls() {
    const FRAMES: usize = 8 * BLOCK;
    for &(name, consts) in UNITS {
        let want: Vec<u32> = render(name, consts, FRAMES)
            .iter()
            .map(|s| s.to_bits())
            .collect();
        for block in [1usize, 8, 16, 32] {
            let (mut controller, _nrt, mut world) = engine(options());
            let def = chaos_def(name, constant_inputs(consts), vec![]);
            controller.add_synthdef_reblocked(def, block);
            controller
                .synth_new("c", ROOT_GROUP_ID, AddAction::Tail)
                .expect("synth_new");
            let got = drain(&mut world, FRAMES / BLOCK);
            assert_bits(&got, &want, &format!("{name} reblocked to {block}"));
        }
    }
}

/// Each constructor rejects an input list one short of its full arity.
#[test]
fn units_reject_short_input_lists() {
    for &(name, consts) in UNITS {
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

// ---------------------------------------------------------------------------------------------
// scsynth's output
// ---------------------------------------------------------------------------------------------

#[rustfmt::skip]
const FB_SINE_N_SLOW_BLOCK: [u32; 64] = [
    0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3de0_d372,
    0x3de0_d372, 0x3de0_d372, 0x3de0_d372, 0x3de0_d372, 0x3de0_d372, 0x3de0_d372, 0x3f14_f2dc,
    0x3f14_f2dc, 0x3f14_f2dc, 0x3f14_f2dc, 0x3f14_f2dc, 0x3f14_f2dc, 0x3f14_f2dc, 0x3f71_3508,
    0x3f71_3508, 0x3f71_3508, 0x3f71_3508, 0x3f71_3508, 0x3f71_3508, 0x3f71_3508, 0x3f73_ad91,
    0x3f73_ad91, 0x3f73_ad91, 0x3f73_ad91, 0x3f73_ad91, 0x3f73_ad91, 0x3f73_ad91, 0x3f0c_2fa8,
    0x3f0c_2fa8, 0x3f0c_2fa8, 0x3f0c_2fa8, 0x3f0c_2fa8, 0x3f0c_2fa8, 0x3f0c_2fa8, 0xbe01_76ba,
    0xbe01_76ba, 0xbe01_76ba, 0xbe01_76ba, 0xbe01_76ba, 0xbe01_76ba, 0xbe01_76ba, 0xbf45_6d6f,
    0xbf45_6d6f, 0xbf45_6d6f, 0xbf45_6d6f, 0xbf45_6d6f, 0xbf45_6d6f, 0xbf7d_2a5d, 0xbf7d_2a5d,
    0xbf7d_2a5d, 0xbf7d_2a5d, 0xbf7d_2a5d, 0xbf7d_2a5d, 0xbf7d_2a5d, 0xbede_9933, 0xbede_9933,
    0xbede_9933,
];

const FB_SINE_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3f41_bfcd),
    (1023, 0x3df0_86a6),
    (2047, 0x3f7f_cac4),
    (4095, 0xbf73_4ec6),
];

#[rustfmt::skip]
const FB_SINE_N_FAST_BLOCK: [u32; 64] = [
    0x3de0_d372, 0x3f14_f2dc, 0x3f71_3508, 0x3f73_ad91, 0x3f0c_2fa8, 0xbe01_76ba, 0xbf45_6d6f,
    0xbf7d_2a5d, 0xbede_9933, 0x3f24_b2e7, 0x3f7b_6626, 0x3f63_953e, 0x3ecb_e9ab, 0xbe9a_0be5,
    0xbf61_595b, 0xbf6e_7170, 0xbe3d_7b63, 0x3f58_7442, 0x3f7d_688f, 0x3f2e_b3dd, 0x3d57_aa62,
    0xbf22_3e7f, 0xbf7f_5c0a, 0xbf26_5e34, 0x3ec4_f8c4, 0x3f5d_5600, 0x3f7e_17ee, 0x3f33_2b97,
    0x3d9f_2809, 0xbf1c_d4be, 0xbf7e_b11c, 0xbf2d_2abe, 0x3eb0_f6c8, 0x3f57_0667, 0x3f7f_55de,
    0x3f3c_683b, 0x3e06_8812, 0xbf10_d868, 0xbf7c_79ca, 0xbf3a_f131, 0x3e85_39aa, 0x3f47_e0e6,
    0x3f7f_d5e7, 0x3f4e_9926, 0x3e7b_6dd4, 0xbeeb_1849, 0xbf74_3351, 0xbf54_9119, 0x3d94_67e0,
    0x3f20_a524, 0x3f76_bceb, 0x3f6d_0188, 0x3ef5_ce15, 0xbe54_f61c, 0xbf53_50cc, 0xbf77_ed6a,
    0xbea5_3ecf, 0x3f3e_d059, 0x3f7f_d68f, 0x3f4d_f5f9, 0x3e77_158a, 0xbeed_366c, 0xbf74_985f,
    0xbf53_b501,
];

const FB_SINE_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf0b_73ee),
    (1023, 0x3f7e_3803),
    (2047, 0xbe1e_2a60),
    (4095, 0x3f4f_e009),
];

#[rustfmt::skip]
const FB_SINE_L_SLOW_BLOCK: [u32; 64] = [
    0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd,
    0x3dcf_b870, 0x3dd2_a413, 0x3dd5_8fb5, 0x3dd8_7b58, 0x3ddb_66fb, 0x3dde_529e, 0x3de0_d372,
    0x3e36_e7f9, 0x3e7d_6638, 0x3ea1_f23c, 0x3ec5_315c, 0x3ee8_707c, 0x3f05_d7ce, 0x3f14_f2dc,
    0x3f22_672d, 0x3f2f_db7e, 0x3f3d_4fcf, 0x3f4a_c420, 0x3f58_3871, 0x3f65_acc2, 0x3f71_3508,
    0x3f71_9147, 0x3f71_ed85, 0x3f72_49c4, 0x3f72_a603, 0x3f73_0241, 0x3f73_5e80, 0x3f73_ad91,
    0x3f64_95df, 0x3f55_7e2d, 0x3f46_667b, 0x3f37_4ec9, 0x3f28_3717, 0x3f19_1f65, 0x3f0c_2fa8,
    0x3ee6_0b6c, 0x3eb3_b788, 0x3e81_63a4, 0x3e1e_1f81, 0x3d65_dee8, 0xbd2c_c035, 0xbe01_76ba,
    0xbe61_bfea, 0xbea1_048d, 0xbed1_2925, 0xbf00_a6df, 0xbf18_b92b, 0xbf45_6d6f, 0xbf4d_8e52,
    0xbf55_af35, 0xbf5d_d017, 0xbf65_f0fa, 0xbf6e_11dd, 0xbf76_32bf, 0xbf7d_2a5d, 0xbf68_7a06,
    0xbf53_c9af,
];

const FB_SINE_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3ea0_30d8),
    (1023, 0x3f0c_8d0d),
    (2047, 0x3f69_89ce),
    (4095, 0xbf18_916f),
];

#[rustfmt::skip]
const FB_SINE_L_FAST_BLOCK: [u32; 64] = [
    0x3dcc_cccd, 0x3de0_d372, 0x3f14_f2dc, 0x3f71_3508, 0x3f73_ad91, 0x3f0c_2fa8, 0xbe01_76ba,
    0xbf45_6d6f, 0xbf7d_2a5d, 0xbede_9933, 0x3f24_b2e7, 0x3f7b_6626, 0x3f63_953e, 0x3ecb_e9ab,
    0xbe9a_0be5, 0xbf61_595b, 0xbf6e_7170, 0xbe3d_7b63, 0x3f58_7442, 0x3f7d_688f, 0x3f2e_b3dd,
    0x3d57_aa62, 0xbf22_3e7f, 0xbf7f_5c0a, 0xbf26_5e34, 0x3ec4_f8c4, 0x3f5d_5600, 0x3f7e_17ee,
    0x3f33_2b97, 0x3d9f_2809, 0xbf1c_d4be, 0xbf7e_b11c, 0xbf2d_2abe, 0x3eb0_f6c8, 0x3f57_0667,
    0x3f7f_55de, 0x3f3c_683b, 0x3e06_8812, 0xbf10_d868, 0xbf7c_79ca, 0xbf3a_f131, 0x3e85_39aa,
    0x3f47_e0e6, 0x3f7f_d5e7, 0x3f4e_9926, 0x3e7b_6dd4, 0xbeeb_1849, 0xbf74_3351, 0xbf54_9119,
    0x3d94_67e0, 0x3f20_a524, 0x3f76_bceb, 0x3f6d_0188, 0x3ef5_ce15, 0xbe54_f61c, 0xbf53_50cc,
    0xbf77_ed6a, 0xbea5_3ecf, 0x3f3e_d059, 0x3f7f_d68f, 0x3f4d_f5f9, 0x3e77_158a, 0xbeed_366c,
    0xbf74_985f,
];

const FB_SINE_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf7f_c7c2),
    (1023, 0x3f3b_077e),
    (2047, 0x3f06_5b3a),
    (4095, 0x3f7f_c084),
];

#[rustfmt::skip]
const FB_SINE_C_SLOW_BLOCK: [u32; 64] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x3dcc_cccd,
    0x3dcc_9e3c, 0x3dcc_3257, 0x3dcb_b8d1, 0x3dcb_615e, 0x3dcb_5bb0, 0x3dcb_d77c, 0x3dcc_cccd,
    0x3dc6_0e51, 0x3db4_a6ce, 0x3da1_35a6, 0x3d94_5a38, 0x3d96_b3e3, 0x3db0_e20a, 0x3de0_d372,
    0x3e1e_c2f3, 0x3e5e_3549, 0x3e94_a469, 0x3ebd_42d4, 0x3ee6_39f0, 0x3f06_66e5, 0x3f14_f2dc,
    0x3f24_bd49, 0x3f34_a77f, 0x3f44_1fb8, 0x3f52_942d, 0x3f5f_7317, 0x3f6a_2ab0, 0x3f71_3508,
    0x3f77_4e80, 0x3f7b_b0aa, 0x3f7e_34fd, 0x3f7e_b4f3, 0x3f7d_0a05, 0x3f79_0dac, 0x3f73_ad91,
    0x3f6a_d995, 0x3f5f_53ac, 0x3f51_73be, 0x3f41_91b1, 0x3f30_056b, 0x3f1d_26d1, 0x3f0c_2fa8,
    0x3eed_41ae, 0x3ebd_5f2f, 0x3e8a_2481, 0x3e29_fca6, 0x3d7a_ca8d, 0xbe01_76ba, 0xbe67_50c9,
    0xbea8_b396, 0xbedd_f7d3, 0xbf08_c7fd, 0xbf20_cb78, 0xbf36_13ca, 0xbf45_6d6f, 0xbf55_2ab2,
    0xbf63_a31d,
];

const FB_SINE_C_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbf26_b873),
    (1023, 0x3f76_16e5),
    (2047, 0x3ec8_d13d),
    (4095, 0x3d5c_bf84),
];

#[rustfmt::skip]
const FB_SINE_C_FAST_BLOCK: [u32; 64] = [
    0x3dcc_cccd, 0x3dcc_cccd, 0x3de0_d372, 0x3f14_f2dc, 0x3f71_3508, 0x3f73_ad91, 0x3f0c_2fa8,
    0xbe01_76ba, 0xbf45_6d6f, 0xbf7d_2a5d, 0xbede_9933, 0x3f24_b2e7, 0x3f7b_6626, 0x3f63_953e,
    0x3ecb_e9ab, 0xbe9a_0be5, 0xbf61_595b, 0xbf6e_7170, 0xbe3d_7b63, 0x3f58_7442, 0x3f7d_688f,
    0x3f2e_b3dd, 0x3d57_aa62, 0xbf22_3e7f, 0xbf7f_5c0a, 0xbf26_5e34, 0x3ec4_f8c4, 0x3f5d_5600,
    0x3f7e_17ee, 0x3f33_2b97, 0x3d9f_2809, 0xbf1c_d4be, 0xbf7e_b11c, 0xbf2d_2abe, 0x3eb0_f6c8,
    0x3f57_0667, 0x3f7f_55de, 0x3f3c_683b, 0x3e06_8812, 0xbf10_d868, 0xbf7c_79ca, 0xbf3a_f131,
    0x3e85_39aa, 0x3f47_e0e6, 0x3f7f_d5e7, 0x3f4e_9926, 0x3e7b_6dd4, 0xbeeb_1849, 0xbf74_3351,
    0xbf54_9119, 0x3d94_67e0, 0x3f20_a524, 0x3f76_bceb, 0x3f6d_0188, 0x3ef5_ce15, 0xbe54_f61c,
    0xbf53_50cc, 0xbf77_ed6a, 0xbea5_3ecf, 0x3f3e_d059, 0x3f7f_d68f, 0x3f4d_f5f9, 0x3e77_158a,
    0xbeed_366c,
];

const FB_SINE_C_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf35_1bbb),
    (1023, 0x3e47_1a48),
    (2047, 0x3f71_9024),
    (4095, 0x3f46_a64b),
];

#[rustfmt::skip]
const FB_SINE_N_RESEED: [u32; 64] = [
    0x3f19_999a, 0x3f19_999a, 0x3f19_999a, 0x3f19_999a, 0x3e23_2450, 0x3e23_2450, 0x3e23_2450,
    0x3e23_2450, 0x3e23_2450, 0x3e23_2450, 0x3e23_2450, 0x3f15_fa71, 0x3f15_fa71, 0x3f15_fa71,
    0x3f15_fa71, 0x3f15_fa71, 0x3f15_fa71, 0x3f15_fa71, 0x3f71_3ddb, 0x3f71_3ddb, 0x3f71_3ddb,
    0x3f71_3ddb, 0x3f71_3ddb, 0x3f71_3ddb, 0x3f71_3ddb, 0x3f73_ad4c, 0x3f73_ad4c, 0x3f73_ad4c,
    0x3f73_ad4c, 0x3f73_ad4c, 0x3f73_ad4c, 0x3f73_ad4c, 0x3f0c_2fad, 0x3f0c_2fad, 0x3f0c_2fad,
    0x3f0c_2fad, 0x3f0c_2fad, 0x3f0c_2fad, 0xbe01_76bc, 0xbe01_76bc, 0xbe01_76bc, 0xbe01_76bc,
    0xbe01_76bc, 0xbe01_76bc, 0xbe01_76bc, 0xbf45_6d6f, 0xbf45_6d6f, 0xbf45_6d6f, 0xbf45_6d6f,
    0xbf45_6d6f, 0xbf45_6d6f, 0xbf45_6d6f, 0xbf7d_2a5d, 0xbf7d_2a5d, 0xbf7d_2a5d, 0xbf7d_2a5d,
    0xbf7d_2a5d, 0xbf7d_2a5d, 0xbf7d_2a5d, 0xbede_9933, 0xbede_9933, 0xbede_9933, 0xbede_9933,
    0xbede_9933,
];

#[rustfmt::skip]
const FB_SINE_L_RESEED: [u32; 64] = [
    0x3c93_0398, 0x3e2c_e6bb, 0x3ea3_b682, 0x3ef0_f9a6, 0x3f19_999a, 0x3f09_25dc, 0x3ef1_643b,
    0x3ed0_7cbf, 0x3eaf_9543, 0x3e8e_adc6, 0x3e5b_8c94, 0x3e23_2450, 0x3e62_d671, 0x3e91_4449,
    0x3eb1_1d59, 0x3ed0_f66a, 0x3ef0_cf7a, 0x3f08_5445, 0x3f15_fa71, 0x3f23_499b, 0x3f30_98c5,
    0x3f3d_e7f0, 0x3f4b_371a, 0x3f58_8644, 0x3f65_d56e, 0x3f71_3ddb, 0x3f71_98c6, 0x3f71_f3b1,
    0x3f72_4e9d, 0x3f72_a988, 0x3f73_0473, 0x3f73_5f5e, 0x3f73_ad4c, 0x3f64_95a5, 0x3f55_7dfe,
    0x3f46_6657, 0x3f37_4eb0, 0x3f28_3708, 0x3f0c_2fad, 0x3ee6_0b75, 0x3eb3_b790, 0x3e81_63aa,
    0x3e1e_1f8a, 0x3d65_defa, 0xbd2c_c032, 0xbe01_76bc, 0xbe61_bfec, 0xbea1_048e, 0xbed1_2926,
    0xbf00_a6df, 0xbf18_b92b, 0xbf30_cb77, 0xbf45_6d6f, 0xbf4d_8e52, 0xbf55_af35, 0xbf5d_d017,
    0xbf65_f0fa, 0xbf6e_11dd, 0xbf76_32bf, 0xbf7d_2a5d, 0xbf68_7a06, 0xbf53_c9af, 0xbf3f_1958,
    0xbf2a_6901,
];

#[rustfmt::skip]
const FB_SINE_C_RESEED: [u32; 64] = [
    0xbf70_0467, 0xbf79_7c49, 0xbf7f_387c, 0xbf80_335c, 0x3f19_999a, 0x3f2e_f699, 0x3f36_71f9,
    0x3f34_1d2e, 0x3f2c_09ab, 0x3f22_48e3, 0x3f1a_ec4a, 0x3f19_999a, 0x3f1c_1564, 0x3f1f_224a,
    0x3f22_d06c, 0x3f27_2fec, 0x3f2c_50ed, 0x3f32_4390, 0x3f38_0fc2, 0x3f40_c99f, 0x3f4b_d9a3,
    0x3f58_0c3e, 0x3f64_2de3, 0x3f6f_0b02, 0x3f77_700c, 0x3f7b_c1dc, 0x3f7e_09fb, 0x3f7e_7b06,
    0x3f7c_fe48, 0x3f79_7d0e, 0x3f73_e0a2, 0x3f6c_1251, 0x3f63_910b, 0x3f57_20fd, 0x3f47_f5a2,
    0x3f36_8169, 0x3f23_36c5, 0x3f0e_8827, 0x3ecb_ea70, 0x3e9c_d5ba, 0x3e52_f259, 0x3dcd_9453,
    0xbbf5_8fe1, 0xbdea_a210, 0xbe5f_02da, 0xbe9a_0bf8, 0xbecb_06da, 0xbefd_3474, 0xbf17_514c,
    0xbf2e_af8a, 0xbf43_bbdc, 0xbf55_7d2b, 0xbf61_595b, 0xbf6c_4eaf, 0xbf75_36dd, 0xbf7b_6db3,
    0xbf7e_4efd, 0xbf7d_3687, 0xbf77_8020, 0xbf6e_7170, 0xbf5e_19d1, 0xbf47_d829, 0xbf2c_e57f,
    0xbf0e_7ad9,
];

#[rustfmt::skip]
const FB_SINE_N_RESEED_Y: [u32; 64] = [
    0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3dcc_cccd, 0x3f12_a771, 0x3f12_a771, 0x3f12_a771,
    0x3f12_a771, 0x3f12_a771, 0x3f12_a771, 0x3f12_a771, 0x3f70_2b7c, 0x3f70_2b7c, 0x3f70_2b7c,
    0x3f70_2b7c, 0x3f70_2b7c, 0x3f70_2b7c, 0x3f70_2b7c, 0x3f74_a3e1, 0x3f74_a3e1, 0x3f74_a3e1,
    0x3f74_a3e1, 0x3f74_a3e1, 0x3f74_a3e1, 0x3f74_a3e1, 0x3f0e_f1e5, 0x3f0e_f1e5, 0x3f0e_f1e5,
    0x3f0e_f1e5, 0x3f0e_f1e5, 0x3f0e_f1e5, 0x3f0e_f1e5, 0xbde7_598a, 0xbde7_598a, 0xbde7_598a,
    0xbde7_598a, 0xbde7_598a, 0xbde7_598a, 0xbf43_003c, 0xbf43_003c, 0xbf43_003c, 0xbf43_003c,
    0xbf43_003c, 0xbf43_003c, 0xbf43_003c, 0xbf7d_c451, 0xbf7d_c451, 0xbf7d_c451, 0xbf7d_c451,
    0xbf7d_c451, 0xbf7d_c451, 0xbf7d_c451, 0xbee7_a5e2, 0xbee7_a5e2, 0xbee7_a5e2, 0xbee7_a5e2,
    0xbee7_a5e2, 0xbee7_a5e2, 0xbee7_a5e2, 0x3f20_1b7e, 0x3f20_1b7e, 0x3f20_1b7e, 0x3f20_1b7e,
    0x3f20_1b7e,
];

#[rustfmt::skip]
const FB_SINE_C_RESEED_Y: [u32; 64] = [
    0xbf70_0467, 0xbf79_7c49, 0xbf7f_387c, 0xbf80_335c, 0x3dcc_cccd, 0x3e1c_4bad, 0x3e25_fdc7,
    0x3e13_6b98, 0x3de9_0803, 0x3db2_6bcb, 0x3da2_e04c, 0x3dcc_cccd, 0x3e20_603b, 0x3e71_ae29,
    0x3ea9_0662, 0x3edc_9c50, 0x3f07_bb95, 0x3f1e_ba9c, 0x3f2e_f060, 0x3f3e_e60d, 0x3f4d_a5aa,
    0x3f5a_ec94, 0x3f66_7825, 0x3f70_05b7, 0x3f77_52a6, 0x3f7b_971c, 0x3f7e_5dc4, 0x3f7f_0334,
    0x3f7d_8533, 0x3f79_e188, 0x3f74_15f8, 0x3f6c_204c, 0x3f63_9300, 0x3f57_251b, 0x3f47_fa53,
    0x3f36_858d, 0x3f23_39b0, 0x3f0e_89a3, 0x3ecb_ea14, 0x3e9c_d52e, 0x3e52_f134, 0x3dcd_9260,
    0xbbf5_a55d, 0xbdea_a2ba, 0xbe5f_02e7, 0xbe9a_0bef, 0xbecb_06cd, 0xbefd_3467, 0xbf17_5146,
    0xbf2e_af86, 0xbf43_bbda, 0xbf55_7d2a, 0xbf61_595b, 0xbf6c_4eaf, 0xbf75_36de, 0xbf7b_6db3,
    0xbf7e_4efd, 0xbf7d_3688, 0xbf77_8020, 0xbf6e_7170, 0xbf5e_19d1, 0xbf47_d829, 0xbf2c_e57f,
    0xbf0e_7ad9,
];
#[rustfmt::skip]
const HENON_N_SLOW_BLOCK: [u32; 64] = [
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3e99_999a,
    0x3e99_999a, 0x3e99_999a, 0x3e99_999a, 0x3e99_999a, 0x3e99_999a, 0x3e99_999a, 0x3f83_126f,
    0x3f83_126f, 0x3f83_126f, 0x3f83_126f, 0x3f83_126f, 0x3f83_126f, 0x3f83_126f, 0xbec1_8a0d,
    0xbec1_8a0d, 0xbec1_8a0d, 0xbec1_8a0d, 0xbec1_8a0d, 0xbec1_8a0d, 0xbec1_8a0d, 0x3f8d_b747,
    0x3f8d_b747, 0x3f8d_b747, 0x3f8d_b747, 0x3f8d_b747, 0x3f8d_b747, 0x3f8d_b747, 0xbf54_5af8,
    0xbf54_5af8, 0xbf54_5af8, 0xbf54_5af8, 0xbf54_5af8, 0xbf54_5af8, 0xbf54_5af8, 0x3ebc_d5b8,
    0x3ebc_d5b8, 0x3ebc_d5b8, 0x3ebc_d5b8, 0x3ebc_d5b8, 0x3ebc_d5b8, 0x3ebc_d5b8, 0x3f0f_8a9a,
    0x3f0f_8a9a, 0x3f0f_8a9a, 0x3f0f_8a9a, 0x3f0f_8a9a, 0x3f0f_8a9a, 0x3f2b_a578, 0x3f2b_a578,
    0x3f2b_a578, 0x3f2b_a578, 0x3f2b_a578, 0x3f2b_a578, 0x3f2b_a578, 0x3f09_f085, 0x3f09_f085,
    0x3f09_f085,
];

const HENON_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3f9e_a210),
    (1023, 0x3f2e_13ac),
    (2047, 0x3e28_d7c5),
    (4095, 0x3c9f_a789),
];

#[rustfmt::skip]
const HENON_N_FAST_BLOCK: [u32; 64] = [
    0x3e99_999a, 0x3f83_126f, 0xbec1_8a0d, 0x3f8d_b747, 0xbf54_5af8, 0x3ebc_d5b8, 0x3f0f_8a9a,
    0x3f2b_a578, 0x3f09_f085, 0x3f4b_7033, 0x3e8e_178a, 0x3f90_b6c2, 0xbf34_cb80, 0x3f24_1287,
    0x3e5a_2d3a, 0x3f90_79d5, 0xbf38_3d6d, 0x3f1d_0d67, 0x3e83_ad82, 0x3f8b_b47b, 0xbf17_3180,
    0x3f56_cf7c, 0xbe26_d273, 0x3f9b_772e, 0xbf8e_9c2f, 0xbebf_368d, 0x3ef0_e477, 0x3f13_fb35,
    0x3f2c_606d, 0x3f09_e5c3, 0x3f4b_b884, 0x3e8c_cf08, 0x3f91_0123, 0xbf36_d426, 0x3f20_33d0,
    0x3e73_301d, 0x3f8d_ec61, 0xbf26_5f56, 0x3f3d_c7be, 0x3d11_f71e, 0x3f9c_3d4e, 0xbf89_9fd7,
    0xbe81_287b, 0x3f16_9e42, 0x3ee1_204d, 0x3f67_e4d9, 0xbc89_fbf9, 0x3fa2_bbb6, 0xbfa2_4b7e,
    0xbf5e_8a19, 0xbee0_6ba5, 0x3ef0_c2c0, 0x3f0f_163f, 0x3f34_25cf, 0x3ef2_e536, 0x3f65_61fd,
    0x3c96_04ad, 0x3fa2_58ea, 0xbf9f_929b, 0xbf4b_9ab3, 0xbe84_e5e8, 0x3f2a_c587, 0x3e99_2937,
    0x3f89_946e,
];

const HENON_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3f55_ffe5),
    (1023, 0xbf1f_df64),
    (2047, 0x3f5d_aa03),
    (4095, 0x3ea1_79ec),
];

#[rustfmt::skip]
const HENON_C_SLOW_BLOCK: [u32; 64] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x3e99_999a,
    0x3e9d_73b3, 0x3ea7_dd4c, 0x3eb7_1f5c, 0x3ec9_82d8, 0x3edd_50b6, 0x3ef0_d1ec, 0x3f00_0000,
    0x3f09_cd74, 0x3f15_c5c9, 0x3f22_64c9, 0x3f2e_263f, 0x3f37_85f4, 0x3f3c_ffb4, 0x3f3d_70a4,
    0x3f35_f349, 0x3f26_5ac9, 0x3f12_6e5c, 0x3efb_ea6d, 0x3ed9_6d1e, 0x3ec4_f337, 0x3ec4_47c3,
    0x3ee1_ca6d, 0x3f0e_21d7, 0x3f32_c702, 0x3f57_c1f6, 0x3f75_fff5, 0x3f83_371f, 0x3f82_1474,
    0x3f66_411a, 0x3f2e_d6e8, 0x3ed2_be31, 0x3e05_39d0, 0xbdef_31fa, 0xbe91_9477, 0xbea9_6665,
    0xbe6e_2c23, 0x3b2d_f5b1, 0x3ea1_5a71, 0x3f25_282c, 0x3f6e_d441, 0x3f8f_3374, 0x3f93_6846,
    0x3f80_dfbe, 0x3f34_8e6a, 0x3e9e_7db6, 0xbdf1_6d5a, 0xbf03_b5a6, 0xbf74_bae7, 0xbf78_4fdb,
    0xbf60_1a8c, 0xbf34_77a3, 0xbefb_8795, 0xbe88_b756, 0xbd84_df6b, 0x3d87_3e90, 0x3e41_ac06,
    0x3e9f_ad94,
];

const HENON_C_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3f1f_2700),
    (1023, 0x3cb6_db46),
    (2047, 0x3f5c_8e8d),
    (4095, 0x3f30_a106),
];

#[rustfmt::skip]
const HENON_C_FAST_BLOCK: [u32; 64] = [
    0x3e99_999a, 0x3f00_0000, 0x3f3d_70a4, 0x3ec4_47c3, 0x3f82_1474, 0xbea9_6665, 0x3f93_6846,
    0xbf74_bae7, 0x3d87_3e90, 0x3f35_049d, 0x3ea3_bfa2, 0x3f88_d2b7, 0xbf00_f330, 0x3f77_28b4,
    0xbee9_83e6, 0x3f7f_9889, 0xbf08_4bea, 0x3f67_1652, 0xbe99_dabf, 0x3f92_7b4a, 0xbf6c_7269,
    0x3e18_9648, 0x3f31_1ba1, 0x3ebf_cf21, 0x3f81_6a91, 0xbea3_3514, 0x3f94_9dc3, 0xbf7b_a12f,
    0xbb8c_7ef5, 0x3f34_8128, 0x3e9a_fa22, 0x3f8a_a83e, 0xbf0d_5175, 0x3f65_fa95, 0xbe97_4738,
    0x3f92_da4c, 0xbf6e_7120, 0x3e04_c0ef, 0x3f32_719a, 0x3eb7_a3a6, 0x3f83_b6a7, 0xbebf_e709,
    0x3f8e_56f2, 0xbf57_fc42, 0x3eac_939f, 0x3f16_7c61, 0x3f1e_0a8f, 0x3f24_8d78, 0x3f1b_54fc,
    0x3f2d_6a69, 0x3f0a_234e, 0x3f4b_ab69, 0x3e8d_2e5d, 0x3f90_eccb, 0xbf36_44d6, 0x3f21_45c1,
    0x3e6c_55ab, 0x3f8e_a53f, 0xbf2b_61b6, 0x3f34_f600, 0x3dcc_0244, 0x3f99_5db0, 0xbf7a_dfc5,
    0x3c75_046e,
];

const HENON_C_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf07_eace),
    (1023, 0x3e8d_46b5),
    (2047, 0xbf65_f2e8),
    (4095, 0x3f30_7eb6),
];

#[rustfmt::skip]
const HENON_N_LATCH: [u32; 192] = [
    0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd,
    0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd,
    0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd,
    0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd,
    0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd,
    0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd,
    0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f4c_cccd, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f00_0000, 0x3f00_0000, 0x3f4c_cccd,
];

#[rustfmt::skip]
const HENON_C_LATCH: [u32; 192] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0xb95e_d099,
    0xba5a_12f7, 0xbaf0_0000, 0xbb50_97b5, 0xbb9f_4260, 0xbbe0_0000, 0xbc14_d099, 0xbc3d_a130,
    0xbc6a_0000, 0xbc8c_bda2, 0xbca5_d098, 0xbcc0_0000, 0xbcdb_12f7, 0xbcf6_d099, 0xbd09_8000,
    0xbd17_b426, 0xbd25_e84d, 0xbd34_0000, 0xbd41_ded1, 0xbd4f_684d, 0xbd5c_8000, 0xbd69_097c,
    0xbd74_e84d, 0xbd80_0000, 0xbd85_1a14, 0xbd89_b426, 0xbd8d_c000, 0xbd91_2f69, 0xbd93_f426,
    0xbd96_0000, 0xbd97_44be, 0xbd97_b426, 0xbd97_4000, 0xbd95_da13, 0xbd93_7426, 0xbd90_0000,
    0xbd8b_6f68, 0xbd85_b426, 0xbd7d_8000, 0xbd6d_0979, 0xbd59_e84b, 0xbd44_0000, 0xbd2b_3423,
    0xbd0f_684a, 0xbce1_0000, 0xbc9c_bd98, 0xbc23_a126, 0x0000_0000, 0xb95e_d099, 0xba5a_12f7,
    0xbaf0_0000, 0xbb50_97b5, 0xbb9f_4260, 0xbbe0_0000, 0xbc14_d099, 0xbc3d_a130, 0xbc6a_0000,
    0xbc8c_bda2, 0xbca5_d098, 0xbcc0_0000, 0xbcdb_12f7, 0xbcf6_d099, 0xbd09_8000, 0xbd17_b426,
    0xbd25_e84d, 0xbd34_0000, 0xbd41_ded1, 0xbd4f_684d, 0xbd5c_8000, 0xbd69_097c, 0xbd74_e84d,
    0xbd80_0000, 0xbd85_1a14, 0xbd89_b426, 0xbd8d_c000, 0xbd91_2f69, 0xbd93_f426, 0xbd96_0000,
    0xbd97_44be, 0xbd97_b426, 0xbd97_4000, 0xbd95_da13, 0xbd93_7426, 0xbd90_0000, 0xbd8b_6f68,
    0xbd85_b426, 0xbd7d_8000, 0xbd6d_0979, 0xbd59_e84b, 0xbd44_0000, 0xbd2b_3423, 0xbd0f_684a,
    0xbce1_0000, 0xbc9c_bd98, 0xbc23_a126, 0x3f00_0000, 0x3f00_e107, 0x3f01_e925, 0x3f03_1652,
    0x3f04_6684, 0x3f05_d7b0, 0x3f07_67cf, 0x3f09_14d6, 0x3f0a_dcbb, 0x3f0c_bd75, 0x3f0e_b4fa,
    0x3f10_c142, 0x3f12_e042, 0x3f15_0ff0, 0x3f17_4e44, 0x3f19_9933, 0x3f1b_eeb5, 0x3f1e_4cbf,
    0x3f20_b148, 0x3f23_1a46, 0x3f25_85b1, 0x3f27_f17d, 0x3f2a_5ba2, 0x3f2c_c217, 0x3f2f_22d1,
    0x3f31_7bc8, 0x3f33_caf0, 0x3f36_0e42, 0x3f38_43b3, 0x3f3a_693a, 0x3f3c_7ccd, 0x3f3e_7c63,
    0x3f40_65f2, 0x3f42_3771, 0x3f43_eed6, 0x3f45_8a17, 0x3f47_072b, 0x3f48_6409, 0x3f49_9ea6,
    0x3f4a_b4fa, 0x3f4b_a4fb, 0x3f4c_6c9e, 0x3f4d_09db, 0x3f4d_7aa9, 0x3f4d_bcfc, 0x3f4d_cecd,
    0x3f4d_ae11, 0x3f4d_58bf, 0x3f4c_cccd,
];
#[rustfmt::skip]
const LATOOCARFIAN_L_SLOW_BLOCK: [u32; 64] = [
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000,
    0x3f25_316c, 0x3f4a_62d8, 0x3f6f_9443, 0x3f8a_62d8, 0x3f9c_fb8e, 0x3faf_9443, 0x3fbf_84e0,
    0x3faa_08c4, 0x3f94_8ca9, 0x3f7e_211a, 0x3f53_28e3, 0x3f28_30ac, 0x3efa_70ea, 0x3eb0_c766,
    0x3e7f_3689, 0x3e1c_de46, 0x3d6a_180d, 0xbd1f_48fe, 0xbe0a_2a82, 0xbe6c_82c5, 0xbea0_6737,
    0xbe71_7e31, 0xbe22_2df5, 0xbda5_bb71, 0xbb63_5f22, 0x3d97_857f, 0x3e1b_12fc, 0x3e5f_0e9e,
    0x3e83_060e, 0x3e96_84cc, 0x3eaa_038a, 0x3ebd_8249, 0x3ed1_0107, 0x3ee4_7fc5, 0x3ef5_358d,
    0x3f14_4aec, 0x3f2d_fb12, 0x3f47_ab38, 0x3f61_5b5e, 0x3f7b_0b83, 0x3f8a_5dd5, 0x3f95_602e,
    0x3f8e_afe4, 0x3f87_ff9a, 0x3f81_4f51, 0x3f75_3e0e, 0x3f67_dd7a, 0x3f4f_058c, 0x3f2d_1b2b,
    0x3f0b_30ca, 0x3ed2_8cd2, 0x3e8e_b80f, 0x3e15_c69a, 0x3c61_d15c, 0xbdcc_5629, 0xbe2a_ab67,
    0xbe6f_2bb9,
];

const LATOOCARFIAN_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbecc_8e8c),
    (1023, 0x3f30_eb12),
    (2047, 0xbe46_6c66),
    (4095, 0xbf50_10a4),
];

#[rustfmt::skip]
const LATOOCARFIAN_L_FAST_BLOCK: [u32; 64] = [
    0x3f00_0000, 0x3fbf_84e0, 0x3eb0_c766, 0xbea0_6737, 0x3e5f_0e9e, 0x3ef5_358d, 0x3f95_602e,
    0x3f4f_058c, 0xbdcc_5629, 0xbf0e_f909, 0x3ec8_05fe, 0xbed1_8d4c, 0x3dea_e469, 0xbf1a_abae,
    0xbf13_ca7b, 0xbfbd_1d92, 0xbe16_3331, 0x3f18_b54e, 0xbee9_a4ac, 0x3e65_8a43, 0xbefa_a85e,
    0xbe94_1032, 0xbfac_5a53, 0xbf1b_060b, 0xbc03_992d, 0xbd27_f2d3, 0xbf83_b701, 0xbf37_56e6,
    0xbf1c_e58d, 0xbed0_4be5, 0xbf11_62f7, 0xbf8f_ff1e, 0xbe9a_b1a0, 0x3e90_febe, 0xbeb1_0495,
    0xbf24_e169, 0xbfaf_4fc6, 0xbe99_5c31, 0x3eba_8a0e, 0xbe8b_7cfa, 0xbeb4_07e5, 0xbf93_8e3b,
    0xbf55_8ef1, 0xbc6a_6605, 0x3ed1_b28a, 0xbf05_2f6f, 0xbc95_ed83, 0xbf79_3e8e, 0xbf38_9213,
    0xbf3e_3f95, 0xbeb2_c0a1, 0xbe71_4e54, 0xbf82_a1a7, 0xbf82_a952, 0x3e53_9acb, 0x3f7d_1106,
    0xbf26_cb8a, 0x3ecd_d803, 0xbe93_19e7, 0x3e9b_83d7, 0xbd84_b66b, 0x3f04_1835, 0x3f21_d9c5,
    0x3fbc_9803,
];

const LATOOCARFIAN_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf5b_626e),
    (1023, 0xbede_24be),
    (2047, 0x3f74_16ee),
    (4095, 0x3f10_d5ce),
];

#[rustfmt::skip]
const LATOOCARFIAN_C_SLOW_BLOCK: [u32; 64] = [
    0x3f15_c92f, 0x3f33_65ed, 0x3f5b_3800, 0x3f87_d097, 0x3fa9_81a1, 0x3fd3_e000, 0x3f00_0000,
    0x3efb_5df4, 0x3ef0_a1cd, 0x3ee4_8a83, 0x3edb_d710, 0x3edb_466c, 0x3ee7_9791, 0x3f00_0000,
    0x3f1c_9e27, 0x3f48_4950, 0x3f7b_84ae, 0x3f97_69b7, 0x3fad_5c62, 0x3fbb_dbf0, 0x3fbf_84e0,
    0x3fb8_155c, 0x3fa6_c2be, 0x3f8e_b1d1, 0x3f66_0ebe, 0x3f2d_d068, 0x3ef5_e467, 0x3eb0_c766,
    0x3e59_52a7, 0x3da6_8a4c, 0xbd27_d415, 0xbe17_b724, 0xbe6f_5c86, 0xbe95_203c, 0xbea0_6737,
    0xbe97_e548, 0xbe72_fc83, 0xbe18_4ca5, 0xbd36_8d4c, 0x3d7c_5c5b, 0x3e1f_fa3e, 0x3e5f_0e9e,
    0x3e88_6499, 0x3e9a_0c4a, 0x3ea7_d06b, 0x3eb5_0302, 0x3ec4_f617, 0x3eda_fbb4, 0x3ef5_358d,
    0x3f10_e921, 0x3f2e_0db8, 0x3f4e_861e, 0x3f6e_cfe5, 0x3f85_b44f, 0x3f95_602e, 0x3f96_73c3,
    0x3f93_eb97, 0x3f8e_5ef8, 0x3f86_6534, 0x3f79_2b31, 0x3f63_0ee6, 0x3f4f_058c, 0x3f33_8d81,
    0x3f12_06cd,
];

const LATOOCARFIAN_C_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3ef0_3ddb),
    (1023, 0x3f82_d964),
    (2047, 0xbd30_4e7a),
    (4095, 0xbf87_0cd1),
];

#[rustfmt::skip]
const LATOOCARFIAN_C_FAST_BLOCK: [u32; 64] = [
    0x3f00_0000, 0x3f00_0000, 0x3fbf_84e0, 0x3eb0_c766, 0xbea0_6737, 0x3e5f_0e9e, 0x3ef5_358d,
    0x3f95_602e, 0x3f4f_058c, 0xbdcc_5629, 0xbf0e_f909, 0x3ec8_05fe, 0xbed1_8d4c, 0x3dea_e469,
    0xbf1a_abae, 0xbf13_ca7b, 0xbfbd_1d92, 0xbe16_3331, 0x3f18_b54e, 0xbee9_a4ac, 0x3e65_8a43,
    0xbefa_a85e, 0xbe94_1032, 0xbfac_5a53, 0xbf1b_060b, 0xbc03_992d, 0xbd27_f2d3, 0xbf83_b701,
    0xbf37_56e6, 0xbf1c_e58d, 0xbed0_4be5, 0xbf11_62f7, 0xbf8f_ff1e, 0xbe9a_b1a0, 0x3e90_febe,
    0xbeb1_0495, 0xbf24_e169, 0xbfaf_4fc6, 0xbe99_5c31, 0x3eba_8a0e, 0xbe8b_7cfa, 0xbeb4_07e5,
    0xbf93_8e3b, 0xbf55_8ef1, 0xbc6a_6605, 0x3ed1_b28a, 0xbf05_2f6f, 0xbc95_ed83, 0xbf79_3e8e,
    0xbf38_9213, 0xbf3e_3f95, 0xbeb2_c0a1, 0xbe71_4e54, 0xbf82_a1a7, 0xbf82_a952, 0x3e53_9acb,
    0x3f7d_1106, 0xbf26_cb8a, 0x3ecd_d803, 0xbe93_19e7, 0x3e9b_83d7, 0xbd84_b66b, 0x3f04_1835,
    0x3f21_d9c5,
];

const LATOOCARFIAN_C_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf69_a27b),
    (1023, 0x3ea3_e254),
    (2047, 0x3e07_690a),
    (4095, 0xbf38_841f),
];

#[rustfmt::skip]
const LATOOCARFIAN_L_RESEED: [u32; 64] = [
    0xbe68_16bc, 0xbded_a486, 0xbbb1_b95f, 0x3dd7_6d5b, 0x3e4c_cccd, 0x3eb7_06ba, 0x3f03_d387,
    0x3f2c_23b1, 0x3f54_73da, 0x3f7c_c404, 0x3f92_8a17, 0x3fa3_d105, 0x3f97_fd3b, 0x3f8c_2971,
    0x3f80_55a7, 0x3f69_03bb, 0x3f51_5c27, 0x3f39_b493, 0x3f25_6e14, 0x3f11_41c9, 0x3efa_2afb,
    0x3ed1_d264, 0x3ea9_79cd, 0x3e81_2136, 0x3e31_913f, 0x3dd8_ce2e, 0x3dda_0b03, 0x3ddb_47d9,
    0x3ddc_84ae, 0x3ddd_c184, 0x3dde_fe5a, 0x3de0_3b2f, 0x3de1_4ac1, 0x3e86_c126, 0x3ed5_2f9b,
    0x3f11_cf08, 0x3f39_0643, 0x3f60_3d7e, 0x3f94_88e3, 0x3f8c_6e53, 0x3f84_53c3, 0x3f78_7266,
    0x3f68_3d45, 0x3f58_0825, 0x3f47_d305, 0x3f39_eea0, 0x3f28_9a06, 0x3f17_456c, 0x3f05_f0d2,
    0x3ee9_386f, 0x3ec6_8f3b, 0x3ea3_e607, 0x3e86_306c, 0x3e7e_8fa2, 0x3e70_be6b, 0x3e62_ed34,
    0x3e55_1bfd, 0x3e47_4ac6, 0x3e39_798e, 0x3e2d_a1a8, 0x3e9c_0476, 0x3ee1_3817, 0x3f13_35dc,
    0x3f35_cfad,
];

#[rustfmt::skip]
const LATOOCARFIAN_C_RESEED: [u32; 64] = [
    0x3ed9_a645, 0x3e8c_a869, 0x3e03_aed9, 0xbb82_5343, 0xbf0e_f909, 0xbf03_fafd, 0xbedf_3c22,
    0xbea8_091c, 0xbe4d_42f6, 0xbd7a_4e73, 0x3da6_e56c, 0x3e4c_cccd, 0x3eb6_3df3, 0x3f0b_05db,
    0x3f3e_0f18, 0x3f6f_61f2, 0x3f8d_12d6, 0x3f9c_c0c2, 0x3fa3_d105, 0x3fa3_8c69, 0x3f9b_d986,
    0x3f8e_df30, 0x3f7d_887d, 0x3f5b_5f0d, 0x3f3b_8fbc, 0x3f25_6e14, 0x3f0e_b888, 0x3eee_4662,
    0x3ebf_826d, 0x3e93_4b82, 0x3e57_8fde, 0x3e16_3c03, 0x3dd8_ce2e, 0x3d8b_417d, 0x3d12_41a3,
    0x3c5f_dfe0, 0x3bd6_2cf1, 0x3c9b_946d, 0x3de1_4ac1, 0x3e62_519a, 0x3ec4_e3cf, 0x3f13_ba9e,
    0x3f46_6749, 0x3f74_70ad, 0x3f8b_e7c6, 0x3f94_88e3, 0x3f96_933c, 0x3f92_58dd, 0x3f89_9539,
    0x3f7c_0787, 0x3f62_bfe3, 0x3f4a_ca6d, 0x3f39_eea0, 0x3f28_25a5, 0x3f14_f7c3, 0x3f01_59f6,
    0x3edc_8273, 0x3eb9_4514, 0x3e9a_e5c8, 0x3e86_306c, 0x3e60_d50b, 0x3e35_7260, 0x3e10_648a,
    0x3def_ae75,
];
#[rustfmt::skip]
const LIN_CONG_L_SLOW_BLOCK: [u32; 64] = [
    0x3e10_0000, 0x3d00_0002, 0xbd9f_fffe, 0xbe3f_ffff, 0xbe97_ffff, 0xbecf_ffff, 0xbf00_0000,
    0xbee8_da74, 0xbed1_b4e8, 0xbeba_8f5c, 0xbea3_69d0, 0xbe8c_4444, 0xbe6a_3d71, 0xbe42_8f5c,
    0xbe0f_a328, 0xbdb9_6de9, 0xbd27_2b02, 0x3c12_1737, 0x3d70_369d, 0x3ddd_f3b6, 0x3e1a_9fbf,
    0x3e52_a392, 0x3e85_53b2, 0x3ea1_559c, 0x3ebd_5785, 0x3ed9_596e, 0x3ef5_5b58, 0x3f06_ae7e,
    0x3f16_15f1, 0x3f25_7d65, 0x3f34_e4d9, 0x3f44_4c4c, 0x3f53_b3c0, 0x3f63_1b33, 0x3f70_4f4d,
    0x3f36_966f, 0x3ef9_bb21, 0x3e86_4964, 0x3d16_bd35, 0xbe41_342d, 0xbed4_0bd4, 0xbf1b_7fce,
    0xbf10_53b2, 0xbf05_2796, 0xbef3_f6f4, 0xbedd_9ebc, 0xbec7_4684, 0xbeb0_ee4c, 0xbe9d_c740,
    0xbe85_3302, 0xbe59_3d8a, 0xbe28_150e, 0xbded_d926, 0xbd8b_882f, 0x3cac_38ba, 0x3d97_340b,
    0x3e01_acf3, 0x3e37_bfe2, 0x3e6d_d2d0, 0x3e91_f2df, 0x3eac_fc56, 0x3ec4_2906, 0x3ee1_e6a2,
    0x3eff_a43f,
];

const LIN_CONG_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbe91_a6cf),
    (1023, 0xbe9c_d079),
    (2047, 0xbd56_db00),
    (4095, 0xbe5a_85d1),
];

#[rustfmt::skip]
const LIN_CONG_L_FAST_BLOCK: [u32; 64] = [
    0xbf00_0000, 0xbe42_8f5c, 0x3e1a_9fbf, 0x3f06_ae7e, 0x3f70_4f4d, 0xbf1b_7fce, 0xbe9d_c740,
    0x3cac_38ba, 0x3ec4_2906, 0x3f48_0c53, 0xbf47_c97b, 0xbeff_3624, 0xbe40_d345, 0x3e1c_883f,
    0x3f07_34d4, 0x3f70_e313, 0xbf1a_dd42, 0xbe9c_61a5, 0x3cc4_ce98, 0x3ec5_d9ba, 0x3f48_fa4f,
    0xbf46_c3b2, 0xbefc_f636, 0xbe3b_e03a, 0x3e21_f9fe, 0x3f08_b41c, 0x3f72_88af, 0xbf19_0d7d,
    0xbe98_655a, 0x3d05_79dd, 0x3eca_ac47, 0x3f4b_a151, 0xbf43_d8cb, 0xbef6_8b06, 0xbe2d_c135,
    0x3e31_8284, 0x3f0c_f9a7, 0x3f77_3b94, 0xbf13_e24d, 0xbe8d_0657, 0x3d69_8ac7, 0x3ed8_6e9b,
    0x3f53_3298, 0xbf3b_85c9, 0xbee4_3acf, 0xbe05_7722, 0x3e5d_d3ff, 0x3f19_29a9, 0xbf7b_5c6a,
    0xbf38_564b, 0xbedd_38ba, 0xbdec_1851, 0x3e6e_c9ab, 0x3f1d_d39f, 0xbf76_3b0e, 0xbf32_b19a,
    0xbed0_ce68, 0xbdb5_77b3, 0x3e86_6a9b, 0x3f26_16cb, 0xbf6d_245d, 0xbf28_b23e, 0xbeba_cfd0,
    0xbd29_622e,
];

const LIN_CONG_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbe85_3142),
    (1023, 0xbf72_f11d),
    (2047, 0xbe65_2650),
    (4095, 0x3f50_db54),
];

#[rustfmt::skip]
const LIN_CONG_C_SLOW_BLOCK: [u32; 64] = [
    0x3e95_c92f, 0x3eb3_65ed, 0x3edb_3800, 0x3f07_d097, 0x3f29_81a1, 0x3f53_e000, 0x3e80_0000,
    0x3e2e_ffa8, 0x3d5d_f749, 0xbdaa_8d71, 0xbe66_0f2a, 0xbeb4_ce5e, 0xbee7_5852, 0xbf00_0000,
    0xbf02_ef64, 0xbef9_e547, 0xbee0_fa6a, 0xbec0_051f, 0xbe9b_ec50, 0xbe73_2dd3, 0xbe42_8f5c,
    0xbe11_a4ae, 0xbdc0_2dc8, 0xbd37_79bc, 0x3ba1_8801, 0x3d62_c2e6, 0x3dda_2d1d, 0x3e1a_9fbf,
    0x3e50_6eb2, 0x3e83_7888, 0x3e9f_1795, 0x3ebb_18a9, 0x3ed7_7fec, 0x3ef4_5187, 0x3f06_ae7e,
    0x3f1a_211f, 0x3f33_e517, 0x3f4f_3922, 0x3f67_5bfa, 0x3f77_8c5b, 0x3f7b_0901, 0x3f70_4f4d,
    0x3f4c_f921, 0x3f13_5279, 0x3e98_d604, 0x3ba9_33f4, 0xbe89_cf54, 0xbefa_6712, 0xbf1b_7fce,
    0xbf29_877c, 0xbf27_f3c9, 0xbf1b_17bb, 0xbf07_465a, 0xbee1_a55a, 0xbe9d_c740, 0xbe86_2ae0,
    0xbe5c_7f8a, 0xbe2c_0493, 0xbdf5_bb1d, 0xbd92_0661, 0xbcb3_714f, 0x3cac_38ba, 0x3d92_f171,
    0x3dfc_2f1a,
];

const LIN_CONG_C_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbf2e_6899),
    (1023, 0xbf1b_c7e9),
    (2047, 0xbec1_cda5),
    (4095, 0xbf06_2fac),
];

#[rustfmt::skip]
const LIN_CONG_C_FAST_BLOCK: [u32; 64] = [
    0x3e80_0000, 0xbf00_0000, 0xbe42_8f5c, 0x3e1a_9fbf, 0x3f06_ae7e, 0x3f70_4f4d, 0xbf1b_7fce,
    0xbe9d_c740, 0x3cac_38ba, 0x3ec4_2906, 0x3f48_0c53, 0xbf47_c97b, 0xbeff_3624, 0xbe40_d345,
    0x3e1c_883f, 0x3f07_34d4, 0x3f70_e313, 0xbf1a_dd42, 0xbe9c_61a5, 0x3cc4_ce98, 0x3ec5_d9ba,
    0x3f48_fa4f, 0xbf46_c3b2, 0xbefc_f636, 0xbe3b_e03a, 0x3e21_f9fe, 0x3f08_b41c, 0x3f72_88af,
    0xbf19_0d7d, 0xbe98_655a, 0x3d05_79dd, 0x3eca_ac47, 0x3f4b_a151, 0xbf43_d8cb, 0xbef6_8b06,
    0xbe2d_c135, 0x3e31_8284, 0x3f0c_f9a7, 0x3f77_3b94, 0xbf13_e24d, 0xbe8d_0657, 0x3d69_8ac7,
    0x3ed8_6e9b, 0x3f53_3298, 0xbf3b_85c9, 0xbee4_3acf, 0xbe05_7722, 0x3e5d_d3ff, 0x3f19_29a9,
    0xbf7b_5c6a, 0xbf38_564b, 0xbedd_38ba, 0xbdec_1851, 0x3e6e_c9ab, 0x3f1d_d39f, 0xbf76_3b0e,
    0xbf32_b19a, 0xbed0_ce68, 0xbdb5_77b3, 0x3e86_6a9b, 0x3f26_16cb, 0xbf6d_245d, 0xbf28_b23e,
    0xbeba_cfd0,
];

const LIN_CONG_C_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf10_52e6),
    (1023, 0x3f20_d106),
    (2047, 0xbf07_dc7d),
    (4095, 0x3ed4_2d08),
];

#[rustfmt::skip]
const LIN_CONG_L_WIDE: [u32; 64] = [
    0x3e30_0000, 0x3dc0_0001, 0x3c80_0007, 0xbd7f_fffb, 0xbe0f_ffff, 0xbe5f_fffe, 0xbe92_4924,
    0xbe33_3328, 0xbd83_a811, 0x3d3e_2c5d, 0x3e20_ea37, 0x3e89_24ac, 0x3ec1_d43b, 0x3ef2_6ab8,
    0x3f07_9f48, 0x3f16_0934, 0x3f24_7320, 0x3f32_dd0c, 0x3f41_46f8, 0x3f4f_b0e4, 0x3f5c_0bae,
    0x3f44_f9c8, 0x3f2d_e7e2, 0x3f16_d5fc, 0x3eff_882d, 0x3ed1_6461, 0x3ea3_4095, 0x3e77_685d,
    0x3e19_cf24, 0x3d70_d7aa, 0xbd05_8d3b, 0xbdfd_f910, 0xbe5c_95c1, 0xbe9d_177d, 0xbec5_3497,
    0xbe73_243e, 0xbdb7_be9d, 0x3d6d_9682, 0x3e52_aa90, 0x3eb4_f7bf, 0x3f00_4d1b, 0x3f20_b74f,
    0x3f04_1dc2, 0x3ecf_0869, 0x3e95_d54e, 0x3e39_4466, 0x3d8d_bc60, 0xbd2e_2018, 0xbe0d_967f,
    0xbd74_4ec2, 0x3c9b_78f1, 0x3dc7_e3d9, 0x3e34_74bb, 0x3e82_7bc5, 0x3ecd_3e61, 0x3ea2_fa61,
    0x3e71_6cc2, 0x3e1c_e4c2, 0x3d90_b984, 0xbc42_b3df, 0xbdc1_667c, 0xbe29_27d1, 0xbdf5_5bdb,
    0xbd98_6813,
];

#[rustfmt::skip]
const LIN_CONG_C_WIDE: [u32; 64] = [
    0x3e95_c92f, 0x3eb3_65ed, 0x3edb_3800, 0x3f07_d097, 0x3f29_81a1, 0x3f53_e000, 0x3e80_0000,
    0x3e41_23e9, 0x3dc0_60b3, 0xbc8b_4ce8, 0xbe02_38e7, 0xbe60_cc75, 0xbe8d_d99b, 0xbe92_4924,
    0xbe76_458c, 0xbe17_56c1, 0xbcbd_52b6, 0x3df1_b67d, 0x3e85_2ac9, 0x3ec6_701a, 0x3ef2_6ab8,
    0x3f0f_09c3, 0x3f24_c4cb, 0x3f38_e5b9, 0x3f49_ebd4, 0x3f56_5662, 0x3f5c_a4ab, 0x3f5c_0bae,
    0x3f52_ab99, 0x3f40_c7e4, 0x3f28_bf69, 0x3f0c_f100, 0x3edf_7707, 0x3ea6_fb94, 0x3e77_685d,
    0x3e0b_0a07, 0x3c24_887f, 0xbdf6_4e27, 0xbe75_8fbd, 0xbeaa_9e35, 0xbec5_3c07, 0xbec5_3497,
    0xbe9d_f61e, 0xbe1f_d84e, 0x3d24_ff7a, 0x3e7e_c981, 0x3ee0_18ee, 0x3f13_398c, 0x3f20_b74f,
    0x3f19_4f72, 0x3efd_8bf1, 0x3eb0_e728, 0x3e36_c3e3, 0x3cba_daef, 0xbe0d_967f, 0xbdf3_cd44,
    0xbd31_0c7a, 0x3d88_b9a6, 0x3e41_01f7, 0x3e99_5a8b, 0x3ec1_3f7d, 0x3ecd_3e61, 0x3ebd_0083,
    0x3e94_9557,
];
#[rustfmt::skip]
const GBMAN_L_SLOW_BLOCK: [u32; 64] = [
    0x3ffb_ffff, 0x3feb_3333, 0x3fda_6666, 0x3fc9_999a, 0x3fb8_cccd, 0x3fa8_0000, 0x3f99_999a,
    0x3f85_1112, 0x3f61_1113, 0x3f38_0002, 0x3f0e_eef1, 0x3ecb_bbc1, 0x3e73_333e, 0x3dcc_cce0,
    0x3d91_1123, 0x3d2a_aace, 0x3c4c_cd53, 0xbc88_8849, 0xbd3b_bb9d, 0xbd99_998b, 0xbdcc_ccc0,
    0x3d77_7783, 0x3e62_2221, 0x3ec3_3331, 0x3f0a_aaa9, 0x3f33_bbb9, 0x3f5c_ccc9, 0x3f7f_fffc,
    0x3f94_8886, 0x3fa9_110f, 0x3fbd_9997, 0x3fd2_221f, 0x3fe6_aaa8, 0x3ffb_3330, 0x4006_6665,
    0x4006_6665, 0x4006_6665, 0x4006_6665, 0x4006_6666, 0x4006_6666, 0x4006_6666, 0x4006_6666,
    0x3ff8_4444, 0x3fe3_bbbc, 0x3fcf_3334, 0x3fba_aaac, 0x3fa6_2224, 0x3f91_999c, 0x3f80_0002,
    0x3f56_eef3, 0x3f2d_dde3, 0x3f04_ccd2, 0x3eb7_7782, 0x3e4a_aac1, 0xbdcc_cca0, 0xbd91_10f6,
    0xbd2a_aa99, 0xbc4c_cd13, 0x3c88_881e, 0x3d3b_bb63, 0x3d99_995b, 0x3dcc_cc80, 0x3e85_5540,
    0x3ed7_7761,
];

const GBMAN_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3ff9_9c8b),
    (1023, 0x3fe7_a35a),
    (2047, 0x4011_2191),
    (4095, 0xbe75_d6aa),
];

#[rustfmt::skip]
const GBMAN_L_FAST_BLOCK: [u32; 64] = [
    0x3f99_999a, 0x3dcc_cce0, 0xbdcc_ccc0, 0x3f7f_fffc, 0x4006_6665, 0x4006_6666, 0x3f80_0002,
    0xbdcc_cca0, 0x3dcc_cc80, 0x3f99_9992, 0x4006_6665, 0x3ff3_3338, 0x3f4c_ccdc, 0xbdcc_cca0,
    0x3e99_9970, 0x3fb3_3326, 0x4006_6665, 0x3fd9_99a4, 0x3f19_99b4, 0xbdcc_cca0, 0x3eff_ffc0,
    0x3fcc_ccba, 0x4006_6665, 0x3fc0_0010, 0x3ecc_cd18, 0xbdcc_cca0, 0x3f33_3308, 0x3fe6_664e,
    0x4006_6665, 0x3fa6_667c, 0x3e4c_cd90, 0xbdcc_cca0, 0x3f66_6630, 0x3fff_ffe2, 0x4006_6665,
    0x3f8c_cce8, 0x3670_0000, 0xbdcc_cca0, 0x3f8c_ccac, 0x400c_ccbb, 0x4006_6665, 0x3f66_66a8,
    0xbe4c_cbb0, 0x3e99_9888, 0x3fbf_ff98, 0x400c_ccbb, 0x3fd9_99de, 0x3f00_00d0, 0xbe4c_cbb0,
    0x3f33_321c, 0x3ff3_3284, 0x400c_ccbb, 0x3fa6_66f2, 0x3dcc_d7c0, 0xbe4c_cbb0, 0x3f8c_cbfa,
    0x4013_32b8, 0x400c_ccbb, 0x3f66_680c, 0xbe99_95c0, 0x3ecc_c5a8, 0x3fd9_96da, 0x4013_32b8,
    0x3fcc_ce96,
];

const GBMAN_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3fd3_c272),
    (1023, 0x404b_5bf5),
    (2047, 0x3f32_0bec),
    (4095, 0x4069_1b69),
];

#[rustfmt::skip]
const QUAD_C_SLOW_BLOCK: [u32; 64] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x3bdf_38e3, 0x3cb9_1c71, 0x3d25_6000, 0x3d59_c71c, 0x3d5d_2e39, 0x3d13_0000, 0x0000_0000,
    0xbdb4_a6aa, 0xbe67_6fff, 0xbec6_4500, 0xbf0b_b555, 0xbf2c_e180, 0xbf3f_f400, 0xbf40_0000,
    0xbf24_69e6, 0xbedc_fa64, 0xbe27_e920, 0x3dea_f389, 0x3eb7_992b, 0x3f05_8fc0, 0x3f10_0000,
    0x3ee8_30a0, 0x3e50_994b, 0xbdff_b2ee, 0xbef0_bc49, 0xbf45_3167, 0xbf76_908c, 0xbf7f_0000,
    0xbf56_1b8e, 0xbf00_eb97, 0xbd8e_9f94, 0x3ec9_8f0a, 0x3f50_82fa, 0x3f8f_7d4b, 0x3f9e_8080,
    0x3f97_3ee8, 0x3f78_4223, 0x3f28_9113, 0x3e9b_5b7d, 0xbd22_4c00, 0xbea1_469d, 0xbee8_eb0c,
    0xbf00_a705, 0xbef0_ee03, 0xbec6_0ccd, 0xbe8e_ec41, 0xbe33_9c65, 0xbdb4_7e15, 0xbe0a_62d6,
    0xbe6d_4723, 0xbeb5_9fa4, 0xbef6_4f4e, 0xbf16_6e1e, 0xbf26_b80e, 0xbf27_7337, 0xbf12_7356,
    0xbed2_78a5,
];

const QUAD_C_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbed4_cf57),
    (1023, 0xbefb_6c9e),
    (2047, 0xbf0a_520f),
    (4095, 0xbeee_4de3),
];

#[rustfmt::skip]
const QUAD_C_FAST_BLOCK: [u32; 64] = [
    0x0000_0000, 0x0000_0000, 0xbf40_0000, 0x3f10_0000, 0xbf7f_0000, 0x3f9e_8080, 0xbee8_eb0c,
    0xbdb4_7e15, 0xbf27_7337, 0x3ea9_f57a, 0xbf78_c53b, 0x3f95_41fb, 0xbf0e_6c79, 0x3ded_47e3,
    0xbf5a_3944, 0x3f54_3ed8, 0xbf64_46b3, 0x3f6f_d4b5, 0xbf4f_25da, 0x3f36_c416, 0xbf74_48b0,
    0x3f8e_b1ab, 0xbf1f_3d03, 0x3e84_93b5, 0xbf71_1fb1, 0x3f8a_1e2f, 0xbf2a_2a24, 0x3eb6_8c45,
    0xbf7a_bb33, 0x3f98_26ad, 0xbf06_958c, 0x3d55_6711, 0xbf4c_a48c, 0x3f30_3b36, 0xbf76_e9be,
    0x3f92_87e6, 0xbf15_92a1, 0x3e33_da92, 0xbf65_10ee, 0x3f72_083a, 0xbf4d_34ac, 0x3f31_b213,
    0xbf76_5a59, 0x3f91_b60e, 0xbf17_ad27, 0x3e46_2b80, 0xbf67_f46b, 0x3f7a_1f6d, 0xbf45_be08,
    0x3f1e_7c0d, 0xbf7c_5eb1, 0x3f9a_94a0, 0xbeff_994a, 0xbacd_5710, 0xbf3f_992b, 0x3f0e_ff16,
    0xbf7f_1f1b, 0x3f9e_af0c, 0xbee7_d7dc, 0xbdbc_b206, 0xbf26_3d67, 0x3ea4_6231, 0xbf77_cd9b,
    0x3f93_d601,
];

const QUAD_C_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3e9d_6a9d),
    (1023, 0x3ecf_d9e4),
    (2047, 0x3f5d_1075),
    (4095, 0x3f32_c72f),
];

#[rustfmt::skip]
const QUAD_C_RESEED: [u32; 64] = [
    0xbe53_d338, 0x3b9e_4986, 0x3e40_c4ba, 0x3e9d_84c5, 0xbf78_c53b, 0xbf60_b6e5, 0xbf27_f258,
    0xbeb8_5037, 0xbd30_8b93, 0x3e76_ed1e, 0x3ee0_e6bf, 0x3f00_0000, 0x3ecf_81bf, 0x3e2a_f7a8,
    0xbe1c_1865, 0xbefb_10c3, 0xbf48_c628, 0xbf78_bca7, 0xbf80_0000, 0xbf55_ea9d, 0xbeff_bf1d,
    0xbd81_7400, 0x3ecd_b1c5, 0x3f52_e6f0, 0x3f90_d600, 0x3fa0_0000, 0x3f98_fba9, 0x3f7c_4c96,
    0x3f2d_26b8, 0x3ea5_5957, 0xbc9f_5427, 0xbe97_5080, 0xbee0_0000, 0xbefb_3186, 0xbeef_558a,
    0xbec9_d338, 0xbe98_11bc, 0xbe4e_f087, 0xbdf8_0000, 0xbe26_167b, 0xbe7c_3f53, 0xbeb4_f4e0,
    0xbeed_427d, 0xbf0e_600f, 0xbf1c_92af, 0xbf1d_3f00, 0xbf0a_f26a, 0xbecd_e87d, 0xbe65_44c6,
    0xbd21_5f93, 0x3df5_85b3, 0x3e63_d8a2, 0x3e77_5556, 0x3e19_1cf3, 0xbd43_bdb4, 0xbe9a_ab5e,
    0xbf10_a7f8, 0xbf49_d474, 0xbf6c_7cde, 0xbf6e_e5f9, 0xbf47_1696, 0xbef1_e2cb, 0xbda7_1ce2,
    0x3ea9_4b17,
];
#[rustfmt::skip]
const CUSP_N_SLOW_BLOCK: [u32; 64] = [
    0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3d4c_ccd0,
    0x3d4c_ccd0, 0x3d4c_ccd0, 0x3d4c_ccd0, 0x3d4c_ccd0, 0x3d4c_ccd0, 0x3d4c_ccd0, 0x3f13_3cd6,
    0x3f13_3cd6, 0x3f13_3cd6, 0x3f13_3cd6, 0x3f13_3cd6, 0x3f13_3cd6, 0x3f13_3cd6, 0xbee1_c1a5,
    0xbee1_c1a5, 0xbee1_c1a5, 0xbee1_c1a5, 0xbee1_c1a5, 0xbee1_c1a5, 0xbee1_c1a5, 0xbe85_f6e8,
    0xbe85_f6e8, 0xbe85_f6e8, 0xbe85_f6e8, 0xbe85_f6e8, 0xbe85_f6e8, 0xbe85_f6e8, 0x3ce6_582d,
    0x3ce6_582d, 0x3ce6_582d, 0x3ce6_582d, 0x3ce6_582d, 0x3ce6_582d, 0x3ce6_582d, 0x3f2e_7026,
    0x3f2e_7026, 0x3f2e_7026, 0x3f2e_7026, 0x3f2e_7026, 0x3f2e_7026, 0x3f2e_7026, 0xbf11_820e,
    0xbf11_820e, 0xbf11_820e, 0xbf11_820e, 0xbf11_820e, 0xbf11_820e, 0xbedd_690d, 0xbedd_690d,
    0xbedd_690d, 0xbedd_690d, 0xbedd_690d, 0xbedd_690d, 0xbedd_690d, 0xbe7f_6ed7, 0xbe7f_6ed7,
    0xbe7f_6ed7,
];

const CUSP_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3e90_a0c7),
    (1023, 0x3e95_4acb),
    (2047, 0xbe88_4f7c),
    (4095, 0x3df6_eb5f),
];

#[rustfmt::skip]
const CUSP_N_FAST_BLOCK: [u32; 64] = [
    0x3d4c_ccd0, 0x3f13_3cd6, 0xbee1_c1a5, 0xbe85_f6e8, 0x3ce6_582d, 0x3f2e_7026, 0xbf11_820e,
    0xbedd_690d, 0xbe7f_6ed7, 0x3d51_1ca3, 0x3f12_1943, 0xbede_e5bd, 0xbe81_dc74, 0x3d30_9cdb,
    0x3f1a_ffcf, 0xbef4_f3bb, 0xbea0_ddea, 0xbd85_21a9, 0x3f03_fc8b, 0xbeba_80ca, 0xbe16_40b3,
    0x3e8b_5d31, 0x3c0e_f525, 0x3f52_90bb, 0xbf39_2165, 0xbf1d_a165, 0xbefb_5990, 0xbea9_98ec,
    0xbdbf_8947, 0x3ed6_8082, 0xbe6b_5089, 0x3db6_a8f9, 0x3edd_7a24, 0xbe7f_a035, 0x3d4f_a51b,
    0x3f12_7c1e, 0xbedf_de34, 0xbe83_41cf, 0x3d1b_9a89, 0x3f21_325a, 0xbf01_f7e3, 0xbeb5_2463,
    0xbe05_40c4, 0x3ea1_13aa, 0xbd86_8df0, 0x3f03_5354, 0xbeb8_c078, 0xbe10_ba04, 0x3e92_47e0,
    0xbc7f_308d, 0x3f43_4bdb, 0xbf28_d5d6, 0xbf0b_01e3, 0xbecc_d726, 0xbe4e_a07d, 0x3e16_076d,
    0x3e8b_a43d, 0x3bfd_96a5, 0x3f55_35fc, 0xbf3b_e499, 0xbf20_b483, 0xbf01_611d, 0xbeb3_91e2,
    0xbe00_383c,
];

const CUSP_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf4f_742f),
    (1023, 0xbd3c_b18e),
    (2047, 0x3e23_ac4d),
    (4095, 0xbd85_ec15),
];

#[rustfmt::skip]
const QUAD_N_SLOW_BLOCK: [u32; 64] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0xbf40_0000,
    0xbf40_0000, 0xbf40_0000, 0xbf40_0000, 0xbf40_0000, 0xbf40_0000, 0xbf40_0000, 0x3f10_0000,
    0x3f10_0000, 0x3f10_0000, 0x3f10_0000, 0x3f10_0000, 0x3f10_0000, 0x3f10_0000, 0xbf7f_0000,
    0xbf7f_0000, 0xbf7f_0000, 0xbf7f_0000, 0xbf7f_0000, 0xbf7f_0000, 0xbf7f_0000, 0x3f9e_8080,
    0x3f9e_8080, 0x3f9e_8080, 0x3f9e_8080, 0x3f9e_8080, 0x3f9e_8080, 0x3f9e_8080, 0xbee8_eb0c,
    0xbee8_eb0c, 0xbee8_eb0c, 0xbee8_eb0c, 0xbee8_eb0c, 0xbee8_eb0c, 0xbee8_eb0c, 0xbdb4_7e15,
    0xbdb4_7e15, 0xbdb4_7e15, 0xbdb4_7e15, 0xbdb4_7e15, 0xbdb4_7e15, 0xbdb4_7e15, 0xbf27_7337,
    0xbf27_7337, 0xbf27_7337, 0xbf27_7337, 0xbf27_7337, 0xbf27_7337, 0x3ea9_f57a, 0x3ea9_f57a,
    0x3ea9_f57a, 0x3ea9_f57a, 0x3ea9_f57a, 0x3ea9_f57a, 0x3ea9_f57a, 0xbf78_c53b, 0xbf78_c53b,
    0xbf78_c53b,
];

const QUAD_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3f97_da5a),
    (1023, 0xbef6_0336),
    (2047, 0x3f34_d172),
    (4095, 0xbf46_4a7a),
];

#[rustfmt::skip]
const QUAD_N_FAST_BLOCK: [u32; 64] = [
    0xbf40_0000, 0x3f10_0000, 0xbf7f_0000, 0x3f9e_8080, 0xbee8_eb0c, 0xbdb4_7e15, 0xbf27_7337,
    0x3ea9_f57a, 0xbf78_c53b, 0x3f95_41fb, 0xbf0e_6c79, 0x3ded_47e3, 0xbf5a_3944, 0x3f54_3ed8,
    0xbf64_46b3, 0x3f6f_d4b5, 0xbf4f_25da, 0x3f36_c416, 0xbf74_48b0, 0x3f8e_b1ab, 0xbf1f_3d03,
    0x3e84_93b5, 0xbf71_1fb1, 0x3f8a_1e2f, 0xbf2a_2a24, 0x3eb6_8c45, 0xbf7a_bb33, 0x3f98_26ad,
    0xbf06_958c, 0x3d55_6711, 0xbf4c_a48c, 0x3f30_3b36, 0xbf76_e9be, 0x3f92_87e6, 0xbf15_92a1,
    0x3e33_da92, 0xbf65_10ee, 0x3f72_083a, 0xbf4d_34ac, 0x3f31_b213, 0xbf76_5a59, 0x3f91_b60e,
    0xbf17_ad27, 0x3e46_2b80, 0xbf67_f46b, 0x3f7a_1f6d, 0xbf45_be08, 0x3f1e_7c0d, 0xbf7c_5eb1,
    0x3f9a_94a0, 0xbeff_994a, 0xbacd_5710, 0xbf3f_992b, 0x3f0e_ff16, 0xbf7f_1f1b, 0x3f9e_af0c,
    0xbee7_d7dc, 0xbdbc_b206, 0xbf26_3d67, 0x3ea4_6231, 0xbf77_cd9b, 0x3f93_d601, 0xbf12_2e22,
    0x3e16_9b21,
];

const QUAD_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3f91_f085),
    (1023, 0x3f9c_9d32),
    (2047, 0x3f5e_f9b6),
    (4095, 0x3f91_171b),
];

#[rustfmt::skip]
const LIN_CONG_N_SLOW_BLOCK: [u32; 64] = [
    0xbf00_0000, 0xbf00_0000, 0xbf00_0000, 0xbf00_0000, 0xbf00_0000, 0xbf00_0000, 0xbe42_8f5c,
    0xbe42_8f5c, 0xbe42_8f5c, 0xbe42_8f5c, 0xbe42_8f5c, 0xbe42_8f5c, 0xbe42_8f5c, 0x3e1a_9fbf,
    0x3e1a_9fbf, 0x3e1a_9fbf, 0x3e1a_9fbf, 0x3e1a_9fbf, 0x3e1a_9fbf, 0x3e1a_9fbf, 0x3f06_ae7e,
    0x3f06_ae7e, 0x3f06_ae7e, 0x3f06_ae7e, 0x3f06_ae7e, 0x3f06_ae7e, 0x3f06_ae7e, 0x3f70_4f4d,
    0x3f70_4f4d, 0x3f70_4f4d, 0x3f70_4f4d, 0x3f70_4f4d, 0x3f70_4f4d, 0x3f70_4f4d, 0xbf1b_7fce,
    0xbf1b_7fce, 0xbf1b_7fce, 0xbf1b_7fce, 0xbf1b_7fce, 0xbf1b_7fce, 0xbf1b_7fce, 0xbe9d_c740,
    0xbe9d_c740, 0xbe9d_c740, 0xbe9d_c740, 0xbe9d_c740, 0xbe9d_c740, 0xbe9d_c740, 0x3cac_38ba,
    0x3cac_38ba, 0x3cac_38ba, 0x3cac_38ba, 0x3cac_38ba, 0x3cac_38ba, 0x3ec4_2906, 0x3ec4_2906,
    0x3ec4_2906, 0x3ec4_2906, 0x3ec4_2906, 0x3ec4_2906, 0x3ec4_2906, 0x3f48_0c53, 0x3f48_0c53,
    0x3f48_0c53,
];

const LIN_CONG_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbe1d_a350),
    (1023, 0xbda2_fd3f),
    (2047, 0x3db2_9c4e),
    (4095, 0x3ca1_41bb),
];

#[rustfmt::skip]
const LIN_CONG_N_FAST_BLOCK: [u32; 64] = [
    0xbe42_8f5c, 0x3e1a_9fbf, 0x3f06_ae7e, 0x3f70_4f4d, 0xbf1b_7fce, 0xbe9d_c740, 0x3cac_38ba,
    0x3ec4_2906, 0x3f48_0c53, 0xbf47_c97b, 0xbeff_3624, 0xbe40_d345, 0x3e1c_883f, 0x3f07_34d4,
    0x3f70_e313, 0xbf1a_dd42, 0xbe9c_61a5, 0x3cc4_ce98, 0x3ec5_d9ba, 0x3f48_fa4f, 0xbf46_c3b2,
    0xbefc_f636, 0xbe3b_e03a, 0x3e21_f9fe, 0x3f08_b41c, 0x3f72_88af, 0xbf19_0d7d, 0xbe98_655a,
    0x3d05_79dd, 0x3eca_ac47, 0x3f4b_a151, 0xbf43_d8cb, 0xbef6_8b06, 0xbe2d_c135, 0x3e31_8284,
    0x3f0c_f9a7, 0x3f77_3b94, 0xbf13_e24d, 0xbe8d_0657, 0x3d69_8ac7, 0x3ed8_6e9b, 0x3f53_3298,
    0xbf3b_85c9, 0xbee4_3acf, 0xbe05_7722, 0x3e5d_d3ff, 0x3f19_29a9, 0xbf7b_5c6a, 0xbf38_564b,
    0xbedd_38ba, 0xbdec_1851, 0x3e6e_c9ab, 0x3f1d_d39f, 0xbf76_3b0e, 0xbf32_b19a, 0xbed0_ce68,
    0xbdb5_77b3, 0x3e86_6a9b, 0x3f26_16cb, 0xbf6d_245d, 0xbf28_b23e, 0xbeba_cfd0, 0xbd29_622e,
    0x3ea1_079f,
];

const LIN_CONG_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3d97_3bc1),
    (1023, 0xbf2f_1377),
    (2047, 0x3de9_269a),
    (4095, 0xbf3e_18fa),
];

#[rustfmt::skip]
const GBMAN_N_SLOW_BLOCK: [u32; 64] = [
    0x3f99_999a, 0x3f99_999a, 0x3f99_999a, 0x3f99_999a, 0x3f99_999a, 0x3f99_999a, 0x3dcc_cce0,
    0x3dcc_cce0, 0x3dcc_cce0, 0x3dcc_cce0, 0x3dcc_cce0, 0x3dcc_cce0, 0x3dcc_cce0, 0xbdcc_ccc0,
    0xbdcc_ccc0, 0xbdcc_ccc0, 0xbdcc_ccc0, 0xbdcc_ccc0, 0xbdcc_ccc0, 0xbdcc_ccc0, 0x3f7f_fffc,
    0x3f7f_fffc, 0x3f7f_fffc, 0x3f7f_fffc, 0x3f7f_fffc, 0x3f7f_fffc, 0x3f7f_fffc, 0x4006_6665,
    0x4006_6665, 0x4006_6665, 0x4006_6665, 0x4006_6665, 0x4006_6665, 0x4006_6665, 0x4006_6666,
    0x4006_6666, 0x4006_6666, 0x4006_6666, 0x4006_6666, 0x4006_6666, 0x4006_6666, 0x3f80_0002,
    0x3f80_0002, 0x3f80_0002, 0x3f80_0002, 0x3f80_0002, 0x3f80_0002, 0x3f80_0002, 0xbdcc_cca0,
    0xbdcc_cca0, 0xbdcc_cca0, 0xbdcc_cca0, 0xbdcc_cca0, 0xbdcc_cca0, 0x3dcc_cc80, 0x3dcc_cc80,
    0x3dcc_cc80, 0x3dcc_cc80, 0x3dcc_cc80, 0x3dcc_cc80, 0x3dcc_cc80, 0x3f99_9992, 0x3f99_9992,
    0x3f99_9992,
];

const GBMAN_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3fd9_a1fc),
    (1023, 0x3f55_4f70),
    (2047, 0x3fae_bfee),
    (4095, 0x3fcb_7cfe),
];

#[rustfmt::skip]
const GBMAN_N_FAST_BLOCK: [u32; 64] = [
    0x3dcc_cce0, 0xbdcc_ccc0, 0x3f7f_fffc, 0x4006_6665, 0x4006_6666, 0x3f80_0002, 0xbdcc_cca0,
    0x3dcc_cc80, 0x3f99_9992, 0x4006_6665, 0x3ff3_3338, 0x3f4c_ccdc, 0xbdcc_cca0, 0x3e99_9970,
    0x3fb3_3326, 0x4006_6665, 0x3fd9_99a4, 0x3f19_99b4, 0xbdcc_cca0, 0x3eff_ffc0, 0x3fcc_ccba,
    0x4006_6665, 0x3fc0_0010, 0x3ecc_cd18, 0xbdcc_cca0, 0x3f33_3308, 0x3fe6_664e, 0x4006_6665,
    0x3fa6_667c, 0x3e4c_cd90, 0xbdcc_cca0, 0x3f66_6630, 0x3fff_ffe2, 0x4006_6665, 0x3f8c_cce8,
    0x3670_0000, 0xbdcc_cca0, 0x3f8c_ccac, 0x400c_ccbb, 0x4006_6665, 0x3f66_66a8, 0xbe4c_cbb0,
    0x3e99_9888, 0x3fbf_ff98, 0x400c_ccbb, 0x3fd9_99de, 0x3f00_00d0, 0xbe4c_cbb0, 0x3f33_321c,
    0x3ff3_3284, 0x400c_ccbb, 0x3fa6_66f2, 0x3dcc_d7c0, 0xbe4c_cbb0, 0x3f8c_cbfa, 0x4013_32b8,
    0x400c_ccbb, 0x3f66_680c, 0xbe99_95c0, 0x3ecc_c5a8, 0x3fd9_96da, 0x4013_32b8, 0x3fcc_ce96,
    0x3e99_a498,
];

const GBMAN_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x4025_d1b4),
    (1023, 0x3f6d_9f44),
    (2047, 0x4070_edd4),
    (4095, 0x3fc2_2f7e),
];

#[rustfmt::skip]
const STANDARD_N_SLOW_BLOCK: [u32; 64] = [
    0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf30_3071,
    0xbf30_3071, 0xbf30_3071, 0xbf30_3071, 0xbf30_3071, 0xbf30_3071, 0xbf30_3071, 0xbe8a_f246,
    0xbe8a_f246, 0xbe8a_f246, 0xbe8a_f246, 0xbe8a_f246, 0xbe8a_f246, 0xbe8a_f246, 0x3ec5_3366,
    0x3ec5_3366, 0x3ec5_3366, 0x3ec5_3366, 0x3ec5_3366, 0x3ec5_3366, 0x3ec5_3366, 0x3f3e_6ed0,
    0x3f3e_6ed0, 0x3f3e_6ed0, 0x3f3e_6ed0, 0x3f3e_6ed0, 0x3f3e_6ed0, 0x3f3e_6ed0, 0x3f5f_8c35,
    0x3f5f_8c35, 0x3f5f_8c35, 0x3f5f_8c35, 0x3f5f_8c35, 0x3f5f_8c35, 0x3f5f_8c35, 0x3f61_0fae,
    0x3f61_0fae, 0x3f61_0fae, 0x3f61_0fae, 0x3f61_0fae, 0x3f61_0fae, 0x3f61_0fae, 0x3f44_5fc2,
    0x3f44_5fc2, 0x3f44_5fc2, 0x3f44_5fc2, 0x3f44_5fc2, 0x3f44_5fc2, 0x3ee2_7b6a, 0x3ee2_7b6a,
    0x3ee2_7b6a, 0x3ee2_7b6a, 0x3ee2_7b6a, 0x3ee2_7b6a, 0x3ee2_7b6a, 0xbe48_2f7f, 0xbe48_2f7f,
    0xbe48_2f7f,
];

const STANDARD_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbec0_22cf),
    (1023, 0x3e75_6515),
    (2047, 0xbf41_4aa0),
    (4095, 0xbf24_0b43),
];

#[rustfmt::skip]
const STANDARD_N_FAST_BLOCK: [u32; 64] = [
    0xbf30_3071, 0xbe8a_f246, 0x3ec5_3366, 0x3f3e_6ed0, 0x3f5f_8c35, 0x3f61_0fae, 0x3f44_5fc2,
    0x3ee2_7b6a, 0xbe48_2f7f, 0xbf26_5ff4, 0xbf52_160a, 0xbf52_464b, 0xbf27_198b, 0xbe4e_8ef0,
    0x3ee0_252a, 0x3f43_d95c, 0x3f60_ca4c, 0x3f5f_4772, 0x3f3d_eb55, 0x3ec2_f944, 0xbe8d_92ad,
    0xbf30_d897, 0xbf57_a004, 0xbf57_a935, 0xbf30_fc3e, 0xbe8e_3732, 0x3ec2_618a, 0x3f3d_c1be,
    0x3f5f_2382, 0x3f60_8af4, 0x3f43_43e3, 0x3edd_718f, 0xbe55_ef98, 0xbf27_f6c7, 0xbf52_8fc5,
    0xbf52_0a08, 0xbf25_f450, 0xbe44_400a, 0x3ee3_f62c, 0x3f44_bd62, 0x3f61_535b, 0x3f5f_f4d3,
    0x3f3f_5ce9, 0x3ec9_66ac, 0xbe85_d67e, 0xbf2e_e43d, 0xbf56_801c, 0xbf56_6168, 0xbf2e_6d57,
    0xbe83_b773, 0x3ecb_4264, 0x3f3f_dc2f, 0x3f60_5f15, 0x3f62_0ad7, 0x3f46_6d09, 0x3eeb_d096,
    0xbe2d_e5a0, 0xbf23_6a33, 0xbf51_7189, 0xbf53_684f, 0xbf2a_f8b0, 0xbe70_7905, 0x3ed3_179d,
    0x3f40_cc13,
];

const STANDARD_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3f59_1871),
    (1023, 0x3e60_88f9),
    (2047, 0xbe53_9bf4),
    (4095, 0x3e90_451c),
];

#[rustfmt::skip]
const LATOOCARFIAN_N_SLOW_BLOCK: [u32; 64] = [
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3fbf_84e0,
    0x3fbf_84e0, 0x3fbf_84e0, 0x3fbf_84e0, 0x3fbf_84e0, 0x3fbf_84e0, 0x3fbf_84e0, 0x3eb0_c766,
    0x3eb0_c766, 0x3eb0_c766, 0x3eb0_c766, 0x3eb0_c766, 0x3eb0_c766, 0x3eb0_c766, 0xbea0_6737,
    0xbea0_6737, 0xbea0_6737, 0xbea0_6737, 0xbea0_6737, 0xbea0_6737, 0xbea0_6737, 0x3e5f_0e9e,
    0x3e5f_0e9e, 0x3e5f_0e9e, 0x3e5f_0e9e, 0x3e5f_0e9e, 0x3e5f_0e9e, 0x3e5f_0e9e, 0x3ef5_358d,
    0x3ef5_358d, 0x3ef5_358d, 0x3ef5_358d, 0x3ef5_358d, 0x3ef5_358d, 0x3ef5_358d, 0x3f95_602e,
    0x3f95_602e, 0x3f95_602e, 0x3f95_602e, 0x3f95_602e, 0x3f95_602e, 0x3f95_602e, 0x3f4f_058c,
    0x3f4f_058c, 0x3f4f_058c, 0x3f4f_058c, 0x3f4f_058c, 0x3f4f_058c, 0xbdcc_5629, 0xbdcc_5629,
    0xbdcc_5629, 0xbdcc_5629, 0xbdcc_5629, 0xbdcc_5629, 0xbdcc_5629, 0xbf0e_f909, 0xbf0e_f909,
    0xbf0e_f909,
];

const LATOOCARFIAN_N_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbf2e_9593),
    (1023, 0x3f1f_51e9),
    (2047, 0xbed3_ca4c),
    (4095, 0xbe07_fa94),
];

#[rustfmt::skip]
const LATOOCARFIAN_N_FAST_BLOCK: [u32; 64] = [
    0x3fbf_84e0, 0x3eb0_c766, 0xbea0_6737, 0x3e5f_0e9e, 0x3ef5_358d, 0x3f95_602e, 0x3f4f_058c,
    0xbdcc_5629, 0xbf0e_f909, 0x3ec8_05fe, 0xbed1_8d4c, 0x3dea_e469, 0xbf1a_abae, 0xbf13_ca7b,
    0xbfbd_1d92, 0xbe16_3331, 0x3f18_b54e, 0xbee9_a4ac, 0x3e65_8a43, 0xbefa_a85e, 0xbe94_1032,
    0xbfac_5a53, 0xbf1b_060b, 0xbc03_992d, 0xbd27_f2d3, 0xbf83_b701, 0xbf37_56e6, 0xbf1c_e58d,
    0xbed0_4be5, 0xbf11_62f7, 0xbf8f_ff1e, 0xbe9a_b1a0, 0x3e90_febe, 0xbeb1_0495, 0xbf24_e169,
    0xbfaf_4fc6, 0xbe99_5c31, 0x3eba_8a0e, 0xbe8b_7cfa, 0xbeb4_07e5, 0xbf93_8e3b, 0xbf55_8ef1,
    0xbc6a_6605, 0x3ed1_b28a, 0xbf05_2f6f, 0xbc95_ed83, 0xbf79_3e8e, 0xbf38_9213, 0xbf3e_3f95,
    0xbeb2_c0a1, 0xbe71_4e54, 0xbf82_a1a7, 0xbf82_a952, 0x3e53_9acb, 0x3f7d_1106, 0xbf26_cb8a,
    0x3ecd_d803, 0xbe93_19e7, 0x3e9b_83d7, 0xbd84_b66b, 0x3f04_1835, 0x3f21_d9c5, 0x3fbc_9803,
    0x3de0_ca4a,
];

const LATOOCARFIAN_N_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf2b_67fc),
    (1023, 0xbd3b_b644),
    (2047, 0x3f5e_f32b),
    (4095, 0xbef1_9075),
];

#[rustfmt::skip]
const CUSP_L_SLOW_BLOCK: [u32; 64] = [
    0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3e80_0000, 0x3e80_0000,
    0x3e62_2222, 0x3e44_4445, 0x3e26_6667, 0x3e08_8889, 0x3dd5_5557, 0x3d99_999c, 0x3d4c_ccd0,
    0x3e01_9f39, 0x3e50_0b3e, 0x3e8f_3ba2, 0x3eb6_71a4, 0x3edd_a7a7, 0x3f02_6ed5, 0x3f13_3cd6,
    0x3eda_9bb1, 0x3e8e_bdb5, 0x3e05_bf72, 0xbc8f_e429, 0xbe29_b87c, 0xbea0_ba3a, 0xbee1_c1a5,
    0xbed4_5ebf, 0xbec6_fbd9, 0xbeb9_98f2, 0xbeac_360c, 0xbe9e_d326, 0xbe91_7040, 0xbe85_f6e8,
    0xbe60_a831, 0xbe35_6292, 0xbe0a_1cf3, 0xbdbd_aea7, 0xbd4e_46d3, 0xbc04_c15a, 0x3ce6_582d,
    0x3dfc_b300, 0x3e5f_e7fa, 0x3ea0_bb3a, 0x3ed1_8278, 0x3f01_24da, 0x3f19_8879, 0x3f2e_7026,
    0x3eff_8efd, 0x3ea2_3dae, 0x3e09_d8bf, 0xbd43_277c, 0xbe6b_6c7d, 0xbf11_820e, 0xbf0c_6ebf,
    0xbf07_5b71, 0xbf02_4823, 0xbefa_69a8, 0xbef0_430c, 0xbee6_1c6f, 0xbedd_690d, 0xbecf_bf26,
    0xbec2_153e,
];

const CUSP_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3dd7_c471),
    (1023, 0x3e3b_eba5),
    (2047, 0xbeae_439e),
    (4095, 0xbdee_c58e),
];

#[rustfmt::skip]
const CUSP_L_FAST_BLOCK: [u32; 64] = [
    0x3e80_0000, 0x3d4c_ccd0, 0x3f13_3cd6, 0xbee1_c1a5, 0xbe85_f6e8, 0x3ce6_582d, 0x3f2e_7026,
    0xbf11_820e, 0xbedd_690d, 0xbe7f_6ed7, 0x3d51_1ca3, 0x3f12_1943, 0xbede_e5bd, 0xbe81_dc74,
    0x3d30_9cdb, 0x3f1a_ffcf, 0xbef4_f3bb, 0xbea0_ddea, 0xbd85_21a9, 0x3f03_fc8b, 0xbeba_80ca,
    0xbe16_40b3, 0x3e8b_5d31, 0x3c0e_f525, 0x3f52_90bb, 0xbf39_2165, 0xbf1d_a165, 0xbefb_5990,
    0xbea9_98ec, 0xbdbf_8947, 0x3ed6_8082, 0xbe6b_5089, 0x3db6_a8f9, 0x3edd_7a24, 0xbe7f_a035,
    0x3d4f_a51b, 0x3f12_7c1e, 0xbedf_de34, 0xbe83_41cf, 0x3d1b_9a89, 0x3f21_325a, 0xbf01_f7e3,
    0xbeb5_2463, 0xbe05_40c4, 0x3ea1_13aa, 0xbd86_8df0, 0x3f03_5354, 0xbeb8_c078, 0xbe10_ba04,
    0x3e92_47e0, 0xbc7f_308d, 0x3f43_4bdb, 0xbf28_d5d6, 0xbf0b_01e3, 0xbecc_d726, 0xbe4e_a07d,
    0x3e16_076d, 0x3e8b_a43d, 0x3bfd_96a5, 0x3f55_35fc, 0xbf3b_e499, 0xbf20_b483, 0xbf01_611d,
    0xbeb3_91e2,
];

const CUSP_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3f68_6a78),
    (1023, 0xbe9b_3259),
    (2047, 0x3e48_39ca),
    (4095, 0x3ea0_fbc7),
];

#[rustfmt::skip]
const QUAD_L_SLOW_BLOCK: [u32; 64] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0xbde0_0000, 0xbe60_0000, 0xbea8_0000, 0xbee0_0000, 0xbf0c_0000, 0xbf28_0000, 0xbf40_0000,
    0xbf0f_0000, 0xbebc_0000, 0xbe34_0001, 0x3c7f_ffe4, 0x3e53_fffe, 0x3ecb_ffff, 0x3f10_0000,
    0x3eab_a000, 0x3ddd_0002, 0xbdf4_7ffd, 0xbeb1_7fff, 0xbf12_efff, 0xbf4d_1fff, 0xbf7f_0000,
    0xbf2b_9530, 0xbeb0_54c1, 0xbc97_f212, 0x3e9d_567f, 0x3f22_160f, 0x3f75_80df, 0x3f9e_8080,
    0x3f7d_ca62, 0x3f3e_93c4, 0x3efe_ba4c, 0x3e80_4d0f, 0x3b6f_e9ab, 0xbe79_1ad2, 0xbee8_eb0c,
    0xbecd_8808, 0xbeb2_2505, 0xbe96_c201, 0xbe76_bdfb, 0xbe3f_f7f4, 0xbe09_31ed, 0xbdb4_7e15,
    0xbe2e_c3b9, 0xbe81_a434, 0xbeab_e68b, 0xbed6_28e2, 0xbf00_359d, 0xbf27_7337, 0xbf02_a32e,
    0xbebb_a64b, 0xbe64_0c72, 0xbda1_989e, 0x3d84_e7a9, 0x3e55_b3f8, 0x3ea9_f57a, 0x3e11_3af9,
    0xbd45_d408,
];

const QUAD_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3e91_8c7d),
    (1023, 0x3f39_7d38),
    (2047, 0x3d9b_cebd),
    (4095, 0x3eed_dab7),
];

#[rustfmt::skip]
const QUAD_L_FAST_BLOCK: [u32; 64] = [
    0x0000_0000, 0xbf40_0000, 0x3f10_0000, 0xbf7f_0000, 0x3f9e_8080, 0xbee8_eb0c, 0xbdb4_7e15,
    0xbf27_7337, 0x3ea9_f57a, 0xbf78_c53b, 0x3f95_41fb, 0xbf0e_6c79, 0x3ded_47e3, 0xbf5a_3944,
    0x3f54_3ed8, 0xbf64_46b3, 0x3f6f_d4b5, 0xbf4f_25da, 0x3f36_c416, 0xbf74_48b0, 0x3f8e_b1ab,
    0xbf1f_3d03, 0x3e84_93b5, 0xbf71_1fb1, 0x3f8a_1e2f, 0xbf2a_2a24, 0x3eb6_8c45, 0xbf7a_bb33,
    0x3f98_26ad, 0xbf06_958c, 0x3d55_6711, 0xbf4c_a48c, 0x3f30_3b36, 0xbf76_e9be, 0x3f92_87e6,
    0xbf15_92a1, 0x3e33_da92, 0xbf65_10ee, 0x3f72_083a, 0xbf4d_34ac, 0x3f31_b213, 0xbf76_5a59,
    0x3f91_b60e, 0xbf17_ad27, 0x3e46_2b80, 0xbf67_f46b, 0x3f7a_1f6d, 0xbf45_be08, 0x3f1e_7c0d,
    0xbf7c_5eb1, 0x3f9a_94a0, 0xbeff_994a, 0xbacd_5710, 0xbf3f_992b, 0x3f0e_ff16, 0xbf7f_1f1b,
    0x3f9e_af0c, 0xbee7_d7dc, 0xbdbc_b206, 0xbf26_3d67, 0x3ea4_6231, 0xbf77_cd9b, 0x3f93_d601,
    0xbf12_2e22,
];

const QUAD_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf76_8252),
    (1023, 0xbf7d_bc6c),
    (2047, 0xbf5e_2b0a),
    (4095, 0xbf75_ed97),
];

#[rustfmt::skip]
const HENON_L_SLOW_BLOCK: [u32; 64] = [
    0x3ef1_1111, 0x3ee2_2222, 0x3ed3_3333, 0x3ec4_4445, 0x3eb5_5556, 0x3ea6_6667, 0x3e99_999a,
    0x3ecf_a89f, 0x3f02_dbd2, 0x3f1d_e354, 0x3f38_ead6, 0x3f53_f258, 0x3f6e_f9db, 0x3f83_126f,
    0x3f51_cd6c, 0x3f1d_75fb, 0x3ed2_3d14, 0x3e53_1c64, 0x3adf_4fa4, 0xbe4f_9f25, 0xbec1_8a0d,
    0xbe25_4b59, 0x3d61_f5a0, 0x3e8b_2314, 0x3efa_0775, 0x3f34_75eb, 0x3f6b_e81b, 0x3f8d_b747,
    0x3f53_212b, 0x3f0a_d3c7, 0x3e85_0cc8, 0xbcb8_dfef, 0xbe9c_28c6, 0xbf16_61c6, 0xbf54_5af8,
    0xbf27_9e1e, 0xbef5_c28a, 0xbe9c_48d7, 0xbe05_9e48, 0x3d35_5475, 0x3e60_4883, 0x3ebc_d5b8,
    0x3ecb_29a4, 0x3ed9_7d91, 0x3ee7_d17e, 0x3ef6_256b, 0x3f02_3cac, 0x3f09_66a2, 0x3f0f_8a9a,
    0x3f13_a3da, 0x3f17_bd1b, 0x3f1b_d65b, 0x3f1f_ef9b, 0x3f24_08dc, 0x3f2b_a578, 0x3f26_bb14,
    0x3f21_d0b1, 0x3f1c_e64e, 0x3f17_fbea, 0x3f13_1187, 0x3f0e_2724, 0x3f09_f085, 0x3f13_7dcf,
    0x3f1d_0b18,
];

const HENON_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbe40_276d),
    (1023, 0x3f06_8d17),
    (2047, 0x3f3f_547a),
    (4095, 0x3e60_f027),
];

#[rustfmt::skip]
const HENON_L_FAST_BLOCK: [u32; 64] = [
    0x3e99_999a, 0x3f83_126f, 0xbec1_8a0d, 0x3f8d_b747, 0xbf54_5af8, 0x3ebc_d5b8, 0x3f0f_8a9a,
    0x3f2b_a578, 0x3f09_f085, 0x3f4b_7033, 0x3e8e_178a, 0x3f90_b6c2, 0xbf34_cb80, 0x3f24_1287,
    0x3e5a_2d3a, 0x3f90_79d5, 0xbf38_3d6d, 0x3f1d_0d67, 0x3e83_ad82, 0x3f8b_b47b, 0xbf17_3180,
    0x3f56_cf7c, 0xbe26_d273, 0x3f9b_772e, 0xbf8e_9c2f, 0xbebf_368d, 0x3ef0_e477, 0x3f13_fb35,
    0x3f2c_606d, 0x3f09_e5c3, 0x3f4b_b884, 0x3e8c_cf08, 0x3f91_0123, 0xbf36_d426, 0x3f20_33d0,
    0x3e73_301d, 0x3f8d_ec61, 0xbf26_5f56, 0x3f3d_c7be, 0x3d11_f71e, 0x3f9c_3d4e, 0xbf89_9fd7,
    0xbe81_287b, 0x3f16_9e42, 0x3ee1_204d, 0x3f67_e4d9, 0xbc89_fbf9, 0x3fa2_bbb6, 0xbfa2_4b7e,
    0xbf5e_8a19, 0xbee0_6ba5, 0x3ef0_c2c0, 0x3f0f_163f, 0x3f34_25cf, 0x3ef2_e536, 0x3f65_61fd,
    0x3c96_04ad, 0x3fa2_58ea, 0xbf9f_929b, 0xbf4b_9ab3, 0xbe84_e5e8, 0x3f2a_c587, 0x3e99_2937,
    0x3f89_946e,
];

const HENON_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3f55_ffe5),
    (1023, 0xbf1f_df64),
    (2047, 0x3f5d_aa03),
    (4095, 0x3ea1_79ec),
];

#[rustfmt::skip]
const LORENZ_L_SLOW_BLOCK: [u32; 64] = [
    0x3b83_126f, 0x3b83_126f, 0x3b83_126f, 0x3b83_126f, 0x3b83_126f, 0x3b83_126f, 0x3b83_126f,
    0x3b80_bde6, 0x3b7c_d2bb, 0x3b78_29aa, 0x3b73_8099, 0x3b6e_d788, 0x3b6a_2e77, 0x3b66_2fd6,
    0x3b75_6e54, 0x3b82_5669, 0x3b89_f5a7, 0x3b91_94e6, 0x3b99_3425, 0x3ba0_d363, 0x3ba7_5be2,
    0x3bb9_0f76, 0x3bca_c309, 0x3bdc_769c, 0x3bee_2a2f, 0x3bff_ddc2, 0x3c08_c8aa, 0x3c10_5ec5,
    0x3c21_01ab, 0x3c31_a492, 0x3c42_4778, 0x3c52_ea5e, 0x3c63_8d45, 0x3c74_302b, 0x3c81_3954,
    0x3c90_592b, 0x3c9f_7902, 0x3cae_98d9, 0x3cbd_b8b0, 0x3ccc_d888, 0x3cdb_f85f, 0x3ce8_ef17,
    0x3d02_1fb7, 0x3d0f_c7e3, 0x3d1d_700f, 0x3d2b_183b, 0x3d38_c067, 0x3d46_6893, 0x3d52_1d4b,
    0x3d6a_ae6e, 0x3d81_9fc9, 0x3d8d_e85a, 0x3d9a_30ec, 0x3da6_797e, 0x3dbd_4968, 0x3dd3_2cb6,
    0x3de9_1005, 0x3dfe_f353, 0x3e0a_6b51, 0x3e15_5cf8, 0x3e20_4e9f, 0x3e29_b00a, 0x3e3c_9baa,
    0x3e4f_874a,
];

const LORENZ_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbec9_56ec),
    (1023, 0xbed7_e955),
    (2047, 0xbdac_2c7c),
    (4095, 0x3d42_0663),
];

#[rustfmt::skip]
const LORENZ_L_FAST_BLOCK: [u32; 64] = [
    0x3b83_126f, 0x3b66_2fd6, 0x3ba7_5be2, 0x3c10_5ec5, 0x3c81_3954, 0x3ce8_ef17, 0x3d52_1d4b,
    0x3dbd_4968, 0x3e29_b00a, 0x3e95_b6bd, 0x3efb_65ed, 0x3f37_8f90, 0x3f46_8c26, 0x3f0d_e6f8,
    0x3e75_2e0b, 0x3b8c_8ca5, 0xbe09_4813, 0xbe58_54a2, 0xbe84_2633, 0xbe95_d82b, 0xbea5_6ab2,
    0xbeb3_9362, 0xbebe_d262, 0xbec4_c052, 0xbec3_a9e5, 0xbebb_ea62, 0xbeb0_1146, 0xbea3_b85d,
    0xbe9a_0e61, 0xbe95_0dff, 0xbe95_77c5, 0xbe9b_1865, 0xbea4_f25b, 0xbeb1_2f37, 0xbebd_0c3b,
    0xbec5_31a7, 0xbec6_cabc, 0xbec0_fcdd, 0xbeb5_939f, 0xbea8_25e2, 0xbe9c_68d9, 0xbe94_f8e0,
    0xbe93_12a9, 0xbe96_d83b, 0xbe9f_9686, 0xbeab_c967, 0xbeb8_f47d, 0xbec3_b5ec, 0xbec8_958e,
    0xbec5_94a4, 0xbebb_781c, 0xbead_8971, 0xbe9f_ec5e, 0xbe95_f417, 0xbe91_785d, 0xbe93_024e,
    0xbe9a_2c70, 0xbea5_c6c6, 0xbeb3_b0c0, 0xbec0_b55d, 0xbec8_fbb5, 0xbec9_7085, 0xbec1_8a6b,
    0xbeb3_daec,
];

const LORENZ_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbc0d_c1b0),
    (1023, 0x3eaf_e46e),
    (2047, 0x3e77_7b85),
    (4095, 0xbebd_963a),
];

#[rustfmt::skip]
const STANDARD_L_SLOW_BLOCK: [u32; 64] = [
    0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf57_419f, 0xbf57_419f,
    0xbf51_8f1e, 0xbf4b_dc9c, 0xbf46_2a1b, 0xbf40_779a, 0xbf3a_c518, 0xbf35_1297, 0xbf30_3071,
    0xbf20_a060, 0xbf11_1050, 0xbf01_803f, 0xbee3_e05d, 0xbec4_c03b, 0xbea5_a01a, 0xbe8a_f246,
    0xbe33_d990, 0xbda3_9d26, 0x3c81_e34d, 0x3de4_8ecd, 0x3e54_5263, 0x3e9b_2eb0, 0x3ec5_3366,
    0x3edf_fc39, 0x3efa_c50c, 0x3f0a_c6f0, 0x3f18_2b59, 0x3f25_8fc3, 0x3f32_f42c, 0x3f3e_6ed0,
    0x3f43_4319, 0x3f48_1763, 0x3f4c_ebac, 0x3f51_bff6, 0x3f56_943f, 0x3f5b_6888, 0x3f5f_8c35,
    0x3f5f_c4b7, 0x3f5f_fd38, 0x3f60_35ba, 0x3f60_6e3c, 0x3f60_a6bd, 0x3f60_df3f, 0x3f61_0fae,
    0x3f5c_e0b1, 0x3f58_b1b4, 0x3f54_82b7, 0x3f50_53ba, 0x3f4c_24bd, 0x3f44_5fc2, 0x3f38_4020,
    0x3f2c_207e, 0x3f20_00dc, 0x3f13_e13a, 0x3f07_c199, 0x3ef7_43ed, 0x3ee2_7b6a, 0x3eb2_db49,
    0x3e83_3b29,
];

const STANDARD_L_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0xbf06_98b4),
    (1023, 0xbe74_2879),
    (2047, 0xbf1b_2fa2),
    (4095, 0xbe59_d815),
];

#[rustfmt::skip]
const STANDARD_L_FAST_BLOCK: [u32; 64] = [
    0xbf57_419f, 0xbf30_3071, 0xbe8a_f246, 0x3ec5_3366, 0x3f3e_6ed0, 0x3f5f_8c35, 0x3f61_0fae,
    0x3f44_5fc2, 0x3ee2_7b6a, 0xbe48_2f7f, 0xbf26_5ff4, 0xbf52_160a, 0xbf52_464b, 0xbf27_198b,
    0xbe4e_8ef0, 0x3ee0_252a, 0x3f43_d95c, 0x3f60_ca4c, 0x3f5f_4772, 0x3f3d_eb55, 0x3ec2_f944,
    0xbe8d_92ad, 0xbf30_d897, 0xbf57_a004, 0xbf57_a935, 0xbf30_fc3e, 0xbe8e_3732, 0x3ec2_618a,
    0x3f3d_c1be, 0x3f5f_2382, 0x3f60_8af4, 0x3f43_43e3, 0x3edd_718f, 0xbe55_ef98, 0xbf27_f6c7,
    0xbf52_8fc5, 0xbf52_0a08, 0xbf25_f450, 0xbe44_400a, 0x3ee3_f62c, 0x3f44_bd62, 0x3f61_535b,
    0x3f5f_f4d3, 0x3f3f_5ce9, 0x3ec9_66ac, 0xbe85_d67e, 0xbf2e_e43d, 0xbf56_801c, 0xbf56_6168,
    0xbf2e_6d57, 0xbe83_b773, 0x3ecb_4264, 0x3f3f_dc2f, 0x3f60_5f15, 0x3f62_0ad7, 0x3f46_6d09,
    0x3eeb_d096, 0xbe2d_e5a0, 0xbf23_6a33, 0xbf51_7189, 0xbf53_684f, 0xbf2a_f8b0, 0xbe70_7905,
    0x3ed3_179d,
];

const STANDARD_L_FAST_LATER: [(usize, u32); 4] = [
    (511, 0xbf61_c820),
    (1023, 0xbee3_cfdf),
    (2047, 0x3ef0_5b29),
    (4095, 0xbf01_acc6),
];

#[rustfmt::skip]
const CUSP_L_RESEED: [u32; 64] = [
    0x3e40_51a6, 0x3eaa_c8e8, 0x3ef5_68fd, 0x3f20_0489, 0x3f40_0000, 0x3f0b_e73d, 0x3eaf_9cf2,
    0x3e0e_d6d6, 0xbd83_186f, 0xbe88_f7a3, 0xbef1_292a, 0xbf25_3c19, 0xbf20_cad9, 0xbf1c_599a,
    0xbf17_e85a, 0xbf13_771b, 0xbf0f_05db, 0xbf0a_949c, 0xbf06_c5d3, 0xbf01_409b, 0xbef7_76c6,
    0xbeec_6c55, 0xbee1_61e5, 0xbed6_5775, 0xbecb_4d05, 0xbec1_d65b, 0xbeb2_3151, 0xbea2_8c46,
    0xbe92_e73c, 0xbe83_4231, 0xbe67_3a4d, 0xbe47_f038, 0xbe2d_1e6f, 0xbde6_675f, 0xbd65_23c0,
    0x3a21_cf56, 0x3d6a_323b, 0x3de8_ee9c, 0x3e60_06b2, 0x3e4f_fa26, 0x3e3f_ed9b, 0x3e2f_e10f,
    0x3e1f_d483, 0x3e0f_c7f7, 0x3dff_76d6, 0x3de3_f39d, 0x3e18_0712, 0x3e3e_1456, 0x3e64_219a,
    0x3e85_176f, 0x3e98_1e10, 0x3eab_24b2, 0x3ebb_7386, 0x3e94_f0fa, 0x3e5c_dcdd, 0x3e0f_d7c5,
    0x3d85_a55a, 0xbc23_26b2, 0xbdae_6f06, 0xbe19_3be2, 0xbdb6_9ff2, 0xbceb_207c, 0x3d02_1f67,
    0x3dbc_e786,
];

#[rustfmt::skip]
const QUAD_L_RESEED: [u32; 64] = [
    0xbea7_dde4, 0xbde7_e6ca, 0x3dcf_a9fa, 0x3ea1_ceb0, 0x3f00_0000, 0x3e90_0000, 0x3d80_0002,
    0xbe1f_fffe, 0xbebf_ffff, 0xbf17_ffff, 0xbf4f_ffff, 0xbf80_0000, 0xbf2c_0000, 0xbeb0_0001,
    0xbc80_0012, 0x3e9f_fffe, 0x3f23_ffff, 0x3f77_ffff, 0x3fa0_0000, 0x3f80_8000, 0x3f42_0000,
    0x3f03_0000, 0x3e88_0001, 0x3ca0_0016, 0xbe67_fffd, 0xbee0_0000, 0xbec8_6000, 0xbeb0_c000,
    0xbe99_2000, 0xbe81_8000, 0xbe53_c001, 0xbe24_8001, 0xbdf8_0000, 0xbe45_a4c0, 0xbe87_a4c0,
    0xbeac_7720, 0xbed1_4980, 0xbef6_1be0, 0xbf1d_3f00, 0xbefa_9812, 0xbeba_b224, 0xbe75_986b,
    0xbdeb_991e, 0x3c1f_f4d5, 0x3e09_cb2a, 0x3e77_5556, 0x3d8f_d027, 0xbdcf_0a5f, 0xbe8b_7939,
    0xbee3_2fda, 0xbf1d_733e, 0xbf49_4e8f, 0xbf6e_e5f9, 0xbf24_b52c, 0xbeb5_08bd, 0xbd82_9c8c,
    0x3e67_74ef, 0x3f04_0e09, 0x3f4e_3ed6, 0x3f86_eb31, 0x3f4c_9cfa, 0x3f0b_6391, 0x3e94_5450,
    0x3d0f_0bf5,
];

#[rustfmt::skip]
const STANDARD_L_RESEED: [u32; 64] = [
    0xbe89_b309, 0xbe96_3c21, 0xbea2_c539, 0xbeaf_4e52, 0xbeba_0cf9, 0xbea4_7075, 0xbe8e_d3f0,
    0xbe72_6ed7, 0xbe47_35ce, 0xbe1b_fcc5, 0xbde1_8778, 0xbd97_6ed5, 0xbcac_67e6, 0x3d02_75c5,
    0x3dad_8fbe, 0x3e0c_f24d, 0x3e43_1cbb, 0x3e79_4729, 0x3e93_da56, 0x3e9c_3654, 0x3ea4_9252,
    0x3eac_ee4f, 0x3eb5_4a4d, 0x3ebd_a64b, 0x3ec6_0248, 0x3ecd_2c90, 0x3ebe_e9a8, 0x3eb0_a6c0,
    0x3ea2_63d8, 0x3e94_20f1, 0x3e85_de09, 0x3e6f_3642, 0x3e56_c390, 0x3e1d_2392, 0x3dc7_0729,
    0x3d27_8e5a, 0xbc7b_c673, 0xbd92_b8ca, 0xbe34_60f4, 0xbe55_054c, 0xbe75_a9a5, 0xbe8b_26fe,
    0xbe9b_792b, 0xbeab_cb57, 0xbebc_1d83, 0xbeca_1ace, 0xbec3_f244, 0xbebd_c9ba, 0xbeb7_a130,
    0xbeb1_78a6, 0xbeab_501c, 0xbea5_2792, 0xbe9f_e040, 0xbe85_f74d, 0xbe58_1cb5, 0xbe24_4acf,
    0xbde0_f1d4, 0xbd72_9c12, 0xbc0d_51f1, 0x3d0e_569a, 0x3da4_7363, 0x3e00_ddbc, 0x3e2f_81c7,
    0x3e5e_25d2,
];

#[rustfmt::skip]
const LORENZ_L_RESEED: [u32; 64] = [
    0x3e35_de39, 0x3e0e_ae78, 0x3dce_fd70, 0x3d80_9def, 0x3cf5_c28f, 0x3cf1_63c9, 0x3ced_0503,
    0x3ce8_a63d, 0x3ce4_4777, 0x3cdf_e8b1, 0x3cdb_89eb, 0x3cd7_caf8, 0x3ce6_1233, 0x3cf4_596d,
    0x3d01_5054, 0x3d08_73f1, 0x3d0f_978e, 0x3d16_bb2b, 0x3d1c_d9b2, 0x3d2d_65fe, 0x3d3d_f24b,
    0x3d4e_7e98, 0x3d5f_0ae4, 0x3d6f_9731, 0x3d80_11bf, 0x3d87_294d, 0x3d96_9ce6, 0x3da6_107f,
    0x3db5_8417, 0x3dc4_f7b0, 0x3dd4_6b49, 0x3de3_dee1, 0x3df1_1d65, 0x3e06_4f09, 0x3e14_0f60,
    0x3e21_cfb7, 0x3e2f_900e, 0x3e3d_5064, 0x3e56_da2a, 0x3e6e_05d3, 0x3e82_98be, 0x3e8e_2e92,
    0x3e99_c467, 0x3ea5_5a3b, 0x3eb0_f00f, 0x3eba_de33, 0x3ecb_8209, 0x3edc_25de, 0x3eec_c9b4,
    0x3efd_6d89, 0x3f07_08af, 0x3f0f_5a9a, 0x3f16_7c3f, 0x3f1d_79ec, 0x3f24_7798, 0x3f2b_7545,
    0x3f32_72f2, 0x3f39_709e, 0x3f40_6e4b, 0x3f46_6c4d, 0x3f44_1638, 0x3f41_c022, 0x3f3f_6a0d,
    0x3f3d_13f8,
];
#[rustfmt::skip]
const CUSP_N_RESEED: [u32; 64] = [
    0x3f40_0000, 0x3f40_0000, 0x3f40_0000, 0x3f40_0000, 0xbf25_3c19, 0xbf25_3c19, 0xbf25_3c19,
    0xbf25_3c19, 0xbf25_3c19, 0xbf25_3c19, 0xbf25_3c19, 0xbf06_c5d3, 0xbf06_c5d3, 0xbf06_c5d3,
    0xbf06_c5d3, 0xbf06_c5d3, 0xbf06_c5d3, 0xbf06_c5d3, 0xbec1_d65b, 0xbec1_d65b, 0xbec1_d65b,
    0xbec1_d65b, 0xbec1_d65b, 0xbec1_d65b, 0xbec1_d65b, 0xbe2d_1e6f, 0xbe2d_1e6f, 0xbe2d_1e6f,
    0xbe2d_1e6f, 0xbe2d_1e6f, 0xbe2d_1e6f, 0xbe2d_1e6f, 0x3e60_06b2, 0x3e60_06b2, 0x3e60_06b2,
    0x3e60_06b2, 0x3e60_06b2, 0x3e60_06b2, 0x3de3_f39d, 0x3de3_f39d, 0x3de3_f39d, 0x3de3_f39d,
    0x3de3_f39d, 0x3de3_f39d, 0x3de3_f39d, 0x3ebb_7386, 0x3ebb_7386, 0x3ebb_7386, 0x3ebb_7386,
    0x3ebb_7386, 0x3ebb_7386, 0x3ebb_7386, 0xbe19_3be2, 0xbe19_3be2, 0xbe19_3be2, 0xbe19_3be2,
    0xbe19_3be2, 0xbe19_3be2, 0xbe19_3be2, 0x3e87_af78, 0x3e87_af78, 0x3e87_af78, 0x3e87_af78,
    0x3e87_af78,
];

#[rustfmt::skip]
const QUAD_N_RESEED: [u32; 64] = [
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0xbf80_0000, 0xbf80_0000, 0xbf80_0000,
    0xbf80_0000, 0xbf80_0000, 0xbf80_0000, 0xbf80_0000, 0x3fa0_0000, 0x3fa0_0000, 0x3fa0_0000,
    0x3fa0_0000, 0x3fa0_0000, 0x3fa0_0000, 0x3fa0_0000, 0xbee0_0000, 0xbee0_0000, 0xbee0_0000,
    0xbee0_0000, 0xbee0_0000, 0xbee0_0000, 0xbee0_0000, 0xbdf8_0000, 0xbdf8_0000, 0xbdf8_0000,
    0xbdf8_0000, 0xbdf8_0000, 0xbdf8_0000, 0xbdf8_0000, 0xbf1d_3f00, 0xbf1d_3f00, 0xbf1d_3f00,
    0xbf1d_3f00, 0xbf1d_3f00, 0xbf1d_3f00, 0x3e77_5556, 0x3e77_5556, 0x3e77_5556, 0x3e77_5556,
    0x3e77_5556, 0x3e77_5556, 0x3e77_5556, 0xbf6e_e5f9, 0xbf6e_e5f9, 0xbf6e_e5f9, 0xbf6e_e5f9,
    0xbf6e_e5f9, 0xbf6e_e5f9, 0xbf6e_e5f9, 0x3f86_eb31, 0x3f86_eb31, 0x3f86_eb31, 0x3f86_eb31,
    0x3f86_eb31, 0x3f86_eb31, 0x3f86_eb31, 0xbf31_6a24, 0xbf31_6a24, 0xbf31_6a24, 0xbf31_6a24,
    0xbf31_6a24,
];

#[rustfmt::skip]
const LATOOCARFIAN_N_RESEED: [u32; 64] = [
    0x3e4c_cccd, 0x3e4c_cccd, 0x3e4c_cccd, 0x3e4c_cccd, 0x3fa3_d105, 0x3fa3_d105, 0x3fa3_d105,
    0x3fa3_d105, 0x3fa3_d105, 0x3fa3_d105, 0x3fa3_d105, 0x3f25_6e14, 0x3f25_6e14, 0x3f25_6e14,
    0x3f25_6e14, 0x3f25_6e14, 0x3f25_6e14, 0x3f25_6e14, 0x3dd8_ce2e, 0x3dd8_ce2e, 0x3dd8_ce2e,
    0x3dd8_ce2e, 0x3dd8_ce2e, 0x3dd8_ce2e, 0x3dd8_ce2e, 0x3de1_4ac1, 0x3de1_4ac1, 0x3de1_4ac1,
    0x3de1_4ac1, 0x3de1_4ac1, 0x3de1_4ac1, 0x3de1_4ac1, 0x3f94_88e3, 0x3f94_88e3, 0x3f94_88e3,
    0x3f94_88e3, 0x3f94_88e3, 0x3f94_88e3, 0x3f39_eea0, 0x3f39_eea0, 0x3f39_eea0, 0x3f39_eea0,
    0x3f39_eea0, 0x3f39_eea0, 0x3f39_eea0, 0x3e86_306c, 0x3e86_306c, 0x3e86_306c, 0x3e86_306c,
    0x3e86_306c, 0x3e86_306c, 0x3e86_306c, 0x3e2d_a1a8, 0x3e2d_a1a8, 0x3e2d_a1a8, 0x3e2d_a1a8,
    0x3e2d_a1a8, 0x3e2d_a1a8, 0x3e2d_a1a8, 0x3f8c_55dd, 0x3f8c_55dd, 0x3f8c_55dd, 0x3f8c_55dd,
    0x3f8c_55dd,
];

#[rustfmt::skip]
const LATOOCARFIAN_N_RESEED_Y: [u32; 64] = [
    0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f00_0000, 0x3f88_1d2a, 0x3f88_1d2a, 0x3f88_1d2a,
    0x3f88_1d2a, 0x3f88_1d2a, 0x3f88_1d2a, 0x3f88_1d2a, 0x3f76_498a, 0x3f76_498a, 0x3f76_498a,
    0x3f76_498a, 0x3f76_498a, 0x3f76_498a, 0x3f76_498a, 0xbe2e_25fb, 0xbe2e_25fb, 0xbe2e_25fb,
    0xbe2e_25fb, 0xbe2e_25fb, 0xbe2e_25fb, 0xbe2e_25fb, 0xbf60_e53c, 0xbf60_e53c, 0xbf60_e53c,
    0xbf60_e53c, 0xbf60_e53c, 0xbf60_e53c, 0xbf60_e53c, 0x3f0e_cdf0, 0x3f0e_cdf0, 0x3f0e_cdf0,
    0x3f0e_cdf0, 0x3f0e_cdf0, 0x3f0e_cdf0, 0xbeed_1d3e, 0xbeed_1d3e, 0xbeed_1d3e, 0xbeed_1d3e,
    0xbeed_1d3e, 0xbeed_1d3e, 0xbeed_1d3e, 0x3e2b_2531, 0x3e2b_2531, 0x3e2b_2531, 0x3e2b_2531,
    0x3e2b_2531, 0x3e2b_2531, 0x3e2b_2531, 0xbf17_a528, 0xbf17_a528, 0xbf17_a528, 0xbf17_a528,
    0xbf17_a528, 0xbf17_a528, 0xbf17_a528, 0xbef2_7287, 0xbef2_7287, 0xbef2_7287, 0xbef2_7287,
    0xbef2_7287,
];

#[rustfmt::skip]
const STANDARD_N_RESEED: [u32; 64] = [
    0xbe48_2f7f, 0xbe48_2f7f, 0xbe48_2f7f, 0xbe48_2f7f, 0xbd97_6ed5, 0xbd97_6ed5, 0xbd97_6ed5,
    0xbd97_6ed5, 0xbd97_6ed5, 0xbd97_6ed5, 0xbd97_6ed5, 0x3e93_da56, 0x3e93_da56, 0x3e93_da56,
    0x3e93_da56, 0x3e93_da56, 0x3e93_da56, 0x3e93_da56, 0x3ecd_2c90, 0x3ecd_2c90, 0x3ecd_2c90,
    0x3ecd_2c90, 0x3ecd_2c90, 0x3ecd_2c90, 0x3ecd_2c90, 0x3e56_c390, 0x3e56_c390, 0x3e56_c390,
    0x3e56_c390, 0x3e56_c390, 0x3e56_c390, 0x3e56_c390, 0xbe34_60f4, 0xbe34_60f4, 0xbe34_60f4,
    0xbe34_60f4, 0xbe34_60f4, 0xbe34_60f4, 0xbeca_1ace, 0xbeca_1ace, 0xbeca_1ace, 0xbeca_1ace,
    0xbeca_1ace, 0xbeca_1ace, 0xbeca_1ace, 0xbe9f_e040, 0xbe9f_e040, 0xbe9f_e040, 0xbe9f_e040,
    0xbe9f_e040, 0xbe9f_e040, 0xbe9f_e040, 0x3d0e_569a, 0x3d0e_569a, 0x3d0e_569a, 0x3d0e_569a,
    0x3d0e_569a, 0x3d0e_569a, 0x3d0e_569a, 0x3eb1_b41e, 0x3eb1_b41e, 0x3eb1_b41e, 0x3eb1_b41e,
    0x3eb1_b41e,
];

#[rustfmt::skip]
const STANDARD_N_RESEED_Y: [u32; 64] = [
    0xbe48_2f7f, 0xbe48_2f7f, 0xbe48_2f7f, 0xbe48_2f7f, 0xbf17_be37, 0xbf17_be37, 0xbf17_be37,
    0xbf17_be37, 0xbf17_be37, 0xbf17_be37, 0xbf17_be37, 0xbd22_d4fc, 0xbd22_d4fc, 0xbd22_d4fc,
    0xbd22_d4fc, 0xbd22_d4fc, 0xbd22_d4fc, 0xbd22_d4fc, 0x3f0d_8a23, 0x3f0d_8a23, 0x3f0d_8a23,
    0x3f0d_8a23, 0x3f0d_8a23, 0x3f0d_8a23, 0x3f0d_8a23, 0x3f54_e420, 0x3f54_e420, 0x3f54_e420,
    0x3f54_e420, 0x3f54_e420, 0x3f54_e420, 0x3f54_e420, 0x3f73_1dd8, 0x3f73_1dd8, 0x3f73_1dd8,
    0x3f73_1dd8, 0x3f73_1dd8, 0x3f73_1dd8, 0xbf7b_7cdd, 0xbf7b_7cdd, 0xbf7b_7cdd, 0xbf7b_7cdd,
    0xbf7b_7cdd, 0xbf7b_7cdd, 0xbf7b_7cdd, 0xbf65_9507, 0xbf65_9507, 0xbf65_9507, 0xbf65_9507,
    0xbf65_9507, 0xbf65_9507, 0xbf65_9507, 0xbf35_b811, 0xbf35_b811, 0xbf35_b811, 0xbf35_b811,
    0xbf35_b811, 0xbf35_b811, 0xbf35_b811, 0xbe8a_e2e1, 0xbe8a_e2e1, 0xbe8a_e2e1, 0xbe8a_e2e1,
    0xbe8a_e2e1,
];
#[rustfmt::skip]
const STANDARD_N_WIDE_SLOW_BLOCK: [u32; 64] = [
    0xbfa8_be61, 0xbfa8_be61, 0xbfa8_be61, 0xbfa8_be61, 0xbfa8_be61, 0xbfa8_be61, 0xbdd3_473e,
    0xbdd3_473e, 0xbdd3_473e, 0xbdd3_473e, 0xbdd3_473e, 0xbdd3_473e, 0xbdd3_473e, 0xbf40_587b,
    0xbf40_587b, 0xbf40_587b, 0xbf40_587b, 0xbf40_587b, 0xbf40_587b, 0xbf40_587b, 0xbf7c_bf71,
    0xbf7c_bf71, 0xbf7c_bf71, 0xbf7c_bf71, 0xbf7c_bf71, 0xbf7c_bf71, 0xbf7c_bf71, 0x3d3f_b4f4,
    0x3d3f_b4f4, 0x3d3f_b4f4, 0x3d3f_b4f4, 0x3d3f_b4f4, 0x3d3f_b4f4, 0x3d3f_b4f4, 0x3ed5_b651,
    0x3ed5_b651, 0x3ed5_b651, 0x3ed5_b651, 0x3ed5_b651, 0x3ed5_b651, 0x3ed5_b651, 0x3cb8_e13c,
    0x3cb8_e13c, 0x3cb8_e13c, 0x3cb8_e13c, 0x3cb8_e13c, 0x3cb8_e13c, 0x3cb8_e13c, 0xbf20_910c,
    0xbf20_910c, 0xbf20_910c, 0xbf20_910c, 0xbf20_910c, 0xbf20_910c, 0x3d3a_c2f4, 0x3d3a_c2f4,
    0x3d3a_c2f4, 0x3d3a_c2f4, 0x3d3a_c2f4, 0x3d3a_c2f4, 0x3d3a_c2f4, 0x3e32_8f46, 0x3e32_8f46,
    0x3e32_8f46,
];

const STANDARD_N_WIDE_SLOW_LATER: [(usize, u32); 4] = [
    (511, 0x3dea_bdb2),
    (1023, 0xbf6c_9415),
    (2047, 0xbea8_53b7),
    (4095, 0xbf65_5ff9),
];

#[rustfmt::skip]
const STANDARD_N_WIDE_FAST_BLOCK: [u32; 64] = [
    0xbdd3_473e, 0xbf40_587b, 0xbf7c_bf71, 0x3d3f_b4f4, 0x3ed5_b651, 0x3cb8_e13c, 0xbf20_910c,
    0x3d3a_c2f4, 0x3e32_8f46, 0xbe8c_df63, 0xbf02_cf72, 0xbf6f_4c03, 0xbf61_25c1, 0x3f71_11ae,
    0x3f76_882e, 0xbf34_ae4d, 0xbf77_8e72, 0x3d83_53d0, 0x3f39_e086, 0xbf3e_fa6b, 0x3f11_fd19,
    0x3f53_f88a, 0x3f36_2b09, 0xbeea_9b18, 0xbd91_3733, 0xbf26_c7cb, 0xbf70_2f62, 0x3f69_fdc8,
    0x3e8b_ec7b, 0xbeeb_bd2e, 0x3ec8_48b5, 0xbf3f_0298, 0x3f65_6e0e, 0x3eaf_6214, 0xbe67_fc54,
    0xbbb6_5c4e, 0x3f45_9be5, 0x3f24_902e, 0xbe2a_609d, 0x3f6a_59a2, 0xbeb8_9f79, 0xbf4f_57a8,
    0x3f23_5934, 0xbf4b_f311, 0x3f3f_00ef, 0xbefc_2d52, 0x3dba_7c18, 0xbe9d_5f85, 0xbf07_1b79,
    0x3f76_3ad3, 0x3f24_e288, 0xbe99_1ed0, 0x3ee8_aae2, 0xbe98_4643, 0x3f0c_ae61, 0xbd4c_2652,
    0x3ea1_0e76, 0x3dc9_7ef9, 0x3e4c_3d35, 0xbeba_d769, 0x3da0_b323, 0x3f41_21a0, 0xbf43_4c48,
    0xbf33_8f2d,
];

const STANDARD_N_WIDE_FAST_LATER: [(usize, u32); 4] = [
    (511, 0x3f77_8449),
    (1023, 0xbf2b_8eef),
    (2047, 0x3d88_b149),
    (4095, 0xbda4_d8e9),
];
