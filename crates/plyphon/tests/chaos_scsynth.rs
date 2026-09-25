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

/// The ported units at their pinned inputs (`freq` first).
const UNITS: &[(&str, &[f32])] = &[
    ("FBSineN", FB_SINE),
    ("FBSineL", FB_SINE),
    ("FBSineC", FB_SINE),
    ("HenonN", HENON),
    ("HenonC", HENON),
    ("LatoocarfianL", LATOOCARFIAN),
    ("LatoocarfianC", LATOOCARFIAN),
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
// Arity
// ---------------------------------------------------------------------------------------------

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
