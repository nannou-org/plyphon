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

/// The ported units at their pinned inputs (`freq` first).
fn units() -> [(&'static str, &'static [f32]); 3] {
    [
        ("FBSineN", FB_SINE),
        ("FBSineL", FB_SINE),
        ("FBSineC", FB_SINE),
    ]
}

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
// Arity
// ---------------------------------------------------------------------------------------------

/// Each constructor rejects an input list one short of its full arity.
#[test]
fn units_reject_short_input_lists() {
    for (name, consts) in units() {
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
