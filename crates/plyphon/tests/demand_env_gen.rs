//! `DemandEnvGen`: an envelope whose segments are demanded from `Dseq`s. Each expected output is
//! what scsynth's own `DemandUGens.cpp` computes for the same inputs (built without floating-point
//! contraction, so `a * b + c` rounds twice as the source reads): the audio-rate renders as a hash
//! of every sample's bit pattern plus spot samples, the control-rate renders as every value.
//!
//! `gate` and `reset` are parameters (`Param::control`, or `Param::audio` for the audio-rate-gate
//! calc), changed at the start of the listed blocks.

use plyphon::{
    AddAction, Event, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const BLOCK: usize = 64;

/// One `DemandEnvGen` input: `Dseq(items, repeats)` or a constant.
enum Src {
    Seq(&'static [f32], f32),
    Const(f32),
    /// `Dwhite(lo, hi, inf)`, drawing from the synth's stream.
    White(f32, f32),
}

struct Spec {
    /// `.ar` (else `.kr`).
    ar: bool,
    /// `gate` and `reset` are audio-rate (else control-rate) parameters.
    audio_params: bool,
    level: Src,
    dur: Src,
    shape: Src,
    curve: Src,
    level_scale: f32,
    level_bias: f32,
    time_scale: f32,
    done_action: f32,
    gate: f32,
    reset: f32,
    /// `(block, value)` changes to `gate`.
    gate_changes: &'static [(usize, f32)],
    /// `(block, value)` changes to `reset`.
    reset_changes: &'static [(usize, f32)],
    blocks: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            ar: true,
            audio_params: false,
            level: Src::Const(0.0),
            dur: Src::Const(0.0),
            shape: Src::Const(1.0),
            curve: Src::Const(0.0),
            level_scale: 1.0,
            level_bias: 0.0,
            time_scale: 1.0,
            done_action: 0.0,
            gate: 1.0,
            reset: 1.0,
            gate_changes: &[],
            reset_changes: &[],
            blocks: 1,
        }
    }
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// What a render produced: the audio-rate output (every sample), or the control-rate output as
/// `K2A` shows it (the first sample of block `b` is the value computed in block `b - 1`, and of
/// block 0 the constructor's value); and whether the synth ended.
struct Rendered {
    out: Vec<f32>,
    ended: bool,
}

fn render(spec: Spec) -> Rendered {
    let mut units = Vec::new();
    let input = |src: &Src, units: &mut Vec<UnitSpec>| match *src {
        Src::Const(v) => c(v),
        Src::White(lo, hi) => {
            units.push(UnitSpec::new(
                "Dwhite",
                Rate::Demand,
                vec![c(f32::INFINITY), c(lo), c(hi)],
                1,
            ));
            u(units.len() as u32 - 1)
        }
        Src::Seq(items, repeats) => {
            let mut inputs = vec![c(repeats)];
            inputs.extend(items.iter().map(|&v| c(v)));
            units.push(UnitSpec::new("Dseq", Rate::Demand, inputs, 1));
            u(units.len() as u32 - 1)
        }
    };
    let level = input(&spec.level, &mut units);
    let dur = input(&spec.dur, &mut units);
    let shape = input(&spec.shape, &mut units);
    let curve = input(&spec.curve, &mut units);
    let env = units.len() as u32;
    units.push(UnitSpec::new(
        "DemandEnvGen",
        if spec.ar { Rate::Audio } else { Rate::Control },
        vec![
            level,
            dur,
            shape,
            curve,
            InputRef::Param(0),
            InputRef::Param(1),
            c(spec.level_scale),
            c(spec.level_bias),
            c(spec.time_scale),
            c(spec.done_action),
        ],
        1,
    ));
    let mut src = env;
    if !spec.ar {
        units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(env)], 1));
        src += 1;
    }
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0));
    let param = if spec.audio_params {
        Param::audio
    } else {
        Param::control
    };
    let (mut controller, mut nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "e".to_string(),
        params: vec![param("gate", spec.gate), param("reset", spec.reset)],
        units,
    });
    let node = controller
        .synth_new("e", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let blocks = if spec.ar {
        spec.blocks
    } else {
        spec.blocks + 1
    };
    let mut out = Vec::new();
    let mut buf = [0.0f32; BLOCK];
    for block in 0..blocks {
        for &(at, v) in spec.gate_changes {
            if at == block {
                controller.set_control(node, 0, v).expect("set_control");
            }
        }
        for &(at, v) in spec.reset_changes {
            if at == block {
                controller.set_control(node, 1, v).expect("set_control");
            }
        }
        world.fill(&mut buf, 1);
        if spec.ar {
            out.extend_from_slice(&buf);
        } else {
            out.push(buf[0]);
        }
    }
    nrt.process();
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        if matches!(event, Event::NodeEnded(n) if n.node == node) {
            ended = true;
        }
    }
    Rendered { out, ended }
}

/// FNV-1a over the samples' bit patterns (little-endian bytes).
fn hash(samples: &[f32]) -> u32 {
    let mut h = 2_166_136_261u32;
    for s in samples {
        let b = s.to_bits();
        for i in 0..4 {
            h ^= (b >> (8 * i)) & 0xff;
            h = h.wrapping_mul(16_777_619);
        }
    }
    h
}

fn bits(samples: &[f32]) -> Vec<u32> {
    samples.iter().map(|s| s.to_bits()).collect()
}

/// Every shape in turn: step, linear, exponential, sine, welch, curve -4, curve 0.0005 (linear),
/// squared, cubed, then an unknown shape (9999) that holds. The levels run out after ten segments:
/// the end level holds, the next calc releases and applies doneAction 2.
const SHAPE_LEVELS: &[f32] = &[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.95, 0.15];
const SHAPES: &[f32] = &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 9999.0];

#[test]
fn every_shape_at_audio_rate_then_done_action() {
    // The default reset of 1 resets in the constructor's calc, discarding a second pull of the
    // first level; the first segment is a step to 0.9.
    let r = render(Spec {
        level: Src::Seq(SHAPE_LEVELS, 1.0),
        dur: Src::Seq(
            &[
                0.0005, 0.0007, 0.0004, 0.0009, 0.0006, 0.0008, 0.0005, 0.0007, 0.0003, 0.0004,
            ],
            1.0,
        ),
        shape: Src::Seq(SHAPES, 1.0),
        curve: Src::Seq(&[0.0, 0.0, 0.0, 0.0, 0.0, -4.0, 0.0005, 0.0, 0.0, 0.0], 1.0),
        done_action: 2.0,
        blocks: 12,
        ..Spec::default()
    });
    // The release fires in block 5; the synth is freed after it.
    let live = 6 * BLOCK;
    assert_eq!(hash(&r.out[..live]), 0x86d6_68e7);
    let spots = [
        (0, 0x3f66_6666),
        (24, 0x3f60_e72f),
        (40, 0x3f08_f3c0),
        (57, 0x3e53_7f65),
        (80, 0x3f4b_9cfd),
        (100, 0x3efb_91f6),
        (130, 0x3f0b_231d),
        (150, 0x3f1e_7993),
        (170, 0x3ed8_b8d4),
        (200, 0x3f05_4beb),
        (230, 0x3f0a_5d33),
        (250, 0x3f32_93c3),
        (280, 0x3e19_999a),
        (383, 0x3e19_999a),
    ];
    for (i, b) in spots {
        assert_eq!(r.out[i].to_bits(), b, "sample {i}");
    }
    assert!(r.out[live..].iter().all(|&s| s == 0.0));
    assert!(r.ended);
}

#[test]
fn every_shape_at_control_rate_then_done_action() {
    // The same sequence at control rate (one step per block), with curve 4 and -0.0005.
    let r = render(Spec {
        ar: false,
        level: Src::Seq(SHAPE_LEVELS, 1.0),
        dur: Src::Seq(
            &[
                0.005, 0.007, 0.004, 0.009, 0.006, 0.008, 0.005, 0.007, 0.003, 0.004,
            ],
            1.0,
        ),
        shape: Src::Seq(SHAPES, 1.0),
        curve: Src::Seq(&[0.0, 0.0, 0.0, 0.0, 0.0, 4.0, -0.0005, 0.0, 0.0, 0.0], 1.0),
        done_action: 2.0,
        blocks: 60,
        ..Spec::default()
    });
    // The release fires in block 44, whose own value K2A would show only in block 45.
    let expected = [
        0x3f66_6666,
        0x3f66_6666,
        0x3f66_6666,
        0x3f66_6666,
        0x3f42_8f5c,
        0x3f1e_b852,
        0x3ef5_c28f,
        0x3eae_147b,
        0x3e4c_cccd,
        0x3ea2_8cc4,
        0x3f01_0413,
        0x3f4c_cccc,
        0x3fa2_8cc3,
        0x3f99_8350,
        0x3f81_088f,
        0x3f40_7a42,
        0x3f01_5ae6,
        0x3eae_280b,
        0x3e9b_e9a6,
        0x3ee5_0f3e,
        0x3f12_29ec,
        0x3f28_9601,
        0x3f32_c4a3,
        0x3f2f_55c8,
        0x3f2e_0ba1,
        0x3f2b_8895,
        0x3f26_a419,
        0x3f1d_1c97,
        0x3f0a_8d15,
        0x3ecc_cccd,
        0x3eee_eeef,
        0x3f08_8888,
        0x3f19_9999,
        0x3f2a_aaaa,
        0x3f20_1386,
        0x3f15_d337,
        0x3f0b_e9bd,
        0x3f02_5718,
        0x3ef2_3692,
        0x3f44_2c21,
        0x3f94_9997,
        0x3f94_9997,
        0x3f94_9997,
        0x3f94_9997,
        0x3e19_999a,
    ];
    assert_eq!(bits(&r.out[..expected.len()]), expected);
    assert!(r.out[expected.len()..].iter().all(|&s| s == 0.0));
    assert!(r.ended);
}

#[test]
fn scales_and_nan_shape_and_curve_fall_back() {
    // levelScale 2, levelBias 0.1, timeScale 1.5; no reset, so the constructor's level (0.5,
    // unscaled) starts the first segment. The shape stream (exponential, curve) and the curve
    // stream (3) run out: later segments keep the last shape and curve. A 0.0001 s segment is under
    // two samples, so it is linear; the levels run out at block 23, releasing with doneAction 0.
    let r = render(Spec {
        ar: false,
        level: Src::Seq(&[0.5, 1.0, 0.25, 2.0, 0.75, 0.1, 0.3], 1.0),
        dur: Src::Seq(&[0.004, 0.006, 0.0001, 0.001, 0.005, 0.004], 1.0),
        shape: Src::Seq(&[2.0, 5.0], 1.0),
        curve: Src::Seq(&[3.0], 1.0),
        level_scale: 2.0,
        level_bias: 0.1,
        time_scale: 1.5,
        reset: 0.0,
        blocks: 40,
        ..Spec::default()
    });
    let expected = [
        0x3f30_1475,
        0x3f72_3845,
        0x3fa6_9a0b,
        0x3fe5_2e6f,
        0x401d_a23d,
        0x401a_4a81,
        0x4015_28e5,
        0x400d_486c,
        0x4001_3103,
        0x3fdd_42b9,
        0x3fa4_468e,
        0x3f19_999a,
        0x4083_3333,
        0x3fcc_cccd,
        0x3fa4_5686,
        0x3f77_c080,
        0x3f26_d3f3,
        0x3eab_cecb,
        0x3c9f_5b0f,
        0x3e54_d3df,
        0x3eca_de2e,
        0x3f15_a936,
        0x3f45_e355,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
        0x3f33_3334,
    ];
    assert_eq!(bits(&r.out), expected);
    assert!(!r.ended);
}

/// Linear segments 0 -> 1 -> 0.5 -> 0 ... of 0.002 s and 0.003 s. `reset` rises to 1 at block 3
/// (discard the first level and restart), falls at 4, rises to 2 at block 6 (jump to the first
/// level), falls at 7. `gate` closes at block 9, reopens at 11 and goes to 0.5 at 13, releasing at
/// the end of that segment with doneAction 2.
fn reset_and_gate(audio_params: bool) -> Spec {
    Spec {
        audio_params,
        level: Src::Seq(&[0.0, 1.0, 0.5], f32::INFINITY),
        dur: Src::Seq(&[0.002, 0.003], f32::INFINITY),
        done_action: 2.0,
        reset: 0.0,
        gate_changes: &[(9, 0.0), (11, 1.0), (13, 0.5)],
        reset_changes: &[(3, 1.0), (4, 0.0), (6, 2.0), (7, 0.0)],
        blocks: 24,
        ..Spec::default()
    }
}

#[test]
fn a_control_rate_gate_resets_on_every_sample_of_a_rising_block() {
    // With a control-rate gate the previous reset is only updated after the block, so a rising
    // reset restarts the envelope on every sample of block 3 (each sample one step of a new segment
    // toward 1), and at block 6 jumps to 0 on every sample. Closing the gate at block 9 holds from
    // block 10; reopening at 11 runs from block 12. The release fires in block 14.
    let r = render(reset_and_gate(false));
    let live = 15 * BLOCK;
    assert_eq!(hash(&r.out[..live]), 0x0e27_4965);
    let spots = [
        (0, 0x3caa_aaaa),
        (96, 0x3f80_e05f),
        (192, 0x3f2b_d516),
        (193, 0x3f2c_b588),
        (194, 0x3f2d_93a4),
        (256, 0x3f54_f147),
        (384, 0x3c2a_aaaa),
        (385, 0x3c2a_aaaa),
        (416, 0x3c2a_aaaa),
        (448, 0x3caa_aaaa),
        (640, 0x3f2a_f247),
        (736, 0x3f2a_f247),
        (768, 0x3f2a_085b),
        (959, 0xbbab_3a62),
    ];
    for (i, b) in spots {
        assert_eq!(r.out[i].to_bits(), b, "sample {i}");
    }
    assert!(r.out[live..].iter().all(|&s| s == 0.0));
    assert!(r.ended);
}

#[test]
fn an_audio_rate_gate_reads_gate_and_reset_every_sample() {
    // With an audio-rate gate and reset the rising reset restarts the envelope once, on the first
    // sample of blocks 3 and 6. The release fires in block 13.
    let r = render(reset_and_gate(true));
    let live = 14 * BLOCK;
    assert_eq!(hash(&r.out[..live]), 0x4970_ca9b);
    let spots = [
        (0, 0x3caa_aaaa),
        (192, 0x3f2b_d516),
        (193, 0x3f2c_b7e5),
        (194, 0x3f2d_9ab4),
        (256, 0x3f64_88e6),
        (384, 0x3c2a_aaaa),
        (385, 0x3caa_aaaa),
        (416, 0x3eaf_ffff),
        (640, 0x3ed6_0a3f),
        (736, 0x3ed6_0a3f),
        (768, 0x3ed3_5d52),
        (895, 0xbbab_3a62),
    ];
    for (i, b) in spots {
        assert_eq!(r.out[i].to_bits(), b, "sample {i}");
    }
    assert!(r.out[live..].iter().all(|&s| s == 0.0));
    assert!(r.ended);
}

/// Curve-shape segments of 0.0005 s whose shape (always 5) and curve are drawn from `Dwhite`s.
fn drawn_shape_and_curve(audio_params: bool) -> Spec {
    Spec {
        audio_params,
        level: Src::Seq(&[0.2, 0.9, 0.1, 0.7], f32::INFINITY),
        dur: Src::Const(0.0005),
        shape: Src::White(5.0, 5.5),
        curve: Src::White(-4.0, 4.0),
        reset: 0.0,
        blocks: 4,
        ..Spec::default()
    }
}

#[test]
fn shape_is_demanded_before_curve_with_a_control_rate_gate() {
    let r = render(drawn_shape_and_curve(false));
    assert_eq!(hash(&r.out), 0xb53c_3065);
    let spots = [
        (0, 0x3ebf_1598),
        (16, 0x3f5c_ab4d),
        (32, 0x3f21_06d5),
        (64, 0x3f16_a07b),
        (128, 0x3f27_7828),
        (240, 0x3df6_a1ee),
    ];
    for (i, b) in spots {
        assert_eq!(r.out[i].to_bits(), b, "sample {i}");
    }
}

#[test]
fn curve_is_demanded_before_shape_with_an_audio_rate_gate() {
    // The constructor's calc (the first segment) still demands the shape first.
    let r = render(drawn_shape_and_curve(true));
    assert_eq!(hash(&r.out), 0x3f66_764b);
    let spots = [
        (0, 0x3ebf_1598),
        (16, 0x3f5c_ab4d),
        (32, 0x3eb2_8446),
        (64, 0x3f14_9cb2),
        (128, 0x3f08_495c),
        (240, 0x3e3d_6693),
    ];
    for (i, b) in spots {
        assert_eq!(r.out[i].to_bits(), b, "sample {i}");
    }
}

#[test]
fn a_dur_stream_that_ends_first_stops_without_releasing() {
    // Dseq([0.01], 1) durations: after the one linear segment the NaN duration stops the envelope
    // with a release that never comes due, so it holds (doneAction 2 never fires).
    let r = render(Spec {
        ar: false,
        level: Src::Seq(&[0.5, 1.0, 0.25], 1.0),
        dur: Src::Seq(&[0.01], 1.0),
        done_action: 2.0,
        reset: 0.0,
        blocks: 20,
        ..Spec::default()
    });
    let mut expected = vec![
        0x3f11_1111,
        0x3f22_2222,
        0x3f33_3333,
        0x3f44_4444,
        0x3f55_5555,
        0x3f66_6666,
        0x3f77_7777,
    ];
    expected.resize(21, 0x3f84_4444);
    assert_eq!(bits(&r.out), expected);
    assert!(!r.ended);
}

#[test]
fn a_closed_gate_holds_the_level() {
    // gate 0 at construction (running off), opened at block 5, closed at 12, reopened at 15: while
    // closed the level holds; the squared-shape segments continue where they stopped.
    let r = render(Spec {
        ar: false,
        level: Src::Seq(&[0.2, 0.6, 0.4], f32::INFINITY),
        dur: Src::Seq(&[0.004], f32::INFINITY),
        shape: Src::Const(6.0),
        gate: 0.0,
        reset: 0.0,
        gate_changes: &[(5, 1.0), (12, 0.0), (15, 1.0)],
        blocks: 24,
        ..Spec::default()
    });
    let expected = [
        0x3e4c_cccd,
        0x3e4c_cccd,
        0x3e4c_cccd,
        0x3e4c_cccd,
        0x3e4c_cccd,
        0x3e4c_cccd,
        0x3e4c_cccd,
        0x3e9e_78d5,
        0x3ee2_bd19,
        0x3f19_9999,
        0x3f47_ed91,
        0x3f13_1f91,
        0x3ecc_ccd0,
        0x3e83_842b,
        0x3e83_842b,
        0x3e83_842b,
        0x3e83_842b,
        0x3e69_01b8,
        0x3e4c_ccce,
        0x3e32_699a,
        0x3eb5_df5e,
        0x3f19_9996,
        0x3f68_981a,
        0x3f20_e9d5,
        0x3ecc_ccd5,
    ];
    assert_eq!(bits(&r.out), expected);
}
