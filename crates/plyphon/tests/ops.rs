//! Exercise `BinaryOpUGen` and `UnaryOpUGen`: an amplitude-scaled sine (`SinOsc.ar(freq) * amp`)
//! and a rectified sine (`SinOsc.ar(freq).abs()`), plus the unsupported-operator error path.

use plyphon::BuildError;
use plyphon::controller::SynthNewError;
use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f32 = 48_000.0;

fn render(world: &mut plyphon::World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

/// `SinOsc.ar(440) * amp`, with `amp` a control parameter.
fn amped_sine() -> SynthDef {
    SynthDef {
        name: "amped".to_string(),
        params: vec![Param::control("freq", 440.0), Param::control("amp", 0.5)],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            // BinaryOpUGen, special index 2 = multiply: SinOsc * amp.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Unit { unit: 0, output: 0 }, InputRef::Param(1)],
                num_outputs: 1,
                special_index: 2,
            },
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    }
}

/// Render one block of `DC(a) <op> DC(b)` (or unary `<op>(DC(a))` when `b` is `None`).
fn op_block(special_index: i16, a: f32, b: Option<f32>) -> f32 {
    let (name, inputs) = match b {
        Some(b) => (
            "BinaryOpUGen",
            vec![InputRef::Unit { unit: 0, output: 0 }, InputRef::Constant(b)],
        ),
        None => ("UnaryOpUGen", vec![InputRef::Unit { unit: 0, output: 0 }]),
    };
    let def = SynthDef {
        name: "op".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(a)], 1),
            UnitSpec {
                name: name.to_string(),
                rate: Rate::Audio,
                inputs,
                num_outputs: 1,
                special_index,
            },
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    controller
        .synth_new("op", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let mut blk = [0.0f32; 64];
    world.fill(&mut blk, 1);
    assert!(
        blk.windows(2).all(|w| w[0] == w[1]),
        "constant operands must give a constant block (op {special_index})"
    );
    blk[0]
}

#[test]
fn hot_op_dispatch_matches_definitions() {
    // The monomorphised fast paths (and a couple of fn-pointer fallbacks) against their
    // definitions - guards the dispatch's index mapping.
    let (a, b) = (0.75f32, 0.4f32);
    let binary: &[(i16, f32)] = &[
        (0, a + b),
        (1, a - b),
        (2, a * b),
        (4, a / b),
        (12, a.min(b)),
        (13, a.max(b)),
        (5, a % b),          // fallback path (opMod)
        (38, (a - b).abs()), // fallback path (opAbsDif)
    ];
    for &(op, expected) in binary {
        let got = op_block(op, a, Some(b));
        assert!(
            (got - expected).abs() < 1e-6,
            "binary op {op}: got {got}, expected {expected}"
        );
    }
    let neg = -0.6f32;
    let unary: &[(i16, f32, f32)] = &[
        (0, a, -a),
        (5, neg, neg.abs()),
        (12, a, a * a),
        (13, a, a * a * a),
        (15, a, a.exp()), // fallback path (opExp)
    ];
    for &(op, x, expected) in unary {
        let got = op_block(op, x, None);
        assert!(
            (got - expected).abs() < 1e-6,
            "unary op {op}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn binary_op_multiply_scales_amplitude() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(amped_sine());
    let node = controller
        .synth_new("amped", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let a = render(&mut world, SR as usize / 8);
    assert!((peak(&a) - 0.5).abs() < 0.05, "peak {} != ~0.5", peak(&a));

    // Drop the amplitude via the `amp` parameter (index 1).
    controller.set_control(node, 1, 0.25).expect("set_control");
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 8);
    assert!((peak(&b) - 0.25).abs() < 0.05, "peak {} != ~0.25", peak(&b));
}

#[test]
fn unary_op_abs_rectifies() {
    let def = SynthDef {
        name: "rect".to_string(),
        params: vec![Param::control("freq", 440.0)],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            // UnaryOpUGen, special index 5 = abs.
            UnitSpec {
                name: "UnaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Unit { unit: 0, output: 0 }],
                num_outputs: 1,
                special_index: 5,
            },
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    controller
        .synth_new("rect", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let out = render(&mut world, SR as usize / 8);
    assert!(
        out.iter().all(|&x| x >= -1e-6),
        "rectified output went negative"
    );
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    assert!(
        mean > 0.2,
        "rectified sine should have a positive DC offset, got {mean}"
    );
}

#[test]
fn unary_op_midicps_converts_constant() {
    // UnaryOpUGen special index 17 = midicps, applied to the constant MIDI note 69 -> 440 Hz. With a
    // constant (scalar) input the op evaluates once and fills the block, so `Out.ar` emits a steady
    // 440.0 DC signal.
    let def = SynthDef {
        name: "mc".to_string(),
        params: vec![],
        units: vec![
            UnitSpec {
                name: "UnaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Constant(69.0)],
                num_outputs: 1,
                special_index: 17,
            },
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
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    controller
        .synth_new("mc", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let out = render(&mut world, 512);
    assert!(
        out.iter().all(|&x| (x - 440.0).abs() < 1e-2),
        "midicps(69) should emit a steady 440.0, got e.g. {}",
        out[0]
    );
}

#[test]
fn binary_op_clip2_bounds_a_loud_sine() {
    // SinOsc.ar(440) * 2 clipped to ±0.5 via BinaryOpUGen special index 42 (clip2). The audio-rate
    // signal against a constant bound must clip cleanly without panicking.
    let def = SynthDef {
        name: "clipped".to_string(),
        params: vec![Param::control("freq", 440.0)],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(2.0),
                ],
                num_outputs: 1,
                special_index: 2, // multiply -> amplitude 2.0
            },
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.5),
                ],
                num_outputs: 1,
                special_index: 42, // clip2 -> ±0.5
            },
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    controller
        .synth_new("clipped", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let out = render(&mut world, SR as usize / 8);
    assert!(
        out.iter().all(|&x| x.abs() <= 0.5 + 1e-6),
        "clip2 must bound the signal to ±0.5, peak {}",
        peak(&out)
    );
    // The 2x sine overshoots ±0.5 most of the time, so the bound is actually exercised.
    assert!(
        (peak(&out) - 0.5).abs() < 1e-3,
        "clipping should reach the ±0.5 bound, peak {}",
        peak(&out)
    );
}

#[test]
fn unsupported_op_is_rejected() {
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![],
        units: vec![UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![InputRef::Constant(1.0), InputRef::Constant(2.0)],
            num_outputs: 1,
            special_index: 999,
        }],
    };
    let (mut controller, _nrt, _world) = engine(Options::default());
    controller.add_synthdef(def);
    let result = controller.synth_new("bad", ROOT_GROUP_ID, AddAction::Tail, &[]);
    assert!(matches!(
        result,
        Err(SynthNewError::Build(BuildError::UnsupportedOp(999)))
    ));
}
