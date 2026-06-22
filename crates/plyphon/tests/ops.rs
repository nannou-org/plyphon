//! Exercise `BinaryOpUGen` and `UnaryOpUGen`: an amplitude-scaled sine (`SinOsc.ar(freq) * amp`)
//! and a rectified sine (`SinOsc.ar(freq).abs()`), plus the unsupported-operator error path.

use plyphon::controller::SynthNewError;
use plyphon::error::BuildError;
use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, engine,
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
        params: vec![
            Param {
                name: "freq".to_string(),
                default: 440.0,
            },
            Param {
                name: "amp".to_string(),
                default: 0.5,
            },
        ],
        ugens: vec![
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            // BinaryOpUGen, special index 2 = multiply: SinOsc * amp.
            UgenSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Ugen { ugen: 0, output: 0 }, InputRef::Param(1)],
                num_outputs: 1,
                special_index: 2,
            },
            UgenSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 1, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn binary_op_multiply_scales_amplitude() {
    let (mut controller, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(amped_sine());
    let node = controller
        .synth_new("amped", ROOT_GROUP_ID, AddAction::Tail)
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
        params: vec![Param {
            name: "freq".to_string(),
            default: 440.0,
        }],
        ugens: vec![
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            // UnaryOpUGen, special index 5 = abs.
            UgenSpec {
                name: "UnaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Ugen { ugen: 0, output: 0 }],
                num_outputs: 1,
                special_index: 5,
            },
            UgenSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 1, output: 0 },
                ],
                0,
            ),
        ],
    };
    let (mut controller, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    controller
        .synth_new("rect", ROOT_GROUP_ID, AddAction::Tail)
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
fn unsupported_op_is_rejected() {
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![],
        ugens: vec![UgenSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![InputRef::Constant(1.0), InputRef::Constant(2.0)],
            num_outputs: 1,
            special_index: 999,
        }],
    };
    let (mut controller, _world) = engine(Options::default());
    controller.add_synthdef(def);
    let result = controller.synth_new("bad", ROOT_GROUP_ID, AddAction::Tail);
    assert!(matches!(
        result,
        Err(SynthNewError::Build(BuildError::UnsupportedOp(999)))
    ));
}
