//! Build a SynthDef in SCgf form (as `sclang` would emit, with a `Control` UGen for `freq`), encode
//! it via the `scgf` crate, load it through plyphon's converter, and confirm it folds into plyphon's
//! parameter model and plays.

use plyphon::synthdef::read::parse;
use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, engine};
use scgf::{Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen, encode};

const SR: f32 = 48_000.0;

/// `SynthDef(\sine, { |freq=440| Out.ar(0, SinOsc.ar(freq)) })` in SCgf form.
fn sine_scgf() -> Vec<u8> {
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "sine".to_string(),
            constants: vec![0.0, 0.0], // phase, bus
            param_values: vec![440.0],
            param_names: vec![ParamName {
                name: "freq".to_string(),
                index: 0,
            }],
            ugens: vec![
                Ugen {
                    name: "Control".to_string(),
                    rate: Rate::Control,
                    special_index: 0,
                    inputs: vec![],
                    outputs: vec![Rate::Control],
                },
                Ugen {
                    name: "SinOsc".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Ugen { ugen: 0, output: 0 }, // freq <- Control
                        Input::Constant { index: 0 },       // phase <- 0
                    ],
                    outputs: vec![Rate::Audio],
                },
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 1 },       // bus <- 0
                        Input::Ugen { ugen: 1, output: 0 }, // sig <- SinOsc
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![],
        }],
    };
    encode(&file).expect("encode SCgf")
}

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR).floor();
    let w = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in samples {
        let s = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
}

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

#[test]
fn scgf_folds_control_into_param_and_plays() {
    let bytes = sine_scgf();
    let defs = parse(&bytes).expect("load SCgf");
    assert_eq!(defs.len(), 1);
    let def = &defs[0];

    // The Control UGen folded into a `freq` parameter; only SinOsc and Out survive.
    assert_eq!(def.name, "sine");
    assert_eq!(def.params.len(), 1);
    assert_eq!(def.params[0].name, "freq");
    assert_eq!(def.params[0].default, 440.0);
    assert_eq!(def.units.len(), 2);
    assert_eq!(def.units[0].name, "SinOsc");
    assert_eq!(def.units[1].name, "Out");
    // SinOsc's freq input now references the folded parameter.
    assert!(matches!(def.units[0].inputs[0], InputRef::Param(0)));

    // Instantiate and play it: 440 Hz, then retune to 330 Hz via the folded parameter.
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def.clone());
    let node = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 440.0) > 5.0 * goertzel(&a, 880.0),
        "expected 440 Hz"
    );

    controller.set_control(node, 0, 330.0).expect("set_control");
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&b, 330.0) > 5.0 * goertzel(&b, 660.0),
        "expected 330 Hz"
    );
}
