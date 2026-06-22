//! Drive the engine entirely through OSC: receive a SynthDef with `/d_recv`, start it with
//! `/s_new`, retune it with `/n_set` (by control name), and stop it with `/n_free` - checking the
//! audio out of a `World` at each step.

use plyphon::{Options, ROOT_GROUP_ID, engine};
use plyphon_osc::OscDispatcher;
use rosc::{OscMessage, OscPacket, OscType};
use scgf::{Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen};

const SR: f32 = 48_000.0;

fn sine_scgf() -> Vec<u8> {
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "sine".to_string(),
            constants: vec![0.0, 0.0],
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
                        Input::Ugen { ugen: 0, output: 0 },
                        Input::Constant { index: 0 },
                    ],
                    outputs: vec![Rate::Audio],
                },
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 1 },
                        Input::Ugen { ugen: 1, output: 0 },
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![],
        }],
    };
    scgf::encode(&file).expect("encode SCgf")
}

fn osc(addr: &str, args: Vec<OscType>) -> Vec<u8> {
    rosc::encoder::encode(&OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    }))
    .expect("encode OSC")
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
fn drives_engine_over_osc() {
    let (controller, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut dispatcher = OscDispatcher::new(controller);

    // Receive the SynthDef and start a synth (node 1000) at the tail of the root group.
    dispatcher
        .apply_bytes(&osc("/d_recv", vec![OscType::Blob(sine_scgf())]))
        .expect("/d_recv");
    dispatcher
        .apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(1000),
                OscType::Int(1), // addToTail
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .expect("/s_new");

    let a = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&a, 440.0) > 5.0 * goertzel(&a, 880.0),
        "expected 440 Hz"
    );

    // Retune by control name.
    dispatcher
        .apply_bytes(&osc(
            "/n_set",
            vec![
                OscType::Int(1000),
                OscType::String("freq".to_string()),
                OscType::Float(330.0),
            ],
        ))
        .expect("/n_set");
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 2);
    assert!(
        goertzel(&b, 330.0) > 5.0 * goertzel(&b, 660.0),
        "expected 330 Hz"
    );

    // Free the node; after flushing the in-flight block the output is silent.
    dispatcher
        .apply_bytes(&osc("/n_free", vec![OscType::Int(1000)]))
        .expect("/n_free");
    let _ = render(&mut world, 1024);
    let c = render(&mut world, SR as usize / 4);
    assert!(
        c.iter().all(|s| s.abs() < 1e-6),
        "expected silence after /n_free"
    );

    dispatcher.controller().drain_trash();
}
