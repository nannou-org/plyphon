//! Exercise control-rate unit outputs: a `Line.kr` ramp feeding a `SinOsc.ar` frequency input, so a
//! control-rate signal drives an audio-rate unit. The pitch should glide from 220 Hz to 660 Hz.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

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

/// `SinOsc.ar(Line.kr(220, 660, 1.0))` - the control-rate `Line` output feeds SinOsc's freq input.
fn glide_def() -> SynthDef {
    SynthDef {
        name: "glide".to_string(),
        params: vec![],
        units: vec![
            // Line.kr(220, 660, 1.0): a control-rate ramp (one output value per block).
            UnitSpec {
                name: "Line".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Constant(220.0),
                    InputRef::Constant(660.0),
                    InputRef::Constant(1.0),
                ],
                num_outputs: 1,
                special_index: 0,
            },
            // SinOsc.ar(freq = Line.kr output) - the freq input is control-rate.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
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

#[test]
fn control_rate_line_glides_sine_frequency() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(glide_def());
    controller
        .synth_new("glide", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    // A short window near the start: the ramp is still near 220 Hz.
    let early = render(&mut world, (SR * 0.05) as usize);
    // Advance well past the 1.0 s ramp so it holds at 660 Hz, then sample another short window.
    let _ = render(&mut world, (SR * 1.1) as usize);
    let late = render(&mut world, (SR * 0.05) as usize);

    assert!(
        goertzel(&early, 220.0) > 3.0 * goertzel(&early, 660.0),
        "early window should be near 220 Hz"
    );
    assert!(
        goertzel(&late, 660.0) > 3.0 * goertzel(&late, 220.0),
        "late window should be near 660 Hz"
    );
}
