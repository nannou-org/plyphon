//! Exercise the `LPF`/`HPF` Butterworth filters: a 200 Hz + 4000 Hz mix through a 1 kHz low-pass
//! keeps the low tone and attenuates the high one; a 1 kHz high-pass does the reverse.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, engine};

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

/// `(SinOsc.ar(200) + SinOsc.ar(4000))` through `filter.ar(in, 1000) -> Out`.
fn filtered_mix(filter: &str) -> SynthDef {
    SynthDef {
        name: "filtered".to_string(),
        params: vec![],
        ugens: vec![
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(200.0), InputRef::Constant(0.0)],
                1,
            ),
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(4000.0), InputRef::Constant(0.0)],
                1,
            ),
            // BinaryOpUGen add (special index 0): mix the two tones.
            UgenSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Ugen { ugen: 0, output: 0 },
                    InputRef::Ugen { ugen: 1, output: 0 },
                ],
                num_outputs: 1,
                special_index: 0,
            },
            // filter.ar(mix, 1000)
            UgenSpec::new(
                filter,
                Rate::Audio,
                vec![
                    InputRef::Ugen { ugen: 2, output: 0 },
                    InputRef::Constant(1000.0),
                ],
                1,
            ),
            UgenSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 3, output: 0 },
                ],
                0,
            ),
        ],
    }
}

fn render_filtered(filter: &str) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(filtered_mix(filter));
    controller
        .synth_new("filtered", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    // Skip the filter's start-up transient, then analyse.
    let _ = render(&mut world, SR as usize / 20);
    render(&mut world, SR as usize / 5)
}

#[test]
fn low_pass_keeps_low_tone() {
    let out = render_filtered("LPF");
    assert!(
        out.iter().all(|s| s.is_finite()),
        "filter output not finite"
    );
    let low = goertzel(&out, 200.0);
    let high = goertzel(&out, 4000.0);
    assert!(
        low > 8.0 * high,
        "LPF should keep 200 Hz, attenuate 4000 Hz: low={low}, high={high}"
    );
}

#[test]
fn high_pass_keeps_high_tone() {
    let out = render_filtered("HPF");
    assert!(
        out.iter().all(|s| s.is_finite()),
        "filter output not finite"
    );
    let low = goertzel(&out, 200.0);
    let high = goertzel(&out, 4000.0);
    assert!(
        high > 8.0 * low,
        "HPF should keep 4000 Hz, attenuate 200 Hz: low={low}, high={high}"
    );
}
