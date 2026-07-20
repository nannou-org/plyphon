//! `PitchShift`: a granular pitch shifter. Four overlapping windowed grains replay the recent input
//! faster or slower than it arrived, so a tone comes out transposed by `pitchRatio`.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
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

/// `SinOsc(freq) -> PitchShift(in, 0.2, pitch_ratio, 0, 0) -> Out`.
fn render(freq: f32, pitch_ratio: f32, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "ps".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(freq), c(0.0)], 1),
            UnitSpec::new(
                "PitchShift",
                Rate::Audio,
                vec![u(0), c(0.2), c(pitch_ratio), c(0.0), c(0.0)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
    });
    controller
        .synth_new("ps", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}

#[test]
fn pitch_shift_transposes_up_an_octave() {
    // A 220 Hz sine pitched up by 2 comes out dominated by 440 Hz.
    let out = render(220.0, 2.0, SR as usize / 2);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "PitchShift must stay finite"
    );
    assert!(
        out.iter().all(|&s| s.abs() < 4.0),
        "PitchShift stays bounded"
    );
    let up = goertzel(&out, 440.0);
    let orig = goertzel(&out, 220.0);
    // A granular shifter leaks some energy at the original pitch; the transposed peak still dominates.
    assert!(
        up > 2.0 * orig,
        "pitchRatio 2 should transpose 220 up to 440 (440={up}, 220={orig})"
    );
    assert!(up > 0.05, "the shifted tone should sound (440={up})");
}

#[test]
fn pitch_shift_down_an_octave() {
    // pitchRatio 0.5 drops 440 to 220.
    let out = render(440.0, 0.5, SR as usize / 2);
    let down = goertzel(&out, 220.0);
    let orig = goertzel(&out, 440.0);
    assert!(
        down > 2.0 * orig,
        "pitchRatio 0.5 should transpose 440 down to 220 (220={down}, 440={orig})"
    );
}

#[test]
fn pitch_shift_unity_preserves_pitch() {
    // pitchRatio 1 leaves the pitch alone: the 330 Hz input stays at 330.
    let out = render(330.0, 1.0, SR as usize / 2);
    let at = goertzel(&out, 330.0);
    assert!(
        at > 4.0 * goertzel(&out, 660.0) && at > 4.0 * goertzel(&out, 165.0),
        "unity ratio keeps 330 (330={at})"
    );
    assert!(
        at > 0.1,
        "unity pitch shift should pass the tone (330={at})"
    );
}
