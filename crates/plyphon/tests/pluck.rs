//! `Pluck`: a Karplus-Strong plucked string - a cubic comb whose feedback runs through a one-zero
//! damping filter, excited by a noise burst gated in for one delay period on each rising trigger.

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

fn rms(s: &[f32]) -> f32 {
    (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt()
}

/// `WhiteNoise -> Pluck(in, Impulse(trig_hz), maxdelay, 1/freq, decay, coef) -> Out`.
fn render(freq: f32, trig_hz: f32, decay: f32, coef: f32, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "p".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new("Impulse", Rate::Audio, vec![c(trig_hz), c(0.0)], 1),
            UnitSpec::new(
                "Pluck",
                Rate::Audio,
                vec![u(0), u(1), c(0.05), c(1.0 / freq), c(decay), c(coef)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
        ],
    });
    controller
        .synth_new("p", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}

#[test]
fn pluck_rings_at_pitch_and_decays() {
    // One pluck at the start (Impulse fires once over the render); the string rings near its
    // fundamental `1/delaytime` = 220 Hz and dies away as the damping filter dulls each period.
    let out = render(220.0, 1.0, 3.0, 0.3, SR as usize / 4);
    assert!(out.iter().all(|s| s.is_finite()), "Pluck must stay finite");
    assert!(out.iter().all(|&s| s.abs() < 4.0), "Pluck stays bounded");

    let at = goertzel(&out, 220.0);
    assert!(
        at > 4.0 * goertzel(&out, 330.0),
        "the string should ring near 220 (220={at}, 330={})",
        goertzel(&out, 330.0)
    );
    assert!(at > 0.01, "the pluck should sound (220={at})");

    // It decays: the onset is louder than the tail.
    let quarter = out.len() / 4;
    let onset = rms(&out[..quarter]);
    let tail = rms(&out[out.len() - quarter..]);
    assert!(
        onset > 1.5 * tail,
        "the pluck should decay (onset={onset}, tail={tail})"
    );
}

#[test]
fn pluck_is_silent_without_a_trigger() {
    // With the trigger held at 0, no excitation ever enters the line, so the string stays silent.
    let out = render_no_trig(220.0, 2.0, 0.5, SR as usize / 8);
    assert!(
        out.iter().all(|&s| s.abs() < 1e-6),
        "an un-plucked string is silent"
    );
}

/// Like [`render`] but the trigger is a constant 0 (never plucks).
fn render_no_trig(freq: f32, decay: f32, coef: f32, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "p".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new(
                "Pluck",
                Rate::Audio,
                vec![u(0), c(0.0), c(0.05), c(1.0 / freq), c(decay), c(coef)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
    });
    controller
        .synth_new("p", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}
