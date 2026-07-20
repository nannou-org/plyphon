//! `FreeVerb`/`FreeVerb2`: the freeverb reverb - eight parallel damped combs into four series
//! allpasses, mono and true-stereo.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

fn out_ch(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
}

fn rms(s: &[f32]) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt()
}

fn render(units: Vec<UnitSpec>, channels: usize, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "r".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames * channels];
    world.fill(&mut out, channels);
    out
}

#[test]
fn freeverb_rings_out_and_decays() {
    // A single impulse into a fully-wet reverb produces a decaying tail that outlasts the impulse.
    let out = render(
        vec![
            UnitSpec::new("Impulse", Rate::Audio, vec![c(1.0), c(0.0)], 1),
            UnitSpec::new(
                "FreeVerb",
                Rate::Audio,
                vec![u(0), c(1.0), c(0.8), c(0.5)], // mix=1 (wet), room=0.8, damp=0.5
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 2,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "FreeVerb must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 4.0), "FreeVerb stays bounded");

    // A tail exists well after the impulse, and it decays (early tail louder than late tail).
    let n = out.len();
    let early = rms(&out[n / 8..n / 4]);
    let late = rms(&out[3 * n / 4..]);
    assert!(early > 1e-4, "the reverb tail should sound (early={early})");
    assert!(
        early > 2.0 * late,
        "the reverb should decay (early={early}, late={late})"
    );
}

#[test]
fn freeverb_dry_passes_the_input() {
    // mix = 0 is fully dry: the output equals the input sample for sample.
    let out = render(
        vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            UnitSpec::new(
                "FreeVerb",
                Rate::Audio,
                vec![u(0), c(0.0), c(0.5), c(0.5)],
                1,
            ),
            // reference: the same SinOsc straight out on a second synth path is awkward; instead
            // compare against a freshly-built dry SinOsc of the same phase below.
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        256,
    );
    let dry = render(
        vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
        1,
        256,
    );
    for (i, (a, b)) in out.iter().zip(&dry).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "dry FreeVerb should equal the input at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn freeverb2_spreads_to_stereo() {
    // FreeVerb2 takes two inputs and yields two decorrelated reverb channels.
    let out = render(
        vec![
            UnitSpec::new("Impulse", Rate::Audio, vec![c(1.0), c(0.0)], 1),
            UnitSpec::new("Impulse", Rate::Audio, vec![c(1.3), c(0.0)], 1),
            UnitSpec::new(
                "FreeVerb2",
                Rate::Audio,
                vec![u(0), u(1), c(1.0), c(0.8), c(0.5)],
                2,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![c(0.0), out_ch(2, 0), out_ch(2, 1)],
                0,
            ),
        ],
        2,
        SR as usize / 2,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "FreeVerb2 must stay finite"
    );
    assert!(
        out.iter().all(|&s| s.abs() < 4.0),
        "FreeVerb2 stays bounded"
    );
    let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
    let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
    assert!(
        rms(&ch0) > 1e-4 && rms(&ch1) > 1e-4,
        "both reverb channels should sound"
    );
    assert!(
        ch0.iter().zip(&ch1).any(|(a, b)| (a - b).abs() > 1e-4),
        "the two reverb channels should differ (stereo)"
    );
}
