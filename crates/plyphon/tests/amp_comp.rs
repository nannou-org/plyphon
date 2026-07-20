//! `AmpComp`/`AmpCompA`: frequency-dependent amplitude compensation.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// The steady output value of `unit` (a constant-parameter compensation).
fn value(unit: UnitSpec) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "a".to_string(),
        params: vec![],
        units: vec![
            unit,
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    });
    controller
        .synth_new("a", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; 64];
    world.fill(&mut out, 1);
    out[63]
}

fn amp_comp(freq: f32, root: f32, exp: f32) -> UnitSpec {
    UnitSpec::new("AmpComp", Rate::Audio, vec![c(freq), c(root), c(exp)], 1)
}

fn amp_comp_a(freq: f32, root: f32, min_amp: f32, root_amp: f32) -> UnitSpec {
    UnitSpec::new(
        "AmpCompA",
        Rate::Audio,
        vec![c(freq), c(root), c(min_amp), c(root_amp)],
        1,
    )
}

#[test]
fn amp_comp_is_unity_at_root_and_falls_with_freq() {
    // (root/freq)^exp: at freq == root the gain is 1; above root it drops.
    assert!(
        (amp_comp_value_unity() - 1.0).abs() < 1e-4,
        "AmpComp is unity at the root frequency"
    );
    let octave_up = value(amp_comp(880.0, 440.0, 0.3333));
    // (1/2)^(1/3) ~= 0.7937.
    assert!(
        (octave_up - 0.7937).abs() < 0.01,
        "an octave up compensates to ~0.794 (got {octave_up})"
    );
    assert!(octave_up < 1.0, "higher freqs get less gain");
}

fn amp_comp_value_unity() -> f32 {
    value(amp_comp(440.0, 440.0, 0.3333))
}

#[test]
fn amp_comp_a_hits_root_amp_at_root() {
    // The A-weighting curve is rescaled so the gain equals `rootAmp` exactly at the root frequency.
    let at_root = value(amp_comp_a(1000.0, 1000.0, 0.32, 1.0));
    assert!(
        (at_root - 1.0).abs() < 1e-3,
        "AmpCompA equals rootAmp (1.0) at the root (got {at_root})"
    );
    // A very different frequency gives a different, finite, positive gain.
    let elsewhere = value(amp_comp_a(120.0, 1000.0, 0.32, 1.0));
    assert!(
        elsewhere.is_finite() && elsewhere > 0.0,
        "finite gain: {elsewhere}"
    );
    assert!(
        (elsewhere - at_root).abs() > 1e-3,
        "the curve varies with frequency (root={at_root}, other={elsewhere})"
    );
}
