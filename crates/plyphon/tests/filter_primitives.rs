//! Exercise the primitive filters (`OnePole`, `OneZero`, `Integrator`, `LeakDC`, `TwoPole`,
//! `TwoZero`, `Decay`, `Decay2`) and the resonant biquads (`RLPF`, `RHPF`, `BPF`, `BRF`, `Resonz`,
//! `Ringz`) through the engine: each is checked for the frequency emphasis or time-domain shape
//! scsynth's kernel produces.

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

/// Play `units` (the last of which must be an `Out`) and render, skipping the start-up transient.
fn run(units: Vec<UnitSpec>) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let _ = render(&mut world, SR as usize / 20);
    render(&mut world, SR as usize / 5)
}

fn sin(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

fn out(unit: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![InputRef::Constant(0.0), InputRef::Unit { unit, output: 0 }],
        0,
    )
}

/// `filter.ar((SinOsc.ar(200) + SinOsc.ar(4000)), ..extra) -> Out`. Unit 3 is the filter.
fn on_mix(filter: &str, extra: Vec<InputRef>) -> Vec<f32> {
    let mut inputs = vec![InputRef::Unit { unit: 2, output: 0 }];
    inputs.extend(extra);
    run(vec![
        sin(200.0),
        sin(4000.0),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
            ],
            num_outputs: 1,
            special_index: 0, // add
        },
        UnitSpec::new(filter, Rate::Audio, inputs, 1),
        out(3),
    ])
}

fn ratio(out: &[f32]) -> (f32, f32) {
    (goertzel(out, 200.0), goertzel(out, 4000.0))
}

#[test]
fn one_pole_lowpasses_with_positive_coef() {
    let o = on_mix("OnePole", vec![InputRef::Constant(0.9)]);
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        low > 4.0 * high,
        "OnePole(0.9) should favour 200: {low} vs {high}"
    );
}

#[test]
fn one_zero_highpasses_with_negative_coef() {
    let o = on_mix("OneZero", vec![InputRef::Constant(-0.5)]);
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        high > 4.0 * low,
        "OneZero(-0.5) should favour 4000: {low} vs {high}"
    );
}

#[test]
fn integrator_boosts_lows() {
    let o = on_mix("Integrator", vec![InputRef::Constant(0.9)]);
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        low > 4.0 * high,
        "Integrator(0.9) should favour 200: {low} vs {high}"
    );
}

#[test]
fn two_pole_resonates_at_its_centre() {
    let o = on_mix(
        "TwoPole",
        vec![InputRef::Constant(4000.0), InputRef::Constant(0.95)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        high > 4.0 * low,
        "TwoPole@4000 should favour 4000: {low} vs {high}"
    );
}

#[test]
fn two_zero_notches_its_centre() {
    let o = on_mix(
        "TwoZero",
        vec![InputRef::Constant(4000.0), InputRef::Constant(0.99)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        low > 4.0 * high,
        "TwoZero@4000 should notch 4000: {low} vs {high}"
    );
}

#[test]
fn leak_dc_removes_dc_but_keeps_the_tone() {
    // (SinOsc.ar(200) * 0.5 + 0.5) has a 0.5 DC offset; LeakDC should strip it while keeping 200 Hz.
    let out = run(vec![
        sin(200.0),
        // SinOsc * 0.5
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.5),
            ],
            num_outputs: 1,
            special_index: 2, // mul
        },
        // + 0.5 DC
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(0.5),
            ],
            num_outputs: 1,
            special_index: 0, // add
        },
        UnitSpec::new(
            "LeakDC",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Constant(0.995),
            ],
            1,
        ),
        out(3),
    ]);
    assert!(out.iter().all(|s| s.is_finite()));
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    assert!(
        mean.abs() < 0.02,
        "LeakDC should remove the 0.5 DC offset, mean {mean}"
    );
    assert!(
        goertzel(&out, 200.0) > 0.1,
        "the 200 Hz tone should survive DC blocking"
    );
}

#[test]
fn decay_turns_impulses_into_one_sided_decays() {
    // Impulse.ar(50) -> Decay(0.05): a train of positive exponential decays, never negative.
    let out = run(vec![
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(50.0), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec::new(
            "Decay",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.05),
            ],
            1,
        ),
        out(1),
    ]);
    assert!(out.iter().all(|s| s.is_finite()));
    assert!(
        out.iter().all(|&s| s >= -1e-6),
        "Decay of positive impulses stays non-negative"
    );
    let energy = out.iter().map(|s| s * s).sum::<f32>();
    assert!(energy > 0.1, "the decays should carry energy, got {energy}");
}

#[test]
fn decay2_shapes_a_smooth_attack_decay() {
    // Impulse.ar(20) -> Decay2(0.01, 0.2): each impulse becomes an attack-then-decay bump. The
    // attack smooths the onset, so the peak is well below the raw impulse's 1.0.
    let out = run(vec![
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(20.0), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec::new(
            "Decay2",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.01),
                InputRef::Constant(0.2),
            ],
            1,
        ),
        out(1),
    ]);
    assert!(out.iter().all(|s| s.is_finite()));
    // The slow (decay) branch never falls below the fast (attack) branch, so the envelope is
    // non-negative; the difference-of-decays stays bounded well within unity-ish range.
    assert!(
        out.iter().all(|&s| s >= -1e-6),
        "the attack-decay envelope stays non-negative"
    );
    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s));
    assert!(
        peak > 0.01 && peak < 1.5,
        "envelope peak should be audible and bounded, got {peak}"
    );
    let energy = out.iter().map(|s| s * s).sum::<f32>();
    assert!(
        energy > 0.01,
        "the envelope should carry energy, got {energy}"
    );
}

#[test]
fn rlpf_keeps_lows() {
    let o = on_mix(
        "RLPF",
        vec![InputRef::Constant(1000.0), InputRef::Constant(1.0)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        low > 4.0 * high,
        "RLPF@1000 should keep 200: {low} vs {high}"
    );
}

#[test]
fn rhpf_keeps_highs() {
    let o = on_mix(
        "RHPF",
        vec![InputRef::Constant(1000.0), InputRef::Constant(1.0)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        high > 4.0 * low,
        "RHPF@1000 should keep 4000: {low} vs {high}"
    );
}

#[test]
fn bpf_passes_its_band() {
    let o = on_mix(
        "BPF",
        vec![InputRef::Constant(4000.0), InputRef::Constant(1.0)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        high > 4.0 * low,
        "BPF@4000 should pass 4000, reject 200: {low} vs {high}"
    );
}

#[test]
fn brf_rejects_its_band() {
    let o = on_mix(
        "BRF",
        vec![InputRef::Constant(4000.0), InputRef::Constant(1.0)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        low > 4.0 * high,
        "BRF@4000 should reject 4000, keep 200: {low} vs {high}"
    );
}

#[test]
fn resonz_boosts_its_centre() {
    let o = on_mix(
        "Resonz",
        vec![InputRef::Constant(4000.0), InputRef::Constant(0.1)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        high > 4.0 * low,
        "Resonz@4000 should boost 4000: {low} vs {high}"
    );
}

#[test]
fn ringz_rings_at_its_frequency() {
    let o = on_mix(
        "Ringz",
        vec![InputRef::Constant(4000.0), InputRef::Constant(0.1)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        high > 4.0 * low,
        "Ringz@4000 should ring at 4000: {low} vs {high}"
    );
}
