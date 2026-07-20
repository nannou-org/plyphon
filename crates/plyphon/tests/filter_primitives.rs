//! Exercise the primitive filters (`OnePole`, `OneZero`, `Integrator`, `LeakDC`, `TwoPole`,
//! `TwoZero`, `Decay`, `Decay2`), the resonant biquads (`RLPF`, `RHPF`, `BPF`, `BRF`, `Resonz`,
//! `Ringz`) and the fixed-coefficient/delay filters (`LPZ1`, `HPZ1`, `LPZ2`, `HPZ2`, `BPZ2`, `BRZ2`,
//! `Slope`, `Delay1`, `Delay2`, `Slew`, `APF`) through the engine: each is checked for the frequency
//! emphasis or time-domain shape scsynth's kernel produces.

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
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail, &[])
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

/// `filter.ar((SinOsc.ar(f1) + SinOsc.ar(f2)), ..extra) -> Out`, returning `(|f1|, |f2|)`.
fn on_two(filter: &str, f1: f32, f2: f32, extra: Vec<InputRef>) -> (f32, f32) {
    let mut inputs = vec![InputRef::Unit { unit: 2, output: 0 }];
    inputs.extend(extra);
    let out = run(vec![
        sin(f1),
        sin(f2),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
            ],
            num_outputs: 1,
            special_index: 0,
        },
        UnitSpec::new(filter, Rate::Audio, inputs, 1),
        out(3),
    ]);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "{filter} output not finite"
    );
    (goertzel(&out, f1), goertzel(&out, f2))
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

#[test]
fn lpz1_and_lpz2_favour_lows() {
    // These are gentle low-passes with the cutoff at Nyquist, so they only discriminate against a
    // near-Nyquist tone: 1 kHz survives, 22 kHz is attenuated.
    for f in ["LPZ1", "LPZ2"] {
        let (low, high) = on_two(f, 1000.0, 22_000.0, vec![]);
        assert!(
            low > 4.0 * high,
            "{f} should favour 1 kHz over 22 kHz: {low} vs {high}"
        );
    }
}

#[test]
fn hpz1_and_hpz2_favour_highs() {
    for f in ["HPZ1", "HPZ2"] {
        let o = on_mix(f, vec![]);
        assert!(o.iter().all(|s| s.is_finite()));
        let (low, high) = ratio(&o);
        assert!(high > 4.0 * low, "{f} should favour 4000: {low} vs {high}");
    }
}

#[test]
fn bpz2_favours_the_mid_band() {
    // BPZ2's `0.5*(x0 - x2)` has zeros at DC and Nyquist and a peak at SR/4, so of 200 vs 4000 it
    // strongly favours the higher tone.
    let o = on_mix("BPZ2", vec![]);
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(high > 4.0 * low, "BPZ2 should favour 4000: {low} vs {high}");
}

#[test]
fn brz2_notches_quarter_nyquist() {
    // BRZ2's `0.5*(x0 + x2)` notches SR/4 (12 kHz at 48 kHz), so a 12 kHz tone is killed relative to
    // a 1 kHz tone.
    let out = run(vec![
        sin(1000.0),
        sin(12_000.0),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
            ],
            num_outputs: 1,
            special_index: 0,
        },
        UnitSpec::new(
            "BRZ2",
            Rate::Audio,
            vec![InputRef::Unit { unit: 2, output: 0 }],
            1,
        ),
        out(3),
    ]);
    assert!(out.iter().all(|s| s.is_finite()));
    let low = goertzel(&out, 1000.0);
    let high = goertzel(&out, 12_000.0);
    assert!(
        low > 4.0 * high,
        "BRZ2 should notch 12 kHz: {low} vs {high}"
    );
}

#[test]
fn slope_boosts_highs_like_a_differentiator() {
    // Slope is the sample-rate-scaled difference, so its gain rises with frequency: 4000 dominates.
    let o = on_mix("Slope", vec![]);
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(high > 4.0 * low, "Slope should boost 4000: {low} vs {high}");
}

#[test]
fn delay1_and_delay2_pass_signal_transparently() {
    // A pure sample delay is magnitude-flat, so both tones survive unchanged.
    for f in ["Delay1", "Delay2"] {
        let o = on_mix(f, vec![]);
        assert!(o.iter().all(|s| s.is_finite()));
        let (low, high) = ratio(&o);
        assert!(
            low > 0.1 && high > 0.1,
            "{f} should pass both tones: {low}, {high}"
        );
    }
}

#[test]
fn apf_passes_all_frequencies() {
    // An all-pass is magnitude-flat: both tones survive its phase shift.
    let o = on_mix(
        "APF",
        vec![InputRef::Constant(1000.0), InputRef::Constant(0.9)],
    );
    assert!(o.iter().all(|s| s.is_finite()));
    let (low, high) = ratio(&o);
    assert!(
        low > 0.1 && high > 0.1,
        "APF should pass both tones: {low}, {high}"
    );
}

#[test]
fn slew_limits_the_rate_of_change() {
    // A 200 Hz sine (amplitude 1) slew-limited to ±100/s can only climb ~0.25 over a half-cycle, so
    // its amplitude is throttled well below the input's.
    let out = run(vec![
        sin(200.0),
        UnitSpec::new(
            "Slew",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(100.0),
                InputRef::Constant(100.0),
            ],
            1,
        ),
        out(1),
    ]);
    assert!(out.iter().all(|s| s.is_finite()));
    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!(
        peak > 0.05 && peak < 0.5,
        "Slew should throttle the amplitude, peak {peak}"
    );
}
