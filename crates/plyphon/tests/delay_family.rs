//! The interpolating and feedback delays: `DelayL`/`DelayC` (fractional taps) and `CombN/L/C` /
//! `AllpassN/L/C` (recirculating). Driven by a DC step, each has a deterministic response: a linear
//! tap splits the step's edge across two samples, a comb builds a decaying staircase spaced by the
//! delay, and an allpass adds an immediate feed-forward term the comb lacks.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

/// `ln(0.001)`, scsynth's `log001`, for recomputing the comb/allpass feedback coefficient in-test.
const LOG001: f32 = -6.907_755_4;

/// `sc_CalcFeedback`: the coefficient for a -60 dB decay over `decaytime` for a loop of `delaytime`.
fn calc_feedback(delaytime: f32, decaytime: f32) -> f32 {
    if delaytime == 0.0 || decaytime == 0.0 {
        return 0.0;
    }
    (LOG001 * delaytime / decaytime.abs())
        .exp()
        .copysign(decaytime)
}

/// `DC.ar(1.0) -> name.ar(in, ...tail) -> Out`, rendered for `total` samples.
fn run(name: &str, tail: Vec<InputRef>, total: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut inputs = vec![InputRef::Unit { unit: 0, output: 0 }];
    inputs.extend(tail);
    controller.add_synthdef(SynthDef {
        name: "d".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(name, Rate::Audio, inputs, 1),
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
    });
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; total];
    world.fill(&mut buf, 1);
    buf
}

#[test]
fn linear_delay_splits_the_step_edge() {
    // DelayL by K + 0.5 samples: the step's rising edge lands half on sample K and full from K+1
    // (linear interpolation of a 0->1 edge at the half-sample position).
    let k = 100usize;
    let delaytime = (k as f32 + 0.5) / SR;
    let out = run(
        "DelayL",
        vec![InputRef::Constant(0.02), InputRef::Constant(delaytime)],
        200,
    );
    assert!(
        out[..k].iter().all(|&s| s.abs() < 1e-6),
        "pre-delay silence"
    );
    assert!(
        (out[k] - 0.5).abs() < 0.05,
        "half-sample edge should be ~0.5, got {}",
        out[k]
    );
    assert!(
        out[k + 1..150].iter().all(|&s| (s - 1.0).abs() < 1e-4),
        "settles to the delayed level"
    );
}

#[test]
fn cubic_delay_is_bounded_and_delays_the_step() {
    // DelayC by K + 0.5: cubic interpolation, so the edge may briefly over/undershoot, but it is 0
    // well before the tap, bounded, and settled to 1 a couple of samples after.
    let k = 100usize;
    let delaytime = (k as f32 + 0.5) / SR;
    let out = run(
        "DelayC",
        vec![InputRef::Constant(0.02), InputRef::Constant(delaytime)],
        200,
    );
    assert!(out.iter().all(|s| s.is_finite()), "finite");
    assert!(out.iter().all(|&s| s.abs() < 1.5), "bounded");
    assert!(
        out[..k - 2].iter().all(|&s| s.abs() < 1e-6),
        "silent before the tap"
    );
    assert!(
        out[k + 2..150].iter().all(|&s| (s - 1.0).abs() < 1e-4),
        "settles to the delayed level"
    );
}

#[test]
fn comb_builds_a_decaying_staircase() {
    // CombN fed a step: silence for the first D samples, then a staircase 1, 1+f, 1+f+f^2, ... whose
    // successive increments shrink by the feedback coefficient f each delay period.
    let d = 50usize;
    let decaytime = 0.05f32;
    let delaytime = (d as f32 + 0.5) / SR; // + 0.5 so the integer tap is unambiguously D
    let feedbk = calc_feedback(d as f32 / SR, decaytime);
    let out = run(
        "CombN",
        vec![
            InputRef::Constant(0.01),
            InputRef::Constant(delaytime),
            InputRef::Constant(decaytime),
        ],
        250,
    );
    assert!(out.iter().all(|s| s.is_finite()), "finite");
    assert!(out[..d].iter().all(|&s| s.abs() < 1e-6), "silent before D");
    // Plateau midpoints: 1, 1+f, 1+f+f^2. The increments are f and f^2.
    let (p1, p2, p3) = (out[d + d / 2], out[2 * d + d / 2], out[3 * d + d / 2]);
    assert!((p1 - 1.0).abs() < 1e-4, "first plateau ~1, got {p1}");
    let inc1 = p2 - p1;
    let inc2 = p3 - p2;
    assert!(
        (inc1 - feedbk).abs() < 0.02,
        "first increment ~f={feedbk}, got {inc1}"
    );
    assert!(
        (inc2 / inc1 - feedbk).abs() < 0.02,
        "increments should shrink by f={feedbk}, ratio {}",
        inc2 / inc1
    );
}

#[test]
fn allpass_adds_the_feedforward_term() {
    // AllpassN fed a step: unlike a comb (silent before D), the allpass immediately emits -f from its
    // feed-forward path for the first D samples, then recirculates. Stays finite and bounded.
    let d = 50usize;
    let decaytime = 0.05f32;
    let delaytime = (d as f32 + 0.5) / SR;
    let feedbk = calc_feedback(d as f32 / SR, decaytime);
    let out = run(
        "AllpassN",
        vec![
            InputRef::Constant(0.01),
            InputRef::Constant(delaytime),
            InputRef::Constant(decaytime),
        ],
        250,
    );
    assert!(out.iter().all(|s| s.is_finite()), "finite");
    assert!(out.iter().all(|&s| s.abs() < 4.0), "bounded");
    assert!(
        (out[d / 2] + feedbk).abs() < 0.02,
        "feed-forward term should be -f={}, got {}",
        -feedbk,
        out[d / 2]
    );
}

#[test]
fn interpolating_combs_and_allpasses_stay_bounded() {
    // The linear/cubic feedback variants are harder to pin to a closed form, but must run stably: a
    // step in gives a finite, bounded, non-silent response that recirculates past the delay.
    let d = 60usize;
    let delaytime = (d as f32 + 0.3) / SR;
    for name in ["CombL", "CombC", "AllpassL", "AllpassC"] {
        let out = run(
            name,
            vec![
                InputRef::Constant(0.02),
                InputRef::Constant(delaytime),
                InputRef::Constant(0.08),
            ],
            600,
        );
        assert!(out.iter().all(|s| s.is_finite()), "{name} finite");
        assert!(out.iter().all(|&s| s.abs() < 8.0), "{name} bounded");
        assert!(
            out[2 * d..].iter().any(|&s| s.abs() > 0.1),
            "{name} should recirculate energy past the delay"
        );
    }
}
