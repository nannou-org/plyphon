//! The signal-measurement units: `Peak`, `RunningMin`, `RunningMax`, `PeakFollower`, `MostChange`,
//! `LeastChange`, `LastValue`. Each is driven through a `World` and its running statistic checked.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;
const AMP: f32 = 0.5;

fn render(units: Vec<UnitSpec>, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "m".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("m", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}

/// `SinOsc.ar(freq) * AMP` as unit 0.
fn sine(freq: f32) -> Vec<UnitSpec> {
    vec![
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(AMP),
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
    ]
}

fn out_unit(src: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: src,
                output: 0,
            },
        ],
        0,
    )
}

#[test]
fn peak_holds_the_running_maximum_abs() {
    // Peak.ar(SinOsc * AMP, 0) climbs to the signal's peak amplitude and never decreases.
    let mut units = sine(300.0);
    units.push(UnitSpec::new(
        "Peak",
        Rate::Audio,
        vec![
            InputRef::Unit { unit: 1, output: 0 },
            InputRef::Constant(0.0),
        ],
        1,
    ));
    units.push(out_unit(2));
    let out = render(units, SR as usize / 4);

    assert!(
        out.windows(2).all(|w| w[1] >= w[0] - 1e-6),
        "Peak must be non-decreasing without a reset"
    );
    let last = *out.last().unwrap();
    assert!(
        (last - AMP).abs() < 0.02,
        "Peak should reach the amplitude {AMP}, got {last}"
    );
}

#[test]
fn running_max_and_min_track_the_extremes() {
    for (name, expected, ord_ok) in [
        ("RunningMax", AMP, 1.0f32),   // non-decreasing toward +AMP
        ("RunningMin", -AMP, -1.0f32), // non-increasing toward -AMP
    ] {
        let mut units = sine(300.0);
        units.push(UnitSpec::new(
            name,
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(0.0),
            ],
            1,
        ));
        units.push(out_unit(2));
        let out = render(units, SR as usize / 4);
        assert!(
            out.windows(2).all(|w| (w[1] - w[0]) * ord_ok >= -1e-6),
            "{name} must be monotonic"
        );
        let last = *out.last().unwrap();
        assert!(
            (last - expected).abs() < 0.02,
            "{name} should reach {expected}, got {last}"
        );
    }
}

#[test]
fn peak_follower_with_zero_decay_is_a_rectifier() {
    // With decay 0 the follower has no release, so its output is |in| every sample - a full-wave
    // rectifier: never negative, peaking at AMP, crossing zero at twice the input frequency.
    let mut units = sine(300.0);
    units.push(UnitSpec::new(
        "PeakFollower",
        Rate::Audio,
        vec![
            InputRef::Unit { unit: 1, output: 0 },
            InputRef::Constant(0.0),
        ],
        1,
    ));
    units.push(out_unit(2));
    let out = render(units, SR as usize / 4);
    assert!(
        out.iter().all(|&s| s >= 0.0),
        "rectified output must be >= 0"
    );
    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s));
    assert!(
        (peak - AMP).abs() < 0.02,
        "peak should be {AMP}, got {peak}"
    );
}

#[test]
fn peak_follower_releases_slowly_with_decay() {
    // A high decay coefficient keeps the level near the peak between the sine's zero crossings, so the
    // minimum output stays well above zero (unlike the decay-0 rectifier, which touches zero).
    let mut units = sine(300.0);
    units.push(UnitSpec::new(
        "PeakFollower",
        Rate::Audio,
        vec![
            InputRef::Unit { unit: 1, output: 0 },
            InputRef::Constant(0.999),
        ],
        1,
    ));
    units.push(out_unit(2));
    // Skip the initial ramp-up.
    let out = render(units, SR as usize / 4);
    let tail = &out[out.len() / 2..];
    let min = tail.iter().fold(f32::MAX, |m, &s| m.min(s));
    assert!(
        min > 0.4,
        "slow release should hold near the peak, min {min}"
    );
}

#[test]
fn most_and_least_change_pick_the_right_input() {
    // a = DC(0.2) (never changes), b = SinOsc * AMP (changes every sample). MostChange follows b;
    // LeastChange follows the constant a.
    for (name, constant_out) in [("MostChange", false), ("LeastChange", true)] {
        let mut units = vec![UnitSpec::new(
            "DC",
            Rate::Audio,
            vec![InputRef::Constant(0.2)],
            1,
        )];
        // units 1,2 = SinOsc, *AMP (offset by the DC at index 0).
        units.push(UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(300.0), InputRef::Constant(0.0)],
            1,
        ));
        units.push(UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(AMP),
            ],
            num_outputs: 1,
            special_index: 2,
        });
        units.push(UnitSpec::new(
            name,
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 }, // a = DC
                InputRef::Unit { unit: 2, output: 0 }, // b = sine
            ],
            1,
        ));
        units.push(out_unit(3));
        let out = render(units, SR as usize / 4);
        // Ignore the first sample (a tie seeds the winner).
        let tail = &out[1..];
        if constant_out {
            assert!(
                tail.iter().all(|&s| (s - 0.2).abs() < 1e-6),
                "LeastChange should hold the unchanging input 0.2"
            );
        } else {
            let span = tail.iter().fold(f32::MIN, |m, &s| m.max(s))
                - tail.iter().fold(f32::MAX, |m, &s| m.min(s));
            assert!(
                span > 0.5,
                "MostChange should follow the varying sine, span {span}"
            );
        }
    }
}

#[test]
fn last_value_quantises_with_hysteresis() {
    // A rising ramp through LastValue(_, 0.25) yields a coarse staircase: only a handful of distinct
    // held values, each step at least ~0.25 apart.
    let dur = (SR as usize / 4) as f32 / SR;
    let units = vec![
        UnitSpec::new(
            "Line",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Constant(1.0),
                InputRef::Constant(dur),
                InputRef::Constant(0.0),
            ],
            1,
        ),
        UnitSpec::new(
            "LastValue",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.25),
            ],
            1,
        ),
        out_unit(1),
    ];
    let out = render(units, SR as usize / 4);
    assert!(
        out.windows(2).all(|w| w[1] >= w[0] - 1e-6),
        "output should be non-decreasing for a rising ramp"
    );
    // Count distinct held values.
    let mut distinct = 1;
    for w in out.windows(2) {
        if (w[1] - w[0]).abs() > 1e-6 {
            distinct += 1;
        }
    }
    assert!(
        (2..=6).contains(&distinct),
        "quantiser should produce a few steps, got {distinct}"
    );
}
