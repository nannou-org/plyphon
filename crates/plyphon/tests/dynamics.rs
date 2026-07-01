//! Exercise the dynamics units: `Compander` compresses signals above its threshold and passes those
//! below it, and `DetectSilence` flags (and could free) a synth once its input goes quiet.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn run(units: Vec<UnitSpec>, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "d".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let sizes = [64usize, 128, 480, 512];
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

fn sin(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

fn peak(s: &[f32]) -> f32 {
    s.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

/// `Compander(scaled sine, scaled sine, 0.3, 1, 0.5, 0.002, 0.01)`, returning the settled output.
fn compressed(amp: f32) -> Vec<f32> {
    let mut out = run(
        vec![
            sin(300.0),
            // sine * amp -> the level we feed as both input and side-chain.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(amp),
                ],
                num_outputs: 1,
                special_index: 2,
            },
            UnitSpec::new(
                "Compander",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 }, // in
                    InputRef::Unit { unit: 1, output: 0 }, // control (side-chain)
                    InputRef::Constant(0.3),               // thresh
                    InputRef::Constant(1.0),               // slopeBelow (unity: passes)
                    InputRef::Constant(0.5),               // slopeAbove (2:1 compression)
                    InputRef::Constant(0.002),             // clampTime (attack)
                    InputRef::Constant(0.01),              // relaxTime (release)
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
        SR as usize / 4,
    );
    // The settled tail (after the follower has caught up).
    out.split_off(out.len() / 2)
}

#[test]
fn compander_compresses_above_threshold() {
    // A full-scale sine (peak ~1 >> thresh 0.3) is compressed 2:1: gain ~ (1/0.3)^-0.5 ~ 0.55.
    let loud = compressed(1.0);
    let p = peak(&loud);
    assert!(
        (0.4..0.75).contains(&p),
        "loud signal should be compressed to ~0.55, got {p}"
    );
}

#[test]
fn compander_passes_below_threshold() {
    // A quiet sine (peak 0.1 < thresh 0.3, unity slopeBelow) passes at unity gain.
    let quiet = compressed(0.1);
    let p = peak(&quiet);
    assert!(
        (0.08..0.12).contains(&p),
        "quiet signal should pass ~unchanged, got {p}"
    );
}

#[test]
fn detect_silence_flags_when_the_input_goes_quiet() {
    // A sine gated by a fast Line(1 -> 0) is loud, then silent. DetectSilence should output 0 while
    // it sounds and 1 once it has been silent for `time`.
    let out = run(
        vec![
            sin(200.0),
            // Line(1, 0, 0.02): a 20 ms fade to silence, then holds 0.
            UnitSpec::new(
                "Line",
                Rate::Control,
                vec![
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.02),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // sine * env -> decays to silence.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2,
            },
            // DetectSilence(decayed, amp=0.01, time=0.05, doneAction=0).
            UnitSpec::new(
                "DetectSilence",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(0.01),
                    InputRef::Constant(0.05),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 3, output: 0 },
                ],
                0,
            ),
        ],
        SR as usize / 4,
    );
    assert!(
        out.iter().all(|&x| x == 0.0 || x == 1.0),
        "DetectSilence must be 0 or 1"
    );
    assert_eq!(out[0], 0.0, "should not fire while the sound is present");
    assert_eq!(*out.last().unwrap(), 1.0, "should flag silence by the end");
}

/// `<name>(SinOsc(400) * amp, level, dur) -> Out`, returning the settled tail (past the look-ahead
/// latency and gain glide).
fn look_ahead(name: &str, amp: f32, level: f32, dur: f32) -> Vec<f32> {
    let mut out = run(
        vec![
            sin(400.0),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(amp),
                ],
                num_outputs: 1,
                special_index: 2,
            },
            UnitSpec::new(
                name,
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(level),
                    InputRef::Constant(dur),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
        SR as usize / 4,
    );
    out.split_off(out.len() / 2)
}

#[test]
fn limiter_caps_loud_and_passes_quiet() {
    // Loud (amp 2 > level 0.5) is capped to ~0.5; quiet (amp 0.3 < 0.5) passes at ~0.3.
    let loud = peak(&look_ahead("Limiter", 2.0, 0.5, 0.01));
    let quiet = peak(&look_ahead("Limiter", 0.3, 0.5, 0.01));
    assert!(
        (0.4..0.65).contains(&loud),
        "loud signal should be limited to ~level, got {loud}"
    );
    assert!(
        (0.25..0.35).contains(&quiet),
        "quiet signal should pass unchanged, got {quiet}"
    );
}

#[test]
fn normalizer_drives_peak_to_level() {
    // Both a loud (amp 2) and a quiet (amp 0.1) sine are driven to the target level (0.5).
    let loud = peak(&look_ahead("Normalizer", 2.0, 0.5, 0.01));
    let quiet = peak(&look_ahead("Normalizer", 0.1, 0.5, 0.01));
    assert!(
        (0.4..0.65).contains(&loud),
        "loud signal should be normalized to ~0.5, got {loud}"
    );
    assert!(
        (0.4..0.65).contains(&quiet),
        "quiet signal should be boosted to ~0.5, got {quiet}"
    );
}
