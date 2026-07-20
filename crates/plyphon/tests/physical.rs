//! Exercise the physical-model units: `Spring` (a driven damped mass-spring) resonates, and
//! `Ball`/`TBall` (a ball bouncing on a moving floor) run stably - `TBall` firing a spike at each
//! bounce.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn run(units: Vec<UnitSpec>, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "p".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("p", ROOT_GROUP_ID, AddAction::Tail, &[])
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

fn out(unit: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![InputRef::Constant(0.0), InputRef::Unit { unit, output: 0 }],
        0,
    )
}

/// A floor that oscillates around 0 (a SinOsc scaled), so a ball stays near it and keeps bouncing.
fn floor() -> [UnitSpec; 2] {
    [
        sin(4.0),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.2),
            ],
            num_outputs: 1,
            special_index: 2, // mul
        },
    ]
}

#[test]
fn spring_resonates_when_driven() {
    // Spring(WhiteNoise, 400, 0.2): a damped mass-spring driven by noise - stable and audible.
    let out = run(
        vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new(
                "Spring",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(400.0),
                    InputRef::Constant(0.2),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize / 2,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "Spring output not finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 1.0e4), "Spring diverged");
    assert!(out.iter().any(|&s| s.abs() > 0.01), "Spring was silent");
}

#[test]
fn ball_bounces_stably() {
    // Ball on an oscillating floor stays bounded and keeps moving.
    let [f0, f1] = floor();
    let out = run(
        vec![
            f0,
            f1,
            UnitSpec::new(
                "Ball",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(1.0),  // gravity
                    InputRef::Constant(0.1),  // damping
                    InputRef::Constant(0.01), // friction
                ],
                1,
            ),
            out(2),
        ],
        SR as usize / 2,
    );
    assert!(out.iter().all(|s| s.is_finite()), "Ball output not finite");
    assert!(out.iter().all(|&s| s.abs() < 100.0), "Ball diverged");
    let (min, max) = out
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    assert!(max - min > 0.05, "Ball did not move ({min}..{max})");
}

#[test]
fn tball_fires_on_bounces() {
    // TBall outputs 0 most of the time and a nonzero collision velocity at each bounce.
    let [f0, f1] = floor();
    let out = run(
        vec![
            f0,
            f1,
            UnitSpec::new(
                "TBall",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(4.0), // gravity
                    InputRef::Constant(0.1), // damping
                    InputRef::Constant(0.0), // friction 0 -> no sticky trap, so it bounces
                ],
                1,
            ),
            out(2),
        ],
        SR as usize / 2,
    );
    assert!(out.iter().all(|s| s.is_finite()), "TBall output not finite");
    assert!(out.contains(&0.0), "TBall should be silent between bounces");
    assert!(
        out.iter().any(|&s| s != 0.0),
        "TBall should fire at bounces"
    );
    assert!(out.iter().all(|&s| s.abs() < 100.0), "TBall diverged");
}
