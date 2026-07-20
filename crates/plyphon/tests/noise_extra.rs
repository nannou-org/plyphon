//! The chaotic/deterministic noise generators: `Crackle`, `Logistic`, `Hasher`, `MantissaMask`.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

fn render(units: Vec<UnitSpec>, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "n".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("n", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}

#[test]
fn crackle_is_bounded_chaos() {
    // The Crackle map stays finite and non-silent, and (near chaosParam 1.5) is not a constant.
    let out = render(
        vec![
            UnitSpec::new("Crackle", Rate::Audio, vec![c(1.5)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
        SR as usize / 8,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "Crackle must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 4.0), "Crackle stays bounded");
    let first = out[100];
    assert!(
        out.iter().any(|&s| (s - first).abs() > 1e-3),
        "Crackle should be chaotic, not constant"
    );
}

#[test]
fn logistic_stays_in_unit_interval() {
    // The logistic map y = r*y*(1-y) with r < 4 stays within [0, 1]; iterated at audio rate it varies.
    let out = render(
        vec![
            UnitSpec::new(
                "Logistic",
                Rate::Audio,
                vec![c(3.7), c(SR), c(0.5)], // chaosParam, freq (>= SR: iterate every sample), init
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
        SR as usize / 8,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "Logistic must stay finite"
    );
    assert!(
        out.iter().all(|&s| (0.0..=1.0).contains(&s)),
        "the logistic map stays in [0, 1]"
    );
    let first = out[100];
    assert!(
        out.iter().any(|&s| (s - first).abs() > 1e-3),
        "at r=3.7 the map should be chaotic"
    );
}

#[test]
fn logistic_freq_holds_between_iterations() {
    // At a low freq the map is held between iterations, so the output is piecewise-constant (adjacent
    // samples are usually equal).
    let out = render(
        vec![
            UnitSpec::new("Logistic", Rate::Audio, vec![c(3.7), c(1000.0), c(0.5)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
        4800,
    );
    // sr/freq = 48 samples per step, so far more equal-neighbour pairs than distinct ones.
    let holds = out.windows(2).filter(|w| w[0] == w[1]).count();
    let steps = out.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(
        holds > 10 * steps,
        "Logistic should hold its value between iterations (holds={holds}, steps={steps})"
    );
}

#[test]
fn hasher_is_deterministic_and_bounded() {
    // Hasher maps equal inputs to equal outputs. Feeding a constant yields a constant in [-1, 1); the
    // same constant on a second run yields the identical value.
    let run = |v: f32| {
        render(
            vec![
                UnitSpec::new("Hasher", Rate::Audio, vec![c(v)], 1),
                UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
            ],
            64,
        )
    };
    let a = run(0.321);
    assert!(
        a.iter().all(|&s| (-1.0..1.0).contains(&s)),
        "Hasher in [-1, 1)"
    );
    assert!(
        a.iter().all(|&s| s == a[0]),
        "a constant input hashes to a constant output"
    );
    let b = run(0.321);
    assert_eq!(a[0], b[0], "Hasher is deterministic");
    let d = run(0.322);
    assert_ne!(a[0], d[0], "different inputs hash differently");
}

#[test]
fn mantissa_mask_quantizes() {
    // MantissaMask keeps only the top `bits` mantissa bits, so its output only takes a few distinct
    // values as a smooth sine sweeps - far fewer than the input's continuum.
    let out = render(
        vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
            UnitSpec::new("MantissaMask", Rate::Audio, vec![u(0), c(2.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        2048,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "MantissaMask stays finite"
    );
    // The masked output stays close to the input magnitude but is coarsely quantized.
    let mut distinct: Vec<f32> = out.to_vec();
    distinct.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distinct.dedup();
    assert!(
        distinct.len() < 64,
        "2-bit mantissa masking should collapse the sine to few levels ({} distinct)",
        distinct.len()
    );
    assert!(
        out.iter().any(|&s| s.abs() > 0.1),
        "the masked sine still sounds"
    );
}
