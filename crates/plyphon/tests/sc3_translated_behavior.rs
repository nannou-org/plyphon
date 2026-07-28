//! Constructor-state and first-block smoke vectors for the translated SC3 calculation units.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

/// Shorthand for a constant graph input.
fn constant(value: f32) -> InputRef {
    InputRef::Constant(value)
}

/// Shorthand for output zero of an earlier unit.
fn wire(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// Appends a mono output and renders the supplied graph for two blocks.
fn render(mut units: Vec<UnitSpec>, source: u32) -> Vec<f32> {
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![constant(0.0), wire(source)],
        0,
    ));
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: 48_000.0,
        block_size: 64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "translated-behavior".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("translated-behavior", ROOT_GROUP_ID, AddAction::Tail)
        .expect("spawn translated unit");
    let mut output = vec![0.0; 64];
    world.fill(&mut output, 1);
    output
}

/// Compares equal-length samples using a tight absolute tolerance.
fn assert_close(actual: &[f32], expected: &[f32]) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "sample {index}: expected {expected}, got {actual}"
        );
    }
}

/// Pins zero-attack envelope behavior at the first input sample.
#[test]
fn env_detect_zero_attack_tracks_the_first_audio_sample() {
    let output = render(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![constant(-0.75)], 1),
            UnitSpec::new(
                "EnvDetect",
                Rate::Audio,
                vec![wire(0), constant(0.0), constant(0.0)],
                1,
            ),
        ],
        1,
    );
    assert!(output.iter().all(|&sample| sample == 0.75));
}

/// Pins the translated constructor phase for every band-limited oscillator.
#[test]
fn blit_variants_preserve_translated_constructor_phase() {
    let impulse = render(
        vec![UnitSpec::new(
            "BlitB3",
            Rate::Audio,
            vec![constant(12_000.0)],
            1,
        )],
        0,
    );
    assert_close(&impulse[..4], &[0.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]);

    let saw = render(
        vec![UnitSpec::new(
            "BlitB3Saw",
            Rate::Audio,
            vec![constant(440.0), constant(0.0)],
            1,
        )],
        0,
    );
    assert_close(
        &saw[..5],
        &[-0.2, -1.0 / 30.0, 7.0 / 15.0, -1.0 / 30.0, -0.2],
    );

    let square = render(
        vec![UnitSpec::new(
            "BlitB3Square",
            Rate::Audio,
            vec![constant(440.0), constant(0.0)],
            1,
        )],
        0,
    );
    assert_close(&square[..5], &[0.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0]);

    let triangle = render(
        vec![UnitSpec::new(
            "BlitB3Tri",
            Rate::Audio,
            vec![constant(440.0), constant(0.0), constant(0.0)],
            1,
        )],
        0,
    );
    assert_close(
        &triangle[..5],
        &[0.0, 1.0 / 30.0, 2.0 / 15.0, 1.0 / 30.0, 0.0],
    );
}

/// Proves the translated filter families emit finite, non-silent audio.
#[test]
fn translated_filters_emit_finite_nonzero_audio() {
    for (name, controls) in [
        (
            "DFM1",
            vec![
                constant(1_000.0),
                constant(0.2),
                constant(1.0),
                constant(0.0),
                constant(0.0),
            ],
        ),
        ("MoogLadder", vec![constant(1_000.0), constant(0.2)]),
        ("MoogVCF", vec![constant(1_000.0), constant(0.2)]),
    ] {
        let mut inputs = vec![wire(0)];
        inputs.extend(controls);
        let output = render(
            vec![
                UnitSpec::new(
                    "SinOsc",
                    Rate::Audio,
                    vec![constant(220.0), constant(0.0)],
                    1,
                ),
                UnitSpec::new(name, Rate::Audio, inputs, 1),
            ],
            1,
        );
        assert!(
            output.iter().all(|sample| sample.is_finite()),
            "{name} emitted a non-finite sample"
        );
        assert!(
            output.iter().any(|sample| sample.abs() > 1e-8),
            "{name} emitted only silence"
        );
    }
}
