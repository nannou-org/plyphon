//! Diagnostic guard units: `CheckBadValues` classifies each sample's IEEE-754 category as a code,
//! and `Sanitize` swaps any bad sample for a replacement value.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;

/// The smallest positive subnormal `f32` (`1.4e-45`), an unambiguous FP_SUBNORMAL.
const SUBNORMAL: f32 = f32::from_bits(1);

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

/// `Out.ar(0, Unit{src})`.
fn out(src: u32) -> UnitSpec {
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

/// The first output sample of a one-synth graph.
fn first_sample(world: &mut World) -> f32 {
    let mut buf = vec![0.0f32; 64];
    world.fill(&mut buf, 1);
    buf[0]
}

/// `CheckBadValues.ar(DC.ar(value), 0, 0)` - the classification code for `value`.
fn check(value: f32) -> f32 {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "c".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(value)], 1),
            UnitSpec::new(
                "CheckBadValues",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out(1),
        ],
    });
    controller
        .synth_new("c", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    first_sample(&mut world)
}

#[test]
fn check_bad_values_classifies() {
    assert_eq!(check(0.5), 0.0, "a normal value is ok (0)");
    assert_eq!(check(0.0), 0.0, "zero is ok (0)");
    assert_eq!(check(f32::NAN), 1.0, "NaN is code 1");
    assert_eq!(check(f32::INFINITY), 2.0, "+inf is code 2");
    assert_eq!(check(f32::NEG_INFINITY), 2.0, "-inf is code 2");
    assert_eq!(check(SUBNORMAL), 3.0, "a subnormal is code 3");
}

/// `Sanitize.ar(DC.ar(value), replace)` - the passed-through or replaced sample.
fn sanitize(value: f32, replace: f32) -> f32 {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(value)], 1),
            UnitSpec::new(
                "Sanitize",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(replace),
                ],
                1,
            ),
            out(1),
        ],
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    first_sample(&mut world)
}

#[test]
fn sanitize_replaces_bad_values() {
    assert_eq!(sanitize(0.5, -9.0), 0.5, "a good value passes through");
    assert_eq!(sanitize(0.0, -9.0), 0.0, "zero passes through");
    assert_eq!(sanitize(f32::NAN, -9.0), -9.0, "NaN is replaced");
    assert_eq!(sanitize(f32::INFINITY, -9.0), -9.0, "inf is replaced");
    assert_eq!(sanitize(SUBNORMAL, -9.0), -9.0, "a subnormal is replaced");
}
