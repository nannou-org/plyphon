//! Exact build-time ABI coverage for the translated spec-100 calc units.

use plyphon::{BuildError, GraphDef, InputRef, Rate, RateInfo, SynthDef, UnitRegistry, UnitSpec};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn compile(units: Vec<UnitSpec>) -> Result<GraphDef, BuildError> {
    let rate = RateInfo::new(SR, BLOCK);
    SynthDef {
        name: "translated-abi".to_string(),
        params: vec![],
        units,
    }
    .compile(
        &UnitRegistry::with_builtins(),
        &rate,
        &rate,
        64,
        32,
        None,
        1,
    )
}

fn constants(count: usize) -> Vec<InputRef> {
    vec![InputRef::Constant(0.0); count]
}

fn unit(name: &str, rate: Rate, inputs: usize, outputs: usize) -> UnitSpec {
    UnitSpec::new(name, rate, constants(inputs), outputs)
}

#[test]
fn translated_units_accept_exact_shapes_and_rates() {
    for (name, inputs) in [
        ("DFM1", 6),
        ("MoogVCF", 3),
        ("BlitB3", 1),
        ("BlitB3Saw", 2),
        ("BlitB3Square", 2),
        ("BlitB3Tri", 3),
    ] {
        compile(vec![unit(name, Rate::Audio, inputs, 1)])
            .unwrap_or_else(|error| panic!("{name} exact ABI rejected: {error}"));
    }

    compile(vec![unit("MoogLadder", Rate::Audio, 3, 1)]).expect("MoogLadder audio ABI");
    compile(vec![unit("MoogLadder", Rate::Control, 3, 1)]).expect("MoogLadder control ABI");

    compile(vec![
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.25)], 1),
        UnitSpec::new(
            "EnvDetect",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(100.0),
                InputRef::Constant(0.0),
            ],
            1,
        ),
    ])
    .expect("EnvDetect exact ABI");
}

#[test]
fn translated_units_reject_wrong_input_output_special_and_rate() {
    assert_eq!(
        compile(vec![unit("DFM1", Rate::Audio, 5, 1)]).map(|_| ()),
        Err(BuildError::WrongInputCount)
    );
    assert_eq!(
        compile(vec![unit("BlitB3", Rate::Audio, 1, 2)]).map(|_| ()),
        Err(BuildError::WrongOutputCount {
            expected: 1,
            actual: 2,
        })
    );

    let mut special = unit("MoogVCF", Rate::Audio, 3, 1);
    special.special_index = 7;
    assert_eq!(
        compile(vec![special]).map(|_| ()),
        Err(BuildError::UnsupportedOp(7))
    );
    assert_eq!(
        compile(vec![unit("BlitB3Saw", Rate::Control, 2, 1)]).map(|_| ()),
        Err(BuildError::UnsupportedUnitRate)
    );
    assert_eq!(
        compile(vec![unit("EnvDetect", Rate::Audio, 3, 1)]).map(|_| ()),
        Err(BuildError::UnsupportedUnitRate)
    );
}
