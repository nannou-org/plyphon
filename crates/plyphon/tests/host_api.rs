//! Host-facing controller and registry API checks.

use plyphon::{
    InputRef, Options, Param, Rate, SynthDef, SynthNewError, UnitRegistry, UnitSpec, engine,
};

fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![Param::control("freq", 440.0)],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn registry_enumerates_calc_and_demand_names_separately() {
    let registry = UnitRegistry::with_builtins();

    assert!(registry.names().any(|name| name == "SinOsc"));
    assert!(registry.names().any(|name| name == "Out"));
    assert!(registry.names().any(|name| name == "BinaryOpUGen"));
    assert!(!registry.names().any(|name| name == "Dseq"));

    assert!(registry.demand_names().any(|name| name == "Dseq"));
    assert!(!registry.demand_names().any(|name| name == "SinOsc"));
}

#[test]
fn controller_exposes_read_only_registry() {
    let (controller, _, _) = engine(Options::default());

    assert!(controller.registry().names().any(|name| name == "SinOsc"));
    assert!(
        controller
            .registry()
            .demand_names()
            .any(|name| name == "Dseq")
    );
}

#[test]
fn ensure_compiled_is_idempotent_and_reports_unknown_defs() {
    let (mut controller, _, _) = engine(Options {
        command_capacity: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def());

    let first = controller.ensure_compiled("sine").expect("first compile");
    let second = controller
        .ensure_compiled("sine")
        .expect("second compile must be idempotent");
    assert_eq!(first, second);

    match controller.ensure_compiled("nope") {
        Err(SynthNewError::UnknownDef(name)) => assert_eq!(name, "nope"),
        other => panic!("expected UnknownDef, got {other:?}"),
    }
}
