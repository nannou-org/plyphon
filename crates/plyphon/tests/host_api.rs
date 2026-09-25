//! Host-facing controller and registry API checks.

use plyphon::{
    InputRef, Options, Param, Rate, SynthDef, SynthNewError, UnitRegistry, UnitSpec, World, engine,
};

fn sine_def() -> SynthDef {
    named_sine("sine", 440.0)
}

/// `SinOsc.ar(freq) -> Out.ar(0, sig)` under `name`, with `freq`'s default as the only content that
/// varies between otherwise identical definitions.
fn named_sine(name: &str, freq: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![Param::control("freq", freq)],
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

/// Drive the engine for one control block so queued def-table commands are applied - the point at
/// which a cleared or replaced slot drops its `Arc<GraphDef>` and a retired def becomes reapable.
fn render(world: &mut World) {
    let mut buf = vec![0.0f32; 64];
    world.fill(&mut buf, 1);
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

#[test]
fn freed_specialized_def_id_is_reused() {
    // A host that mints a fresh def name per specialization needs the id space bounded by *live*
    // definitions: freeing a name releases its id once the audio thread has released the slot, and
    // the next unseen name takes it back before minting.
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(named_sine("a", 440.0));
    let first = controller.ensure_compiled("a").expect("compile a");

    controller.free_def("a").expect("free a");
    render(&mut world); // apply FreeGraphDef: the resident slot drops its reference
    controller.reap_retired_defs();

    controller.add_synthdef(named_sine("b", 220.0));
    let reused = controller.ensure_compiled("b").expect("compile b");
    assert_eq!(
        reused, first,
        "an unseen name must take the freed id before minting a new one"
    );

    // Nothing else was freed, so the next unseen name mints instead of reusing.
    controller.add_synthdef(named_sine("c", 110.0));
    let minted = controller.ensure_compiled("c").expect("compile c");
    assert_ne!(
        minted, first,
        "only a *freed* id is reclaimed; an unrelated name still mints"
    );
}

#[test]
fn redefined_def_keeps_its_id_after_reap() {
    // Retirement has two producers and only one of them releases an id. A redefinition redefines
    // the name's slot in place, so reaping it must not hand that id to the next unseen name.
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(named_sine("a", 440.0));
    let original = controller.ensure_compiled("a").expect("compile a");

    controller.add_synthdef(named_sine("a", 220.0));
    let redefined = controller.ensure_compiled("a").expect("recompile a");
    assert_eq!(
        redefined, original,
        "a redefinition keeps the name's def id (the same slot is redefined)"
    );

    render(&mut world); // apply the replacing DefineGraphDef: the superseded form becomes reapable
    controller.reap_retired_defs();
    assert_eq!(
        controller.retired_defs_len(),
        0,
        "the superseded form is reclaimed once the audio thread has replaced its slot"
    );

    controller.add_synthdef(named_sine("b", 110.0));
    let minted = controller.ensure_compiled("b").expect("compile b");
    assert_ne!(
        minted, original,
        "a redefinition-retired id must never reach the free list"
    );
}
