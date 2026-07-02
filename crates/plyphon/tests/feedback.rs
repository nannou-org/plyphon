//! Feedback buses: `LocalIn`/`LocalOut` form a one-block feedback loop. A comb `sum = injection +
//! coef*LocalIn`, fed back via `LocalOut`, decays by `coef` each block once the injection stops.

use plyphon::{
    AddAction, BuildError, InputRef, Options, ROOT_GROUP_ID, Rate, RateInfo, SynthDef,
    UnitRegistry, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
const COEF: f32 = 0.5;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    }
}

/// Render one control block; return its first sample.
fn one(world: &mut World) -> f32 {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf[0]
}

#[test]
fn feedback_comb_decays_by_coef() {
    // Unit order matters: LocalIn is calc-ordered *before* LocalOut, so it reads last block's write
    // (the one-block delay). sum = DC(In.kr(bus 0)) + COEF*LocalIn; Out and LocalOut both take sum.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "comb".to_string(),
        params: vec![],
        units: vec![
            // 0: LocalIn.ar(1) - last block's fed-back sum.
            UnitSpec::new("LocalIn", Rate::Audio, vec![], 1),
            // 1: MulAdd.ar(LocalIn, COEF, 0) - scale the feedback.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(COEF),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 2: In.kr(bus 0) - the host's injection level.
            UnitSpec::new("In", Rate::Control, vec![InputRef::Constant(0.0)], 1),
            // 3: DC.ar(injection) - lift it to audio rate.
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 2, output: 0 }],
                1,
            ),
            // 4: sum = injection + COEF*LocalIn  (BinaryOpUGen add, special_index 0).
            UnitSpec::new(
                "BinaryOpUGen",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                1,
            ),
            // 5: Out.ar(0, sum).
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 4, output: 0 },
                ],
                0,
            ),
            // 6: LocalOut.ar(sum) - feed the sum back for next block.
            UnitSpec::new(
                "LocalOut",
                Rate::Audio,
                vec![InputRef::Unit { unit: 4, output: 0 }],
                0,
            ),
        ],
    });
    controller.set_control_bus(0, 1.0).unwrap(); // inject 1.0
    controller
        .synth_new("comb", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // Block 1: LocalIn reads silence -> sum = injection = 1.0.
    assert!(
        (one(&mut world) - 1.0).abs() < 1e-6,
        "first block is the injection"
    );
    controller.set_control_bus(0, 0.0).unwrap(); // stop injecting
    // Each subsequent block: sum = COEF * last block's sum.
    let mut expected = COEF;
    for _ in 0..6 {
        let got = one(&mut world);
        assert!(
            (got - expected).abs() < 1e-5,
            "feedback should decay by COEF: got {got}, expected {expected}"
        );
        expected *= COEF;
    }
}

/// Compile `def` with the built-in registry, returning the result so a test can assert the error.
fn try_compile(def: &SynthDef) -> Result<(), BuildError> {
    let rate = RateInfo::new(SR, BLOCK);
    def.compile(
        &UnitRegistry::with_builtins(),
        &rate,
        &rate,
        64,
        32,
        None,
        1,
    )
    .map(|_| ())
}

#[test]
fn local_bus_channel_mismatch_rejected() {
    // LocalIn declares 1 channel; LocalOut writes 2 -> mismatch.
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("LocalIn", Rate::Audio, vec![], 1),
            UnitSpec::new(
                "LocalOut",
                Rate::Audio,
                vec![InputRef::Constant(0.0), InputRef::Constant(0.0)],
                0,
            ),
        ],
    };
    assert_eq!(
        try_compile(&def),
        Err(BuildError::LocalBusMismatch {
            local_in: 1,
            local_out: 2
        })
    );
}

#[test]
fn multiple_local_in_rejected() {
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("LocalIn", Rate::Audio, vec![], 1),
            UnitSpec::new("LocalIn", Rate::Audio, vec![], 1),
        ],
    };
    assert_eq!(try_compile(&def), Err(BuildError::MultipleLocalBuses));
}

#[test]
fn in_feedback_reads_a_bus() {
    // Write 0.5 to audio bus channel 1, then read it back with InFeedback in the same block.
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 2,
        ..opts()
    });
    controller.add_synthdef(SynthDef {
        name: "fb".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.5)], 1),
            // Out.ar(1, 0.5): write channel 1 (touched, so not silenced).
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(1.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
            // InFeedback.ar(1): read channel 1 - here same-block (Out is ordered first).
            UnitSpec::new("InFeedback", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            // Out.ar(0, inFeedback): route it to channel 0.
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
    });
    controller
        .synth_new("fb", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let mut buf = vec![0.0f32; BLOCK * 2];
    world.fill(&mut buf, 2);
    assert!(
        (buf[0] - 0.5).abs() < 1e-6,
        "InFeedback should read channel 1's value (0.5), got {}",
        buf[0]
    );
}

/// `DC.ar(value) -> Out.ar(bus)`: a constant writer onto a private audio bus.
fn writer_def(name: &str, value: f32, bus: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(value)], 1),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(bus),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

/// `<reader>.ar(bus) -> Out.ar(0)`: routes a private-bus read to the output.
fn reader_def(name: &str, reader: &str, bus: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(reader, Rate::Audio, vec![InputRef::Constant(bus)], 1),
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

/// Engine with one output, no inputs: private audio channels start at bus 1.
fn private_bus_engine() -> (plyphon::Controller, plyphon::Nrt, World) {
    engine(Options {
        input_channels: 0,
        ..opts()
    })
}

#[test]
fn in_before_writer_reads_silence() {
    // The reader runs *before* the writer each block, so the bus holds last block's audio when it
    // reads - which `In.ar` must treat as stale (zero), not re-emit (scsynth's touched check).
    let (mut controller, _nrt, mut world) = private_bus_engine();
    controller.add_synthdef(reader_def("read", "In", 1.0));
    controller.add_synthdef(writer_def("write", 0.8, 1.0));
    controller
        .synth_new("read", ROOT_GROUP_ID, AddAction::Head)
        .unwrap();
    controller
        .synth_new("write", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    for block in 0..4 {
        let got = one(&mut world);
        assert!(
            got.abs() < 1e-6,
            "In.ar before its writer should read silence, got {got} (block {block})"
        );
    }
}

#[test]
fn in_feedback_before_writer_reads_last_block() {
    // Same ordering, but `InFeedback` deliberately reads the stale bus: silence on the first
    // block (nothing written yet), then the previous block's 0.8 forever.
    let (mut controller, _nrt, mut world) = private_bus_engine();
    controller.add_synthdef(reader_def("read", "InFeedback", 1.0));
    controller.add_synthdef(writer_def("write", 0.8, 1.0));
    controller
        .synth_new("read", ROOT_GROUP_ID, AddAction::Head)
        .unwrap();
    controller
        .synth_new("write", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let first = one(&mut world);
    assert!(
        first.abs() < 1e-6,
        "nothing was written before the first block, got {first}"
    );
    for block in 1..4 {
        let got = one(&mut world);
        assert!(
            (got - 0.8).abs() < 1e-6,
            "InFeedback should read last block's 0.8, got {got} (block {block})"
        );
    }
}

#[test]
fn in_after_writer_reads_live() {
    // The writer runs first, so the bus is touched when the reader runs: `In.ar` reads it live.
    let (mut controller, _nrt, mut world) = private_bus_engine();
    controller.add_synthdef(reader_def("read", "In", 1.0));
    controller.add_synthdef(writer_def("write", 0.8, 1.0));
    controller
        .synth_new("write", ROOT_GROUP_ID, AddAction::Head)
        .unwrap();
    controller
        .synth_new("read", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let got = one(&mut world);
    assert!(
        (got - 0.8).abs() < 1e-6,
        "In.ar after its writer should read the live 0.8, got {got}"
    );
}

#[test]
fn in_goes_silent_when_writer_freed() {
    // The "frozen buzz": once the writer is freed the bus keeps its final block, which `In.ar`
    // must stop re-emitting.
    let (mut controller, _nrt, mut world) = private_bus_engine();
    controller.add_synthdef(reader_def("read", "In", 1.0));
    controller.add_synthdef(writer_def("write", 0.8, 1.0));
    controller
        .synth_new("write", ROOT_GROUP_ID, AddAction::Head)
        .unwrap();
    controller
        .synth_new("read", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let live = one(&mut world);
    assert!((live - 0.8).abs() < 1e-6, "live read, got {live}");
    // Free by def name lookup: re-add would complicate - just free every node via the group.
    controller.free_all(ROOT_GROUP_ID).unwrap();
    controller.add_synthdef(reader_def("read2", "In", 1.0));
    controller
        .synth_new("read2", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let got = one(&mut world);
    assert!(
        got.abs() < 1e-6,
        "In.ar of a writer-less bus should read silence, got {got}"
    );
}
