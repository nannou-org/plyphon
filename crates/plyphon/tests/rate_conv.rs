//! Rate-conversion units: DC (constant), A2K (audio->control first sample), T2A (control trigger ->
//! sample-accurate audio), and K2A (control->audio interpolation).

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    }
}

/// One control block of mono output.
fn one_block(world: &mut World) -> Vec<f32> {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf
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

#[test]
fn dc_is_constant() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "dc".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.5)], 1),
            out(0),
        ],
    });
    controller
        .synth_new("dc", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    for x in one_block(&mut world) {
        assert!((x - 0.5).abs() < 1e-6, "DC sample {x} != 0.5");
    }
}

#[test]
fn a2k_takes_first_sample() {
    // DC.ar(0.7) -> A2K.kr -> MulAdd.ar(_, 1, 0) broadcasts the control value back to audio.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "a2k".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.7)], 1),
            UnitSpec::new(
                "A2K",
                Rate::Control,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out(2),
        ],
    });
    controller
        .synth_new("a2k", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    for x in one_block(&mut world) {
        assert!((x - 0.7).abs() < 1e-6, "A2K sample {x} != 0.7");
    }
}

#[test]
fn t2a_places_trigger_at_offset() {
    let offset = 3usize;
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "t2a".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "T2A",
                Rate::Audio,
                vec![InputRef::Constant(1.0), InputRef::Constant(offset as f32)],
                1,
            ),
            out(0),
        ],
    });
    controller
        .synth_new("t2a", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // First block: rising edge (prev 0 -> 1) fires once, at `offset`.
    let block = one_block(&mut world);
    for (i, x) in block.iter().enumerate() {
        let expected = if i == offset { 1.0 } else { 0.0 };
        assert!(
            (x - expected).abs() < 1e-6,
            "T2A sample {i} = {x}, expected {expected}"
        );
    }
    // Second block: level held high, no new edge, so silence.
    for x in one_block(&mut world) {
        assert!(x.abs() < 1e-6, "T2A second block should be silent, got {x}");
    }
}

#[test]
fn k2a_interpolates_control_step() {
    // In.kr(bus 0) -> K2A.ar -> Out. Step the bus between blocks and watch K2A ramp across one
    // block (the canonical control->audio linear interpolation).
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "k2a".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Control, vec![InputRef::Constant(0.0)], 1),
            UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
    });
    controller.set_control_bus(0, 0.0).expect("set bus");
    controller
        .synth_new("k2a", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // Bus held at 0: silent (K2A seeds prev = 0 on the first block).
    for x in one_block(&mut world) {
        assert!(x.abs() < 1e-6, "K2A at rest should be 0, got {x}");
    }

    // Step the bus to 1.0: K2A ramps linearly from prev (0) to cur (1) across the block, so sample
    // i is i / blockSize.
    controller.set_control_bus(0, 1.0).expect("set bus");
    let ramp = one_block(&mut world);
    let step = 1.0 / BLOCK as f32;
    for (i, &x) in ramp.iter().enumerate() {
        let expected = i as f32 * step;
        assert!(
            (x - expected).abs() < 1e-4,
            "K2A ramp sample {i} = {x}, expected {expected}"
        );
    }

    // Bus steady at 1.0: settles to a constant 1.0.
    for x in one_block(&mut world) {
        assert!((x - 1.0).abs() < 1e-6, "K2A settled should be 1.0, got {x}");
    }
}
