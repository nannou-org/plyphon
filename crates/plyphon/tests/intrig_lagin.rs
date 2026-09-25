//! `InTrig` and `LagIn`, the control-bus readers: `InTrig` passes a channel's value only on a block
//! in which it was written, and `LagIn` smooths each channel with a one-pole lag whose coefficient
//! comes from the World's control rate. The `LagIn` values are scsynth's, at 48 kHz with 64-sample
//! blocks.

use plyphon::{
    AddAction, Controller, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

fn world(output_channels: usize) -> (Controller, Nrt, World) {
    engine(Options {
        output_channels,
        ..Options::default()
    })
}

/// `name.kr(inputs)` with `channels` outputs, each held into `Out.ar(i)` by `DC.ar`.
fn reader(def: &str, name: &str, inputs: Vec<InputRef>, channels: u32) -> SynthDef {
    let mut units = vec![UnitSpec::new(
        name,
        Rate::Control,
        inputs,
        channels as usize,
    )];
    for ch in 0..channels {
        units.push(UnitSpec::new(
            "DC",
            Rate::Audio,
            vec![InputRef::Unit {
                unit: 0,
                output: ch,
            }],
            1,
        ));
        units.push(UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![c(ch as f32), u(1 + 2 * ch)],
            0,
        ));
    }
    SynthDef {
        name: def.to_string(),
        params: vec![],
        units,
    }
}

/// `Out.kr(bus, value)`.
fn writer(def: &str, bus: f32, value: f32) -> SynthDef {
    SynthDef {
        name: def.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Control, vec![c(value)], 1),
            UnitSpec::new("Out", Rate::Control, vec![c(bus), u(0)], 0),
        ],
    }
}

/// One block's first frame, `channels` wide.
fn frame(world: &mut World, channels: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; BLOCK * channels];
    world.fill(&mut buf, channels);
    buf[..channels].to_vec()
}

#[test]
fn in_trig_passes_only_channels_written_this_block() {
    let (mut controller, _nrt, mut world) = world(2);
    controller.add_synthdef(writer("w", 3.0, 0.75));
    controller.add_synthdef(reader("r", "InTrig", vec![c(2.0)], 2));
    let w = controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .expect("writer");
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    // Bus 2 is never written; bus 3 is, every block, ahead of the reader.
    assert_eq!(frame(&mut world, 2), [0.0, 0.75]);
    assert_eq!(frame(&mut world, 2), [0.0, 0.75]);
    // Once the writer is gone the bus keeps its value, but no block writes it.
    controller.free(w).expect("free");
    assert_eq!(frame(&mut world, 2), [0.0, 0.0]);
}

#[test]
fn in_trig_reads_only_on_the_first_tick_when_reblocked() {
    // Reblocked to 16 samples, the reader ticks four times per World block and only the first
    // tick reads the bus; the other three output zero.
    let (mut controller, _nrt, mut world) = world(1);
    controller.add_synthdef(writer("w", 3.0, 0.75));
    controller.add_synthdef_reblocked(reader("r", "InTrig", vec![c(3.0)], 1), 16);
    controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .expect("writer");
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    assert!(buf[..16].iter().all(|&v| v == 0.75), "{buf:?}");
    assert!(buf[16..].iter().all(|&v| v == 0.0), "{buf:?}");
}

#[test]
fn in_trig_passes_a_c_set_for_one_block() {
    // scsynth's `/c_set`, `/c_setn` and `/c_fill` mark the channel written in the coming block.
    let (mut controller, _nrt, mut world) = world(2);
    controller.add_synthdef(reader("r", "InTrig", vec![c(2.0)], 2));
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    assert_eq!(frame(&mut world, 2), [0.0, 0.0]);
    controller.set_control_bus(3, 0.5).expect("c_set");
    assert_eq!(frame(&mut world, 2), [0.0, 0.5]);
    assert_eq!(frame(&mut world, 2), [0.0, 0.0]);
    controller
        .set_control_bus_n(2, &[0.25, 0.75])
        .expect("c_setn");
    assert_eq!(frame(&mut world, 2), [0.25, 0.75]);
    assert_eq!(frame(&mut world, 2), [0.0, 0.0]);
}

#[test]
fn out_kr_sums_onto_a_c_set_in_the_same_block() {
    // The `/c_set` counts as the block's first write, so `Out.kr` sums onto it; the block after,
    // `Out.kr` is the first writer again and copies.
    let (mut controller, _nrt, mut world) = world(1);
    controller.add_synthdef(writer("w", 3.0, 0.75));
    controller.add_synthdef(reader("r", "In", vec![c(3.0)], 1));
    controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .expect("writer");
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    assert_eq!(frame(&mut world, 1), [0.75]);
    controller.set_control_bus(3, 0.5).expect("c_set");
    assert_eq!(frame(&mut world, 1), [1.25]);
    assert_eq!(frame(&mut world, 1), [0.75]);
}

#[test]
fn in_trig_ahead_of_its_writer_reads_nothing() {
    // A writer later in the tree touches the channel after the reader has looked at it.
    let (mut controller, _nrt, mut world) = world(1);
    controller.add_synthdef(writer("w", 3.0, 0.75));
    controller.add_synthdef(reader("r", "InTrig", vec![c(3.0)], 1));
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .expect("writer");
    assert_eq!(frame(&mut world, 1), [0.0]);
    assert_eq!(frame(&mut world, 1), [0.0]);
}

/// scsynth's `LagIn.kr(4, 1, 0.1)` stepping towards a bus value of 1 from 0.
const LAG_STEPS: [u32; 4] = [0x3db433a8, 0x3e2c461c, 0x3e773770, 0x3e9dc854];

fn lag_in_steps(reblock: Option<usize>) -> Vec<u32> {
    let (mut controller, _nrt, mut world) = world(1);
    let def = reader("r", "LagIn", vec![c(4.0), c(0.1)], 1);
    match reblock {
        Some(block) => controller.add_synthdef_reblocked(def, block),
        None => controller.add_synthdef(def),
    }
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    assert_eq!(frame(&mut world, 1), [0.0]);
    controller.set_control_bus(4, 1.0).expect("c_set");
    (0..4).map(|_| frame(&mut world, 1)[0].to_bits()).collect()
}

#[test]
fn lag_in_smooths_towards_the_bus_value() {
    assert_eq!(lag_in_steps(None), LAG_STEPS);
}

#[test]
fn lag_in_steps_once_per_world_block_when_reblocked() {
    // The coefficient comes from the World's control rate and only the first tick steps, so a
    // graph reblocked to 16 samples follows the same curve.
    assert_eq!(lag_in_steps(Some(16)), LAG_STEPS);
}

#[test]
fn lag_in_reads_the_bus_unsmoothed_in_its_constructor() {
    let (mut controller, _nrt, mut world) = world(1);
    controller.add_synthdef(reader("r", "LagIn", vec![c(4.0), c(0.1)], 1));
    controller.set_control_bus(4, 0.5).expect("c_set");
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    assert_eq!(frame(&mut world, 1), [0.5]);
}

#[test]
fn lag_in_smooths_at_most_sixteen_channels() {
    // Channel 16 of a 17-channel `LagIn` outputs zero whatever its bus holds.
    let (mut controller, _nrt, mut world) = world(17);
    controller.add_synthdef(reader("r", "LagIn", vec![c(0.0), c(0.0)], 17));
    controller.set_control_bus_n(0, &[1.0; 17]).expect("c_setn");
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    let f = frame(&mut world, 17);
    assert!(f[..16].iter().all(|&v| v == 1.0), "{f:?}");
    assert_eq!(f[16], 0.0);
}

#[test]
fn a_bus_number_that_never_fits_reads_from_channel_zero() {
    // scsynth's bus window starts at control bus 0 and moves only to a bus number whose channels
    // all fit. A negative bus never does, and `readControlBus` only checks the requested channel
    // (`-1 + i`) against the top of the bus, so the channels read buses 0 and 1.
    let (mut controller, _nrt, mut world) = world(2);
    controller.add_synthdef(reader("r", "LagIn", vec![c(-1.0), c(0.0)], 2));
    controller
        .set_control_bus_n(0, &[0.25, 0.5])
        .expect("c_setn");
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .expect("reader");
    assert_eq!(frame(&mut world, 2), [0.25, 0.5]);
}
