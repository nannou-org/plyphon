//! The control family: `TrigControl` (a `/n_set` is seen for one block then resets to 0) and (later)
//! `LagControl`. The value is observed by routing the param through `DC.ar` to the output.

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const BLOCK: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: 48_000.0,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    }
}

fn one(world: &mut World) -> f32 {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf[0]
}

/// `DC.ar(param) -> Out.ar(0)`: the parameter's value, observable as the output.
fn def(name: &str, param: Param) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![param],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Param(0)], 1),
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
fn trig_control_pulses_one_block() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(def("t", Param::trig("t", 0.0)));
    let node = controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    assert_eq!(one(&mut world), 0.0, "default 0: no pulse");
    controller.set_control(node, 0, 1.0).unwrap();
    assert_eq!(
        one(&mut world),
        1.0,
        "the /n_set is seen for exactly the block it lands in"
    );
    assert_eq!(one(&mut world), 0.0, "and resets to 0 the next block");
    assert_eq!(one(&mut world), 0.0, "stays 0 until set again");
}

#[test]
fn trig_control_default_fires_on_first_block() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(def("t2", Param::trig("t", 1.0)));
    controller
        .synth_new("t2", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    assert_eq!(
        one(&mut world),
        1.0,
        "a non-zero default fires on the first block"
    );
    assert_eq!(one(&mut world), 0.0, "then resets to 0");
}

#[test]
fn lag_control_dezippers_a_step() {
    let (mut controller, _nrt, mut world) = engine(opts());
    let lag = 0.01; // seconds
    controller.add_synthdef(def("lag", Param::lag("f", 0.0, lag)));
    let node = controller
        .synth_new("lag", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    assert_eq!(one(&mut world), 0.0, "holds at the seeded default");
    controller.set_control(node, 0, 1.0).unwrap();
    // First block after the step: a partial one-pole move, not the full target.
    let first = one(&mut world);
    assert!(
        first > 0.0 && first < 0.95,
        "de-zippers, not instant: got {first}"
    );
    // After ~lag seconds (lag * controlRate blocks) it settles to within 0.1% - the -60 dB target.
    // This also confirms the coefficient uses the *control* rate: at audio rate it would barely move
    // over the same span.
    let control_rate = 48_000.0 / BLOCK as f32; // = 750 blocks/s
    let blocks = (lag * control_rate).ceil() as usize; // ~8
    let mut last = first;
    for _ in 0..blocks {
        last = one(&mut world);
    }
    assert!(
        last > 0.99,
        "settles toward the target after the lag time: got {last}"
    );
}

#[test]
fn lag_control_zero_lag_is_instant() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(def("lag0", Param::lag("f", 0.0, 0.0)));
    let node = controller
        .synth_new("lag0", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    assert_eq!(one(&mut world), 0.0, "holds at the default");
    controller.set_control(node, 0, 1.0).unwrap();
    assert_eq!(
        one(&mut world),
        1.0,
        "a zero lag follows the step instantly"
    );
}
