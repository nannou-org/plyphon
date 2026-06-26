//! In-graph node control: `FreeSelf` frees the enclosing synth on a rising trigger; `PauseSelf`
//! pauses it (and it can be resumed, proving it was paused rather than freed).

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

/// Render one control block and return its first sample.
fn one(world: &mut World) -> f32 {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf[0]
}

/// `DC.ar(1) -> Out.ar(0)`, plus `In.kr(control bus 0) -> <node_ctl>`. The constant output reads 1
/// while the synth runs and 0 once it is freed or paused; control bus 0 is the trigger.
fn def(name: &str, node_ctl: &str) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
            UnitSpec::new("In", Rate::Control, vec![InputRef::Constant(0.0)], 1),
            UnitSpec::new(
                node_ctl,
                Rate::Control,
                vec![InputRef::Unit { unit: 2, output: 0 }],
                0,
            ),
        ],
    }
}

#[test]
fn free_self_frees_on_trigger() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(def("fs", "FreeSelf"));
    controller.set_control_bus(0, 0.0).unwrap();
    controller
        .synth_new("fs", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    assert_eq!(one(&mut world), 1.0, "running synth outputs the constant");
    // Trigger: FreeSelf fires this block (which still produces output); the free applies after.
    controller.set_control_bus(0, 1.0).unwrap();
    assert_eq!(one(&mut world), 1.0, "firing block still produces output");
    assert_eq!(one(&mut world), 0.0, "freed synth is silent");
    assert_eq!(one(&mut world), 0.0, "stays freed");
}

#[test]
fn pause_self_pauses_and_resumes() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(def("ps", "PauseSelf"));
    controller.set_control_bus(0, 0.0).unwrap();
    let node = controller
        .synth_new("ps", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    assert_eq!(one(&mut world), 1.0, "running synth outputs the constant");
    // Trigger: PauseSelf fires this block (which still produces output); the pause applies after.
    controller.set_control_bus(0, 1.0).unwrap();
    assert_eq!(one(&mut world), 1.0, "firing block still produces output");
    assert_eq!(one(&mut world), 0.0, "paused synth is silent");
    // Resume: audible again, proving the node was paused (not freed) and is still in the tree.
    controller.node_run(node, true).unwrap();
    assert_eq!(one(&mut world), 1.0, "resumed synth produces output again");
}
