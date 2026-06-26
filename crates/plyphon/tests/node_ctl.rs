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

/// Render `n` control blocks, returning each block's first sample.
fn many(world: &mut World, n: usize) -> Vec<f32> {
    (0..n).map(|_| one(world)).collect()
}

/// `Line.kr(0, 1, 0.01, doneAction: 0)`: ramps over ~0.01 s (~7.5 control blocks at 48k/64) then
/// marks itself done *without* freeing the synth - the source a `Done`/`*WhenDone` watcher observes.
fn line_unit() -> UnitSpec {
    UnitSpec::new(
        "Line",
        Rate::Control,
        vec![
            InputRef::Constant(0.0),
            InputRef::Constant(1.0),
            InputRef::Constant(0.01),
            InputRef::Constant(0.0),
        ],
        1,
    )
}

#[test]
fn done_reports_source_completion() {
    // Done.kr(line) outputs 0 then 1; MulAdd.ar broadcasts the control flag to the audio output.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "done".to_string(),
        params: vec![],
        units: vec![
            line_unit(),
            UnitSpec::new(
                "Done",
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
        .synth_new("done", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    assert_eq!(one(&mut world), 0.0, "Done is 0 while the source runs");
    let later = many(&mut world, 20);
    assert_eq!(
        *later.last().unwrap(),
        1.0,
        "Done latches to 1 once the source has finished"
    );
}

#[test]
fn free_self_when_done_frees_at_source_completion() {
    // DC.ar(1) -> Out keeps the synth audible; FreeSelfWhenDone.kr(line) frees it when the line ends.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "fswd".to_string(),
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
            line_unit(),
            UnitSpec::new(
                "FreeSelfWhenDone",
                Rate::Control,
                vec![InputRef::Unit { unit: 2, output: 0 }],
                0,
            ),
        ],
    });
    controller
        .synth_new("fswd", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let blocks = many(&mut world, 20);
    assert_eq!(blocks[0], 1.0, "audible while the line runs");
    assert_eq!(
        *blocks.last().unwrap(),
        0.0,
        "freed once the line completes"
    );
}

/// A constant-output "victim" synth (`DC.ar(1) -> Out`) that another node frees or pauses by id.
fn victim_def(name: &str) -> SynthDef {
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
        ],
    }
}

#[test]
fn free_frees_another_node_by_id() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(victim_def("victim"));
    let victim = controller
        .synth_new("victim", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    // A controller node: In.kr(bus 0) -> Free.kr(trig, victim id). Added at the tail, so it runs
    // after the victim each block.
    controller.add_synthdef(SynthDef {
        name: "freer".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Control, vec![InputRef::Constant(0.0)], 1),
            UnitSpec::new(
                "Free",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(victim as f32),
                ],
                0,
            ),
        ],
    });
    controller.set_control_bus(0, 0.0).unwrap();
    controller
        .synth_new("freer", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    assert_eq!(one(&mut world), 1.0, "victim runs");
    controller.set_control_bus(0, 1.0).unwrap();
    assert_eq!(
        one(&mut world),
        1.0,
        "firing block still produces the victim's output"
    );
    assert_eq!(one(&mut world), 0.0, "victim freed by the other node");
    assert_eq!(one(&mut world), 0.0, "stays freed");
}

#[test]
fn pause_pauses_and_resumes_another_node_by_id() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(victim_def("victim"));
    let victim = controller
        .synth_new("victim", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    // Pause.kr(gate, victim id): pauses on a falling gate, resumes on a rising one.
    controller.add_synthdef(SynthDef {
        name: "pauser".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Control, vec![InputRef::Constant(0.0)], 1),
            UnitSpec::new(
                "Pause",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(victim as f32),
                ],
                0,
            ),
        ],
    });
    controller.set_control_bus(0, 1.0).unwrap();
    controller
        .synth_new("pauser", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    assert_eq!(one(&mut world), 1.0, "victim runs (gate high)");
    // Gate low: Pause fires this block (victim still heard), pauses the victim after.
    controller.set_control_bus(0, 0.0).unwrap();
    assert_eq!(one(&mut world), 1.0, "firing block still produces output");
    assert_eq!(one(&mut world), 0.0, "victim paused");
    // Gate high again: resume queued this (still-silent) block, audible the next.
    controller.set_control_bus(0, 1.0).unwrap();
    assert_eq!(
        one(&mut world),
        0.0,
        "resume queued, victim still paused this block"
    );
    assert_eq!(one(&mut world), 1.0, "victim resumed");
}
