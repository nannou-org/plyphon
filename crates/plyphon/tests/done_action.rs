//! Exercise done actions and the NRT event/trash flow: a `Line.kr(..., doneAction: 2)` amplitude
//! envelope frees its own synth when it finishes; the `Nrt` then drops the freed synth and reports
//! a `NodeEnded` event.

use std::collections::HashSet;

use plyphon::{
    AddAction, Event, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f32 = 48_000.0;

fn render(world: &mut plyphon::World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `SinOsc.ar(440) * Line.kr(1, 0, 0.1, doneAction: code)` -> `Out`. The line ramps the amplitude to
/// zero over 0.1 s, then fires done action `code` on the enclosing synth.
fn done_synth(name: &str, code: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            // Line.kr(1, 0, 0.1, code): amplitude 1 -> 0, then the done action.
            UnitSpec {
                name: "Line".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.1),
                    InputRef::Constant(code),
                ],
                num_outputs: 1,
                special_index: 0,
            },
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            // SinOsc * Line (BinaryOpUGen, special index 2 = multiply).
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2,
            },
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
    }
}

/// `Line.kr(1, 0, 0.1, doneAction: 2)`-driven sine - the original single-synth free-self case.
fn enveloped_sine() -> SynthDef {
    done_synth("env", 2.0)
}

/// `SinOsc.ar(440)` -> `Out` with no done action: a neighbour that keeps running so a done action on
/// another node has something to free, pause, or leave alone.
fn plain_sine(name: &str) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(330.0), InputRef::Constant(0.0)],
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

/// Drain every lifecycle notification the engine has emitted, as `(started, ended, paused)` id sets.
fn poll_events(nrt: &mut Nrt) -> (HashSet<i32>, HashSet<i32>, HashSet<i32>) {
    nrt.process();
    let (mut started, mut ended, mut paused) = (HashSet::new(), HashSet::new(), HashSet::new());
    while let Some(event) = nrt.poll() {
        match event {
            Event::NodeStarted(n) => {
                started.insert(n.node);
            }
            Event::NodeEnded(n) => {
                ended.insert(n.node);
            }
            Event::NodePaused(n) => {
                paused.insert(n.node);
            }
            _ => {}
        }
    }
    (started, ended, paused)
}

#[test]
fn line_done_action_frees_synth() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(enveloped_sine());
    let node = controller
        .synth_new("env", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // It plays before the 0.1 s envelope completes.
    let early = render(&mut world, (SR * 0.05) as usize);
    assert!(
        early.iter().any(|s| s.abs() > 0.05),
        "synth should play before the envelope ends"
    );

    // Render past completion; the done action frees the synth.
    let _ = render(&mut world, (SR * 0.2) as usize);

    // The freed synth's state returns to the rt-pool on the audio thread (no trash); the NRT side
    // still surfaces the lifecycle notifications.
    nrt.process();
    let (mut started, mut ended) = (false, false);
    while let Some(event) = nrt.poll() {
        match event {
            Event::NodeStarted(n) if n.node == node => started = true,
            Event::NodeEnded(n) if n.node == node => ended = true,
            _ => {}
        }
    }
    assert!(started, "expected a NodeStarted notification");
    assert!(
        ended,
        "expected a NodeEnded notification from the done action"
    );

    // With the synth gone, the output is silent.
    let late = render(&mut world, (SR * 0.05) as usize);
    assert!(
        late.iter().all(|s| s.abs() < 1e-6),
        "expected silence after the synth freed itself"
    );
}

fn setup() -> (plyphon::Controller, Nrt, plyphon::World) {
    engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    })
}

/// Code 3: free this synth and the preceding node.
#[test]
fn done_action_3_frees_self_and_prev() {
    let (mut c, mut nrt, mut world) = setup();
    c.add_synthdef(plain_sine("plain"));
    c.add_synthdef(done_synth("d3", 3.0));
    // root -> [a, b]; b's done action frees b and its preceding sibling a.
    let a = c
        .synth_new("plain", ROOT_GROUP_ID, AddAction::Tail)
        .expect("a");
    let b = c
        .synth_new("d3", ROOT_GROUP_ID, AddAction::Tail)
        .expect("b");

    let _ = render(&mut world, (SR * 0.2) as usize);

    let (started, ended, paused) = poll_events(&mut nrt);
    assert!(started.contains(&a) && started.contains(&b));
    assert!(ended.contains(&a), "the preceding node should be freed");
    assert!(ended.contains(&b), "the synth itself should be freed");
    assert!(paused.is_empty());
}

/// Code 4: free this synth and the following node.
#[test]
fn done_action_4_frees_self_and_next() {
    let (mut c, mut nrt, mut world) = setup();
    c.add_synthdef(plain_sine("plain"));
    c.add_synthdef(done_synth("d4", 4.0));
    // root -> [b, a]; b frees b and its following sibling a.
    let b = c
        .synth_new("d4", ROOT_GROUP_ID, AddAction::Tail)
        .expect("b");
    let a = c
        .synth_new("plain", ROOT_GROUP_ID, AddAction::Tail)
        .expect("a");

    let _ = render(&mut world, (SR * 0.2) as usize);

    let (_started, ended, _paused) = poll_events(&mut nrt);
    assert!(ended.contains(&a), "the following node should be freed");
    assert!(ended.contains(&b), "the synth itself should be freed");
}

/// Code 7: free this synth and every preceding node in its group.
#[test]
fn done_action_7_frees_self_to_head() {
    let (mut c, mut nrt, mut world) = setup();
    c.add_synthdef(plain_sine("plain"));
    c.add_synthdef(done_synth("d7", 7.0));
    // root -> [a, b, d]; d frees itself and all preceding siblings.
    let a = c
        .synth_new("plain", ROOT_GROUP_ID, AddAction::Tail)
        .expect("a");
    let b = c
        .synth_new("plain", ROOT_GROUP_ID, AddAction::Tail)
        .expect("b");
    let d = c
        .synth_new("d7", ROOT_GROUP_ID, AddAction::Tail)
        .expect("d");

    let _ = render(&mut world, (SR * 0.2) as usize);

    let (_started, ended, _paused) = poll_events(&mut nrt);
    assert!(
        ended.contains(&a) && ended.contains(&b) && ended.contains(&d),
        "the synth and every preceding sibling should be freed: {ended:?}"
    );
}

/// Code 9: free this synth and pause the preceding node.
#[test]
fn done_action_9_frees_self_and_pauses_prev() {
    let (mut c, mut nrt, mut world) = setup();
    c.add_synthdef(plain_sine("plain"));
    c.add_synthdef(done_synth("d9", 9.0));
    // root -> [a, b]; b frees itself and pauses its preceding sibling a.
    let a = c
        .synth_new("plain", ROOT_GROUP_ID, AddAction::Tail)
        .expect("a");
    let b = c
        .synth_new("d9", ROOT_GROUP_ID, AddAction::Tail)
        .expect("b");

    let _ = render(&mut world, (SR * 0.2) as usize);

    let (_started, ended, paused) = poll_events(&mut nrt);
    assert!(ended.contains(&b), "the synth itself should be freed");
    assert!(
        !ended.contains(&a),
        "the preceding node should not be freed"
    );
    assert!(paused.contains(&a), "the preceding node should be paused");
}

/// Code 13: free this synth and every other node in its group, keeping the group.
#[test]
fn done_action_13_frees_all_in_group() {
    let (mut c, mut nrt, mut world) = setup();
    c.add_synthdef(plain_sine("plain"));
    c.add_synthdef(done_synth("d13", 13.0));
    let g = c.new_group(ROOT_GROUP_ID, AddAction::Tail).expect("g");
    let a = c.synth_new("plain", g, AddAction::Tail).expect("a");
    let b = c.synth_new("d13", g, AddAction::Tail).expect("b");

    let _ = render(&mut world, (SR * 0.2) as usize);

    let (started, ended, _paused) = poll_events(&mut nrt);
    assert!(started.contains(&g));
    assert!(
        ended.contains(&a) && ended.contains(&b),
        "every node in the group should be freed: {ended:?}"
    );
    assert!(!ended.contains(&g), "the enclosing group itself survives");
}

/// Code 14: free the enclosing group and every node within it (this synth included).
#[test]
fn done_action_14_frees_enclosing_group() {
    let (mut c, mut nrt, mut world) = setup();
    c.add_synthdef(plain_sine("plain"));
    c.add_synthdef(done_synth("d14", 14.0));
    let g = c.new_group(ROOT_GROUP_ID, AddAction::Tail).expect("g");
    let a = c.synth_new("plain", g, AddAction::Tail).expect("a");
    let b = c.synth_new("d14", g, AddAction::Tail).expect("b");

    let _ = render(&mut world, (SR * 0.2) as usize);

    let (_started, ended, _paused) = poll_events(&mut nrt);
    assert!(
        ended.contains(&g) && ended.contains(&a) && ended.contains(&b),
        "the enclosing group and everything within it should be freed: {ended:?}"
    );
}
