//! `addReplace` (addAction 4): a new `/s_new` or `/g_new` node takes a target node's exact slot and
//! frees the target (with its whole subtree). Mirrors scsynth's `Node_Replace`: the replaced node's
//! `/n_end` fires before the new node's `/n_go`, the replaced node reports `-1` links (its links are
//! nulled before deletion), and the root group cannot be replaced.

use plyphon::{
    AddAction, Event, InputRef, NodeNotify, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

const SR: f64 = 48_000.0;

fn render(world: &mut World, frames: usize) {
    let mut buf = vec![0.0f32; 128];
    let mut done = 0;
    while done < frames {
        world.fill(&mut buf, 1);
        done += buf.len();
    }
}

fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
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

fn drain(nrt: &mut plyphon::Nrt) -> Vec<Event> {
    nrt.process();
    std::iter::from_fn(|| nrt.poll()).collect()
}

fn ended(events: &[Event], id: i32) -> NodeNotify {
    events
        .iter()
        .find_map(|e| match e {
            Event::NodeEnded(n) if n.node == id => Some(*n),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no /n_end for {id}"))
}

fn started(events: &[Event], id: i32) -> NodeNotify {
    events
        .iter()
        .find_map(|e| match e {
            Event::NodeStarted(n) if n.node == id => Some(*n),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no /n_go for {id}"))
}

fn pos(events: &[Event], id: i32) -> usize {
    events
        .iter()
        .position(|e| matches!(e, Event::NodeStarted(n) | Event::NodeEnded(n) if n.node == id))
        .unwrap_or_else(|| panic!("no lifecycle event for {id}"))
}

#[test]
fn s_new_replace_takes_the_targets_slot() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def());

    // root -> [a, b, c].
    let a = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let b = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let c = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    let _ = drain(&mut nrt);

    // Replace b: the new synth d takes b's slot between a and c.
    let d = controller.synth_new("sine", b, AddAction::Replace).unwrap();
    render(&mut world, 256);
    let events = drain(&mut nrt);

    // scsynth order: the replaced node's /n_end precedes the new node's /n_go.
    assert!(
        pos(&events, b) < pos(&events, d),
        "/n_end(b) must come before /n_go(d), got {events:?}"
    );
    // The replaced node reports -1 links (Node_Replace nulls them before deletion).
    let end_b = ended(&events, b);
    assert_eq!((end_b.parent, end_b.prev, end_b.next), (-1, -1, -1));
    // The new node sits in b's old slot: between a and c, under the root.
    let go_d = started(&events, d);
    assert_eq!(
        (go_d.parent, go_d.prev, go_d.next, go_d.is_group),
        (ROOT_GROUP_ID, a, c, 0)
    );
}

#[test]
fn g_new_replace_deep_frees_the_target_subtree() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def());

    // root -> group g -> synth s.
    let g = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let s = controller.synth_new("sine", g, AddAction::Tail).unwrap();
    render(&mut world, 256);
    let _ = drain(&mut nrt);

    // Replace the group g with a fresh empty group h.
    let h = controller.new_group(g, AddAction::Replace).unwrap();
    render(&mut world, 256);
    let events = drain(&mut nrt);

    // The child s and the group g are both freed; the new group h is announced after them.
    assert!(
        pos(&events, s) < pos(&events, g),
        "child freed before its group"
    );
    assert!(
        pos(&events, g) < pos(&events, h),
        "/n_end(g) before /n_go(h)"
    );
    // h is an empty group in g's old slot under the root.
    let go_h = started(&events, h);
    assert_eq!(
        (go_h.parent, go_h.is_group, go_h.head, go_h.tail),
        (ROOT_GROUP_ID, 1, -1, -1)
    );
}

#[test]
fn replacing_the_root_group_is_a_no_op_and_reclaims_the_graph() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def());
    assert_eq!(world.rt_memory_used(), 0);

    // Targeting the root with addReplace must fail cleanly: the built graph is reclaimed, and no
    // node is created or freed.
    let ghost = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Replace)
        .unwrap();
    render(&mut world, 256);
    let events = drain(&mut nrt);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Event::NodeStarted(n) if n.node == ghost)),
        "no synth should start when replacing the root"
    );
    assert_eq!(
        world.rt_memory_used(),
        0,
        "the rejected replacement's graph block must be reclaimed (no leak)"
    );
}
