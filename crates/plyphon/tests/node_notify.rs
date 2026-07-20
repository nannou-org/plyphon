//! Node-lifecycle notifications carry the full tree position scsynth sends: `/n_go`/`/n_end`/`/n_off`
//! /`/n_on` report node, parent, prev, next, isGroup (plus head/tail for a group), captured at the
//! moment of the event. A deep free reports its descendants before the group, each with the position
//! it held while still linked - a removed predecessor reads back as `-1`, exactly as scsynth's
//! `Node_StateMsg` (before `Node_Remove`) does.

use plyphon::{
    AddAction, Event, InputRef, NodeNotify, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

const SR: f64 = 48_000.0;

fn render(world: &mut World, frames: usize) {
    let mut buf = vec![0.0f32; 256];
    let mut done = 0;
    while done < frames {
        world.fill(&mut buf, 1);
        done += buf.len();
    }
}

/// `SinOsc.ar(440) -> Out.ar(0)`.
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

fn started(events: &[Event], id: i32) -> NodeNotify {
    events
        .iter()
        .find_map(|e| match e {
            Event::NodeStarted(n) if n.node == id => Some(*n),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no /n_go for {id}"))
}

#[test]
fn lifecycle_notifications_carry_full_position() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def());

    // root -> group G -> [A, B].
    let g = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let a = controller.synth_new("sine", g, AddAction::Tail).unwrap();
    let b = controller.synth_new("sine", g, AddAction::Tail).unwrap();
    render(&mut world, 512);
    let go = drain(&mut nrt);

    // The group's /n_go is a group form (isGroup 1, head/tail); it was empty when announced.
    let gg = started(&go, g);
    assert_eq!(
        (gg.parent, gg.is_group, gg.head, gg.tail),
        (ROOT_GROUP_ID, 1, -1, -1),
        "group /n_go: parented to root, group form, announced empty"
    );
    // A is the first (only) child when announced: prev/next -1, parented to G, synth form.
    let aa = started(&go, a);
    assert_eq!((aa.parent, aa.prev, aa.next, aa.is_group), (g, -1, -1, 0));
    // B is appended after A: prev = A.
    let bb = started(&go, b);
    assert_eq!((bb.parent, bb.prev, bb.next, bb.is_group), (g, a, -1, 0));

    // Deep-free the group: descendants are reported before the group (post-order), each with the
    // position it held when removed, then the now-empty group.
    controller.free(g).unwrap();
    render(&mut world, 1024);
    let ends: Vec<NodeNotify> = drain(&mut nrt)
        .into_iter()
        .filter_map(|e| match e {
            Event::NodeEnded(n) => Some(n),
            _ => None,
        })
        .collect();
    let order: Vec<i32> = ends.iter().map(|n| n.node).collect();
    assert_eq!(
        order,
        vec![a, b, g],
        "children (head->tail) freed before the group"
    );

    let end_a = ends[0];
    // A freed first: predecessor none, successor B still present, parent G still present.
    assert_eq!((end_a.parent, end_a.prev, end_a.next), (g, -1, b));
    let end_b = ends[1];
    // B freed next: A already gone reads back as -1, parent G still present.
    assert_eq!((end_b.parent, end_b.prev, end_b.next), (g, -1, -1));
    let end_g = ends[2];
    // The group last: its real position under root, now empty (head/tail -1).
    assert_eq!(
        (end_g.parent, end_g.is_group, end_g.head, end_g.tail),
        (ROOT_GROUP_ID, 1, -1, -1)
    );
}

/// Count the `SynthFailed` terminals for `id`.
fn fails(events: &[Event], id: i32) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, Event::SynthFailed { id: f } if *f == id))
        .count()
}

/// Count the `/n_end`s for `id`.
fn ends(events: &[Event], id: i32) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, Event::NodeEnded(n) if n.node == id))
        .count()
}

/// A duplicate node id fails the create (scsynth's `kSCErr_DuplicateNodeID`), a missing target
/// group fails it too (scsynth's `/fail`), and each failure emits exactly one `SynthFailed`
/// terminal while the live node keeps its slot and identity.
#[test]
fn duplicate_id_and_bad_target_fail_the_create_and_preserve_the_live_node() {
    let opts = Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    };
    let (mut controller, mut nrt, mut world) = engine(opts);
    controller.add_synthdef(sine_def());
    let live = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    started(&drain(&mut nrt), live);

    // Duplicate id, plain add.
    controller
        .synth_new_with_id(live, "sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    // Duplicate id, replace-with-own-id: the duplicate check precedes the vacate, so nothing is
    // freed (scsynth's Node_New runs before Node_Replace).
    controller
        .synth_new_with_id(live, "sine", live, AddAction::Replace)
        .unwrap();
    // Missing target group.
    let bad_target = controller
        .synth_new("sine", 999_999, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    let events = drain(&mut nrt);
    assert_eq!(fails(&events, live), 2, "each duplicate create fails once");
    assert_eq!(fails(&events, bad_target), 1, "bad target fails once");
    assert_eq!(ends(&events, live), 0, "the live node was not disturbed");
    assert_eq!(ends(&events, bad_target), 0, "a failed id never ends");

    // The original node is still linked and reachable by its id.
    controller.free(live).unwrap();
    render(&mut world, 256);
    assert_eq!(ends(&drain(&mut nrt), live), 1);
}

/// A full tree fails the create with a `SynthFailed` terminal rather than silently dropping it.
#[test]
fn full_tree_emits_synth_failed() {
    let opts = Options {
        sample_rate: SR,
        output_channels: 1,
        // Root plus one synth: the second concurrent synth finds the tree full.
        max_nodes: 2,
        ..Options::default()
    };
    let (mut controller, mut nrt, mut world) = engine(opts);
    controller.add_synthdef(sine_def());
    let live = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let full = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    let events = drain(&mut nrt);
    started(&events, live);
    assert_eq!(fails(&events, full), 1);
    assert_eq!(ends(&events, full), 0);
}

/// A duplicate group id is an idempotent no-op (scsynth's `/g_new` tolerates
/// `kSCErr_DuplicateNodeID`): no second group is created, no second `/n_go` is sent, and the
/// original stays linked and reachable.
#[test]
fn duplicate_group_id_is_an_idempotent_noop() {
    let opts = Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    };
    let (mut controller, mut nrt, mut world) = engine(opts);
    let g = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    started(&drain(&mut nrt), g);

    controller
        .new_group_with_id(g, ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    let events = drain(&mut nrt);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Event::NodeStarted(n) if n.node == g)),
        "no second /n_go for the duplicate group"
    );

    controller.free(g).unwrap();
    render(&mut world, 256);
    assert_eq!(
        ends(&drain(&mut nrt), g),
        1,
        "exactly one group remains to end"
    );
}
