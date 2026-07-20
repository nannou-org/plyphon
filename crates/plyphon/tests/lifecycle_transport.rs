//! Reliable critical lifecycle delivery and isolated advisory-event regressions.

use plyphon::{
    AddAction, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

/// One engine control block.
const BLOCK: usize = 64;

/// A synth with no output that stays alive until explicitly freed.
fn idle_def() -> SynthDef {
    SynthDef {
        name: "idle".to_string(),
        params: vec![],
        units: vec![],
    }
}

/// A synth that requests `Done.freeSelf` on its first process block.
fn first_block_free_def() -> SynthDef {
    SynthDef {
        name: "first-block-free".to_string(),
        params: vec![],
        units: vec![UnitSpec::new(
            "Line",
            Rate::Control,
            vec![
                InputRef::Constant(1.0),
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Constant(2.0),
            ],
            1,
        )],
    }
}

/// Render one block so the RT side drains commands and applies autonomous lifecycle work.
fn render(world: &mut plyphon::World) {
    let mut output = [0.0; BLOCK];
    world.fill(&mut output, 1);
}

/// Return engine options with small explicit node/event limits for backpressure tests.
fn options(max_nodes: usize, critical_event_capacity: usize) -> Options {
    Options {
        block_size: BLOCK,
        output_channels: 1,
        max_nodes,
        critical_event_capacity,
        command_capacity: 128,
        ..Options::default()
    }
}

/// A full critical ring delays, but cannot split or lose, an initialized create result.
#[test]
fn initialized_create_is_indivisible_when_lifecycle_ring_is_full() {
    let (mut controller, mut nrt, mut world) = engine(options(8, 1));
    controller.add_synthdef(idle_def());
    controller.ensure_compiled("idle").unwrap();
    let group = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world);

    let synth = controller
        .synth_new("idle", group, AddAction::Tail, &[])
        .unwrap();
    render(&mut world);
    assert!(matches!(
        nrt.poll_critical(),
        Some(Event::NodeStarted(info)) if info.node == group
    ));
    render(&mut world);
    assert!(matches!(
        nrt.poll_critical(),
        Some(Event::NodeStarted(info)) if info.node == synth
    ));
}

/// Tree capacity and placement failures each produce one failure realization.
#[test]
fn full_tree_and_missing_target_emit_exactly_one_synth_failed() {
    let (mut controller, mut nrt, mut world) = engine(options(2, 16));
    controller.add_synthdef(idle_def());
    let live = controller
        .synth_new("idle", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let full = controller
        .synth_new("idle", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let missing = controller
        .synth_new("idle", 999_999, AddAction::Tail, &[])
        .unwrap();
    render(&mut world);

    let events = std::iter::from_fn(|| nrt.poll_critical()).collect::<Vec<_>>();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::NodeStarted(info) if info.node == live))
            .count(),
        1
    );
    for failed in [full, missing] {
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, Event::SynthFailed { id } if *id == failed))
                .count(),
            1,
            "accepted id {failed} must have exactly one failure terminal"
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, Event::NodeEnded(info) if info.node == failed))
        );
    }
}

/// A caller-chosen id collision fails the accepted add without replacing the live owner.
#[test]
fn caller_chosen_duplicate_id_emits_one_synth_failed_and_preserves_the_live_node() {
    let (mut controller, mut nrt, mut world) = engine(options(4, 16));
    controller.add_synthdef(idle_def());
    let live = controller
        .synth_new("idle", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    render(&mut world);
    while nrt.poll_critical().is_some() {}

    controller
        .synth_new_with_id(live, "idle", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world);
    let failed = std::iter::from_fn(|| nrt.poll_critical()).collect::<Vec<_>>();
    assert_eq!(
        failed
            .iter()
            .filter(|event| matches!(event, Event::SynthFailed { id } if *id == live))
            .count(),
        1
    );

    controller.free(live).unwrap();
    render(&mut world);
    assert!(
        std::iter::from_fn(|| nrt.poll_critical())
            .any(|event| matches!(event, Event::NodeEnded(info) if info.node == live))
    );
}

/// A first-block self-free retains start before end across a ring overflow.
#[test]
fn lifecycle_backlog_is_fifo_for_first_block_self_free() {
    let (mut controller, mut nrt, mut world) = engine(options(4, 1));
    controller.add_synthdef(first_block_free_def());
    let id = controller
        .synth_new("first-block-free", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    render(&mut world);
    assert!(matches!(
        nrt.poll_critical(),
        Some(Event::NodeStarted(info)) if info.node == id
    ));
    render(&mut world);
    assert!(matches!(
        nrt.poll_critical(),
        Some(Event::NodeEnded(info)) if info.node == id
    ));
}

/// A stalled consumer retains the complete start/end wave for a retired group.
#[test]
fn stalled_nrt_preserves_full_start_terminal_and_group_retire_wave() {
    let (mut controller, mut nrt, mut world) = engine(options(8, 32));
    controller.add_synthdef(idle_def());
    let group = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let ids = (0..4)
        .map(|_| {
            controller
                .synth_new("idle", group, AddAction::Tail, &[])
                .unwrap()
        })
        .collect::<Vec<_>>();
    render(&mut world);
    controller.free(group).unwrap();
    render(&mut world);

    let events = std::iter::from_fn(|| nrt.poll_critical()).collect::<Vec<_>>();
    for id in ids {
        let start = events
            .iter()
            .position(|event| matches!(event, Event::NodeStarted(info) if info.node == id))
            .expect("start retained");
        let end = events
            .iter()
            .position(|event| matches!(event, Event::NodeEnded(info) if info.node == id))
            .expect("end retained");
        assert!(start < end);
    }
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::NodeEnded(info) if info.node == group))
    );
}

/// Advisory saturation cannot consume or suppress a child ownership terminal.
#[test]
fn advisory_move_pause_flood_cannot_starve_or_drop_critical_lifecycle_events() {
    let (mut controller, mut nrt, mut world) = engine(options(8, 32));
    controller.add_synthdef(idle_def());
    let a = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let b = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let id = controller
        .synth_new("idle", a, AddAction::Tail, &[])
        .unwrap();
    render(&mut world);
    while nrt.poll_critical().is_some() {}
    while nrt.poll_advisory().is_some() {}

    for index in 0..20 {
        controller
            .move_node(id, if index % 2 == 0 { b } else { a }, AddAction::Tail)
            .unwrap();
        controller.node_run(id, index % 2 == 0).unwrap();
    }
    controller.free(id).unwrap();
    render(&mut world);

    assert!(
        std::iter::from_fn(|| nrt.poll_critical())
            .any(|event| matches!(event, Event::NodeEnded(info) if info.node == id))
    );
}

/// The compatibility poll merges isolated rings by the RT emission sequence.
#[test]
fn merged_poll_preserves_move_pause_then_end_emission_order() {
    let (mut controller, mut nrt, mut world) = engine(options(8, 32));
    controller.add_synthdef(idle_def());
    let a = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let b = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let id = controller
        .synth_new("idle", a, AddAction::Tail, &[])
        .unwrap();
    render(&mut world);
    while nrt.poll().is_some() {}

    controller.move_node(id, b, AddAction::Tail).unwrap();
    controller.node_run(id, false).unwrap();
    controller.free(id).unwrap();
    render(&mut world);
    let events = std::iter::from_fn(|| nrt.poll()).collect::<Vec<_>>();
    assert!(matches!(events.as_slice(), [
        Event::NodeMoved(moved),
        Event::NodePaused(paused),
        Event::NodeEnded(ended),
    ] if moved.node == id && paused.node == id && ended.node == id));
}
