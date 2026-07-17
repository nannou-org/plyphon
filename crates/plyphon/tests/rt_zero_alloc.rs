//! Allocator-instrumented coverage for initialized creation and critical lifecycle emission.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use plyphon::{
    AddAction, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

/// Whether system-allocation calls should be counted for the measured RT section.
static TRACKING: AtomicBool = AtomicBool::new(false);
/// Number of allocation/reallocation calls observed while [`TRACKING`] was enabled.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// System allocator wrapper used only by this integration-test executable.
struct CountingAllocator;

// SAFETY: every operation delegates unchanged to `System`; the atomic bookkeeping does not touch
// the allocation or alter allocator semantics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `layout` is forwarded exactly as received from the allocator caller.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are forwarded exactly as received from the allocator caller.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: `layout` is forwarded exactly as received from the allocator caller.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: all arguments are forwarded exactly as received from the allocator caller.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// A synth whose first process tick immediately requests self-free.
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

/// A synth with no output that remains live until its owning group is freed.
fn idle_def() -> SynthDef {
    SynthDef {
        name: "idle".to_string(),
        params: vec![],
        units: vec![],
    }
}

/// The measured RT call allocates a seeded synth and emits start/end without system allocation.
#[test]
fn initialized_create_and_critical_lifecycle_paths_are_rt_zero_alloc() {
    let (mut controller, _nrt, mut world) = engine(Options {
        block_size: 64,
        output_channels: 1,
        max_nodes: 8,
        critical_event_capacity: 1,
        ..Options::default()
    });
    controller.add_synthdef(first_block_free_def());
    controller.ensure_compiled("first-block-free").unwrap();
    let mut output = [0.0; 64];
    world.fill(&mut output, 1);
    controller
        .synth_new_with_initial_controls("first-block-free", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::SeqCst);
    world.fill(&mut output, 1);
    TRACKING.store(false, Ordering::SeqCst);
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        0,
        "initialized creation plus NodeStarted/NodeEnded overflow must not use the system allocator"
    );
}

/// A stalled NRT drain retains the complete near-`4M` ownership wave while advisory traffic
/// saturates its independent ring, without spilling either RT-side queue onto the heap.
#[test]
fn stalled_nrt_preserves_combined_four_m_wave_with_advisory_flood_without_rt_alloc() {
    const MAX_NODES: usize = 8;
    const CRITICAL_CAPACITY: usize = 4 * MAX_NODES;
    const MISSING_TARGET: i32 = 999_999;

    let (mut controller, mut nrt, mut world) = engine(Options {
        block_size: 64,
        output_channels: 1,
        max_nodes: MAX_NODES,
        critical_event_capacity: CRITICAL_CAPACITY,
        command_capacity: 128,
        ..Options::default()
    });
    controller.add_synthdef(idle_def());
    controller.ensure_compiled("idle").unwrap();

    // Fill every non-root tree slot: an umbrella group owns two move targets and the synth cohort.
    // Their starts remain undrained while the next block emits the rest of the ownership wave.
    let umbrella = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let group_a = controller.new_group(umbrella, AddAction::Tail).unwrap();
    let group_b = controller.new_group(umbrella, AddAction::Tail).unwrap();
    let tree_synths = (0..MAX_NODES - 4)
        .map(|_| {
            controller
                .synth_new("idle", group_a, AddAction::Tail)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut output = [0.0; 64];
    world.fill(&mut output, 1);

    // Flood all three advisory classes in the same RT block. The selected synth finishes back at
    // group A's tail, which also makes the exact subtree-terminal order deterministic below.
    let moved = tree_synths[0];
    for index in 0..(MAX_NODES * 2) {
        let target = if index % 2 == 0 { group_b } else { group_a };
        controller
            .move_node(moved, target, AddAction::Tail)
            .unwrap();
        controller.node_run(moved, false).unwrap();
        controller.node_run(moved, true).unwrap();
    }

    // Represent both non-event pending creates (`M`) and initialized event creates (`E == M`).
    // The missing target makes every accepted id realize as exactly one SynthFailed even though the
    // node table is full. Freeing the umbrella then emits the tree cohort's matching terminals.
    let general_failures = (0..MAX_NODES)
        .map(|_| {
            controller
                .synth_new("idle", MISSING_TARGET, AddAction::Tail)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let event_failures = (0..MAX_NODES)
        .map(|_| {
            controller
                .synth_new_with_initial_controls("idle", MISSING_TARGET, AddAction::Tail, &[])
                .unwrap()
        })
        .collect::<Vec<_>>();
    controller.free(umbrella).unwrap();

    ALLOCATIONS.store(0, Ordering::Relaxed);
    TRACKING.store(true, Ordering::SeqCst);
    world.fill(&mut output, 1);
    TRACKING.store(false, Ordering::SeqCst);
    assert_eq!(
        ALLOCATIONS.load(Ordering::Relaxed),
        0,
        "the near-4M critical wave and advisory saturation must not allocate on the RT thread"
    );

    let critical = std::iter::from_fn(|| nrt.poll_critical()).collect::<Vec<_>>();
    let observed = critical
        .iter()
        .map(|event| match event {
            Event::NodeStarted(info) => (0u8, info.node),
            Event::SynthFailed { id } => (1u8, *id),
            Event::NodeEnded(info) => (2u8, info.node),
            Event::NodePaused(_) | Event::NodeResumed(_) | Event::NodeMoved(_) => {
                panic!("advisory event leaked into the critical stream")
            }
        })
        .collect::<Vec<_>>();

    let mut expected = Vec::with_capacity(4 * MAX_NODES - 2);
    expected.extend(
        std::iter::once(umbrella)
            .chain([group_a, group_b])
            .chain(tree_synths.iter().copied())
            .map(|id| (0, id)),
    );
    expected.extend(
        general_failures
            .iter()
            .chain(&event_failures)
            .copied()
            .map(|id| (1, id)),
    );
    expected.extend(
        tree_synths[1..]
            .iter()
            .copied()
            .chain(std::iter::once(moved))
            .chain([group_a, group_b, umbrella])
            .map(|id| (2, id)),
    );
    assert_eq!(expected.len(), 4 * MAX_NODES - 2);
    assert_eq!(
        observed, expected,
        "critical results must be complete and FIFO"
    );

    let advisory = std::iter::from_fn(|| nrt.poll_advisory()).collect::<Vec<_>>();
    assert_eq!(advisory.len(), MAX_NODES);
    assert!(
        advisory
            .iter()
            .any(|event| matches!(event, Event::NodeMoved(_)))
    );
    assert!(
        advisory
            .iter()
            .any(|event| matches!(event, Event::NodePaused(_)))
    );
    assert!(
        advisory
            .iter()
            .any(|event| matches!(event, Event::NodeResumed(_)))
    );
}
