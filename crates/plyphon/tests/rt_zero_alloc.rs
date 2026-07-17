//! Allocator-instrumented coverage for initialized creation and critical lifecycle emission.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

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
