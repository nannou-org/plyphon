//! Allocation checks for warmed SC3 processing callbacks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};
use plyphon_dsp::buffer::SpectrumCoord;

/// Sample rate shared by the real-time callback checks.
const SAMPLE_RATE: f64 = 48_000.0;
/// Control block size shared by the real-time callback checks.
const BLOCK_SIZE: usize = 64;
/// FFT size used by all spectral callback checks.
const FFT_SIZE: usize = 1_024;

/// Serializes allocator measurement windows if the test binary is run without
/// the validation command's `--test-threads=1`.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());
/// Enables allocation counting only around the callback under test.
static COUNTING: AtomicBool = AtomicBool::new(false);
/// Counts successful heap allocations during the current measurement window.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// A [`System`] wrapper that counts successful allocations while enabled.
struct CountingAllocator;

// SAFETY: every operation delegates to `System` with the original pointer and
// layout. The extra atomics neither modify nor retain allocation metadata.
unsafe impl GlobalAlloc for CountingAllocator {
    /// Delegates allocation to `System` and records successful measured calls.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller provides the `GlobalAlloc` contract for `layout`.
        let pointer = unsafe { System.alloc(layout) };
        record_success(pointer);
        pointer
    }

    /// Delegates zeroed allocation to `System` and records successful measured calls.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller provides the `GlobalAlloc` contract for `layout`.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        record_success(pointer);
        pointer
    }

    /// Delegates deallocation to `System` without changing measurement state.
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller provides the `GlobalAlloc` contract for this pair.
        unsafe { System.dealloc(pointer, layout) };
    }

    /// Delegates reallocation to `System` and records a successful replacement.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller provides the `GlobalAlloc` contract for this
        // pointer, layout, and replacement size.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        record_success(replacement);
        replacement
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Records one successful allocation when a measurement window is active.
fn record_success(pointer: *mut u8) {
    if !pointer.is_null() && COUNTING.load(Ordering::Relaxed) {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Disables allocation counting even if a measured callback unwinds.
struct MeasurementWindow;

impl MeasurementWindow {
    /// Resets the counter and begins one allocation measurement window.
    fn begin() -> Self {
        ALLOCATIONS.store(0, Ordering::SeqCst);
        COUNTING.store(true, Ordering::SeqCst);
        Self
    }

    /// Ends the window and returns its successful-allocation count.
    fn finish(self) -> usize {
        COUNTING.store(false, Ordering::SeqCst);
        ALLOCATIONS.load(Ordering::SeqCst)
    }
}

impl Drop for MeasurementWindow {
    /// Ensures counting is disabled if a measured callback unwinds.
    fn drop(&mut self) {
        COUNTING.store(false, Ordering::SeqCst);
    }
}

/// Runs `callback` inside a fresh allocation-counting window.
fn measured_allocations(callback: impl FnOnce()) -> usize {
    let window = MeasurementWindow::begin();
    callback();
    window.finish()
}

/// A prebuilt engine whose next `fill` call contains no construction work.
struct RenderHarness {
    /// Real-time engine side containing the warmed graph.
    world: World,
    /// Preallocated interleaved hardware-output block.
    output: Vec<f32>,
}

impl RenderHarness {
    /// Warms graph installation and one processing block before measurement.
    fn warm(mut world: World) -> Self {
        let mut output = vec![0.0; BLOCK_SIZE];
        world.fill(&mut output, 1);
        Self { world, output }
    }

    /// Processes one block while retaining an observable output value.
    fn process(&mut self) {
        self.world.fill(&mut self.output, 1);
        black_box(self.output[0]);
    }
}

/// Shorthand for a constant UGen input.
fn constant(value: f32) -> InputRef {
    InputRef::Constant(value)
}

/// Shorthand for output zero of an earlier UGen.
fn wire(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// Builds a warmed one-voice graph for one SC3 calculation unit.
fn calc_harness(name: &'static str, inputs: Vec<InputRef>, outputs: usize) -> RenderHarness {
    let mut units = vec![
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![constant(220.0), constant(0.0)],
            1,
        ),
        UnitSpec::new(name, Rate::Audio, inputs, outputs),
    ];
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![constant(0.0), wire(1)],
        0,
    ));

    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
        output_channels: 1,
        ..Options::default()
    });
    let def_name = format!("rt-{name}");
    controller.add_synthdef(SynthDef {
        name: def_name.clone(),
        params: vec![],
        units,
    });
    controller
        .synth_new(&def_name, ROOT_GROUP_ID, AddAction::Tail)
        .expect("SC3 calculation graph compiles");
    RenderHarness::warm(world)
}

/// Builds a warmed graph whose next callback pulls, resets, and shared-RNG
/// reseeds `DNoiseRing`.
fn demand_harness() -> RenderHarness {
    let units = vec![
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![
                constant(0.5),
                constant(0.5),
                constant(1.0),
                constant(8.0),
                constant(1.0),
            ],
            1,
        ),
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![
                constant(SAMPLE_RATE as f32 / (2.0 * BLOCK_SIZE as f32)),
                constant(0.0),
            ],
            1,
        ),
        UnitSpec::new("RandSeed", Rate::Audio, vec![wire(1), constant(42.0)], 1),
        UnitSpec::new(
            "Duty",
            Rate::Audio,
            vec![
                constant(1.0 / SAMPLE_RATE as f32),
                wire(1),
                constant(0.0),
                wire(0),
            ],
            1,
        ),
        UnitSpec::new("Out", Rate::Audio, vec![constant(0.0), wire(3)], 0),
    ];
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "rt-dnoise-ring".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("rt-dnoise-ring", ROOT_GROUP_ID, AddAction::Tail)
        .expect("DNoiseRing graph compiles");
    let mut harness = RenderHarness::warm(world);
    // At 375 Hz the audio trigger is high at block zero, low at block one,
    // and high again in the measured block. This settles `Duty`'s edge state
    // so the next callback exercises both reset and `RandSeed`.
    harness.process();
    harness
}

/// Creates one valid packed spectrum for a ready-frame PV callback.
fn spectrum(seed: f32) -> Buffer {
    let mut samples = vec![0.0; FFT_SIZE];
    samples[0] = seed;
    samples[1] = seed + 0.5;
    for (bin, pair) in samples[2..].chunks_exact_mut(2).enumerate() {
        pair[0] = seed + (bin + 1) as f32 * 0.001;
        pair[1] = (bin as f32 * 0.01).sin();
    }
    let mut buffer = Buffer::from_interleaved(samples, 1, SAMPLE_RATE);
    buffer.set_coord(SpectrumCoord::Polar);
    buffer
}

/// Builds a warmed one-voice graph for one ready-frame PV unit.
fn pv_harness(name: &'static str) -> RenderHarness {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(spectrum(1.0)))
        .expect("install spectrum A");

    let inputs = if name == "PV_Morph" {
        controller
            .buffer_set(1, Box::new(spectrum(2.0)))
            .expect("install spectrum B");
        vec![constant(0.0), constant(1.0), constant(0.5)]
    } else {
        vec![constant(0.0), constant(0.5)]
    };
    let def_name = format!("rt-{name}");
    controller.add_synthdef(SynthDef {
        name: def_name.clone(),
        params: vec![],
        units: vec![
            UnitSpec::new(name, Rate::Control, inputs, 1),
            UnitSpec::new("K2A", Rate::Audio, vec![wire(0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![constant(0.0), wire(1)], 0),
        ],
    });
    controller
        .synth_new(&def_name, ROOT_GROUP_ID, AddAction::Tail)
        .expect("SC3 PV graph compiles");
    RenderHarness::warm(world)
}

/// Returns warmed callbacks covering every SC3 processing unit family.
fn all_family_harnesses() -> Vec<(&'static str, RenderHarness)> {
    vec![
        (
            "EnvDetect",
            calc_harness(
                "EnvDetect",
                vec![wire(0), constant(0.001), constant(0.01)],
                1,
            ),
        ),
        (
            "Decimator",
            calc_harness(
                "Decimator",
                vec![wire(0), constant(12_000.0), constant(8.0)],
                1,
            ),
        ),
        (
            "DFM1",
            calc_harness(
                "DFM1",
                vec![
                    wire(0),
                    constant(1_000.0),
                    constant(0.2),
                    constant(1.0),
                    constant(0.0),
                    constant(0.0),
                ],
                1,
            ),
        ),
        (
            "BMoog",
            calc_harness(
                "BMoog",
                vec![wire(0), constant(1_000.0), constant(0.2), constant(0.0)],
                1,
            ),
        ),
        (
            "MoogLadder",
            calc_harness(
                "MoogLadder",
                vec![wire(0), constant(1_000.0), constant(0.2)],
                1,
            ),
        ),
        (
            "MoogVCF",
            calc_harness(
                "MoogVCF",
                vec![wire(0), constant(1_000.0), constant(0.2)],
                1,
            ),
        ),
        ("BlitB3", calc_harness("BlitB3", vec![constant(440.0)], 1)),
        (
            "BlitB3Saw",
            calc_harness("BlitB3Saw", vec![constant(440.0), constant(0.99)], 1),
        ),
        (
            "BlitB3Square",
            calc_harness("BlitB3Square", vec![constant(440.0), constant(0.99)], 1),
        ),
        (
            "BlitB3Tri",
            calc_harness(
                "BlitB3Tri",
                vec![constant(440.0), constant(0.99), constant(0.99)],
                1,
            ),
        ),
        (
            "Perlin3",
            calc_harness("Perlin3", vec![wire(0), constant(0.25), constant(0.75)], 1),
        ),
        (
            "RosslerL",
            calc_harness(
                "RosslerL",
                vec![
                    constant(6_000.0),
                    constant(0.2),
                    constant(0.2),
                    constant(5.7),
                    constant(0.05),
                    constant(0.1),
                    constant(0.0),
                    constant(0.0),
                ],
                3,
            ),
        ),
        ("DNoiseRing", demand_harness()),
        ("PV_Freeze", pv_harness("PV_Freeze")),
        ("PV_MagSmooth", pv_harness("PV_MagSmooth")),
        ("PV_Morph", pv_harness("PV_Morph")),
    ]
}

/// Proves the test allocator reports an intentional heap allocation.
#[test]
fn counting_allocator_positive_control_observes_heap_allocation() {
    let _serial = MEASUREMENT_LOCK.lock().expect("measurement lock");
    let allocations = measured_allocations(|| {
        let allocation = Vec::<u8>::with_capacity(black_box(64));
        black_box(&allocation);
    });
    assert!(
        allocations > 0,
        "the positive control must exercise the counting allocator"
    );
}

/// Proves every warmed SC3 processing callback performs zero heap allocations.
#[test]
fn sc3_processing_callbacks_are_allocation_free_after_warmup() {
    let _serial = MEASUREMENT_LOCK.lock().expect("measurement lock");
    let mut harnesses = all_family_harnesses();

    for (name, harness) in &mut harnesses {
        let allocations = measured_allocations(|| harness.process());
        assert_eq!(
            allocations, 0,
            "{name} allocated during its warmed processing callback"
        );
    }
}
