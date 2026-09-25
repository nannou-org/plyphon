//! `DelayN` and constructor-time unit memory: a delay line allocated from the unit pool when the synth
//! starts, sized from the first value of `maxdelaytime` (scsynth's constructor `RTAlloc` of
//! `m_dlybuf`). These tests cover the compiled allocation slots, a parameter-sized line, the
//! end-to-end delay, the cold-start guard over recycled (un-zeroed) memory, allocation failure
//! silencing only the failing unit, and every byte returning when the synth ends.

use plyphon::{
    AddAction, BuildError, GraphDef, InputRef, Options, Param, ROOT_GROUP_ID, Rate, RateInfo,
    SynthDef, UnitRegistry, UnitSpec, engine,
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

/// Compile with the built-in registry, returning the `GraphDef` so a test can inspect its layout.
fn compile(def: &SynthDef) -> Result<GraphDef, BuildError> {
    let rate = RateInfo::new(SR, BLOCK);
    def.compile(
        &UnitRegistry::with_builtins(),
        &rate,
        &rate,
        64,
        32,
        None,
        1,
    )
}

/// `DC.ar(level) -> DelayN.ar(in, maxdelay, delay_secs) -> Out.ar(0)`.
fn delay_def(name: &str, level: f32, maxdelay: f32, delay_secs: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(level)], 1),
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(maxdelay),
                    InputRef::Constant(delay_secs),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn delays_allocate_at_start_rather_than_reserve() {
    // Each delay gets its own slot in the per-synth allocation table, in calc order, and reserves no
    // compile-time aux memory: its line is allocated from the unit pool when the synth starts.
    let def = SynthDef {
        name: "two".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.01),
                    InputRef::Constant(0.002),
                ],
                1,
            ),
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.02),
                    InputRef::Constant(0.003),
                ],
                1,
            ),
        ],
    };
    let g = compile(&def).unwrap();
    let slots: Vec<_> = g.units().iter().map(|u| u.pool_slot).collect();
    assert_eq!(slots, vec![None, Some(0), Some(1)]);
    assert_eq!(g.num_pool_slots(), 2);
    assert_eq!(g.layout().aux.len, 0, "no delay reserves compile-time aux");
}

/// `DC.ar(1) -> DelayN.ar(in, maxdelay_param, delay_secs) -> Out.ar(0)`, with `maxdelaytime` wired
/// to a control parameter instead of a constant.
fn param_sized_delay(maxdelay: f32, delay_secs: f32) -> SynthDef {
    SynthDef {
        name: "p".to_string(),
        params: vec![Param::control("maxdelay", maxdelay)],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Param(0),
                    InputRef::Constant(delay_secs),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    }
}

/// Index of the first sample at or above `0.5` (the step where the delayed `DC(1)` arrives).
fn onset(buf: &[f32]) -> Option<usize> {
    buf.iter().position(|&s| s >= 0.5)
}

#[test]
fn maxdelaytime_from_a_parameter_sizes_the_line_at_start() {
    // scsynth reads `maxdelaytime` once in the constructor, whatever it is wired to. A 100-sample delay
    // lands at sample 100 when the parameter allows it; when the parameter caps the line below the
    // requested delay, the tap clamps to the (power-of-two) line instead.
    let delay_secs = 100.0 / SR as f32;
    let render = |def: SynthDef| {
        let (mut controller, _nrt, mut world) = engine(opts());
        controller.add_synthdef(def);
        controller
            .synth_new("p", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();
        let mut buf = vec![0.0f32; 8 * BLOCK];
        world.fill(&mut buf, 1);
        buf
    };

    let roomy = render(param_sized_delay(0.01, delay_secs));
    assert_eq!(onset(&roomy), Some((delay_secs * SR as f32) as usize));

    // `maxdelay = 0` gives a line of `NEXTPOWEROFTWO(1 + 64) = 128` samples, which cannot hold a
    // 1000-sample delay, so the tap clamps to the line and the step arrives early.
    let long = 1000.0 / SR as f32;
    let capped = render(param_sized_delay(0.0, long));
    let clamped = onset(&capped).expect("the clamped delay still passes the signal");
    assert!(
        clamped < 1000,
        "a 128-sample line clamps a 1000-sample delay, got onset {clamped}"
    );
}

#[test]
fn failed_allocation_silences_only_that_unit() {
    // scsynth's `ClearUnitOnMemFailed`: a unit whose `RTAlloc` fails outputs zeros, and the rest of
    // the synth keeps running. Here a DelayN asks for more than the unit pool holds, while a DC in
    // the same synth writes a second channel.
    let (mut controller, nrt, mut world) = engine(Options {
        output_channels: 2,
        unit_pool_bytes: 64 * 1024,
        ..opts()
    });
    controller.add_synthdef(SynthDef {
        name: "big".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(10.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("big", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = vec![0.0f32; 2 * 4 * BLOCK];
    world.fill(&mut buf, 2);

    let (left, right): (Vec<f32>, Vec<f32>) = buf.chunks(2).map(|f| (f[0], f[1])).unzip();
    assert!(left.iter().all(|&s| s == 0.0), "the failed delay is silent");
    assert!(
        right.iter().all(|&s| (s - 1.0).abs() < 1e-6),
        "the rest of the synth keeps running"
    );
    let mut nrt = nrt;
    while let Some(event) = nrt.poll() {
        assert!(
            !matches!(event, plyphon::Event::SynthFailed { .. }),
            "the synth itself starts: {event:?}"
        );
    }
}

#[test]
fn out_of_range_maxdelaytime_silences_without_panicking() {
    // A size far past anything representable (a runaway parameter) must fail the allocation, not
    // overflow the line-length arithmetic.
    for maxdelay in [1.0e30, f32::INFINITY, f32::NAN] {
        let (mut controller, _nrt, mut world) = engine(opts());
        controller.add_synthdef(param_sized_delay(maxdelay, 0.001));
        controller
            .synth_new("p", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();
        let mut buf = vec![0.0f32; 4 * BLOCK];
        world.fill(&mut buf, 1);
        if maxdelay.is_nan() {
            // A NaN `maxdelaytime` clamps to 0, like scsynth's float-to-int conversion of it: the
            // smallest line, which the pool holds.
            continue;
        }
        assert!(
            buf.iter().all(|&s| s == 0.0),
            "maxdelay {maxdelay} is silenced"
        );
    }
}

#[test]
fn delays_dc_by_n_samples_across_blocks() {
    // A constant 1.0 fed through a delay reads back as a step from 0 to 1.0 exactly at the delay
    // length: silence while the read tap is still behind the start of writing, then the (constant)
    // signal. The delay spans more than one control block, proving the line persists across blocks.
    let delay_secs = 100.0 / SR as f32;
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(delay_def("d", 1.0, 0.01, delay_secs));
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let total = 3 * BLOCK;
    let mut buf = vec![0.0f32; total];
    world.fill(&mut buf, 1);

    // The unit truncates `delay_secs * sr` to an integer tap; compute it the same way so the test is
    // robust to f32 rounding (the step lands at exactly this sample).
    let k = (delay_secs * SR as f32) as usize;
    assert!(k > BLOCK, "delay must span >1 block (k = {k})");
    assert!(total > k);
    for (i, &s) in buf.iter().enumerate().take(k) {
        assert!(s.abs() < 1e-6, "pre-delay silence at {i}: {s}");
    }
    for (i, &s) in buf.iter().enumerate().skip(k) {
        assert!((s - 1.0).abs() < 1e-6, "delayed signal at {i}: {s}");
    }
}

#[test]
fn cold_start_clean_over_recycled_memory() {
    // The aux arena is deliberately not zeroed at instantiation. Run one delay synth long enough to
    // fill its line with 1.0, free it, then create an identical synth that reclaims the same (still
    // dirty) pool region. Its cold-start guard must read 0 before its own writes reach the tap - if
    // it instead leaked the previous tenant's 1.0, this would fail.
    let delay_secs = 100.0 / SR as f32;
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(delay_def("d", 1.0, 0.01, delay_secs));

    let id1 = controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    // 40 blocks (2560 samples) >> the 1024-sample line: every slot is overwritten with 1.0.
    let mut sink = vec![0.0f32; BLOCK];
    for _ in 0..40 {
        world.fill(&mut sink, 1);
    }
    controller.free(id1).unwrap();
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // This fill processes the free (dealloc) then the new synth (realloc of the same-sized, dirty
    // region) before running the second instance.
    let total = 3 * BLOCK;
    let mut buf = vec![0.0f32; total];
    world.fill(&mut buf, 1);

    let k = (delay_secs * SR as f32) as usize;
    for (i, &s) in buf.iter().enumerate().take(k) {
        assert!(s.abs() < 1e-6, "stale aux leaked at {i}: {s}");
    }
    for (i, &s) in buf.iter().enumerate().skip(k) {
        assert!(
            (s - 1.0).abs() < 1e-6,
            "second instance's own signal at {i}: {s}"
        );
    }
}

#[test]
fn freeing_a_delay_returns_all_its_memory() {
    // The synth's block and its delay line (allocated at start) both return when it ends: real-time
    // memory use goes back exactly to its pre-create baseline.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(delay_def("d", 1.0, 0.5, 0.1)); // a sizeable line
    let baseline = world.rt_memory_used();

    let id = controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut sink = vec![0.0f32; BLOCK];
    world.fill(&mut sink, 1);
    let after_create = world.rt_memory_used();
    assert!(
        after_create > baseline,
        "the synth and its delay line use memory (baseline {baseline}, after {after_create})"
    );

    controller.free(id).unwrap();
    world.fill(&mut sink, 1);
    assert_eq!(
        world.rt_memory_used(),
        baseline,
        "freeing the synth returns its block and its delay line"
    );
}
