//! `DelayN` and the per-instance auxiliary-memory mechanism: a delay line sized at compile time from
//! a scalar `maxdelaytime` and folded into the synth's single pool block (the safe analogue of
//! scsynth's `RTAlloc`'d `m_dlybuf`). These tests cover the layout maths, the constant-input
//! requirement, the end-to-end delay, the cold-start guard over recycled (un-zeroed) memory, and the
//! single-alloc/single-free invariant.

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

/// The delay-line length (in `f32`s) `DelayN`'s `build` derives for `maxdelay` - the same
/// `NEXTPOWEROFTWO(ceil(maxdelay*SR + 1) + BUFLENGTH)` formula as scsynth's `DelayUnit_AllocDelayLine`.
fn expected_len(maxdelay: f32) -> u64 {
    let base = (maxdelay.max(0.0) as f64 * SR + 1.0).ceil() as i64;
    ((base + BLOCK as i64).max(1) as u64).next_power_of_two()
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
fn compile_reserves_aux_arena() {
    let maxdelay = 0.01;
    let g = compile(&delay_def("d", 1.0, maxdelay, 0.002)).unwrap();
    let aux_bytes = expected_len(maxdelay) as usize * core::mem::size_of::<f32>();

    // Exactly one unit (the DelayN) declares aux memory, at offset 0, sized to the power-of-two line.
    let aux_units: Vec<_> = g.units().iter().filter(|u| u.aux_size != 0).collect();
    assert_eq!(aux_units.len(), 1, "only DelayN reserves aux memory");
    assert_eq!(aux_units[0].aux_size, aux_bytes);
    assert_eq!(aux_units[0].aux_offset, 0);
    // The block's aux span is exactly that one line (already 8-aligned for a power-of-two length).
    assert_eq!(g.layout().aux.len, aux_bytes);
}

#[test]
fn non_constant_maxdelaytime_rejected() {
    // `maxdelaytime` sizes the line, so it must be a compile-time constant. Wiring it to a control
    // parameter (a non-constant) must fail at compile, not silently mis-size the buffer.
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![Param::control("maxdelay", 0.01)],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Param(0),
                    InputRef::Constant(0.002),
                ],
                1,
            ),
        ],
    };
    assert_eq!(
        compile(&def).map(|_| ()),
        Err(BuildError::AuxRequiresConstant { input: 1 })
    );
}

#[test]
fn two_delays_get_distinct_aux_regions() {
    let (max_a, max_b) = (0.01f32, 0.02f32);
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
                    InputRef::Constant(max_a),
                    InputRef::Constant(0.002),
                ],
                1,
            ),
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(max_b),
                    InputRef::Constant(0.003),
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
    };
    let g = compile(&def).unwrap();
    let aux: Vec<_> = g.units().iter().filter(|u| u.aux_size != 0).collect();
    assert_eq!(aux.len(), 2);
    let (size_a, size_b) = (
        expected_len(max_a) as usize * 4,
        expected_len(max_b) as usize * 4,
    );
    // Packed in calc order: first line at 0, second right after it; both within the aux span.
    assert_eq!(aux[0].aux_size, size_a);
    assert_eq!(aux[1].aux_size, size_b);
    assert_eq!(aux[0].aux_offset, 0);
    assert_eq!(
        aux[1].aux_offset, size_a,
        "second line packs after the first"
    );
    assert_eq!(g.layout().aux.len, size_a + size_b);
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
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail, &[])
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
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    // 40 blocks (2560 samples) >> the 1024-sample line: every slot is overwritten with 1.0.
    let mut sink = vec![0.0f32; BLOCK];
    for _ in 0..40 {
        world.fill(&mut sink, 1);
    }
    controller.free(id1).unwrap();
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail, &[])
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
    // The whole delay line lives in the synth's single pool block, so one dealloc reclaims it:
    // rt-pool usage returns exactly to its pre-create baseline (the single-alloc/single-free invariant).
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(delay_def("d", 1.0, 0.5, 0.1)); // a sizeable line
    let baseline = world.rt_memory_used();

    let id = controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let mut sink = vec![0.0f32; BLOCK];
    world.fill(&mut sink, 1);
    let after_create = world.rt_memory_used();
    assert!(
        after_create > baseline,
        "the delay line should enlarge the synth block (baseline {baseline}, after {after_create})"
    );

    controller.free(id).unwrap();
    world.fill(&mut sink, 1);
    assert_eq!(
        world.rt_memory_used(),
        baseline,
        "freeing the synth returns its whole block, delay line included"
    );
}
