//! Sub-sample offset support end to end: a synth scheduled at a sub-sample-accurate time reports the
//! fractional part of its within-block offset through the `SubsampleOffset` UGen (scsynth's
//! `mSubsampleOffset`). Exercises the whole chain: OSC/NTP time -> `Clock::block_offset` ->
//! `World.current_subsample_offset` -> `Graph` -> `ProcessCtx::subsample_offset` -> the UGen.

use plyphon::{
    AddAction, CommandTime, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
/// OSC/NTP units per second (2^32), the fixed-point time base the scheduler uses.
const OSC_PER_SEC: f64 = 4_294_967_296.0;

/// OSC/NTP units per audio sample.
fn units_per_sample() -> f64 {
    OSC_PER_SEC / SR
}

/// The sub-sample offset the engine will report for a command `delta` OSC units into the block,
/// computed exactly as `Clock::block_offset` does (scsynth's `+0.5`-biased fractional remainder).
fn expected_subsample(delta: u64) -> f32 {
    let diff = delta as f64 * (SR / OSC_PER_SEC) + 0.5;
    (diff - diff.floor()) as f32
}

/// `SubsampleOffset.ar -> Out.ar(0)`.
fn sub_def() -> SynthDef {
    SynthDef {
        name: "sub".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("SubsampleOffset", Rate::Audio, vec![], 1),
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

fn engine_64() -> (plyphon::Controller, plyphon::Nrt, World) {
    engine(Options {
        sample_rate: SR,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    })
}

#[test]
fn scheduled_synth_reports_its_subsample_offset() {
    // Schedule the synth 10.3 samples into a block whose start is `base`. scsynth's biased split
    // rounds the integer onset to 10 and reports subsample `(10.3 + 0.5).fract() = 0.8`.
    let (mut controller, _nrt, mut world) = engine_64();
    controller.add_synthdef(sub_def());

    let base = 1_000_000_000_000u64;
    let delta = (10.3 * units_per_sample()) as u64;
    controller.begin_scheduled(CommandTime::At(base + delta));
    controller
        .synth_new("sub", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    controller.end_scheduled();

    // Run exactly the block that starts at `base`; the scheduled synth is created and processed in it.
    let mut buf = vec![0.0f32; BLOCK];
    world.fill_at(&mut buf, 1, base);

    let expected = expected_subsample(delta);
    assert!(
        (0.7..0.9).contains(&expected),
        "sanity: 10.3-sample offset should give ~0.8, got {expected}"
    );
    // The UGen holds its value for the whole block (only OffsetOut acts on the integer onset), so
    // every sample reads the captured sub-sample offset.
    for (i, &s) in buf.iter().enumerate() {
        assert!(
            (s - expected).abs() < 1e-4,
            "sample {i}: SubsampleOffset {s} != expected {expected}"
        );
    }
}

#[test]
fn immediate_synth_reports_zero_subsample_offset() {
    // A synth created with no schedule (immediate) has offset 0, so SubsampleOffset reports 0.
    let (mut controller, _nrt, mut world) = engine_64();
    controller.add_synthdef(sub_def());
    controller
        .synth_new("sub", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    for (i, &s) in buf.iter().enumerate() {
        assert!(
            s.abs() < 1e-6,
            "sample {i}: immediate offset should be 0, got {s}"
        );
    }
}

#[test]
fn subsample_offset_is_stable_across_blocks() {
    // The captured value persists after the first block (it is snapshotted once), even though
    // `ctx.subsample_offset` is 0 on every subsequent block.
    let (mut controller, _nrt, mut world) = engine_64();
    controller.add_synthdef(sub_def());

    let base = 2_000_000_000_000u64;
    let delta = (25.6 * units_per_sample()) as u64;
    controller.begin_scheduled(CommandTime::At(base + delta));
    controller
        .synth_new("sub", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    controller.end_scheduled();

    let expected = expected_subsample(delta);
    let mut buf = vec![0.0f32; BLOCK];
    // First block (creates the synth), then two more advancing blocks.
    world.fill_at(&mut buf, 1, base);
    assert!((buf[32] - expected).abs() < 1e-4, "first block {}", buf[32]);
    for _ in 0..2 {
        world.fill(&mut buf, 1);
        assert!(
            (buf[32] - expected).abs() < 1e-4,
            "later block should hold {expected}, got {}",
            buf[32]
        );
    }
}
