//! Constructor-time unit memory: units whose memory size comes from an input allocate it from the
//! unit pool when the synth starts, reading the input's first value as scsynth's constructors do
//! (`ZIN0`/`IN0` then `RTAlloc`). A failed allocation silences only that unit and marks it done
//! (`ClearUnitOnMemFailed`), and every allocation returns to the pool when the synth ends.

use plyphon::{
    AddAction, CommandTime, Event, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, engine,
};

const BLOCK: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: 48_000.0,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    }
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

fn p(param: u32) -> InputRef {
    InputRef::Param(param)
}

/// `SinOsc.ar(220)` then `unit` (its signal input is unit 0, its size input the `size` parameter),
/// then `Out.ar(0, unit)`.
fn sized_by_param(name: &str, size: f32, unit: UnitSpec) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![Param::control("size", size)],
        units: vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
            unit,
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
    }
}

/// Run `def` for `blocks` blocks, returning the output and whether any synth failed to start.
fn run(def: SynthDef, options: Options, blocks: usize) -> (Vec<f32>, bool) {
    let (mut controller, mut nrt, mut world) = engine(options);
    let name = def.name.clone();
    controller.add_synthdef(def);
    controller
        .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut out = vec![0.0f32; blocks * BLOCK];
    world.fill(&mut out, 1);
    let mut failed = false;
    while let Some(event) = nrt.poll() {
        failed |= matches!(event, Event::SynthFailed { .. });
    }
    (out, failed)
}

#[test]
fn every_start_sized_unit_accepts_a_wired_size() {
    // Each unit's size input is wired to a parameter (a non-constant), read at synth start.
    let cases = [
        (
            "comb",
            0.05,
            UnitSpec::new("CombC", Rate::Audio, vec![u(0), p(0), c(0.01), c(0.5)], 1),
        ),
        (
            "pitch_shift",
            0.1,
            UnitSpec::new(
                "PitchShift",
                Rate::Audio,
                vec![u(0), p(0), c(1.5), c(0.0), c(0.0)],
                1,
            ),
        ),
        (
            "limiter",
            0.001,
            UnitSpec::new("Limiter", Rate::Audio, vec![u(0), c(0.5), p(0)], 1),
        ),
        (
            "normalizer",
            0.001,
            UnitSpec::new("Normalizer", Rate::Audio, vec![u(0), c(0.5), p(0)], 1),
        ),
        (
            "gverb",
            10.0,
            UnitSpec::new(
                "GVerb",
                Rate::Audio,
                vec![
                    u(0),
                    p(0),
                    c(3.0),
                    c(0.5),
                    c(0.5),
                    c(15.0),
                    c(1.0),
                    c(0.7),
                    c(0.5),
                    c(30.0),
                ],
                2,
            ),
        ),
        (
            "median",
            3.0,
            UnitSpec::new("Median", Rate::Audio, vec![p(0), u(0)], 1),
        ),
    ];
    for (name, size, unit) in cases {
        let (out, failed) = run(sized_by_param(name, size, unit), opts(), 32);
        assert!(!failed, "{name}: the synth starts");
        assert!(out.iter().all(|s| s.is_finite()), "{name}: finite output");
        assert!(
            out.iter().any(|&s| s.abs() > 1e-3),
            "{name}: a parameter-sized unit processes"
        );
    }
}

#[test]
fn gverb_room_larger_than_its_maximum_is_clamped_at_start() {
    // scsynth's `gverb_set_roomsize` puts a `roomsize` at or above `maxroomsize` at
    // `maxroomsize - 1`, so the FDN lines stay inside their `maxroomsize`-sized buffers.
    let gverb = UnitSpec::new(
        "GVerb",
        Rate::Audio,
        vec![
            u(0),
            p(0),
            c(3.0),
            c(0.5),
            c(0.5),
            c(15.0),
            c(1.0),
            c(0.7),
            c(0.5),
            c(10.0),
        ],
        2,
    );
    let (out, failed) = run(sized_by_param("big_room", 1.0e9, gverb), opts(), 32);
    assert!(!failed);
    assert!(out.iter().all(|s| s.is_finite()));
    assert!(out.iter().any(|&s| s.abs() > 1e-3));
}

#[test]
fn a_failed_allocation_marks_its_unit_done() {
    // `ClearUnitOnMemFailed` sets the unit's done flag, so `FreeSelfWhenDone` watching it frees the
    // synth - here a DelayN that asks for more than the unit pool holds.
    let def = SynthDef {
        name: "watched".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("DelayN", Rate::Audio, vec![u(0), c(10.0), c(0.0)], 1),
            UnitSpec::new("FreeSelfWhenDone", Rate::Control, vec![u(1)], 1),
        ],
    };
    let (mut controller, mut nrt, mut world) = engine(Options {
        unit_pool_bytes: 64 * 1024,
        ..opts()
    });
    controller.add_synthdef(def);
    let id = controller
        .synth_new("watched", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut out = vec![0.0f32; 4 * BLOCK];
    world.fill(&mut out, 1);
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        assert!(!matches!(event, Event::SynthFailed { .. }));
        ended |= matches!(event, Event::NodeEnded(info) if info.node == id);
    }
    assert!(
        ended,
        "the failed unit reports done and the watcher frees the synth"
    );
}

#[test]
fn memory_returns_when_the_synth_ends() {
    // A unit pool that holds exactly one of these delay lines: the second synth can only allocate
    // once the first has ended and returned its line. Each synth plays its own level, so the output
    // shows which one is sounding.
    let def = |name: &str, level: f32| SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(level)], 1),
            UnitSpec::new("DelayN", Rate::Audio, vec![u(0), c(0.5), c(0.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        // A 0.5 s line is 32768 samples (128 KiB); leave room for one plus pool overhead.
        unit_pool_bytes: 192 * 1024,
        ..opts()
    });
    controller.add_synthdef(def("a", 1.0));
    controller.add_synthdef(def("b", 0.25));
    let mut out = vec![0.0f32; BLOCK];

    let a = controller
        .synth_new("a", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    world.fill(&mut out, 1);
    // DelayN's minimum delay is one sample, so the step lands at sample 1.
    assert!(out[1..].iter().all(|&s| (s - 1.0).abs() < 1e-6), "a plays");

    controller.free(a).unwrap();
    controller
        .synth_new("b", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    world.fill(&mut out, 1);
    assert!(
        out[1..].iter().all(|&s| (s - 0.25).abs() < 1e-6),
        "b allocates the line a returned"
    );
}

#[test]
fn a_cleared_scheduled_synth_returns_its_table_off_the_audio_thread() {
    // A scheduled synth creation carries its allocation table; clearing the schedule routes it to
    // the NRT side rather than dropping it on the audio thread.
    let def = SynthDef {
        name: "later".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("DelayN", Rate::Audio, vec![u(0), c(0.01), c(0.0)], 1),
        ],
    };
    let (mut controller, mut nrt, mut world) = engine(opts());
    controller.add_synthdef(def);
    controller.ensure_compiled("later").unwrap();
    controller.begin_scheduled(CommandTime::At(u64::MAX / 2));
    controller
        .synth_new("later", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller.end_scheduled();
    let mut out = vec![0.0f32; BLOCK];
    world.fill(&mut out, 1);
    controller.clear_sched().unwrap();
    world.fill(&mut out, 1);
    assert_eq!(nrt.process(), 1, "the table reaches the NRT side");
}
