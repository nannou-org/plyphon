//! Demand-rate units: `Duty`/`Demand` consumers pulling `Dseq`/`Dseries`/`Dwhite` sources.
//!
//! Each test drives an audio-rate consumer whose held output *is* the demanded value, so the rendered
//! buffer reads back the produced sequence directly. The whole pull happens on the audio thread; the
//! only off-RT work is `SynthDef` compilation.

use plyphon::{
    AddAction, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Segment duration fed to `Duty` (seconds). At `SR` it is ~96 samples, so sampling the middle of
/// each segment is well clear of any boundary even with `dur * SR` rounding.
const SEG_DUR: f32 = 0.002;
/// Nominal samples per segment, and the offset of a safe mid-segment sample.
const SEG: usize = 96;
const MID: usize = SEG / 2;

/// The value held during segment `k` (sampled at its middle).
fn segment(out: &[f32], k: usize) -> f32 {
    out[MID + k * SEG]
}

/// Render `frames` of `channels`-channel audio, varying the host buffer size to exercise reblocking.
fn render(world: &mut World, channels: usize, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut out = Vec::with_capacity((frames + 512) * channels);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames * channels {
        buf.clear();
        buf.resize(sizes[i % sizes.len()] * channels, 0.0);
        i += 1;
        world.fill(&mut buf, channels);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames * channels);
    out
}

/// `Out.ar(0, Duty.ar(SEG_DUR, 0, level: <last unit>, doneAction))`, appended after the demand
/// `sources`. The `level` input is wired to the last source unit; `done` is the Duty done action.
fn duty_def(name: &str, sources: Vec<UnitSpec>, done: f32) -> SynthDef {
    let level_unit = (sources.len() - 1) as u32;
    let duty = sources.len() as u32;
    let mut units = sources;
    units.push(UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Constant(SEG_DUR),
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: level_unit,
                output: 0,
            },
            InputRef::Constant(done),
        ],
        1,
    ));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: duty,
                output: 0,
            },
        ],
        0,
    ));
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units,
    }
}

fn dseq(items: &[f32], repeats: f32) -> UnitSpec {
    let mut inputs = vec![InputRef::Constant(repeats)];
    inputs.extend(items.iter().map(|&v| InputRef::Constant(v)));
    UnitSpec::new("Dseq", Rate::Demand, inputs, 1)
}

fn start(def: SynthDef) -> (plyphon::Controller, plyphon::Nrt, World, i32) {
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let name = def.name.clone();
    controller.add_synthdef(def);
    let node = controller
        .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (controller, nrt, world, node)
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn dseq_yields_items_in_order_and_loops() {
    // Dseq([0.1, 0.2, 0.3], inf) -> holds each for one segment, then loops.
    let (_c, _n, mut world, _node) = start(duty_def(
        "seq",
        vec![dseq(&[0.1, 0.2, 0.3], f32::INFINITY)],
        0.0,
    ));
    let out = render(&mut world, 1, SEG * 5);
    let got: Vec<f32> = (0..5).map(|k| segment(&out, k)).collect();
    for (k, (g, e)) in got.iter().zip([0.1, 0.2, 0.3, 0.1, 0.2]).enumerate() {
        assert!(
            approx(*g, e),
            "segment {k}: expected {e}, got {g} (all: {got:?})"
        );
    }
}

#[test]
fn dseries_counts_by_step_then_holds_on_exhaustion() {
    // Dseries(length: 4, start: 0.1, step: 0.1) -> 0.1, 0.2, 0.3, 0.4, then NaN (Duty holds 0.4).
    let series = UnitSpec::new(
        "Dseries",
        Rate::Demand,
        vec![
            InputRef::Constant(4.0),
            InputRef::Constant(0.1),
            InputRef::Constant(0.1),
        ],
        1,
    );
    let (_c, _n, mut world, _node) = start(duty_def("series", vec![series], 0.0));
    let out = render(&mut world, 1, SEG * 6);
    for (k, e) in [0.1f32, 0.2, 0.3, 0.4, 0.4, 0.4].into_iter().enumerate() {
        let g = segment(&out, k);
        assert!(approx(g, e), "segment {k}: expected {e}, got {g}");
    }
}

#[test]
fn nested_dseq_flattens_in_order() {
    // Dseq([Dseq([0.1, 0.2]), Dseq([0.3, 0.4])], inf) -> 0.1, 0.2, 0.3, 0.4, then loops. The inner
    // sequences run once each (so they exhaust and the outer advances); the outer loops forever.
    // Exercises the recursive pull and the child-reset on wrap.
    let inner_a = dseq(&[0.1, 0.2], 1.0);
    let inner_b = dseq(&[0.3, 0.4], 1.0);
    let outer = UnitSpec::new(
        "Dseq",
        Rate::Demand,
        vec![
            InputRef::Constant(f32::INFINITY),
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Unit { unit: 1, output: 0 },
        ],
        1,
    );
    let (_c, _n, mut world, _node) = start(duty_def("nested", vec![inner_a, inner_b, outer], 0.0));
    let out = render(&mut world, 1, SEG * 6);
    for (k, e) in [0.1f32, 0.2, 0.3, 0.4, 0.1, 0.2].into_iter().enumerate() {
        let g = segment(&out, k);
        assert!(approx(g, e), "segment {k}: expected {e}, got {g}");
    }
}

#[test]
fn duty_done_action_frees_synth_when_durations_run_out() {
    // dur = Dseq([SEG_DUR, SEG_DUR], 1): after two segments the duration source is exhausted, so the
    // NaN duration fires doneAction 2 (free). The level source keeps the segments distinct.
    let dur = dseq(&[SEG_DUR, SEG_DUR], 1.0);
    let level = dseq(&[0.5, 0.6], 1.0);
    // Wire Duty.dur to the dur Dseq (unit 0) and Duty.level to the level Dseq (unit 1).
    let duty = UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 1, output: 0 },
            InputRef::Constant(2.0),
        ],
        1,
    );
    let out_unit = UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 2, output: 0 },
        ],
        0,
    );
    let def = SynthDef {
        name: "done".to_string(),
        params: vec![],
        units: vec![dur, level, duty, out_unit],
    };
    let (_c, mut nrt, mut world, node) = start(def);

    let out = render(&mut world, 1, SEG * 6);
    assert!(approx(segment(&out, 0), 0.5), "first segment should be 0.5");
    assert!(
        approx(segment(&out, 1), 0.6),
        "second segment should be 0.6"
    );
    // After the durations run out the synth frees itself: the tail is silent.
    assert!(
        out[(SEG * 5)..].iter().all(|s| s.abs() < 1e-6),
        "synth should be silent after its done action frees it"
    );
    // ...and the free surfaces as a NodeEnded notification off the audio thread.
    nrt.process();
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        if matches!(event, Event::NodeEnded { id } if id == node) {
            ended = true;
        }
    }
    assert!(ended, "expected a NodeEnded notification");
}

#[test]
fn dwhite_is_bounded_and_decorrelates_across_instances() {
    // Two instances of one def, one per output channel; each holds a fresh Dwhite(0.2, 0.8) value per
    // segment. Every sample must lie in [0.2, 0.8), and the two instances must differ (per-unit RNG).
    let dwhite = UnitSpec::new(
        "Dwhite",
        Rate::Demand,
        vec![
            InputRef::Constant(f32::INFINITY),
            InputRef::Constant(0.2),
            InputRef::Constant(0.8),
        ],
        1,
    );
    // Channel 0 and channel 1, two instances of the same def.
    let def = |ch: f32| SynthDef {
        name: "rand".to_string(),
        params: vec![],
        units: vec![
            dwhite.clone(),
            UnitSpec::new(
                "Duty",
                Rate::Audio,
                vec![
                    InputRef::Constant(SEG_DUR),
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(ch),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 2,
        ..Options::default()
    });
    let mut a = def(0.0);
    a.name = "rand_a".to_string();
    let mut b = def(1.0);
    b.name = "rand_b".to_string();
    controller.add_synthdef(a);
    controller.add_synthdef(b);
    controller
        .synth_new("rand_a", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller
        .synth_new("rand_b", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let out = render(&mut world, 2, SEG * 8);
    let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
    let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();

    for &s in ch0.iter().chain(ch1.iter()) {
        assert!(
            (0.2..0.8).contains(&s),
            "Dwhite value {s} out of [0.2, 0.8)"
        );
    }
    assert!(ch0 != ch1, "two Dwhite instances should decorrelate");
}

#[test]
fn demand_pulls_one_value_per_trigger() {
    // Demand.ar(Impulse.ar(500), 0, Dseries(inf, 1, 1)): each impulse pulls the next integer, held
    // between triggers. At 500 Hz the period is 96 samples, so segment k holds k + 1.
    let impulse = UnitSpec::new(
        "Impulse",
        Rate::Audio,
        vec![InputRef::Constant(500.0), InputRef::Constant(0.0)],
        1,
    );
    let series = UnitSpec::new(
        "Dseries",
        Rate::Demand,
        vec![
            InputRef::Constant(f32::INFINITY),
            InputRef::Constant(1.0),
            InputRef::Constant(1.0),
        ],
        1,
    );
    let demand = UnitSpec::new(
        "Demand",
        Rate::Audio,
        vec![
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 1, output: 0 },
        ],
        1,
    );
    let out_unit = UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 2, output: 0 },
        ],
        0,
    );
    let def = SynthDef {
        name: "trig".to_string(),
        params: vec![],
        units: vec![impulse, series, demand, out_unit],
    };
    let (_c, _n, mut world, _node) = start(def);
    let out = render(&mut world, 1, SEG * 5);
    for k in 0..5 {
        let g = segment(&out, k);
        assert!(
            approx(g, (k + 1) as f32),
            "segment {k}: expected {}, got {g}",
            k + 1
        );
    }
}
