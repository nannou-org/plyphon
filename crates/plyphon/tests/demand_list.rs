//! The list-selection demand sources: `Dser` (serial, length-counted), `Drand` (random pick) and
//! `Dxrand` (random pick, no immediate repeat). Each is driven by a `Duty` consumer whose held output
//! is the demanded value, so the rendered buffer reads back the produced sequence directly.

use plyphon::{
    AddAction, Controller, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const SR: f64 = 48_000.0;
const SEG_DUR: f32 = 0.002;
const SEG: usize = 96;
const MID: usize = SEG / 2;

/// The value held during segment `k` (sampled at its middle).
fn segment(out: &[f32], k: usize) -> f32 {
    out[MID + k * SEG]
}

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `Out.ar(0, Duty.ar(SEG_DUR, 0, level: <source>, 0))`.
fn drive(source: UnitSpec) -> (Controller, Nrt, World) {
    let duty = UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Constant(SEG_DUR),
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Constant(0.0),
        ],
        1,
    );
    let out = UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 1, output: 0 },
        ],
        0,
    );
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "d".to_string(),
        params: vec![],
        units: vec![source, duty, out],
    });
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    (controller, nrt, world)
}

/// A demand list source `name(length, items...)` from all-constant inputs.
fn source(name: &str, length: f32, items: &[f32]) -> UnitSpec {
    let mut inputs = vec![InputRef::Constant(length)];
    inputs.extend(items.iter().map(|&v| InputRef::Constant(v)));
    UnitSpec::new(name, Rate::Demand, inputs, 1)
}

#[test]
fn dser_cycles_and_counts_values() {
    // Dser(5, [0.1, 0.2, 0.3]) yields exactly 5 values, cycling: 0.1, 0.2, 0.3, 0.1, 0.2, then NaN
    // (Duty holds the last, 0.2).
    let (_c, _n, mut world) = drive(source("Dser", 5.0, &[0.1, 0.2, 0.3]));
    let out = render(&mut world, SEG * 7);
    for (k, e) in [0.1f32, 0.2, 0.3, 0.1, 0.2, 0.2, 0.2]
        .into_iter()
        .enumerate()
    {
        let g = segment(&out, k);
        assert!((g - e).abs() < 1e-5, "segment {k}: expected {e}, got {g}");
    }
}

#[test]
fn drand_picks_from_the_list() {
    // Drand(inf, [0.1, 0.5, 0.9]) - every value is one of the three, and over many draws it visits
    // more than one (it is not stuck).
    let (_c, _n, mut world) = drive(source("Drand", f32::INFINITY, &[0.1, 0.5, 0.9]));
    let out = render(&mut world, SEG * 40);
    let vals: Vec<f32> = (0..40).map(|k| segment(&out, k)).collect();
    for (k, &v) in vals.iter().enumerate() {
        assert!(
            [0.1f32, 0.5, 0.9].iter().any(|&e| (v - e).abs() < 1e-5),
            "segment {k} = {v} not a list item"
        );
    }
    let distinct = vals.iter().filter(|&&v| (v - vals[0]).abs() > 1e-5).count();
    assert!(distinct > 0, "Drand should not be stuck on one value");
}

#[test]
fn dxrand_never_repeats_immediately() {
    // Dxrand(inf, [0.1, 0.5, 0.9]) - every value is a list item, and no two consecutive values are
    // equal.
    let (_c, _n, mut world) = drive(source("Dxrand", f32::INFINITY, &[0.1, 0.5, 0.9]));
    let out = render(&mut world, SEG * 40);
    let vals: Vec<f32> = (0..40).map(|k| segment(&out, k)).collect();
    for (k, &v) in vals.iter().enumerate() {
        assert!(
            [0.1f32, 0.5, 0.9].iter().any(|&e| (v - e).abs() < 1e-5),
            "segment {k} = {v} not a list item"
        );
    }
    for w in vals.windows(2) {
        assert!(
            (w[0] - w[1]).abs() > 1e-5,
            "Dxrand must not repeat immediately: {} then {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn nested_dser_in_dseq_flattens() {
    // Dseq([Dser(2, [0.1, 0.2, 0.3]), 0.9], inf) -> 0.1, 0.2, 0.9, 0.1, 0.2, 0.9, ... The inner Dser
    // emits two values per pass (length 2), then the outer advances to 0.9 and loops.
    let inner = source("Dser", 2.0, &[0.1, 0.2, 0.3]);
    let outer = UnitSpec::new(
        "Dseq",
        Rate::Demand,
        vec![
            InputRef::Constant(f32::INFINITY),
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Constant(0.9),
        ],
        1,
    );
    // Build the graph manually so the inner Dser is unit 0 and the outer Dseq is unit 1.
    let duty = UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Constant(SEG_DUR),
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 1, output: 0 },
            InputRef::Constant(0.0),
        ],
        1,
    );
    let out = UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 2, output: 0 },
        ],
        0,
    );
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "nested".to_string(),
        params: vec![],
        units: vec![inner, outer, duty, out],
    });
    controller
        .synth_new("nested", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let out = render(&mut world, SEG * 6);
    for (k, e) in [0.1f32, 0.2, 0.9, 0.1, 0.2, 0.9].into_iter().enumerate() {
        let g = segment(&out, k);
        assert!((g - e).abs() < 1e-5, "segment {k}: expected {e}, got {g}");
    }
}
