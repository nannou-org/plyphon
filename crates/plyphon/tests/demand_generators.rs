//! The generator demand sources: `Dgeom` (geometric), `Diwhite`/`Dibrown` (integer white/brownian)
//! and `Dbrown` (float brownian). Each is driven by a `Duty` consumer whose held output is the
//! demanded value, so the rendered buffer reads back the produced sequence directly.

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

/// `Out.ar(0, Duty.ar(SEG_DUR, 0, level: <source>, 0))` - the source held one value per segment.
/// Returns the controller and nrt handles too so the caller keeps them alive for the render.
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

fn source(name: &str, inputs: Vec<f32>) -> UnitSpec {
    UnitSpec::new(
        name,
        Rate::Demand,
        inputs.into_iter().map(InputRef::Constant).collect(),
        1,
    )
}

#[test]
fn dgeom_multiplies_by_grow_then_holds() {
    // Dgeom(length: 4, start: 0.1, grow: 2) -> 0.1, 0.2, 0.4, 0.8, then NaN (Duty holds 0.8).
    let (_c, _n, mut world) = drive(source("Dgeom", vec![4.0, 0.1, 2.0]));
    let out = render(&mut world, SEG * 6);
    for (k, e) in [0.1f32, 0.2, 0.4, 0.8, 0.8, 0.8].into_iter().enumerate() {
        let g = segment(&out, k);
        assert!((g - e).abs() < 1e-5, "segment {k}: expected {e}, got {g}");
    }
}

#[test]
fn diwhite_yields_bounded_integers() {
    // Diwhite(inf, 1, 4) -> each segment an integer in [1, 4].
    let (_c, _n, mut world) = drive(source("Diwhite", vec![f32::INFINITY, 1.0, 4.0]));
    let out = render(&mut world, SEG * 12);
    for k in 0..12 {
        let v = segment(&out, k);
        assert!(v.fract() == 0.0, "segment {k} = {v} should be an integer");
        assert!((1.0..=4.0).contains(&v), "segment {k} = {v} out of [1, 4]");
    }
}

#[test]
fn dbrown_walks_within_bounds_and_step() {
    // Dbrown(inf, 0.2, 0.8, 0.1) -> a random walk in [0.2, 0.8], each step no larger than 0.1.
    let (_c, _n, mut world) = drive(source("Dbrown", vec![f32::INFINITY, 0.2, 0.8, 0.1]));
    let out = render(&mut world, SEG * 30);
    let vals: Vec<f32> = (0..30).map(|k| segment(&out, k)).collect();
    for (k, &v) in vals.iter().enumerate() {
        assert!(
            (0.2..=0.8).contains(&v),
            "segment {k} = {v} out of [0.2, 0.8]"
        );
    }
    for w in vals.windows(2) {
        assert!(
            (w[1] - w[0]).abs() < 0.1 + 1e-4,
            "step {} exceeds 0.1",
            (w[1] - w[0]).abs()
        );
    }
    // A random walk should actually move, not sit still.
    let span = vals.iter().cloned().fold(f32::MAX, f32::min)
        ..vals.iter().cloned().fold(f32::MIN, f32::max);
    assert!(span.end - span.start > 0.05, "the walk should wander");
}

#[test]
fn dibrown_walks_integers_within_bounds_and_step() {
    // Dibrown(inf, 1, 5, 1) -> an integer random walk in [1, 5], each step at most 1.
    let (_c, _n, mut world) = drive(source("Dibrown", vec![f32::INFINITY, 1.0, 5.0, 1.0]));
    let out = render(&mut world, SEG * 30);
    let vals: Vec<f32> = (0..30).map(|k| segment(&out, k)).collect();
    for (k, &v) in vals.iter().enumerate() {
        assert!(v.fract() == 0.0, "segment {k} = {v} should be an integer");
        assert!((1.0..=5.0).contains(&v), "segment {k} = {v} out of [1, 5]");
    }
    for w in vals.windows(2) {
        assert!((w[1] - w[0]).abs() <= 1.0, "integer step exceeds 1");
    }
}
