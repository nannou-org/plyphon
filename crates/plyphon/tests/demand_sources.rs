//! The demand sources that repeat, total, reset or select the values of their inputs. Each graph is
//! driven by a `Duty` whose held output is one demanded value per segment, so segment `k` reads back
//! the source's `k`-th value (`Duty` holds its previous value over an exhausted `NaN`). Every
//! expected value is the float bit pattern scsynth's own `DemandUGens.cpp` produces for the same
//! graph (stream 0 of a fresh World, `RGen::init(0)`, for the random picks).

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const SEG_DUR: f32 = 0.002;
const SEG: usize = 96;
const MID: usize = SEG / 2;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// A demand-rate unit `name(inputs)`.
fn dem(name: &str, inputs: Vec<InputRef>) -> UnitSpec {
    UnitSpec::new(name, Rate::Demand, inputs, 1)
}

/// `Dseq(items, repeats)` from constants.
fn dseq(items: &[f32], repeats: f32) -> UnitSpec {
    let mut inputs = vec![c(repeats)];
    inputs.extend(items.iter().map(|&v| c(v)));
    dem("Dseq", inputs)
}

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames];
    for chunk in out.chunks_mut(64) {
        world.fill(chunk, 1);
    }
    out
}

/// Play `units` (demand units; the last is the source under test) through
/// `Out.ar(0, Duty.ar(SEG_DUR, 0, 0, source))` and read back `n` segments.
fn segments(mut units: Vec<UnitSpec>, n: usize) -> Vec<f32> {
    let source = units.len() as u32 - 1;
    let duty = units.len() as u32;
    units.push(UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![c(SEG_DUR), c(0.0), c(0.0), u(source)],
        1,
    ));
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(duty)], 0));
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "d".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let out = render(&mut world, SEG * n);
    (0..n).map(|k| out[MID + k * SEG]).collect()
}

/// What `Duty` holds for a source yielding `values`: an exhausted (`NaN`) value holds the previous
/// one.
fn held(values: &[f32]) -> Vec<u32> {
    let mut prev = 0.0f32;
    values
        .iter()
        .map(|&v| {
            if !v.is_nan() {
                prev = v;
            }
            prev.to_bits()
        })
        .collect()
}

fn assert_segments(units: Vec<UnitSpec>, expected: &[f32]) {
    let got = segments(units, expected.len());
    let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, held(expected), "got {got:08x?}");
}

const NAN: f32 = f32::NAN;

#[test]
fn ddup_repeats_each_value_then_ends() {
    // Dseq([Ddup(Dseq([2, 0, 3], 1), Dseq([0.1, 0.2, 0.3, 0.4], 1)), 9], 2): the value is pulled
    // before the count; a count of 0 still yields the value once; 0.4 is consumed by the pull whose
    // count is exhausted. The outer Dseq then moves to 9 and, on its second pass, resets the Ddup,
    // which resets both of its inputs.
    let units = vec![
        dseq(&[2.0, 0.0, 3.0], 1.0),
        dseq(&[0.1, 0.2, 0.3, 0.4], 1.0),
        dem("Ddup", vec![u(0), u(1)]),
        dem("Dseq", vec![c(2.0), u(2), c(9.0)]),
    ];
    let expected = [
        0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 9.0, 0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 9.0, NAN, NAN,
    ];
    assert_segments(units, &expected);
}

#[test]
fn dstutter_rounds_its_count_and_repeats_at_least_once() {
    // Dstutter(Dseq([3, -2, 2.5], inf), Dseq([0.5, 0.25], inf)): counts 3, -2 (yields once) and
    // 2.5 (rounds to 3).
    let units = vec![
        dseq(&[3.0, -2.0, 2.5], f32::INFINITY),
        dseq(&[0.5, 0.25], f32::INFINITY),
        dem("Dstutter", vec![u(0), u(1)]),
    ];
    let expected = [
        0.5, 0.5, 0.5, 0.25, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.5, 0.25,
    ];
    assert_segments(units, &expected);
}

#[test]
fn ddup_pulls_the_value_before_the_count() {
    // Ddup(Diwhite(inf, 1, 3), Dwhite(inf, 0, 1)): both draw from the synth's stream, value first.
    let units = vec![
        dem("Diwhite", vec![c(f32::INFINITY), c(1.0), c(3.0)]),
        dem("Dwhite", vec![c(f32::INFINITY), c(0.0), c(1.0)]),
        dem("Ddup", vec![u(0), u(1)]),
    ];
    let expected = [
        0x3f5b_887e,
        0x3e1d_9c70,
        0x3e1d_9c70,
        0x3ec1_8be0,
        0x3ec1_8be0,
        0x3f22_5838,
        0x3f22_5838,
        0x3f22_5838,
        0x3dc5_4590,
        0x3dc5_4590,
    ]
    .map(f32::from_bits);
    assert_segments(units, &expected);
}
