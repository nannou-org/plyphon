//! `Dshuf`: a list shuffled once per reset and played in that order `repeats` times. Each graph is
//! driven by a `Duty` whose held output is one demanded value per segment, so segment `k` reads back
//! the source's `k`-th value (`Duty` holds its previous value over an exhausted `NaN`). Every
//! expected value is what scsynth's own `DemandUGens.cpp` yields for the same graph, drawing from
//! stream 0 of a fresh World (`RGen::init(0)`).

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f64 = 48_000.0;
const SEG_DUR: f32 = 0.002;
const SEG: usize = 96;
const MID: usize = SEG / 2;
const NAN: f32 = f32::NAN;

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

/// Play `units` (demand units; the last is the source under test) through
/// `Out.ar(0, Duty.ar(SEG_DUR, 0, 0, source))` and check its segments against `expected`, where a
/// `NaN` (exhaustion) holds the previous value.
fn assert_segments(mut units: Vec<UnitSpec>, expected: &[f32]) {
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
    let mut out = vec![0.0f32; SEG * expected.len()];
    for chunk in out.chunks_mut(64) {
        world.fill(chunk, 1);
    }
    let got: Vec<u32> = (0..expected.len())
        .map(|k| out[MID + k * SEG].to_bits())
        .collect();
    let mut prev = 0.0f32;
    let held: Vec<u32> = expected
        .iter()
        .map(|&v| {
            if !v.is_nan() {
                prev = v;
            }
            prev.to_bits()
        })
        .collect();
    assert_eq!(got, held, "got {got:08x?}");
}

#[test]
fn plays_one_shuffled_order_for_every_repeat() {
    // Dshuf([1, 2, 3, 4, 5, 6, 7, 8], 2.5): the constructor shuffles once; the repeats round to 3
    // passes of that same order, then NaN.
    let items: Vec<InputRef> = (1..=8).map(|v| c(v as f32)).collect();
    let mut inputs = vec![c(2.5)];
    inputs.extend(items);
    let order = [7.0, 2.0, 3.0, 6.0, 4.0, 1.0, 8.0, 5.0];
    let mut expected: Vec<f32> = order.repeat(3);
    expected.extend([NAN, NAN]);
    assert_segments(vec![dem("Dshuf", inputs)], &expected);
}

#[test]
fn nested_items_play_to_their_end_and_a_reset_shuffles_again() {
    // Dseq([Dshuf([10, Dseq([1, 2], 1), 30, 40, Dseq([5], 1)], 2), 99], 2): a nested Dseq is played
    // to its end and reset when the order reaches it again. The outer Dseq's second pass resets the
    // Dshuf, which shuffles its current order again.
    let units = vec![
        dseq(&[1.0, 2.0], 1.0),
        dseq(&[5.0], 1.0),
        dem("Dshuf", vec![c(2.0), c(10.0), u(0), c(30.0), c(40.0), u(1)]),
        dem("Dseq", vec![c(2.0), u(2), c(99.0)]),
    ];
    let expected = [
        1.0, 2.0, 30.0, 10.0, 40.0, 5.0, 1.0, 2.0, 30.0, 10.0, 40.0, 5.0, 99.0, 1.0, 2.0, 5.0,
        40.0, 30.0, 10.0, 1.0, 2.0, 5.0, 40.0, 30.0, 10.0, 99.0, NAN,
    ];
    assert_segments(units, &expected);
}

#[test]
fn shuffles_and_nested_randoms_share_the_stream() {
    // Dshuf([Dwhite(0, 1, 2), 5, 6, 7], 2): the constructor's shuffle draws three times, then the
    // nested Dwhite draws from the same stream each time the order reaches it.
    let units = vec![
        dem("Dwhite", vec![c(2.0), c(0.0), c(1.0)]),
        dem("Dshuf", vec![c(2.0), u(0), c(5.0), c(6.0), c(7.0)]),
    ];
    let expected = [
        7.0,
        5.0,
        6.0,
        f32::from_bits(0x3f0a_0d84),
        f32::from_bits(0x3ec1_8be0),
        7.0,
        5.0,
        6.0,
        f32::from_bits(0x3eb7_798c),
        f32::from_bits(0x3f22_5838),
        NAN,
    ];
    assert_segments(units, &expected);
}

#[test]
fn a_nested_dshuf_keeps_its_own_order_and_reshuffles_on_each_reset() {
    // Dshuf([Dshuf([1, 2, 3, 4], 1), 10, 20, 30], 3): each unit shuffles its own table. The outer
    // order reaches the inner Dshuf once per pass and resets it, so it shuffles its current order
    // again, drawing between the outer's values.
    let units = vec![
        dem("Dshuf", vec![c(1.0), c(1.0), c(2.0), c(3.0), c(4.0)]),
        dem("Dshuf", vec![c(3.0), u(0), c(10.0), c(20.0), c(30.0)]),
    ];
    let expected = [
        20.0, 3.0, 1.0, 4.0, 2.0, 10.0, 30.0, 20.0, 2.0, 4.0, 3.0, 1.0, 10.0, 30.0, 20.0, 2.0, 1.0,
        4.0, 3.0, 10.0, 30.0, NAN,
    ];
    assert_segments(units, &expected);
}

#[test]
fn a_single_item_is_not_shuffled() {
    // Dseq([Dshuf([5], 2), Dwhite(0, 1, 1)], 1): one item makes no draw, so the Dwhite yields the
    // stream's first value.
    let units = vec![
        dem("Dshuf", vec![c(2.0), c(5.0)]),
        dem("Dwhite", vec![c(1.0), c(0.0), c(1.0)]),
        dem("Dseq", vec![c(1.0), u(0), u(1)]),
    ];
    assert_segments(units, &[5.0, 5.0, f32::from_bits(0x3f5b_887e), NAN]);
}
