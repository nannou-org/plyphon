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

#[test]
fn dconst_yields_the_remainder_that_reaches_the_sum() {
    // Dseq([Dconst(1, Dseq([0.3], inf), 0.001), 7], 2): 0.3 three times, then 1 - 0.9 (in float),
    // then the sequence ends; the outer Dseq's second pass resets it.
    let units = vec![
        dseq(&[0.3], f32::INFINITY),
        dem("Dconst", vec![c(1.0), u(0), c(0.001)]),
        dem("Dseq", vec![c(2.0), u(1), c(7.0)]),
    ];
    let r = f32::from_bits(0x3dcc_ccc8);
    let expected = [0.3, 0.3, 0.3, r, 7.0, 0.3, 0.3, 0.3, r, 7.0, NAN, NAN];
    assert_segments(units, &expected);
}

#[test]
fn dconst_ends_within_tolerance() {
    // Dseq([Dconst(1, Dseq([0.48, 0.49, 0.5], 1), 0.05), 7], 1): 0.48 + 0.49 is within 0.05 of 1, so
    // the second value is the remainder 1 - 0.48.
    let units = vec![
        dseq(&[0.48, 0.49, 0.5], 1.0),
        dem("Dconst", vec![c(1.0), u(0), c(0.05)]),
        dem("Dseq", vec![c(1.0), u(1), c(7.0)]),
    ];
    let expected = [0.48, f32::from_bits(0x3f05_1eb8), 7.0, NAN];
    assert_segments(units, &expected);
}

#[test]
fn dconst_ends_when_its_input_does() {
    // Dseq([Dconst(2, Dseq([0.5, 0.5], 1), 0.001), 7], 1): the input runs out before the sum.
    let units = vec![
        dseq(&[0.5, 0.5], 1.0),
        dem("Dconst", vec![c(2.0), u(0), c(0.001)]),
        dem("Dseq", vec![c(1.0), u(1), c(7.0)]),
    ];
    assert_segments(units, &[0.5, 0.5, 7.0, NAN]);
}

#[test]
fn dreset_resets_its_input_after_the_pull_on_a_rising_reset() {
    // Dreset(Dseq([1, 2, 3, 4, 5], inf), Dseq([0, 0, 1, 1, 0, 1, 0, 0], inf)): the value pulled
    // with a rising reset is still yielded; the input restarts on the next demand.
    let units = vec![
        dseq(&[1.0, 2.0, 3.0, 4.0, 5.0], f32::INFINITY),
        dseq(&[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], f32::INFINITY),
        dem("Dreset", vec![u(0), u(1)]),
    ];
    let expected = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0];
    assert_segments(units, &expected);
}

#[test]
fn dswitch1_yields_one_value_from_the_indexed_item() {
    // Dswitch1([Dseq([10, 11, 12], inf), 20, Dseq([30, 31], 1)], Dseq([0, 1, 2, 5, -1, 0, 2, 2, 2],
    // 1)): indices wrap into the list (5 -> 2, -1 -> 2); an exhausted item and then the exhausted
    // index yield NaN.
    let units = vec![
        dseq(&[10.0, 11.0, 12.0], f32::INFINITY),
        dseq(&[30.0, 31.0], 1.0),
        dseq(&[0.0, 1.0, 2.0, 5.0, -1.0, 0.0, 2.0, 2.0, 2.0], 1.0),
        dem("Dswitch1", vec![u(2), u(0), c(20.0), u(1)]),
    ];
    let expected = [10.0, 20.0, 30.0, 31.0, NAN, 11.0, NAN, NAN, NAN, NAN, NAN];
    assert_segments(units, &expected);
}

#[test]
fn dswitch_plays_each_selected_item_to_its_end() {
    // Dswitch([Dseq([1, 2], 1), Dseq([3, 4, 5], 1), 6], Dseq([1, 0, 2, 4, 1], 1)): the constructor
    // selects item 1; each exhausted item pulls the next index (4 wraps to 0). A constant item is
    // never exhausted.
    let units = vec![
        dseq(&[1.0, 2.0], 1.0),
        dseq(&[3.0, 4.0, 5.0], 1.0),
        dseq(&[1.0, 0.0, 2.0, 4.0, 1.0], 1.0),
        dem("Dswitch", vec![u(2), u(0), u(1), c(6.0)]),
    ];
    let expected = [3.0, 4.0, 5.0, 1.0, 2.0, 6.0, 6.0, 6.0];
    assert_segments(units, &expected);
}

#[test]
fn dswitch_pulls_a_reselected_item_before_resetting_it() {
    // Dswitch([Dseq([1, 2], 1)], Dseq([0, 0, 0, 0], 1)): when the exhausted item is selected again it
    // is pulled (yielding NaN) before the previously selected item - itself - is reset, so every
    // third value is NaN. The constructor took the first index.
    let units = vec![
        dseq(&[1.0, 2.0], 1.0),
        dseq(&[0.0, 0.0, 0.0, 0.0], 1.0),
        dem("Dswitch", vec![u(1), u(0)]),
    ];
    let expected = [1.0, 2.0, NAN, 1.0, 2.0, NAN, 1.0, 2.0, NAN, 1.0, 2.0, NAN];
    assert_segments(units, &expected);
}

#[test]
fn dswitch_first_selection_past_the_last_item_is_exhausted() {
    // Dswitch([5, 6], 2): the constructor wraps 2 into [0, 2] and selects one past the last item,
    // where scsynth recurses into itself without end. plyphon treats that selection as exhausted, so
    // the first demand pulls the index again and wraps it into the list (2 -> item 0).
    let units = vec![dem("Dswitch", vec![c(2.0), c(5.0), c(6.0)])];
    assert_segments(units, &[5.0, 5.0, 5.0]);
}

#[test]
fn dswitch_reset_resets_every_input_and_reselects() {
    // Dseq([Dswitch([Dseq([1, 2], 1), Dseq([3, 4], 1)], Dseq([1, 0, 0], 1)), 9], 2): the second pass
    // resets the Dswitch, which resets its index and items and selects from the index again.
    let units = vec![
        dseq(&[1.0, 2.0], 1.0),
        dseq(&[3.0, 4.0], 1.0),
        dseq(&[1.0, 0.0, 0.0], 1.0),
        dem("Dswitch", vec![u(2), u(0), u(1)]),
        dem("Dseq", vec![c(2.0), u(3), c(9.0)]),
    ];
    let expected = [3.0, 4.0, 1.0, 2.0, 9.0, 3.0, 4.0, 1.0, 2.0, 9.0, NAN];
    assert_segments(units, &expected);
}

#[test]
fn dswitch1_reset_resets_every_input() {
    // Dseq([Dswitch1([Dseq([1, 2, 3], inf)], Dseq([0, 0], 1)), 9], 2): the outer Dseq's second pass
    // resets the Dswitch1, which restarts both the index and the item.
    let units = vec![
        dseq(&[1.0, 2.0, 3.0], f32::INFINITY),
        dseq(&[0.0, 0.0], 1.0),
        dem("Dswitch1", vec![u(1), u(0)]),
        dem("Dseq", vec![c(2.0), u(2), c(9.0)]),
    ];
    assert_segments(units, &[1.0, 2.0, 9.0, 1.0, 2.0, 9.0, NAN]);
}

#[test]
fn dwrand_picks_by_weight_from_the_synths_stream() {
    // Dwrand([1, Dseq([10, 11], 1), 3], [0.2, 0.5, 0.3], 8): the constructor picks, then each value
    // picks again; the nested Dseq is played to its end (one repeat) and reset when picked again.
    let units = vec![
        dseq(&[10.0, 11.0], 1.0),
        dem(
            "Dwrand",
            vec![c(8.0), c(3.0), c(0.2), c(0.5), c(0.3), c(1.0), u(0), c(3.0)],
        ),
    ];
    let expected = [
        3.0, 1.0, 1.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 3.0, NAN,
    ];
    assert_segments(units, &expected);
}

#[test]
fn dwrand_without_a_pick_is_exhausted() {
    // Dseq([Dwrand([1, 2], [0, 0], 3), 9], 1): no weight sum reaches the draw, so no item is ever
    // picked. scsynth then reads an uninitialised index; plyphon yields NaN.
    let units = vec![
        dem(
            "Dwrand",
            vec![c(3.0), c(2.0), c(0.0), c(0.0), c(1.0), c(2.0)],
        ),
        dem("Dseq", vec![c(1.0), u(0), c(9.0)]),
    ];
    assert_segments(units, &[9.0, NAN]);
}

#[test]
fn dwrand_shares_the_stream_with_its_items() {
    // Dwrand([Dwhite(0, 1, 2), 5], [0.5, 0.5], inf): picks and the nested Dwhite's values interleave
    // on the one stream.
    let units = vec![
        dem("Dwhite", vec![c(2.0), c(0.0), c(1.0)]),
        dem(
            "Dwrand",
            vec![c(f32::INFINITY), c(2.0), c(0.5), c(0.5), u(0), c(5.0)],
        ),
    ];
    let expected = [
        0x40a0_0000,
        0x3e1d_9c70,
        0x3f0a_0d84,
        0x3eb7_798c,
        0x3f22_5838,
        0x40a0_0000,
        0x3f6f_b806,
        0x3ed2_b5bc,
        0x40a0_0000,
        0x3f59_4fec,
    ]
    .map(f32::from_bits);
    assert_segments(units, &expected);
}
