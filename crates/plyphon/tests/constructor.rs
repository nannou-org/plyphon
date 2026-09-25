//! The constructor pass: on a synth's first block every unit is constructed, in SynthDef order,
//! before any unit's first calc - scsynth's `Graph_FirstCalc`. Each expected value follows the
//! scsynth constructor it cites.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

/// Samples per control block at the default engine options.
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `Out.ar(0, Unit{src})`.
fn out(src: u32) -> UnitSpec {
    UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0)
}

/// `K2A.ar(Unit{src})`.
fn k2a(src: u32) -> UnitSpec {
    UnitSpec::new("K2A", Rate::Audio, vec![u(src)], 1)
}

/// A `BinaryOpUGen` running operator `index` on `a` and `b` at `rate`.
fn binary(rate: Rate, index: i16, a: InputRef, b: InputRef) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate,
        inputs: vec![a, b],
        num_outputs: 1,
        special_index: index,
    }
}

/// Play `units` as one synth and render `blocks` control blocks of output bus 0.
fn render(units: Vec<UnitSpec>, blocks: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * blocks];
    world.fill(&mut buf, 1);
    buf
}

#[test]
fn a_constructor_calc_advances_the_state_one_sample() {
    // `OnePole_Ctor` seeds `y1 = 0` and runs the calc for one sample: `1 + 0.5 * (0 - 1) = 0.5`.
    // That sample is consumed by construction, so the first block starts one step later.
    let buf = render(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("OnePole", Rate::Audio, vec![u(0), c(0.5)], 1),
            out(1),
        ],
        1,
    );
    assert_eq!(buf[0], 0.75);
    assert_eq!(buf[1], 0.875);
}

#[test]
fn a_scalar_unit_runs_only_in_its_constructor() {
    // scsynth leaves scalar-rate units out of the calc list (`SC_GraphDef.cpp`), so an `.ir`
    // `rrand` draws once, in its constructor, and holds that value for the synth's life.
    let buf = render(
        vec![binary(Rate::Scalar, 47, c(0.0), c(1.0)), k2a(0), out(1)],
        4,
    );
    assert!((0.0..1.0).contains(&buf[0]));
    assert!(buf.iter().all(|&s| s == buf[0]), "one draw, held");
}

/// The first output sample of `units[index]` in a synth that starts with `RandSeed.ir(1, 7)`,
/// through `K2A` when that unit is not audio-rate.
fn first_sample_of(units: &[UnitSpec], index: usize) -> f32 {
    let mut all = vec![UnitSpec::new(
        "RandSeed",
        Rate::Scalar,
        vec![c(1.0), c(7.0)],
        1,
    )];
    all.extend_from_slice(units);
    let mut src = index as u32 + 1;
    if units[index].rate != Rate::Audio {
        all.push(k2a(src));
        src = all.len() as u32 - 1;
    }
    all.push(out(src));
    render(all, 1)[0]
}

#[test]
fn every_constructor_runs_before_the_first_calc() {
    // scsynth constructs every unit, in SynthDef order, before any unit's first calc
    // (`Graph_FirstCalc`). An `.ar` `rrand` draws in its constructor and then once per sample, so
    // with a `Rand` after it the draws land as: `rrand` constructor, `Rand` constructor, then
    // `rrand`'s first block - the stream's first, second and third values. Audio-rate `rrand`
    // draws with scsynth's bipolar `frand2` (`rrand_aa`), which is exactly `2 * frand - 1`.
    let rrand = binary(Rate::Audio, 47, c(0.0), c(1.0));
    let rand = UnitSpec::new("Rand", Rate::Scalar, vec![c(0.0), c(1.0)], 1);
    let mixed = [rrand, rand.clone()];

    // Three `Rand`s after the same seed hold the stream's first three values.
    let rands = [rand.clone(), rand.clone(), rand];
    let second = first_sample_of(&rands, 1);
    let third = first_sample_of(&rands, 2);
    assert_ne!(second, third);

    assert_eq!(first_sample_of(&mixed, 1), second, "Rand draws second");
    assert_eq!(
        first_sample_of(&mixed, 0),
        2.0 * third - 1.0,
        "rrand's first block draws third"
    );
}

#[test]
fn delay1_starts_from_its_x1_input() {
    // `Delay1_Ctor` seeds its delayed sample from input 1 (`x1`) and writes it: `m_x1 = IN0(1)`.
    let buf = render(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("Delay1", Rate::Audio, vec![u(0), c(0.5)], 1),
            out(1),
        ],
        1,
    );
    assert_eq!(buf[0], 0.5);
    assert!(buf[1..].iter().all(|&s| s == 1.0));
}
