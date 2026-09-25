//! The random operators - unary `rand` (37), `rand2` (38), `linrand` (39), `bilinrand` (40),
//! `sum3rand` (41) and `coin` (44), binary `rrand` (47) and `exprand` (48) - drawing from the synth's
//! random stream at calc and demand rate. Each expected value is the float bit pattern scsynth's own
//! `RGen` gives after `RGen::init(0)` (stream 0 of a fresh World), following the `UnaryOpUGens.cpp`
//! and `BinaryOpUGens.cpp` kernels.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

/// Samples per control block at the default engine options.
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

fn unary(rate: Rate, op: i16, a: InputRef) -> UnitSpec {
    UnitSpec {
        name: "UnaryOpUGen".to_string(),
        rate,
        inputs: vec![a],
        num_outputs: 1,
        special_index: op,
    }
}

fn binary(rate: Rate, op: i16, a: InputRef, b: InputRef) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate,
        inputs: vec![a, b],
        num_outputs: 1,
        special_index: op,
    }
}

/// Play `units` with unit `src` on output bus 0 (through `K2A` unless it is audio-rate) and render
/// one block.
fn render(mut units: Vec<UnitSpec>, src: u32) -> Vec<f32> {
    let mut src = src;
    if units[src as usize].rate != Rate::Audio {
        units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(src)], 1));
        src = units.len() as u32 - 1;
    }
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0));
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
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf
}

fn bits(samples: &[f32]) -> Vec<u32> {
    samples.iter().map(|s| s.to_bits()).collect()
}

#[test]
fn audio_rate_unary_random_operators_draw_per_sample() {
    // `UnaryOpUGen_Ctor` runs the calc once, so each operator's first block starts with its second
    // draw: `rgen.f() * in` per sample, or `frand() < in` for `coin`.
    let cases: [(i16, f32, [u32; 3]); 6] = [
        (37, 1.0, [0x3d94_8b20, 0x3e1d_9c70, 0x3f0a_0d84]),
        (38, 1.0, [0xbf5a_dd38, 0xbf31_31c8, 0x3da0_d840]),
        (39, 1.0, [0x3e1d_9c70, 0x3eb7_798c, 0x3f22_5838]),
        (40, 1.0, [0xbec5_4cd0, 0x3ca1_2540, 0xbe42_b310]),
        (41, 1.0, [0xbe19_2a10, 0x3d15_9440, 0x3e90_394d]),
        (44, 0.5, [0x3f80_0000, 0x3f80_0000, 0x0000_0000]),
    ];
    for (op, x, expected) in cases {
        let buf = render(
            vec![
                UnitSpec::new("DC", Rate::Audio, vec![c(x)], 1),
                unary(Rate::Audio, op, u(0)),
            ],
            1,
        );
        assert_eq!(bits(&buf[..3]), expected, "operator {op}");
    }
}

#[test]
fn a_scalar_random_operator_draws_once() {
    // A scalar-rate operator runs only in its constructor: its one draw is the stream's first.
    let buf = render(vec![unary(Rate::Scalar, 37, c(1.0))], 0);
    assert!(buf.iter().all(|s| s.to_bits() == 0x3f5b_887e));
}

/// `Duty.kr(1, 0, level)`, with `level` the demand unit at index `level`.
fn duty(level: u32) -> UnitSpec {
    UnitSpec::new(
        "Duty",
        Rate::Control,
        vec![c(1.0), c(0.0), c(0.0), u(level)],
        1,
    )
}

#[test]
fn demand_rate_random_operators_draw_from_the_synths_stream() {
    // `rand_d(2)`: its constructor runs the calc once (draw 1), then `Duty`'s constructor pulls
    // (draw 2).
    let buf = render(vec![unary(Rate::Demand, 37, c(2.0)), duty(0)], 1);
    assert_eq!(buf[0].to_bits(), 0x3e14_8b20, "rand");

    // `rrand_d`/`exprand_d` constructors pull nothing, so `Duty`'s pull is the first draw.
    let buf = render(vec![binary(Rate::Demand, 47, c(1.0), c(3.0)), duty(0)], 1);
    assert_eq!(buf[0].to_bits(), 0x402d_c43f, "rrand");
    let buf = render(vec![binary(Rate::Demand, 48, c(1.0), c(3.0)), duty(0)], 1);
    assert_eq!(buf[0].to_bits(), 0x4024_2f9c, "exprand");
}

#[test]
fn demand_coin_draws_even_for_an_exhausted_operand() {
    // `coin_d` draws before it checks its operand for `NaN`; `rand_d` checks first. With an empty
    // `Dseq` as the operand, the `Rand` after them draws the stream's third value behind `coin`
    // (its constructor and `Duty`'s pull each drew) and the first behind `rand`.
    let empty = UnitSpec::new("Dseq", Rate::Demand, vec![c(0.0), c(1.0)], 1);
    let rand = UnitSpec::new("Rand", Rate::Scalar, vec![c(0.0), c(1.0)], 1);
    let after = |op| {
        render(
            vec![
                empty.clone(),
                unary(Rate::Demand, op, u(0)),
                duty(1),
                rand.clone(),
            ],
            3,
        )[0]
        .to_bits()
    };
    assert_eq!(after(44), 0x3e1d_9c70, "coin draws twice");
    assert_eq!(after(37), 0x3f5b_887e, "rand draws nothing");
}
