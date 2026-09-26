//! `RandSeed` reads its seed as scsynth's `RandSeed_next` does, through `DEMANDINPUT_A`
//! (`NoiseUGens.cpp`): a demand source is pulled once per trigger edge.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

fn dseq(items: &[f32]) -> UnitSpec {
    let mut inputs = vec![c(f32::INFINITY)];
    inputs.extend(items.iter().map(|&v| c(v)));
    UnitSpec::new("Dseq", Rate::Demand, inputs, 1)
}

/// `RandSeed.kr(1, seed)` then `WhiteNoise.ar`, rendered for four blocks.
fn seeded_noise(seed: InputRef, extra: Vec<UnitSpec>) -> Vec<f32> {
    let mut units = extra;
    let seed_unit = units.len() as u32;
    units.push(UnitSpec::new(
        "RandSeed",
        Rate::Control,
        vec![c(1.0), seed],
        1,
    ));
    units.push(UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(seed_unit + 1)],
        0,
    ));
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "seeded".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("seeded", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; 4 * BLOCK];
    world.fill(&mut buf, 1);
    buf
}

#[test]
fn rand_seed_pulls_a_demand_rate_seed() {
    // scsynth reads the seed through `DEMANDINPUT_A`: a `Dseq([7])` seed seeds exactly as the
    // constant 7 does, and not as 0 (what reading the demand input as a control value gives).
    let demand = seeded_noise(u(0), vec![dseq(&[7.0])]);
    let constant = seeded_noise(c(7.0), vec![]);
    let zero = seeded_noise(c(0.0), vec![]);
    assert_eq!(demand, constant);
    assert_ne!(demand, zero);
}
