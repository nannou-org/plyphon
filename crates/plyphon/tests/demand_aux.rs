//! Demand-unit aux memory: a demand unit may reserve a region sized when the SynthDef is compiled
//! (`demand_unit_spec_aux`) and reach it through `DemandCtx::aux_mut`. These tests register a small
//! counting unit that keeps its count in its aux region and pulls its one optional input, then nest
//! it so every pull recurses through regions of units further down the chain. Also checks that a
//! demand unit cannot take a later unit (or itself) as input, which the region lending relies on.

use bytemuck::{Pod, Zeroable};
use plyphon::{
    AddAction, BuildError, InputRef, Options, ROOT_GROUP_ID, Rate, RateInfo, SynthDef,
    UnitRegistry, UnitSpec, engine,
};
use plyphon_unit::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec_aux};
use plyphon_unit::unit::registry::{BuildContext, DemandUnitDef};

const SR: f64 = 48_000.0;
const SEG_DUR: f32 = 0.002;
const SEG: usize = 96;
const MID: usize = SEG / 2;

/// A test unit: each demand adds one to the count in its aux region and yields
/// `count + 10 * input` (the input pulled first, when there is one). Its state holds nothing.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Tally {
    _pad: u32,
}

impl DemandUnit for Tally {
    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let x = if ctx.num_inputs() > 0 {
            ctx.demand(0)
        } else {
            0.0
        };
        let count = &mut ctx.aux_mut::<u32>()[0];
        *count += 1;
        *count as f32 + 10.0 * x
    }
}

struct TallyCtor;

impl DemandUnitDef for TallyCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        Ok(demand_unit_spec_aux(Tally { _pad: 0 }, 4, 4))
    }
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `Tally(input)` at demand rate.
fn tally(inputs: Vec<InputRef>) -> UnitSpec {
    UnitSpec::new("Tally", Rate::Demand, inputs, 1)
}

/// Play `synths` synths of `units` (the last unit is the source) through
/// `Out.ar(0, Duty.ar(SEG_DUR, 0, 0, source))`, and read back `n` segments of their sum.
fn segments(mut units: Vec<UnitSpec>, synths: usize, n: usize) -> Vec<f32> {
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
    controller
        .registry_mut()
        .register_demand("Tally", Box::new(TallyCtor));
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    for _ in 0..synths {
        controller
            .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
            .expect("synth_new");
    }
    let mut out = vec![0.0f32; SEG * n];
    for chunk in out.chunks_mut(64) {
        world.fill(chunk, 1);
    }
    (0..n).map(|k| out[MID + k * SEG]).collect()
}

#[test]
fn nested_units_each_keep_their_own_region() {
    // Tally(Tally(Tally())): demand k (from 1) counts k in each region, so the chain yields
    // k + 10 * (k + 10 * k) = 111 * k. A shared or overlapping region would count more than once
    // per demand. Each region starts zeroed.
    let units = vec![tally(vec![]), tally(vec![u(0)]), tally(vec![u(1)])];
    let got = segments(units, 1, 5);
    assert_eq!(got, [111.0, 222.0, 333.0, 444.0, 555.0]);
}

#[test]
fn each_synth_has_its_own_regions() {
    // Two synths of the same def each count from zero: their sum is twice one synth's.
    let units = vec![tally(vec![]), tally(vec![u(0)])];
    let got = segments(units, 2, 4);
    assert_eq!(got, [22.0, 44.0, 66.0, 88.0]);
}

fn try_compile(units: Vec<UnitSpec>) -> Result<(), BuildError> {
    let rate = RateInfo::new(SR, 64);
    SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    }
    .compile(
        &UnitRegistry::with_builtins(),
        &rate,
        &rate,
        64,
        32,
        None,
        1,
    )
    .map(|_| ())
}

#[test]
fn a_demand_unit_cannot_pull_a_later_unit_or_itself() {
    let dseq = |items: Vec<InputRef>| UnitSpec::new("Dseq", Rate::Demand, items, 1);
    // Dseq(1, <unit 1>) where unit 1 is a later Dseq.
    let forward = vec![dseq(vec![c(1.0), u(1)]), dseq(vec![c(1.0), c(2.0)])];
    assert_eq!(try_compile(forward), Err(BuildError::BadInputRef));
    // Dseq(1, <itself>).
    let cycle = vec![dseq(vec![c(1.0), u(0)])];
    assert_eq!(try_compile(cycle), Err(BuildError::BadInputRef));
    // The same pair in order compiles.
    let ordered = vec![dseq(vec![c(1.0), c(2.0)]), dseq(vec![c(1.0), u(0)])];
    assert_eq!(try_compile(ordered), Ok(()));
}
