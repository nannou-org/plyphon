//! How the bus writers treat the first write to a channel in a block, as scsynth's do: an ordinary
//! graph's `Out.ar` copies its signal over an untouched channel (so `-0.0` stays `-0.0`) and sums
//! onto a touched one; a reblocked graph's `Out.ar` clears the channel and sums; `OffsetOut`'s
//! first block keeps a touched channel's leading samples; and a reblocked graph writes a control
//! bus on its first tick only.

use plyphon::{
    AddAction, CommandTime, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
const NEG_ZERO: u32 = 0x8000_0000;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `<writer>.ar(bus, DC.ar(value))` as the def `name`.
fn dc_to(name: &str, writer: &str, bus: f32, value: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(value)], 1),
            UnitSpec::new(writer, Rate::Audio, vec![c(bus), u(0)], 0),
        ],
    }
}

fn world() -> (plyphon::Controller, plyphon::Nrt, World) {
    engine(Options {
        sample_rate: SR,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    })
}

fn block(world: &mut World) -> Vec<u32> {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf.iter().map(|s| s.to_bits()).collect()
}

#[test]
fn out_copies_over_an_untouched_channel() {
    // `Out_next_a` copies the first writer's signal, so `-0.0` survives; a second writer sums
    // (`-0.0 + 0.0 == 0.0`).
    let (mut controller, _nrt, mut world) = world();
    controller.add_synthdef(dc_to("neg", "Out", 0.0, -0.0));
    controller.add_synthdef(dc_to("pos", "Out", 0.0, 0.0));
    controller
        .synth_new("neg", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    assert!(block(&mut world).iter().all(|&b| b == NEG_ZERO));
    controller
        .synth_new("pos", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    assert!(block(&mut world).iter().all(|&b| b == 0));
}

#[test]
fn replace_out_copies_too() {
    let (mut controller, _nrt, mut world) = world();
    controller.add_synthdef(dc_to("neg", "ReplaceOut", 0.0, -0.0));
    controller
        .synth_new("neg", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    assert!(block(&mut world).iter().all(|&b| b == NEG_ZERO));
}

#[test]
fn reblocked_out_clears_and_sums() {
    // `Out_next_a_reblock` clears the whole channel on the first tick and every tick sums into its
    // slice, so `0.0 + -0.0` lands `0.0`.
    let (mut controller, _nrt, mut world) = world();
    controller.add_synthdef_reblocked(dc_to("neg", "Out", 0.0, -0.0), 16);
    controller
        .synth_new("neg", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    assert!(block(&mut world).iter().all(|&b| b == 0));
}

#[test]
fn offset_out_first_block_keeps_a_touched_channels_leading_samples() {
    // An `Out` writes `-0.0` over the block; an `OffsetOut` created 10 samples in then sums its
    // signal in after the offset and leaves the first 10 samples as they were
    // (`OffsetOut_next_a`'s "just keep the existing bus content").
    let (mut controller, _nrt, mut world) = world();
    controller.add_synthdef(dc_to("neg", "Out", 0.0, -0.0));
    controller.add_synthdef(dc_to("late", "OffsetOut", 0.0, 1.0));
    controller
        .synth_new("neg", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let base = 1_000_000_000_000u64;
    let delta = (10.0 * 4_294_967_296.0 / SR) as u64 + 1;
    controller.begin_scheduled(CommandTime::At(base + delta));
    controller
        .synth_new("late", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller.end_scheduled();
    let mut buf = vec![0.0f32; BLOCK];
    world.fill_at(&mut buf, 1, base);
    let bits: Vec<u32> = buf.iter().map(|s| s.to_bits()).collect();
    assert!(bits[..10].iter().all(|&b| b == NEG_ZERO), "{bits:x?}");
    assert!(buf[10..].iter().all(|&s| s == 1.0), "{buf:?}");
}

#[test]
fn reblocked_control_writers_write_on_the_first_tick_only() {
    // A graph reblocked to 16 samples ticks four times per World block; `Out.kr` writes the control
    // bus on the first tick only (`Out_next_k_reblock`), so the bus holds 1, not 4.
    let (mut controller, _nrt, mut world) = world();
    controller.add_synthdef_reblocked(
        SynthDef {
            name: "w".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("DC", Rate::Control, vec![c(1.0)], 1),
                UnitSpec::new("Out", Rate::Control, vec![c(5.0), u(0)], 0),
            ],
        },
        16,
    );
    controller.add_synthdef(SynthDef {
        name: "r".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Control, vec![c(5.0)], 1),
            UnitSpec::new("DC", Rate::Audio, vec![u(0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
    });
    controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    assert_eq!(buf[0], 1.0);
}
