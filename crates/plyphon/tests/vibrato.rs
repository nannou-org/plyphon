//! `Vibrato`: each expected value is the bit pattern scsynth's `Vibrato_Ctor`/`Vibrato_next`
//! produce at 48 kHz with 64-sample blocks for `Vibrato.ar(DC.ar(440), ...)`, drawing from a
//! fresh stream 0 (`RGen::init(0)`), at a spread of samples across the first four blocks.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const BLOCK: usize = 64;

/// The samples each test checks: across the delay, the onset and several cycle boundaries.
const PICKS: [usize; 13] = [0, 1, 47, 48, 49, 100, 143, 144, 145, 191, 192, 250, 255];

/// `Vibrato.ar(DC.ar(440), args...)` into `Out.ar(0)`: the first four blocks, at [`PICKS`].
fn vibrato(args: [f32; 8]) -> Vec<u32> {
    let c = InputRef::Constant;
    let u = |i| InputRef::Unit { unit: i, output: 0 };
    let mut inputs = vec![u(0)];
    inputs.extend(args.map(c));
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(440.0)], 1),
            UnitSpec::new("Vibrato", Rate::Audio, inputs, 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * 4];
    world.fill(&mut buf, 1);
    PICKS.iter().map(|&i| buf[i].to_bits()).collect()
}

#[test]
fn delay_onset_and_cycle_draws_match_scsynth() {
    // rate 1000 Hz, depth 0.1, delay 1 ms (47 samples, one spent by the constructor), onset
    // 2 ms, rateVariation 0.3, depthVariation 0.5, iphase 0.3.
    assert_eq!(
        vibrato([1000.0, 0.1, 0.001, 0.002, 0.3, 0.5, 0.3, 0.0]),
        [
            0x43dc0000, 0x43dc0000, 0x43dc1fe9, 0x43dc3c74, 0x43dc5393, 0x43d8cb37, 0x43dfbbea,
            0x43db624a, 0x43d92f15, 0x43c788d5, 0x43c5f948, 0x43c67116, 0x43c56f1b,
        ]
    );
}

#[test]
fn a_high_trigger_restarts_in_the_constructor_and_the_first_block() {
    // With `trig` high from the start, the constructor's calc sample sees a rising edge and
    // restarts (drawing again, and keeping the restarted delay and onset), and so does the first
    // block, because the constructor puts the previous trigger back to 0.
    assert_eq!(
        vibrato([1000.0, 0.1, 0.001, 0.002, 0.3, 0.5, 0.3, 1.0]),
        [
            0x43dc0000, 0x43dc0000, 0x43dc0000, 0x43dc49d0, 0x43dc8cd8, 0x43d7acf4, 0x43ef4e61,
            0x43ed4dab, 0x43eae22e, 0x43ee9c54, 0x43efd7cb, 0x43ee656f, 0x43d9d159,
        ]
    );
}

#[test]
fn control_rate_steps_once_per_block() {
    // `Vibrato.kr(DC.kr(440), 50, 0.1, 0, 0.02, 0.3, 0.5)`: the phase steps by `4 * rate` over the
    // 750 Hz control rate, with a 14-block onset; `DC.ar` holds each block's value.
    let c = InputRef::Constant;
    let u = |i| InputRef::Unit { unit: i, output: 0 };
    let mut inputs = vec![u(0)];
    inputs.extend([50.0, 0.1, 0.0, 0.02, 0.3, 0.5, 0.0, 0.0].map(c));
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Control, vec![c(440.0)], 1),
            UnitSpec::new("Vibrato", Rate::Control, inputs, 1),
            UnitSpec::new("DC", Rate::Audio, vec![u(1)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
        ],
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * 24];
    world.fill(&mut buf, 1);
    let blocks: Vec<u32> = buf.chunks(BLOCK).map(|b| b[0].to_bits()).collect();
    assert_eq!(
        blocks,
        [
            0x43dc0000, 0x43dd5e16, 0x43def12d, 0x43e031f7, 0x43e0992d, 0x43df9f80, 0x43dcbda7,
            0x43d80123, 0x43d4037a, 0x43d186cf, 0x43d125a7, 0x43d37a8a, 0x43d91fff, 0x43e345c0,
            0x43ea28eb, 0x43ee2b42, 0x43ef4cca, 0x43ed8d7e, 0x43e8ed5f, 0x43e16c6f, 0x43d7b4b8,
            0x43d00d40, 0x43cb3619, 0x43c92f42,
        ]
    );
}

#[test]
fn without_delay_or_onset_the_vibrato_starts_at_once() {
    assert_eq!(
        vibrato([700.0, 0.05, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0]),
        [
            0x43dc0000, 0x43dd2cf0, 0x43d2b140, 0x43d2e752, 0x43d332f6, 0x43d5bc09, 0x43e865ce,
            0x43e86cc0, 0x43e85ba9, 0x43da46d6, 0x43db5582, 0x43da78c0, 0x43e0b5df,
        ]
    );
}
