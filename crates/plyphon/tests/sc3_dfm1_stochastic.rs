//! Exact Plyphon-owned stochastic evidence for `DFM1`.

use plyphon::{
    AddAction, ControllerBatchCommand, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, engine,
};

const SAMPLE_RATE: f64 = 48_000.0;
const BLOCK_SIZE: usize = 64;
const BUILD_SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;
const ENGINE_SEED_INIT: u64 = 0x853c_49e6_748f_ea9b;
const VOICE_COUNT: usize = 2;
const GOLDEN_START_FRAME: usize = 128;
const GOLDEN_FRAMES: usize = 16;

/// Return a reference to a SynthDef control parameter.
fn parameter(index: u32) -> InputRef {
    InputRef::Param(index)
}

/// Return a reference to output zero of a preceding unit.
fn output(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// Build the two-unit signal path whose unit indices pin the DFM1 build and instance seeds.
fn seeded_dfm1_definition() -> SynthDef {
    SynthDef {
        name: "spec100-dfm1-seeded-golden".to_string(),
        params: vec![
            Param::control("source", 0.375),
            Param::control("frequency", 1_100.0),
            Param::control("resonance", 0.35),
            Param::control("inputGain", 0.8),
            Param::control("type", 0.0),
            Param::control("noiseLevel", 0.025),
            Param::control("out", 0.0),
        ],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![parameter(0)], 1),
            UnitSpec::new(
                "DFM1",
                Rate::Audio,
                vec![
                    output(0),
                    parameter(1),
                    parameter(2),
                    parameter(3),
                    parameter(4),
                    parameter(5),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![parameter(6), output(1)], 0),
        ],
    }
}

/// Compute stable FNV-1a over little-endian `f32::to_bits` bytes.
fn fnv1a_bits(bits: &[u32]) -> u64 {
    bits.iter()
        .flat_map(|bits| bits.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// Render two instances with a fixed block schedule and return channel-major golden-window bits.
fn render_seeded_voices() -> Vec<u32> {
    // The DFM1 is unit 1, so its compile-time placeholder seed is exactly BUILD_SEED_STEP.
    // At spawn the fixed engine ladder replaces it with ENGINE_SEED_INIT + BUILD_SEED_STEP for
    // voice A and ENGINE_SEED_INIT + 2*BUILD_SEED_STEP for voice B.
    assert_eq!(1u64.wrapping_mul(BUILD_SEED_STEP), BUILD_SEED_STEP);
    let voice_a_seed = ENGINE_SEED_INIT.wrapping_add(BUILD_SEED_STEP);
    let voice_b_seed = ENGINE_SEED_INIT.wrapping_add(BUILD_SEED_STEP.wrapping_mul(2));
    assert_ne!(voice_a_seed, voice_b_seed);

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
        output_channels: VOICE_COUNT,
        input_channels: 0,
        ..Options::default()
    });
    controller.add_synthdef(seeded_dfm1_definition());
    let def_id = controller
        .ensure_compiled("spec100-dfm1-seeded-golden")
        .expect("compile seeded DFM1 definition");
    controller
        .try_send_batch(&[
            ControllerBatchCommand::AddSynth {
                id: 1_001,
                def_id,
                target: ROOT_GROUP_ID,
                action: AddAction::Tail,
            },
            ControllerBatchCommand::SetControl {
                node: 1_001,
                param: 6,
                value: 0.0,
            },
            ControllerBatchCommand::AddSynth {
                id: 1_002,
                def_id,
                target: ROOT_GROUP_ID,
                action: AddAction::Tail,
            },
            ControllerBatchCommand::SetControl {
                node: 1_002,
                param: 6,
                value: 1.0,
            },
        ])
        .expect("spawn two routed DFM1 voices");

    let mut rendered = vec![0.0; GOLDEN_START_FRAME * VOICE_COUNT];
    world.fill(&mut rendered, VOICE_COUNT);
    for node in [1_001, 1_002] {
        controller
            .set_control(node, 0, -0.25)
            .expect("schedule source change");
        controller
            .set_control(node, 1, 3_200.0)
            .expect("schedule frequency change");
        controller
            .set_control(node, 2, 0.7)
            .expect("schedule resonance change");
        controller
            .set_control(node, 4, 1.0)
            .expect("schedule filter-type change");
        controller
            .set_control(node, 5, 0.04)
            .expect("schedule non-zero noise change");
    }
    let mut changed = vec![0.0; BLOCK_SIZE * VOICE_COUNT];
    world.fill(&mut changed, VOICE_COUNT);
    rendered.extend(changed);

    let mut bits = Vec::with_capacity(GOLDEN_FRAMES * VOICE_COUNT);
    for channel in 0..VOICE_COUNT {
        for frame in GOLDEN_START_FRAME..GOLDEN_START_FRAME + GOLDEN_FRAMES {
            bits.push(rendered[frame * VOICE_COUNT + channel].to_bits());
        }
    }
    bits
}

/// Pin the engine/build seed ladder, non-zero-noise schedule, exact bits, and voice independence.
#[test]
fn dfm1_seeded_noise_golden_bits_and_voices_decorrelate() {
    let actual = render_seeded_voices();
    let expected: [u32; GOLDEN_FRAMES * VOICE_COUNT] = [
        0x3f15_f0a0,
        0x3f0e_ddd1,
        0x3f01_a775,
        0x3ee5_859d,
        0x3ec3_974b,
        0x3e9d_9f50,
        0x3e6a_40a1,
        0x3e12_40c1,
        0x3d8e_8988,
        0xbc17_7171,
        0xbd9e_b8a2,
        0xbe0b_281a,
        0xbe3e_db24,
        0xbe69_7613,
        0xbe85_4b01,
        0xbe8d_de36,
        0x3f18_91fe,
        0x3f0f_4cf0,
        0x3f03_47e1,
        0x3ee9_7a4e,
        0x3eca_949a,
        0x3ea4_be59,
        0x3e74_e0c1,
        0x3e28_15f0,
        0x3db6_0574,
        0x3c66_7c97,
        0xbd57_c5fa,
        0xbde0_3c7b,
        0xbe25_aa0a,
        0xbe55_910a,
        0xbe7b_90bc,
        0xbe87_15a0,
    ];
    assert_eq!(actual, expected, "DFM1 seeded stochastic window changed");
    assert_eq!(
        fnv1a_bits(&actual),
        0x9800_5192_8125_e7ee,
        "DFM1 seeded stochastic FNV-1a changed"
    );
    assert_ne!(
        &actual[..GOLDEN_FRAMES],
        &actual[GOLDEN_FRAMES..],
        "separately spawned DFM1 voices must decorrelate"
    );
}
