//! `Linen`: the gated linear envelope. Each expected value is the bit pattern scsynth's
//! `Linen_next_k` produces at 48 kHz with 64-sample blocks (a control rate of 750 Hz), for
//! `Linen.kr(gate, 0.01, 0.8, 0.02, 2)`: a 7-calc attack, a hold at 0.8, a 14-calc release that
//! starts the calc after the gate falls, then a free.

use plyphon::{
    AddAction, Event, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const BLOCK: usize = 64;

/// The attack and hold: the constructor outputs 0, then blocks 1-7 climb in steps of 0.8 / 7.
const ATTACK: [u32; 8] = [
    0x3dea0ea1, 0x3e6a0ea1, 0x3eaf8af9, 0x3eea0ea1, 0x3f124925, 0x3f2f8af9, 0x3f4ccccd, 0x3f4ccccd,
];

/// The release from 0.8, starting on the block after the gate falls (blocks 12-25).
const RELEASE: [u32; 14] = [
    0x3f4ccccd, 0x3f3e2be3, 0x3f2f8af9, 0x3f20ea0f, 0x3f124925, 0x3f03a83b, 0x3eea0ea1, 0x3ecccccd,
    0x3eaf8af9, 0x3e924925, 0x3e6a0ea1, 0x3e2f8af9, 0x3dea0ea1, 0x3d6a0ea1,
];

/// `Linen.kr(gate, 0.01, 0.8, 0.02, 2)` held into `Out.ar(0)` by `DC.ar`, with `gate` a param.
fn linen_def(gate: f32) -> SynthDef {
    let c = InputRef::Constant;
    SynthDef {
        name: "linen".to_string(),
        params: vec![Param::control("gate", gate)],
        units: vec![
            UnitSpec::new(
                "Linen",
                Rate::Control,
                vec![InputRef::Param(0), c(0.01), c(0.8), c(0.02), c(2.0)],
                1,
            ),
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![c(0.0), InputRef::Unit { unit: 1, output: 0 }],
                0,
            ),
        ],
    }
}

/// The first sample of each of the next `blocks` blocks.
fn block_values(world: &mut World, blocks: usize) -> Vec<u32> {
    let mut buf = [0.0f32; BLOCK];
    (0..blocks)
        .map(|_| {
            world.fill(&mut buf, 1);
            buf[0].to_bits()
        })
        .collect()
}

#[test]
fn attack_hold_release_then_free_match_scsynth() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(linen_def(1.0));
    let node = controller
        .synth_new("linen", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    let mut held = block_values(&mut world, 10);
    assert_eq!(held[..8], ATTACK);
    assert!(held.drain(7..).all(|b| b == 0x3f4ccccd), "holds 0.8");

    // The gate falls for block 11: `Linen` outputs the held level while switching to its release,
    // which then plays out over blocks 12-25.
    controller.set_control(node, 0, 0.0).expect("set gate");
    let released = block_values(&mut world, 15);
    assert_eq!(released[0], 0x3f4ccccd);
    assert_eq!(released[1..], RELEASE);

    // Block 26 finishes the envelope and fires done action 2.
    let _ = block_values(&mut world, 1);
    nrt.process();
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        ended |= matches!(event, Event::NodeEnded(n) if n.node == node);
    }
    assert!(ended, "the release ends by freeing the synth");
}

#[test]
fn a_gate_at_or_below_minus_one_cuts_off_over_minus_gate_minus_one_seconds() {
    // `gate = -1.5` releases over 0.5 s (375 calcs) instead of `releaseTime`.
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(linen_def(1.0));
    let node = controller
        .synth_new("linen", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let _ = block_values(&mut world, 10);
    controller.set_control(node, 0, -1.5).expect("set gate");
    let released = block_values(&mut world, 4);
    assert_eq!(released, [0x3f4ccccd, 0x3f4ccccd, 0x3f4c40fe, 0x3f4bb52e]);
}

#[test]
fn a_gate_at_or_below_minus_one_at_the_start_releases_at_once() {
    // `gate = -1.01` sets the early-release stage before the constructor calc, which then starts a
    // 0.01 s (7-calc) release from level 0; the envelope finishes on block 8.
    let (mut controller, mut nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(linen_def(-1.01));
    let node = controller
        .synth_new("linen", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let values = block_values(&mut world, 7);
    assert!(values.iter().all(|&b| b == 0), "silent while releasing");
    nrt.process();
    while let Some(event) = nrt.poll() {
        assert!(
            !matches!(event, Event::NodeEnded(n) if n.node == node),
            "still releasing after 7 blocks"
        );
    }
    let _ = block_values(&mut world, 1);
    nrt.process();
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        ended |= matches!(event, Event::NodeEnded(n) if n.node == node);
    }
    assert!(ended, "the early release frees the synth on block 8");
}
