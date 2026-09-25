//! `Flip`: negates every other sample. An even-length block negates its even-indexed samples every
//! block; an odd length (control rate, or an odd block size) follows the parity of scsynth's block
//! counter, which starts at 0 on the World's first block, so the sign pattern runs on unbroken
//! across blocks.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// Render `blocks` blocks of `units` (ending in unit `src`, sent to `Out.ar(0)`) at `block_size`,
/// starting the synth after `idle` empty blocks.
fn render(
    units: Vec<UnitSpec>,
    src: u32,
    block_size: usize,
    idle: usize,
    blocks: usize,
) -> Vec<f32> {
    let mut units = units;
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0));
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        block_size,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    let mut buf = vec![0.0f32; block_size];
    for _ in 0..idle {
        world.fill(&mut buf, 1);
    }
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; block_size * blocks];
    world.fill(&mut out, 1);
    out
}

/// `Flip.ar(DC.ar(1))`.
fn flip_ar() -> Vec<UnitSpec> {
    vec![
        UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
        UnitSpec::new("Flip", Rate::Audio, vec![u(0)], 1),
    ]
}

#[test]
fn even_block_negates_the_even_samples_of_every_block() {
    // `Flip_next_even` ignores the block counter.
    for idle in [0, 1] {
        let out = render(flip_ar(), 1, 4, idle, 2);
        assert_eq!(out, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    }
}

#[test]
fn odd_block_follows_the_block_counter() {
    // `Flip_next_odd`: an even scsynth block counter negates the even-indexed samples, an odd one
    // the odd-indexed ones, so the pattern alternates unbroken across 3-sample blocks.
    let out = render(flip_ar(), 1, 3, 0, 3);
    assert_eq!(out, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    // Started on the World's second block (scsynth's counter 1), it begins unflipped.
    let out = render(flip_ar(), 1, 3, 1, 2);
    assert_eq!(out, [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn control_rate_flips_on_alternate_blocks() {
    // A `.kr` `Flip` has a buffer length of 1, so it negates on even scsynth blocks only: -0.5,
    // 0.5, -0.5. `K2A` ramps linearly from each block's value to the next, starting from the
    // constructor's (also negated) value.
    let units = vec![
        UnitSpec::new("DC", Rate::Control, vec![c(0.5)], 1),
        UnitSpec::new("Flip", Rate::Control, vec![u(0)], 1),
        UnitSpec::new("K2A", Rate::Audio, vec![u(1)], 1),
    ];
    let out = render(units, 2, 2, 0, 3);
    assert_eq!(out, [-0.5, -0.5, -0.5, 0.0, 0.5, 0.0]);
}
