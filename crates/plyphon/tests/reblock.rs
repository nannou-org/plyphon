//! Intra-graph reblocking (scsynth's `Reblock(n)`): a def whose graph runs at a smaller control block
//! than the World's, the calc list ticked `world_block / block` times per World block. For a linear
//! chain this is transparent - the output is identical to the non-reblocked def, sample for sample -
//! since interior units see the same per-sample stream, just chunked finer, and the boundary `In`/
//! `Out` slice the World-block-wide bus per tick. Invalid block sizes are rejected at compile.

use plyphon::{
    AddAction, BuildError, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, SynthNewError,
    UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;

/// Render `frames` of mono audio in one-World-block (64-sample) host buffers.
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + 64);
    let mut buf = vec![0.0f32; 64];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `SinOsc.ar(freq) -> Out.ar(0)` - a purely linear chain.
fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

#[test]
fn reblock_is_transparent_for_a_linear_chain() {
    // The same sine, once at the World block and once reblocked to 16-sample sub-blocks (4 ticks per
    // 64-sample World block). A linear chain is sample-for-sample identical either way.
    let plain = {
        let (mut c, _nrt, mut world) = engine(opts());
        c.add_synthdef(sine_def());
        c.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail).unwrap();
        render(&mut world, 4096)
    };
    let reblocked = {
        let (mut c, _nrt, mut world) = engine(opts());
        c.add_synthdef_reblocked(sine_def(), 16);
        c.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail).unwrap();
        render(&mut world, 4096)
    };

    assert!(
        plain.iter().any(|s| s.abs() > 0.1),
        "the reference sine was silent"
    );
    let max_diff = plain
        .iter()
        .zip(&reblocked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "reblocking a linear chain must be transparent (max sample diff {max_diff})"
    );
}

#[test]
fn invalid_reblock_block_sizes_are_rejected() {
    // Not a power of two, and larger than the 64-sample World block: both must fail at compile
    // (surfaced by the first `synth_new`), not silently mis-run.
    for bad in [17usize, 128] {
        let (mut c, _nrt, _world) = engine(opts());
        c.add_synthdef_reblocked(sine_def(), bad);
        let err = c
            .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
            .expect_err("an invalid reblock size must be rejected");
        assert!(
            matches!(err, SynthNewError::Build(BuildError::InvalidReblock { .. })),
            "expected InvalidReblock for block size {bad}, got {err:?}"
        );
    }
}

#[test]
fn a_power_of_two_reblock_runs_and_sounds() {
    // A reblocked synth produces finite, audible output at the right pitch - the sub-block loop and
    // the boundary slicing carry the signal end to end.
    let (mut c, _nrt, mut world) = engine(opts());
    c.add_synthdef_reblocked(sine_def(), 8);
    c.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail).unwrap();
    let out = render(&mut world, 4096);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "reblocked output was non-finite"
    );
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "reblocked synth was silent"
    );
}
