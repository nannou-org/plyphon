//! Loading reblock/resample out of a scsynth **version-3 binary SynthDef**:
//! `plyphon::synthdef::read::parse` extracts the `Reblock`/`Resample` settings (the v3 trailing
//! fields) and the `Controller` honours them via `add_synthdef_rate`. So a `.scsyndef` compiled with
//! `Reblock(16)` runs reblocked - byte-identical to the same def without it - and `blockSize == -1`
//! (control-driven, which plyphon bakes at compile and so cannot support) falls back to no reblock.

use plyphon::synthdef::read::parse;
use plyphon::{AddAction, Options, ROOT_GROUP_ID, World, engine};
use scgf::{Input, Rate, SynthDef, SynthDefFile, Ugen};

const SR: f64 = 48_000.0;

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

/// A binary SCgf blob for `SinOsc.ar(440) -> Out.ar(0)` with the given reblock/resample tail fields.
/// `encode` writes version 3 whenever those are non-default.
fn sine_scgf(block_size: i32, resample_factor: f32) -> Vec<u8> {
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "sine".to_string(),
            constants: vec![440.0, 0.0], // freq, then 0.0 reused for phase and out bus
            param_values: vec![],
            param_names: vec![],
            ugens: vec![
                Ugen {
                    name: "SinOsc".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![Input::Constant { index: 0 }, Input::Constant { index: 1 }],
                    outputs: vec![Rate::Audio],
                },
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 1 },
                        Input::Ugen { ugen: 0, output: 0 },
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![],
            block_size,
            resample_factor,
            ..Default::default()
        }],
    };
    scgf::encode(&file).expect("encode SCgf")
}

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

/// Parse `blob`, add each def with its parsed graph-rate, start `sine`, and render.
fn run_parsed(blob: &[u8]) -> Vec<f32> {
    let (mut c, _nrt, mut world) = engine(opts());
    for (def, reblock, resample) in parse(blob).expect("parse") {
        c.add_synthdef_rate(def, reblock, resample);
    }
    c.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail).unwrap();
    render(&mut world, 4096)
}

#[test]
fn v3_reblock_field_is_parsed_and_honoured() {
    let reblocked = sine_scgf(16, 1.0);
    // parse pulls Reblock(16) off the v3 tail as `(Some(16), 1)`.
    let parsed = parse(&reblocked).expect("parse");
    assert_eq!((parsed[0].1, parsed[0].2), (Some(16), 1));

    // Running the parsed reblocked def is byte-identical to the same def with no reblock - the setting
    // is honoured end to end and (for this linear chain) transparent.
    let plain = run_parsed(&sine_scgf(0, 1.0));
    let reblocked = run_parsed(&reblocked);
    assert!(plain.iter().any(|s| s.abs() > 0.1), "reference was silent");
    let max_diff = plain
        .iter()
        .zip(&reblocked)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "a parsed v3 Reblock must run reblocked and stay transparent (max diff {max_diff})"
    );
}

#[test]
fn v3_resample_field_is_parsed_and_runs() {
    let resampled = sine_scgf(0, 2.0);
    let parsed = parse(&resampled).expect("parse");
    assert_eq!((parsed[0].1, parsed[0].2), (None, 2));
    let out = run_parsed(&resampled);
    assert!(
        out.iter().all(|s| s.is_finite()) && out.iter().any(|s| s.abs() > 0.1),
        "a parsed v3 Resample def should run and sound"
    );
}

#[test]
fn control_driven_reblock_falls_back_to_none() {
    // blockSize == -1 is the control-driven form (block size from a synth control at instantiation),
    // which plyphon cannot support (the graph block is baked into the layout at compile). It loads
    // and runs *without* reblock.
    let blob = sine_scgf(-1, 1.0);
    let parsed = parse(&blob).expect("parse");
    assert_eq!((parsed[0].1, parsed[0].2), (None, 1));
    let out = run_parsed(&blob);
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "the fallback (no-reblock) def should still sound"
    );
}
