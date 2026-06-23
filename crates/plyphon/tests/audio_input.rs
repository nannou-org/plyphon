//! `In.ar` reading the hardware input bus: feed a tone in through `World::fill_duplex` and confirm
//! `In.ar(inputBus) -> Out.ar(0)` passes it to the output. Driven block-aligned, so it needs no
//! actual duplex stream (cpal has none) - just interleaved input buffers handed to `fill_duplex`.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World, engine,
};

const SR: f32 = 48_000.0;
const BLOCK: usize = 64;

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR).floor();
    let w = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in samples {
        let s = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
}

/// Drive `world` block-by-block with `input` (mono), returning the mono output. Block-aligned, so
/// each `fill_duplex` call processes exactly one control block with its matching input frames.
fn render_duplex(world: &mut World, input: &[f32]) -> Vec<f32> {
    let frames = input.len();
    let mut out = vec![0.0f32; frames];
    let mut f = 0;
    while f < frames {
        let n = BLOCK.min(frames - f);
        world.fill_duplex(&mut out[f..f + n], 1, &input[f..f + n], 1);
        f += n;
    }
    out
}

#[test]
fn in_ar_reads_hardware_input() {
    // One output, one input: audio bus channel 0 is the output, channel 1 is the input.
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        block_size: BLOCK,
        output_channels: 1,
        input_channels: 1,
        ..Options::default()
    });
    let reader = SynthDef {
        name: "thru".to_string(),
        params: vec![],
        ugens: vec![
            // In.ar(1): read the first hardware input channel (just past the single output).
            UgenSpec::new("In", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UgenSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 0, output: 0 },
                ],
                0,
            ),
        ],
    };
    controller.add_synthdef(reader);
    controller
        .synth_new("thru", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // A 440 Hz tone presented as "hardware input".
    let frames = BLOCK * 200;
    let input: Vec<f32> = (0..frames)
        .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / SR).sin() * 0.5)
        .collect();

    let out = render_duplex(&mut world, &input);
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "input was not passed through"
    );
    let m440 = goertzel(&out, 440.0);
    let m880 = goertzel(&out, 880.0);
    assert!(
        m440 > 5.0 * m880,
        "expected the 440 Hz input at the output: m440={m440}, m880={m880}"
    );
}
