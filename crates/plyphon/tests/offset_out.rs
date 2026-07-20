//! `OffsetOut` in the real engine: a synth created at a block boundary (offset 0) writes its signal
//! to the bus exactly like `Out`. Confirms the per-channel carry `aux` the audio form reserves is
//! wired up and harmless on the common path; the sub-block delay-and-carry itself is covered by the
//! `shift_and_carry` unit test in `plyphon-unit`.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

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

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR as f32).floor();
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

#[test]
fn offset_out_writes_like_out_at_offset_zero() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    // `SinOsc.ar(440) -> OffsetOut.ar(0, sig)`.
    controller.add_synthdef(SynthDef {
        name: "offset".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "OffsetOut",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("offset", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    let out = render(&mut world, SR as usize / 4);
    assert!(out.iter().any(|s| s.abs() > 0.1), "OffsetOut was silent");
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "OffsetOut should pass the 440 Hz tone through"
    );
}
