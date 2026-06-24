//! Each oscillator (LFSaw, LFPulse, Impulse, Saw, Pulse) should produce its fundamental, stay in
//! range, and not put energy at a non-harmonic frequency.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f32 = 48_000.0;
const FREQ: f32 = 220.0;

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

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + 256);
    let mut buf = vec![0.0f32; 256];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `<osc>(freq, ...) -> Out.ar(0)` rendered for ~0.25 s.
fn render_osc(name: &str, inputs: Vec<InputRef>) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "osc".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(name, Rate::Audio, inputs, 1),
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
    });
    controller
        .synth_new("osc", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, SR as usize / 4)
}

#[test]
fn oscillators_produce_their_fundamental() {
    let cases: [(&str, Vec<InputRef>); 5] = [
        ("LFSaw", vec![InputRef::Constant(FREQ)]),
        ("LFPulse", vec![InputRef::Constant(FREQ)]),
        ("Impulse", vec![InputRef::Constant(FREQ)]),
        ("Saw", vec![InputRef::Constant(FREQ)]),
        ("Pulse", vec![InputRef::Constant(FREQ)]),
    ];
    for (name, inputs) in cases {
        let out = render_osc(name, inputs);
        assert!(out.iter().any(|s| s.abs() > 0.1), "{name} was silent");
        assert!(
            out.iter().all(|s| s.abs() <= 1.5),
            "{name} ran out of range"
        );
        let fundamental = goertzel(&out, FREQ);
        let off = goertzel(&out, FREQ * 1.5); // 330 Hz: not a harmonic of 220
        assert!(
            fundamental > 5.0 * off,
            "{name}: expected {FREQ} Hz fundamental, got fundamental={fundamental}, off={off}"
        );
    }
}

#[test]
fn band_limited_saw_aliases_less_than_lfsaw() {
    // A high fundamental: the band-limited Saw should put far less energy at an aliased,
    // non-harmonic frequency than the naive LFSaw.
    let high = 6000.0;
    let alias = 1234.0; // not a harmonic of 6000
    let saw = render_osc("Saw", vec![InputRef::Constant(high)]);
    let lfsaw = render_osc("LFSaw", vec![InputRef::Constant(high)]);
    assert!(
        goertzel(&saw, alias) < goertzel(&lfsaw, alias),
        "band-limited Saw should alias less than LFSaw"
    );
}
