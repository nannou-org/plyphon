//! Each oscillator (LFSaw, LFPulse, Impulse, Saw, Pulse, and the LFTri/LFPar/LFCub/VarSaw/SyncSaw/
//! FSinOsc set) should produce its fundamental, stay in range, and not put energy at a non-harmonic
//! frequency.

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
fn new_oscillators_produce_their_fundamental() {
    let cases: [(&str, Vec<InputRef>); 5] = [
        (
            "LFTri",
            vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
        ),
        (
            "LFPar",
            vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
        ),
        (
            "LFCub",
            vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
        ),
        (
            "VarSaw",
            vec![
                InputRef::Constant(FREQ),
                InputRef::Constant(0.0),
                InputRef::Constant(0.5),
            ],
        ),
        (
            "FSinOsc",
            vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
        ),
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
fn syncsaw_is_synced_and_bright() {
    // SyncSaw(220, 440): a saw at 440 Hz hard-synced to 220 Hz. Its energy concentrates near the saw
    // frequency (440, a harmonic of the 220 sync), far above a non-harmonic bin.
    let out = render_osc(
        "SyncSaw",
        vec![InputRef::Constant(FREQ), InputRef::Constant(FREQ * 2.0)],
    );
    assert!(out.iter().any(|s| s.abs() > 0.1), "SyncSaw was silent");
    assert!(
        out.iter().all(|s| s.abs() <= 1.5),
        "SyncSaw ran out of range"
    );
    assert!(
        goertzel(&out, FREQ * 2.0) > 5.0 * goertzel(&out, FREQ * 1.5),
        "SyncSaw should be bright at its saw frequency"
    );
}

#[test]
fn fsinosc_is_a_nearly_pure_sine() {
    // The resonator sine should have almost no energy at the second harmonic.
    let out = render_osc(
        "FSinOsc",
        vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
    );
    assert!(
        goertzel(&out, FREQ) > 20.0 * goertzel(&out, FREQ * 2.0),
        "FSinOsc should be a nearly pure sine"
    );
}

#[test]
fn sinoscfb_brightens_with_feedback() {
    // At feedback 0 SinOscFB is a plain sine; feedback phase-modulates it into a brighter tone.
    let clean = render_osc(
        "SinOscFB",
        vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
    );
    assert!(
        clean.iter().all(|s| s.abs() <= 1.5),
        "SinOscFB out of range"
    );
    assert!(
        goertzel(&clean, FREQ) > 20.0 * goertzel(&clean, FREQ * 2.0),
        "at feedback 0 SinOscFB should be a nearly pure sine"
    );
    let bright = render_osc(
        "SinOscFB",
        vec![InputRef::Constant(FREQ), InputRef::Constant(1.0)],
    );
    assert!(
        bright.iter().all(|s| s.is_finite()),
        "SinOscFB must stay finite"
    );
    // Feedback injects harmonics, so the second harmonic becomes a real fraction of the fundamental.
    assert!(
        goertzel(&bright, FREQ * 2.0) > 0.1 * goertzel(&bright, FREQ),
        "feedback should add harmonics"
    );
    // Still pitched at the fundamental.
    assert!(
        goertzel(&bright, FREQ) > 3.0 * goertzel(&bright, FREQ * 1.5),
        "SinOscFB should stay pitched at {FREQ}"
    );
}

#[test]
fn lfgauss_is_a_normalised_bump() {
    // A looping LFGauss: a train of Gaussian bumps in [0, 1] that peak near 1 and dip toward 0.
    let out = render_osc(
        "LFGauss",
        vec![
            InputRef::Constant(0.01), // 10 ms grain -> 100 Hz
            InputRef::Constant(0.15), // width
            InputRef::Constant(0.0),  // iphase
            InputRef::Constant(1.0),  // loop
        ],
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "LFGauss must stay finite"
    );
    assert!(
        out.iter().all(|&s| (0.0..=1.001).contains(&s)),
        "LFGauss is a normalised bump in [0, 1]"
    );
    let peak = out.iter().cloned().fold(0.0f32, f32::max);
    let min = out.iter().cloned().fold(f32::MAX, f32::min);
    assert!(peak > 0.9, "the Gaussian should peak near 1 (peak {peak})");
    assert!(
        min < 0.1,
        "it should dip toward 0 between bumps (min {min})"
    );
}

#[test]
fn lfgauss_non_looping_fires_once() {
    // Non-looping (loop = 0): a single grain, then the ramp runs off the end and the output stays ~0.
    let out = render_osc(
        "LFGauss",
        vec![
            InputRef::Constant(0.02),
            InputRef::Constant(0.15),
            InputRef::Constant(0.0),
            InputRef::Constant(0.0), // loop off
        ],
    );
    let peak = out.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        peak > 0.9,
        "the single grain should peak near 1 (peak {peak})"
    );
    let tail = &out[out.len() / 2..];
    assert!(
        tail.iter().all(|&s| s < 0.05),
        "after the grain LFGauss should be ~silent"
    );
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
