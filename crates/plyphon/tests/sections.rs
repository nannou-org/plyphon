//! `FOS`/`SOS` - the explicit-coefficient filter sections. The `SOS` test feeds it the RBJ `BLowPass`
//! coefficients scsynth's B-series computes in the language, validating the whole B-series path.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f32 = 48_000.0;

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

/// `<name>.ar(SinOsc.ar(sig_freq), coefs...) -> Out`, rendered for `frames`.
fn render_section(name: &str, coefs: &[f32], sig_freq: f32, frames: usize) -> Vec<f32> {
    let mut inputs = vec![InputRef::Unit { unit: 0, output: 0 }];
    inputs.extend(coefs.iter().map(|&c| InputRef::Constant(c)));
    let units = vec![
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(sig_freq), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec::new(name, Rate::Audio, inputs, 1),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
            ],
            0,
        ),
    ];
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    render(&mut world, frames)
}

/// scsynth's `BLowPass.sc` coefficients `[a0, a1, a2 (= a0), b1, b2]` for `SOS.ar(in, a0, a1, a2, b1, b2)`.
fn blowpass_coefs(freq: f32, rq: f32) -> [f32; 5] {
    let w0 = 2.0 * std::f32::consts::PI * freq / SR;
    let cos_w0 = w0.cos();
    let i = 1.0 - cos_w0;
    let alpha = w0.sin() * 0.5 * rq;
    let b0rz = 1.0 / (1.0 + alpha);
    let a0 = i * 0.5 * b0rz;
    let a1 = i * b0rz;
    let b1 = cos_w0 * 2.0 * b0rz;
    let b2 = -(1.0 - alpha) * b0rz;
    [a0, a1, a0, b1, b2]
}

#[test]
fn fos_unit_gain_is_transparent() {
    // FOS(in, 1, 0, 0) = in.
    let out = render_section("FOS", &[1.0, 0.0, 0.0], 440.0, SR as usize / 4);
    assert!(out.iter().all(|&s| s.abs() <= 1.5), "FOS out of range");
    assert!(
        goertzel(&out, 440.0) > 0.3,
        "a unit-gain FOS should pass the signal"
    );
}

#[test]
fn fos_averager_attenuates_highs() {
    // FOS(in, 0.5, 0.5, 0) = 0.5*(x + x1), a two-point moving-average low-pass (a zero at Nyquist).
    let low = render_section("FOS", &[0.5, 0.5, 0.0], 200.0, SR as usize / 4);
    let high = render_section("FOS", &[0.5, 0.5, 0.0], 22_000.0, SR as usize / 4);
    assert!(
        goertzel(&low, 200.0) > 5.0 * goertzel(&high, 22_000.0),
        "the averager should pass 200 Hz and cut near Nyquist"
    );
}

#[test]
fn sos_as_blowpass_attenuates_highs() {
    // Feed SOS the RBJ BLowPass coefficients (500 Hz cutoff) - the exact form the B-series produces.
    let coefs = blowpass_coefs(500.0, 1.0);
    let low = render_section("SOS", &coefs, 200.0, SR as usize / 4);
    let high = render_section("SOS", &coefs, 5000.0, SR as usize / 4);
    assert!(
        low.iter().chain(high.iter()).all(|s| s.is_finite()),
        "SOS must stay finite"
    );
    assert!(
        low.iter().all(|&s| s.abs() < 2.0),
        "SOS should stay bounded"
    );
    let low_e = goertzel(&low, 200.0);
    let high_e = goertzel(&high, 5000.0);
    assert!(
        low_e > 10.0 * high_e,
        "a 500 Hz low-pass should pass 200 and reject 5000 (low={low_e}, high={high_e})"
    );
}
