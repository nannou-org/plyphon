//! Exercise the noise generators and the RNG: `WhiteNoise` is bounded and broadband, and low-passing
//! it (subtractive synthesis: `LPF.ar(WhiteNoise.ar, 500)`) shifts the energy to low frequencies.
//! `ClipNoise`, `GrayNoise`, `PinkNoise`, `BrownNoise`, `Dust` and `Dust2` each get a
//! distribution/spectrum check against scsynth's behaviour.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

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

fn render_synth(def: SynthDef, frames: usize, settle: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let name = def.name.clone();
    controller.add_synthdef(def);
    controller
        .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = Vec::with_capacity(frames + settle + 512);
    let mut buf = Vec::new();
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut i = 0;
    while out.len() < frames + settle {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.split_off(settle).into_iter().take(frames).collect()
}

#[test]
fn white_noise_is_bounded_and_broadband() {
    let def = SynthDef {
        name: "raw".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
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
    };
    let out = render_synth(def, SR as usize / 2, 0);
    assert!(
        out.iter().all(|&x| (-1.0..1.0).contains(&x)),
        "noise out of range"
    );
    let rms = (out.iter().map(|&x| x * x).sum::<f32>() / out.len() as f32).sqrt();
    // Uniform [-1, 1) has RMS 1/sqrt(3) ~= 0.577.
    assert!(
        (0.45..0.7).contains(&rms),
        "white noise rms {rms} not ~0.577"
    );
    // Broadband: low and high single-bin energies are the same order of magnitude.
    let low = goertzel(&out, 500.0);
    let high = goertzel(&out, 8000.0);
    assert!(
        low > 0.2 * high && high > 0.2 * low,
        "noise not broadband: low={low}, high={high}"
    );
}

#[test]
fn lowpassed_white_noise_has_more_low_energy() {
    let def = SynthDef {
        name: "noise".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new(
                "LPF",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(500.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    };
    let out = render_synth(def, SR as usize / 2, SR as usize / 20);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "filtered noise not finite"
    );
    // Average several bins to smooth out single-bin variance.
    let low: f32 = (100..400)
        .step_by(50)
        .map(|f| goertzel(&out, f as f32))
        .sum();
    let high: f32 = (6000..9000)
        .step_by(500)
        .map(|f| goertzel(&out, f as f32))
        .sum();
    assert!(
        low > 4.0 * high,
        "lowpassed noise should favour low frequencies: low={low}, high={high}"
    );
}

/// `name.ar(inputs) -> Out`, rendered for 1 s after a short settle.
fn noise_out(name: &str, inputs: Vec<InputRef>) -> Vec<f32> {
    let def = SynthDef {
        name: "n".to_string(),
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
    };
    render_synth(def, SR as usize, SR as usize / 50)
}

fn band(out: &[f32], range: std::ops::Range<u32>, step: usize) -> f32 {
    range.step_by(step).map(|f| goertzel(out, f as f32)).sum()
}

#[test]
fn clip_noise_is_plus_or_minus_one() {
    let out = noise_out("ClipNoise", vec![]);
    assert!(
        out.iter().all(|&x| x == 1.0 || x == -1.0),
        "ClipNoise must be exactly ±1"
    );
    let pos = out.iter().filter(|&&x| x > 0.0).count() as f32 / out.len() as f32;
    assert!(
        (pos - 0.5).abs() < 0.05,
        "ClipNoise should be balanced, +ve fraction {pos}"
    );
}

#[test]
fn gray_noise_is_bounded_and_broadband() {
    let out = noise_out("GrayNoise", vec![]);
    assert!(
        out.iter().all(|&x| (-1.0..=1.0).contains(&x)),
        "GrayNoise out of range"
    );
    let rms = (out.iter().map(|&x| x * x).sum::<f32>() / out.len() as f32).sqrt();
    assert!(rms > 0.05, "GrayNoise should be audible, rms {rms}");
    let low = goertzel(&out, 500.0);
    let high = goertzel(&out, 8000.0);
    assert!(
        low > 0.1 * high && high > 0.1 * low,
        "GrayNoise not broadband: {low}, {high}"
    );
}

#[test]
fn pink_noise_favours_lows() {
    let out = noise_out("PinkNoise", vec![]);
    assert!(out.iter().all(|s| s.is_finite()));
    assert!(
        out.iter().all(|&x| x.abs() < 2.0),
        "PinkNoise wildly out of range"
    );
    // Pink is -3 dB/octave, so low bins clearly beat high bins.
    let low = band(&out, 100..400, 50);
    let high = band(&out, 6000..9000, 500);
    assert!(
        low > 2.0 * high,
        "PinkNoise should favour lows: low={low}, high={high}"
    );
}

#[test]
fn brown_noise_favours_lows_steeply() {
    let out = noise_out("BrownNoise", vec![]);
    assert!(
        out.iter().all(|&x| (-1.0..=1.0).contains(&x)),
        "BrownNoise out of range"
    );
    // Brown is -6 dB/octave, an even steeper low-frequency tilt.
    let low = band(&out, 50..300, 50);
    let high = band(&out, 6000..9000, 500);
    assert!(
        low > 10.0 * high,
        "BrownNoise should strongly favour lows: low={low}, high={high}"
    );
}

#[test]
fn dust_is_sparse_unipolar_impulses() {
    // Dust.ar(200): ~200 impulses/second in [0, 1), the rest silence.
    let out = noise_out("Dust", vec![InputRef::Constant(200.0)]);
    assert!(
        out.iter().all(|&x| (0.0..1.0).contains(&x)),
        "Dust should be unipolar in [0, 1)"
    );
    let nonzero = out.iter().filter(|&&x| x != 0.0).count();
    assert!(
        (50..500).contains(&nonzero),
        "Dust(200) over 1 s should fire ~200 impulses, got {nonzero}"
    );
}

#[test]
fn dust2_is_sparse_bipolar_impulses() {
    // Dust2.ar(200): ~200 impulses/second in [-1, 1), both signs present.
    let out = noise_out("Dust2", vec![InputRef::Constant(200.0)]);
    assert!(
        out.iter().all(|&x| (-1.0..1.0).contains(&x)),
        "Dust2 should be bipolar in [-1, 1)"
    );
    let nonzero = out.iter().filter(|&&x| x != 0.0).count();
    assert!(
        (50..500).contains(&nonzero),
        "Dust2(200) over 1 s should fire ~200 impulses, got {nonzero}"
    );
    assert!(
        out.iter().any(|&x| x > 0.0) && out.iter().any(|&x| x < 0.0),
        "Dust2 should produce both polarities"
    );
}
