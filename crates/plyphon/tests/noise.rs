//! Exercise `WhiteNoise` and the RNG: raw noise is bounded and broadband, and low-passing it
//! (subtractive synthesis: `LPF.ar(WhiteNoise.ar, 500)`) shifts the energy to low frequencies.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, engine};

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
    let (mut controller, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let name = def.name.clone();
    controller.add_synthdef(def);
    controller
        .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail)
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
        ugens: vec![
            UgenSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
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
        ugens: vec![
            UgenSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UgenSpec::new(
                "LPF",
                Rate::Audio,
                vec![
                    InputRef::Ugen { ugen: 0, output: 0 },
                    InputRef::Constant(500.0),
                ],
                1,
            ),
            UgenSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 1, output: 0 },
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
