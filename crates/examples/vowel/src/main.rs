//! A synthesized sung vowel with `Formant`, via cpal.
//!
//! Three `Formant` oscillators share one fundamental and voice the three formants of an "ah" vowel
//! (F1 ~ 700, F2 ~ 1220, F3 ~ 2600 Hz). Each is a pitch-synchronous grain train, so summed and weighted
//! like a vocal-tract response they form a vowel-coloured drone. A slow `SinOsc.kr` vibrato wobbles the
//! fundamental to make it sound sung. Showcases `Formant` (and, among this batch, the oscillators
//! `SinOscFB`/`Blip`/`LFGauss` added alongside it). Unlike the `feedback` example (which feeds voices
//! back through a `LocalIn`/`LocalOut` bus), the timbre here comes entirely from the formant grains.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The sung pitch (Hz) - a low G2-ish drone.
const FUND_HZ: f32 = 98.0;
/// Vibrato rate (Hz) and depth (Hz) applied to the fundamental.
const VIBRATO_HZ: f32 = 5.5;
const VIBRATO_DEPTH: f32 = 3.0;
/// The three "ah" formants: (formant frequency, bandwidth, weight). Bandwidths exceed the fundamental
/// so each grain is shorter than a pitch period (a real formant bandwidth); weights fall with formant
/// number and fold in the master gain.
const FORMANTS: &[(f32, f32, f32)] = &[
    (700.0, 110.0, 0.12),
    (1220.0, 130.0, 0.08),
    (2600.0, 150.0, 0.05),
];
/// A gentle master gain.
const GAIN: f32 = 0.4;

/// Multiply unit `src` (output 0) by `k`.
fn scale(src: u32, k: f32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![
            InputRef::Unit {
                unit: src,
                output: 0,
            },
            InputRef::Constant(k),
        ],
        num_outputs: 1,
        special_index: 2, // multiply
    }
}

/// Add units `a` and `b` (output 0 of each).
fn add(a: u32, b: u32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![
            InputRef::Unit { unit: a, output: 0 },
            InputRef::Unit { unit: b, output: 0 },
        ],
        num_outputs: 1,
        special_index: 0, // add
    }
}

/// Build a `World` playing the sung vowel.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mut units = vec![
        // 0: SinOsc.kr(VIBRATO_HZ) -> a slow [-1, 1] vibrato.
        UnitSpec::new(
            "SinOsc",
            Rate::Control,
            vec![InputRef::Constant(VIBRATO_HZ), InputRef::Constant(0.0)],
            1,
        ),
        // 1: fundamental = FUND_HZ + vibrato * VIBRATO_DEPTH.
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Control,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(VIBRATO_DEPTH),
                InputRef::Constant(FUND_HZ),
            ],
            num_outputs: 1,
            special_index: 0,
        },
    ];

    // 2,3,4: one Formant per vowel formant, sharing the fundamental at unit 1.
    for &(formfreq, bw, _weight) in FORMANTS {
        units.push(UnitSpec::new(
            "Formant",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(formfreq),
                InputRef::Constant(bw),
            ],
            1,
        ));
    }
    // 5,6,7: weight each formant (weights fold in the master gain).
    for (i, &(_, _, weight)) in FORMANTS.iter().enumerate() {
        units.push(scale(2 + i as u32, weight * GAIN));
    }
    // 8: F1 + F2, then 9: + F3 -> the summed vowel.
    units.push(add(5, 6));
    units.push(add(8, 7));

    // 10: Out.ar(0, [vowel; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 9, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "vowel".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("vowel", ROOT_GROUP_ID, AddAction::Tail);

    world
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // cpal's AudioWorklet backend re-instantiates this module on the audio thread, re-running
    // `main` there; only set up audio on the main browser thread.
    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!("a sung 'ah' vowel for 14s...");

    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 14);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The vowel should sound, stay finite and bounded, and carry energy at each formant far above the
    /// spectral valleys between them.
    #[test]
    fn vowel_voices_its_formants() {
        let mut world = build(SR, 1);
        let frames = (SR * 2.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 2.0),
            "output should stay bounded"
        );
        let rms = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.01, "the vowel should be audible, rms {rms}");

        // Each formant band carries more energy than a valley between formants (900 Hz sits between F1
        // and F2, well away from any of the three peaks).
        let valley = goertzel(&out, 900.0);
        for &(formfreq, _, _) in FORMANTS {
            assert!(
                goertzel(&out, formfreq) > 2.0 * valley,
                "formant at {formfreq} Hz should stand above the spectral valley"
            );
        }
    }
}
