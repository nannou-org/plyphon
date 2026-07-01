//! Karplus-Strong plucked strings from the delay family, via cpal.
//!
//! A periodic `Impulse` clock plucks the string: each pluck is a short noise burst (`WhiteNoise`
//! windowed by a `Decay` envelope) fed into a `CombL` whose delay time is one period of the string's
//! pitch. The comb's feedback recirculates that excitation, so it rings at the pitch and decays over
//! `STRING_DECAY` seconds - the classic Karplus-Strong string. An `AllpassC` adds a little diffusion
//! for body. Showcases the recirculating delays `CombL`/`AllpassC` (and their `N`/`C` siblings).
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// How often the string is plucked (Hz).
const PLUCK_HZ: f32 = 2.2;
/// The string's pitch (Hz); the comb delay is one period, `1 / STRING_HZ`.
const STRING_HZ: f32 = 196.0;
/// How long the string rings after each pluck (seconds) - the comb's `decaytime`.
const STRING_DECAY: f32 = 3.5;
/// Length of the excitation noise burst (seconds) - the `Decay` envelope's time.
const EXCITE_DECAY: f32 = 0.006;
/// A gentle master gain.
const GAIN: f32 = 0.25;

/// A control-rate `BinaryOpUGen` multiply (`a * b`).
fn mul(a: u32, b: u32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![
            InputRef::Unit { unit: a, output: 0 },
            InputRef::Unit { unit: b, output: 0 },
        ],
        num_outputs: 1,
        special_index: 2, // multiply
    }
}

/// Build a `World` playing the plucked-string phrase.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mut units = vec![
        // 0: Impulse.ar(PLUCK_HZ) -> the pluck clock.
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(PLUCK_HZ), InputRef::Constant(0.0)],
            1,
        ),
        // 1: WhiteNoise.ar -> the excitation source.
        UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
        // 2: Decay.ar(clock, EXCITE_DECAY) -> a short window on each pluck.
        UnitSpec::new(
            "Decay",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(EXCITE_DECAY),
            ],
            1,
        ),
        // 3: excitation = noise * window -> a brief noise burst per pluck.
        mul(1, 2),
        // 4: CombL.ar(excitation, maxdelay=0.05, 1/STRING_HZ, STRING_DECAY) -> the ringing string.
        UnitSpec::new(
            "CombL",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(0.05),
                InputRef::Constant(1.0 / STRING_HZ),
                InputRef::Constant(STRING_DECAY),
            ],
            1,
        ),
        // 5: AllpassC.ar(string, maxdelay=0.02, 0.008, 0.1) -> a little diffusion for body.
        UnitSpec::new(
            "AllpassC",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 4, output: 0 },
                InputRef::Constant(0.02),
                InputRef::Constant(0.008),
                InputRef::Constant(0.1),
            ],
            1,
        ),
        // 6: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Constant(0.5),
            ],
            num_outputs: 1,
            special_index: 2,
        },
    ];
    // 7: Out.ar(0, [string; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 6, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "string".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("string", ROOT_GROUP_ID, AddAction::Tail);

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
    println!("a plucked comb-filter string for 15s...");

    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 15);
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

    /// The string should sound, stay finite and bounded, and - because the comb resonates at
    /// `STRING_HZ` and notches the frequencies between its harmonics - concentrate energy at the
    /// pitch rather than at a non-harmonic frequency 1.5x above it.
    #[test]
    fn comb_string_rings_at_its_pitch() {
        let mut world = build(SR, 1);
        let frames = (SR * 2.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "output should stay bounded"
        );
        let rms = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.01, "the string should be audible, rms {rms}");

        // Comb resonances sit at multiples of STRING_HZ; 1.5x is a notch between them.
        let fundamental = goertzel(&out, STRING_HZ);
        let notch = goertzel(&out, STRING_HZ * 1.5);
        assert!(
            fundamental > 2.0 * notch,
            "energy should concentrate at the pitch (fund {fundamental} vs notch {notch})"
        );
    }
}
