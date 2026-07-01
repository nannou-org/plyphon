//! Modal bell synthesis with `Klank`, via cpal.
//!
//! An `Impulse` clock strikes a `Klank` - a bank of decaying resonators tuned to the *inharmonic*
//! partials of a struck bell (the tubular-bell ratios 1, 2.76, 5.40, 8.93, 11.34). Each strike is a
//! broadband impulse that excites every mode at once, so the bank rings out a metallic bell tone that
//! decays over several seconds. Showcases the additive/modal resonator banks `Klang`/`Klank`.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// How often the bell is struck (Hz).
const STRIKE_HZ: f32 = 1.1;
/// The bell's fundamental (Hz).
const BELL_HZ: f32 = 240.0;
/// Inharmonic partial ratios of a tubular bell (frequency, relative amplitude, ring time in seconds).
const MODES: &[(f32, f32, f32)] = &[
    (1.00, 1.00, 3.2),
    (2.76, 0.62, 2.6),
    (5.40, 0.44, 2.0),
    (8.93, 0.30, 1.5),
    (11.34, 0.22, 1.1),
];
/// Scales the strike impulse into the bank (a high-Q resonator bank has a lot of gain).
const STRIKE_AMP: f32 = 1.2;
/// A gentle master gain.
const GAIN: f32 = 0.3;

/// Build a `World` playing the struck bell.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    // 0: Impulse.ar(STRIKE_HZ) -> the strike clock; 1: scale it into an excitation.
    let mut units = vec![
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(STRIKE_HZ), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(STRIKE_AMP),
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
    ];

    // 2: Klank(excitation, freqscale=BELL_HZ, freqoffset=0, decayscale=1, [ratio, amp, ring]...).
    // Each mode's stored freq is a ratio; freqscale multiplies it up to BELL_HZ * ratio.
    let mut klank_inputs = vec![
        InputRef::Unit { unit: 1, output: 0 },
        InputRef::Constant(BELL_HZ),
        InputRef::Constant(0.0),
        InputRef::Constant(1.0),
    ];
    for &(ratio, amp, ring) in MODES {
        klank_inputs.push(InputRef::Constant(ratio));
        klank_inputs.push(InputRef::Constant(amp));
        klank_inputs.push(InputRef::Constant(ring));
    }
    units.push(UnitSpec::new("Klank", Rate::Audio, klank_inputs, 1));

    // 3: tame the level.
    units.push(UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![
            InputRef::Unit { unit: 2, output: 0 },
            InputRef::Constant(0.08),
        ],
        num_outputs: 1,
        special_index: 2,
    });

    // 4: Out.ar(0, [bell; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 3, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "bell".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("bell", ROOT_GROUP_ID, AddAction::Tail);

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
    println!("a struck modal bell for 16s...");

    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 16);
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

    /// The bell should sound, stay finite and bounded, ring at its fundamental, and be percussive -
    /// each strike is a loud transient that decays, so a windowed peak varies a lot.
    #[test]
    fn bell_rings_and_decays() {
        let mut world = build(SR, 1);
        let frames = (SR * 3.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "output should stay bounded"
        );
        let rms = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.01, "the bell should be audible, rms {rms}");

        // Rings at the fundamental (240) far more than at a non-modal frequency.
        let fund = goertzel(&out, BELL_HZ);
        let off = goertzel(&out, BELL_HZ * 1.7);
        assert!(
            fund > 4.0 * off,
            "should ring at {BELL_HZ} (fund {fund}, off {off})"
        );

        // Percussive: windowed peaks vary a lot between strikes and decays.
        let win = SR as usize / 20;
        let peaks: Vec<f32> = out
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, &s| m.max(s.abs())))
            .collect();
        let loud = peaks.iter().cloned().fold(0.0f32, f32::max);
        let quiet = peaks.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            loud > 2.0 * quiet.max(1e-4),
            "the strikes should be percussive (loud={loud}, quiet={quiet})"
        );
    }
}
