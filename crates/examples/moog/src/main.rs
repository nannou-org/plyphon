//! A resonant acid bassline with `MoogFF`, via cpal.
//!
//! A bright `Saw` bass (bouncing between two octaves on a slow pulse) runs through a `MoogFF` - the
//! Moog-ladder resonant low-pass - whose cutoff a slow triangle LFO sweeps up and down while the
//! feedback `gain` sits high enough to make the resonant peak "wah" as it passes each harmonic: the
//! classic acid-bass sound. Showcases the filter family's `MoogFF` (see also `Formlet`/`MidEQ`/`SOS`/
//! the `Lag` cascades/`Hilbert`/`FreqShift`).
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The low bass note (Hz); a slow pulse jumps it up an octave.
const BASS_HZ: f32 = 55.0;
/// How fast the bass bounces between the two octaves (Hz).
const BOUNCE_HZ: f32 = 2.0;
/// Filter-sweep LFO rate (Hz) and its cutoff range (Hz).
const SWEEP_HZ: f32 = 0.13;
const CUTOFF_LO: f32 = 300.0;
const CUTOFF_HI: f32 = 2900.0;
/// Moog feedback resonance (0-4); high, but short of self-oscillation.
const RESONANCE: f32 = 3.0;
/// A gentle master gain.
const GAIN: f32 = 0.35;

/// A `MulAdd.kr(src, mul, add)` = `src * mul + add`.
fn mul_add(src: u32, mul: f32, add: f32) -> UnitSpec {
    UnitSpec {
        name: "MulAdd".to_string(),
        rate: Rate::Control,
        inputs: vec![
            InputRef::Unit {
                unit: src,
                output: 0,
            },
            InputRef::Constant(mul),
            InputRef::Constant(add),
        ],
        num_outputs: 1,
        special_index: 0,
    }
}

/// Build a `World` playing the resonant acid bass.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let span = (CUTOFF_HI - CUTOFF_LO) * 0.5;
    let mid = (CUTOFF_HI + CUTOFF_LO) * 0.5;

    let mut units = vec![
        // 0: LFPulse.kr(BOUNCE_HZ) -> a 0/1 octave toggle.
        UnitSpec::new(
            "LFPulse",
            Rate::Control,
            vec![
                InputRef::Constant(BOUNCE_HZ),
                InputRef::Constant(0.0),
                InputRef::Constant(0.5),
            ],
            1,
        ),
        // 1: bass freq = BASS_HZ + BASS_HZ * toggle -> {55, 110} Hz.
        mul_add(0, BASS_HZ, BASS_HZ),
        // 2: Saw.ar(freq) -> the raw bright bass.
        UnitSpec::new(
            "Saw",
            Rate::Audio,
            vec![InputRef::Unit { unit: 1, output: 0 }],
            1,
        ),
        // 3: LFTri.kr(SWEEP_HZ) -> a slow [-1, 1] cutoff LFO.
        UnitSpec::new(
            "LFTri",
            Rate::Control,
            vec![InputRef::Constant(SWEEP_HZ), InputRef::Constant(0.0)],
            1,
        ),
        // 4: cutoff = LFO mapped to [CUTOFF_LO, CUTOFF_HI].
        mul_add(3, span, mid),
        // 5: MoogFF.ar(bass, cutoff, RESONANCE, 0) -> the resonant ladder.
        UnitSpec::new(
            "MoogFF",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 4, output: 0 },
                InputRef::Constant(RESONANCE),
                InputRef::Constant(0.0),
            ],
            1,
        ),
        // 6: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Constant(0.2),
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
    ];
    // 7: Out.ar(0, [bass; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 6, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "moog".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("moog", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("a resonant acid bass for 16s...");

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

    /// The bass should sound, stay finite and bounded, and the filter sweep should modulate the level
    /// over time (windowed RMS swings as the resonant cutoff opens and closes).
    #[test]
    fn moog_sweeps_and_stays_bounded() {
        let mut world = build(SR, 1);
        let frames = (SR * 6.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "output should stay bounded"
        );
        let rms = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.01, "the bass should be audible, rms {rms}");

        // The resonant sweep modulates the level: windowed RMS swings substantially over the buffer.
        let win = SR as usize / 10;
        let rmss: Vec<f32> = out
            .chunks(win)
            .map(|c| (c.iter().map(|&s| s * s).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        let loud = rmss.iter().cloned().fold(0.0f32, f32::max);
        let quiet = rmss.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            loud > 1.5 * quiet.max(1e-4),
            "the filter sweep should modulate the level (loud={loud}, quiet={quiet})"
        );
    }
}
