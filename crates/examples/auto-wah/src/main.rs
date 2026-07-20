//! An envelope-following auto-wah, via cpal.
//!
//! A rhythmic plucked source (a `Saw` shaped by a `Decay2` envelope, retriggered by an `Impulse`
//! clock) is tracked by a `PeakFollower` - an amplitude envelope follower. That amplitude drives an
//! `RLPF` cutoff, so the filter opens sharply on each pluck and closes as the note decays: a classic
//! envelope-controlled "wah". Showcases the signal-measurement units (`PeakFollower`, `Peak`,
//! `RunningMin`/`RunningMax`, `MostChange`/`LeastChange`, `LastValue`).
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// How often the source is plucked (Hz).
const PLUCK_HZ: f32 = 2.5;
/// The plucked note's pitch (Hz).
const NOTE_HZ: f32 = 110.0;
/// The pluck envelope's attack and decay (seconds).
const ATTACK: f32 = 0.005;
const DECAY: f32 = 0.35;
/// How fast the follower releases (its `decay` coefficient, near 1 for a smooth envelope).
const FOLLOW: f32 = 0.9995;
/// Filter cutoff base and range (Hz); the follower (~0..1) maps to `[BASE, BASE + RANGE]`.
const CUTOFF_BASE: f32 = 250.0;
const CUTOFF_RANGE: f32 = 4500.0;
/// A gentle master gain.
const GAIN: f32 = 0.3;

/// Build a `World` playing the auto-wah.
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
        // 1: Decay2.ar(clock, ATTACK, DECAY) -> the pluck envelope.
        UnitSpec::new(
            "Decay2",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(ATTACK),
                InputRef::Constant(DECAY),
            ],
            1,
        ),
        // 2: Saw.ar(NOTE_HZ) -> the tone.
        UnitSpec::new("Saw", Rate::Audio, vec![InputRef::Constant(NOTE_HZ)], 1),
        // 3: source = saw * envelope.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
        // 4: PeakFollower.ar(source, FOLLOW) -> the amplitude envelope.
        UnitSpec::new(
            "PeakFollower",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(FOLLOW),
            ],
            1,
        ),
        // 5: cutoff = follower * RANGE + BASE.
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 4, output: 0 },
                InputRef::Constant(CUTOFF_RANGE),
                InputRef::Constant(CUTOFF_BASE),
            ],
            num_outputs: 1,
            special_index: 0,
        },
        // 6: RLPF(source, cutoff, rq=0.2) -> the wah-filtered voice.
        UnitSpec::new(
            "RLPF",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Constant(0.2),
            ],
            1,
        ),
        // 7: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 6, output: 0 },
                InputRef::Constant(0.6),
            ],
            num_outputs: 1,
            special_index: 2,
        },
    ];
    // 8: Out.ar(0, [voice; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 7, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "wah".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("wah", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("an envelope-following auto-wah for 14s...");

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

    /// The wah should sound, stay finite and bounded, and be rhythmic - the follower-driven cutoff
    /// makes each pluck a loud transient, so a windowed peak varies a lot across the render.
    #[test]
    fn auto_wah_sounds_and_pulses() {
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
        assert!(rms > 0.01, "the wah should be audible, rms {rms}");

        let win = SR as usize / 20; // 50 ms
        let peaks: Vec<f32> = out
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, &s| m.max(s.abs())))
            .collect();
        let loud = peaks.iter().cloned().fold(0.0f32, f32::max);
        let quiet = peaks.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            loud > 3.0 * quiet.max(1e-4),
            "the plucks should pulse (loud={loud}, quiet={quiet})"
        );
    }
}
