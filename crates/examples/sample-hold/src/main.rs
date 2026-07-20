//! A self-playing sample-and-hold sequence, via cpal.
//!
//! An `Impulse` clock ticks steadily. On each tick, `Latch` samples a slow `SinOsc.kr` pitch contour
//! and holds it, so the melody steps through the contour rather than gliding. The held value is mapped
//! to a frequency (`MulAdd`) driving a `Saw`, and each tick also fires a `Decay2` pluck envelope. It
//! showcases the in-graph trigger units (`Latch` here; `Trig`/`Gate`/`ToggleFF`/`Stepper`/`Phasor`/...
//! are ported alongside).
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// Steps per second (the clock rate).
const TEMPO: f32 = 7.0;
/// How fast the pitch contour drifts (Hz).
const CONTOUR_RATE: f32 = 0.11;
/// Centre frequency and half-range of the melody (Hz): held value in `[-1, 1]` maps to `[MID-HALF, MID+HALF]`.
const MID: f32 = 420.0;
const HALF: f32 = 320.0;
/// A gentle master gain.
const GAIN: f32 = 0.3;

/// Build a `World` playing the sample-and-hold sequence.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mut units = vec![
        // 0: Impulse.ar(TEMPO) -> the clock.
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(TEMPO), InputRef::Constant(0.0)],
            1,
        ),
        // 1: SinOsc.kr(CONTOUR_RATE) -> slow pitch contour in [-1, 1].
        UnitSpec::new(
            "SinOsc",
            Rate::Control,
            vec![InputRef::Constant(CONTOUR_RATE), InputRef::Constant(0.0)],
            1,
        ),
        // 2: Latch(contour, clock) -> stepped pitch held between ticks.
        UnitSpec::new(
            "Latch",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Unit { unit: 0, output: 0 },
            ],
            1,
        ),
        // 3: MulAdd(held, HALF, MID) -> frequency.
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Constant(HALF),
                InputRef::Constant(MID),
            ],
            num_outputs: 1,
            special_index: 0,
        },
        // 4: Saw.ar(freq) -> the tone.
        UnitSpec::new(
            "Saw",
            Rate::Audio,
            vec![InputRef::Unit { unit: 3, output: 0 }],
            1,
        ),
        // 5: Decay2(clock, 0.005, 0.18) -> pluck envelope.
        UnitSpec::new(
            "Decay2",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.005),
                InputRef::Constant(0.18),
            ],
            1,
        ),
        // 6: tone * envelope.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 4, output: 0 },
                InputRef::Unit { unit: 5, output: 0 },
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
    ];
    // 7: Out.ar(0, [voice; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 6, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "sh".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("sh", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("sample-and-hold sequence for 15s...");

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

    /// The sequence should sound, stay bounded, and be rhythmic - the pluck envelope makes the level
    /// pulse, so a windowed peak varies a lot across the render.
    #[test]
    fn sequence_plucks_and_steps() {
        let mut world = build(SR, 1);
        let frames = (SR * 2.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 2.0),
            "output should stay bounded"
        );

        // Split into short windows; a plucked sequence has loud (note onset) and quiet (tail) windows.
        let win = SR as usize / 40; // 25 ms
        let peaks: Vec<f32> = out
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, &s| m.max(s.abs())))
            .collect();
        let loud = peaks.iter().cloned().fold(0.0f32, f32::max);
        let quiet = peaks.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            loud > 0.1,
            "the sequence should be clearly audible, peak {loud}"
        );
        assert!(
            loud > 3.0 * quiet.max(1e-4),
            "the pluck envelope should make the level pulse (loud={loud}, quiet={quiet})"
        );
    }
}
