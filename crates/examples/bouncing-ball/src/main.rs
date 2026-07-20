//! Physical-model percussion: a `TBall` bounces on an oscillating floor and each collision rings a
//! `Ringz` resonator, via cpal.
//!
//! A slow `SinOsc` is the moving floor; `TBall` models a ball bouncing on it under gravity, emitting
//! the collision velocity as a spike at each bounce. Those spikes excite a `Ringz` resonator, so you
//! hear a metallic ping per bounce - an entirely physically-modelled rhythm. Showcases `TBall`/`Ball`/
//! `Spring`.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// How fast the floor oscillates (Hz).
const FLOOR_RATE: f32 = 2.5;
/// Floor amplitude (how far it moves).
const FLOOR_AMP: f32 = 0.3;
/// Gravity pulling the ball down (higher = harder, louder bounces).
const GRAVITY: f32 = 50.0;
/// How much to amplify the (small) collision-velocity spikes before they excite the resonator.
const EXCITE: f32 = 8.0;
/// The resonator's pitch (Hz).
const RING_FREQ: f32 = 900.0;
/// A gentle master gain.
const GAIN: f32 = 0.3;

/// Build a `World` playing the bouncing-ball percussion.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mut units = vec![
        // 0: SinOsc.ar(FLOOR_RATE) -> the moving floor (before scaling).
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(FLOOR_RATE), InputRef::Constant(0.0)],
            1,
        ),
        // 1: floor * FLOOR_AMP.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(FLOOR_AMP),
            ],
            num_outputs: 1,
            special_index: 2,
        },
        // 2: TBall(floor, gravity, damping, friction=0) -> a spike at each bounce.
        UnitSpec::new(
            "TBall",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(GRAVITY),
                InputRef::Constant(0.2),
                InputRef::Constant(0.0),
            ],
            1,
        ),
        // 3: amplify the small bounce spikes into a usable excitation.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Constant(EXCITE),
            ],
            num_outputs: 1,
            special_index: 2,
        },
        // 4: Ringz(excitation, RING_FREQ, 0.3) -> a metallic ping per bounce.
        UnitSpec::new(
            "Ringz",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(RING_FREQ),
                InputRef::Constant(0.3),
            ],
            1,
        ),
        // 5: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 4, output: 0 },
                InputRef::Constant(0.15),
            ],
            num_outputs: 1,
            special_index: 2,
        },
    ];
    // 6: Out.ar(0, [ping; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 5, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "ball".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("ball", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("a bouncing-ball resonator for 14s...");

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

    /// The percussion should sound, stay finite and bounded, and be intermittent - loud around a
    /// bounce/ring, quiet between - so a windowed peak varies a lot across the render.
    #[test]
    fn bounces_ring_the_resonator() {
        let mut world = build(SR, 1);
        let frames = (SR * 3.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "output should stay bounded"
        );

        let win = SR as usize / 20; // 50 ms
        let peaks: Vec<f32> = out
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, &s| m.max(s.abs())))
            .collect();
        let loud = peaks.iter().cloned().fold(0.0f32, f32::max);
        let quiet = peaks.iter().cloned().fold(f32::MAX, f32::min);
        assert!(loud > 0.05, "the bounces should be audible, peak {loud}");
        assert!(
            loud > 3.0 * quiet.max(1e-4),
            "the rhythm should be intermittent (loud={loud}, quiet={quiet})"
        );
    }
}
