//! A cavernous space with `Pluck` and `GVerb`, via cpal.
//!
//! Two Karplus-Strong plucked strings (`Pluck`) - a low root and a fifth above - are re-plucked by
//! two `Impulse` clocks running at slightly different (incommensurate) rates, so their overlaps drift
//! and never quite repeat. Their sum is sent through a large `GVerb` feedback-delay-network reverb,
//! which smears the plucks into a slowly-evolving cavern. Showcases the delay/reverb family's `Pluck`
//! and `GVerb` (alongside `BufDelay*`/`DelTapWr`/`DelTapRd`/`PitchShift`/`FreeVerb`/`FreeVerb2`).
//!
//! The whole patch is in-engine (no control plane), like the sine and moog examples; `GVerb` is
//! stereo, so this plays in stereo (and folds to mono if the device is mono).

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The two string pitches (Hz): a low A and the E a fifth above.
const ROOT_HZ: f32 = 110.0;
const FIFTH_HZ: f32 = 164.81;
/// The two re-pluck clock rates (Hz); their ratio drifts, so the pattern never quite repeats.
const CLOCK_A: f32 = 0.5;
const CLOCK_B: f32 = 0.37;
/// String decay time (seconds) and damping.
const DECAY: f32 = 5.0;
const COEF: f32 = 0.2;
/// A gentle master gain.
const GAIN: f32 = 0.4;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// A `BinaryOpUGen` with the given operator `special_index` (0 add, 2 multiply).
fn binop(a: InputRef, b: InputRef, op: i16) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![a, b],
        num_outputs: 1,
        special_index: op,
    }
}

/// A `Pluck.ar(noise, Impulse.kr(clock_hz), maxdelaytime, 1/freq, decay, coef)`.
fn pluck(noise: u32, clock: u32, freq: f32) -> UnitSpec {
    UnitSpec::new(
        "Pluck",
        Rate::Audio,
        vec![
            u(noise),
            u(clock),
            c(0.05),
            c(1.0 / freq),
            c(DECAY),
            c(COEF),
        ],
        1,
    )
}

/// Build a `World` playing the reverberant plucks.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mut units = vec![
        // 0: shared excitation noise.
        UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
        // 1: clock A -> 2: root string.
        UnitSpec::new("Impulse", Rate::Control, vec![c(CLOCK_A), c(0.0)], 1),
        pluck(0, 1, ROOT_HZ),
        // 3: clock B -> 4: fifth string.
        UnitSpec::new("Impulse", Rate::Control, vec![c(CLOCK_B), c(0.0)], 1),
        pluck(0, 3, FIFTH_HZ),
        // 5: sum the two strings, 6: tame the level before the reverb.
        binop(u(2), u(4), 0),
        binop(u(5), c(0.5), 2),
        // 7: GVerb.ar(sum, roomsize, revtime, damping, inputbw, spread, dry, early, tail, maxroomsize).
        UnitSpec::new(
            "GVerb",
            Rate::Audio,
            vec![
                u(6),
                c(40.0),  // roomsize (const)
                c(5.0),   // revtime
                c(0.4),   // damping
                c(0.5),   // inputbw
                c(20.0),  // spread (const)
                c(0.35),  // drylevel
                c(0.6),   // earlyreflevel
                c(0.5),   // taillevel
                c(300.0), // maxroomsize (const)
            ],
            2,
        ),
    ];

    // 8: Out - stereo uses both GVerb channels; mono sums them (unit 8) first.
    if out_channels >= 2 {
        units.push(UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                c(0.0),
                InputRef::Unit { unit: 7, output: 0 },
                InputRef::Unit { unit: 7, output: 1 },
            ],
            0,
        ));
    } else {
        units.push(binop(
            InputRef::Unit { unit: 7, output: 0 },
            InputRef::Unit { unit: 7, output: 1 },
            0,
        ));
        units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(8)], 0));
    }

    controller.add_synthdef(SynthDef {
        name: "reverb".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("reverb", ROOT_GROUP_ID, AddAction::Tail);

    world
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!("reverberant plucked strings for 20s...");

    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 20);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// The reverberant plucks should sound, stay finite and bounded, and spread across the stereo
    /// field (GVerb's two channels are not identical).
    #[test]
    fn reverb_sounds_and_spreads() {
        let mut world = build(SR, 2);
        let frames = (SR * 6.0) as usize;
        let mut out = vec![0.0f32; frames * 2];
        world.fill(&mut out, 2);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 8.0),
            "output should stay bounded"
        );
        let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
        let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
        let rms = |c: &[f32]| (c.iter().map(|&s| s * s).sum::<f32>() / c.len() as f32).sqrt();
        assert!(
            rms(&ch0) > 0.005 && rms(&ch1) > 0.005,
            "the plucks should sound"
        );
        assert!(
            ch0.iter().zip(&ch1).any(|(a, b)| (a - b).abs() > 1e-4),
            "the reverb should spread across the stereo field"
        );
    }
}
