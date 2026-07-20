//! A chaotic drone: a `CuspN` chaotic oscillator through a resonant filter whose cutoff wanders under
//! a slow `LatoocarfianN` map, via cpal.
//!
//! `CuspN.ar` iterates the cusp map fast enough to give a gritty, aperiodic waveform. A slow
//! `LatoocarfianN` map (a second chaos generator) wanders the `RLPF` cutoff, so the timbre drifts
//! unpredictably. Showcases the chaotic map generators (`CuspN`/`QuadN`/`GbmanN`/`StandardN`/
//! `LatoocarfianN`/`LinCongN`).
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The chaotic oscillator's iteration rate (Hz) - how fast the map advances. Run fast enough to give
/// a gritty, broadband audio texture rather than a slow staircase.
const OSC_FREQ: f32 = 2500.0;
/// How fast the cutoff-wandering map iterates (Hz).
const MOD_FREQ: f32 = 6.0;
/// Cutoff centre and half-range (Hz); the map (~[-1.5, 1.5]) maps to `[MID-HALF, MID+HALF]`.
const CUTOFF_MID: f32 = 900.0;
const CUTOFF_HALF: f32 = 600.0;
/// A gentle master gain (kept low - the chaotic source is hot and the filter resonates).
const GAIN: f32 = 0.15;

/// Build a `World` playing the chaotic drone.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mut units = vec![
        // 0: CuspN.ar(OSC_FREQ, a=1, b=1.9, xi=0) -> a chaotic waveform.
        UnitSpec::new(
            "CuspN",
            Rate::Audio,
            vec![
                InputRef::Constant(OSC_FREQ),
                InputRef::Constant(1.0),
                InputRef::Constant(1.9),
                InputRef::Constant(0.0),
            ],
            1,
        ),
        // 1: LatoocarfianN.kr(MOD_FREQ, 1, 3, 0.5, 0.5, 0.5, 0.5) -> slow wandering value.
        UnitSpec::new(
            "LatoocarfianN",
            Rate::Control,
            vec![
                InputRef::Constant(MOD_FREQ),
                InputRef::Constant(1.0),
                InputRef::Constant(3.0),
                InputRef::Constant(0.5),
                InputRef::Constant(0.5),
                InputRef::Constant(0.5),
                InputRef::Constant(0.5),
            ],
            1,
        ),
        // 2: MulAdd(mod, HALF, MID) -> cutoff.
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Control,
            inputs: vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(CUTOFF_HALF),
                InputRef::Constant(CUTOFF_MID),
            ],
            num_outputs: 1,
            special_index: 0,
        },
        // 3: RLPF(chaos, cutoff, rq=0.3) -> resonant filtered drone.
        UnitSpec::new(
            "RLPF",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Constant(0.3),
            ],
            1,
        ),
        // 4: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(0.2),
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
    ];
    // 5: Out.ar(0, [drone; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 4, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "chaos".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("chaos", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("a chaotic drone for 14s...");

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

    /// The drone should sound, stay finite and bounded, and be aperiodic - a chaotic oscillator has
    /// broadband energy, so its waveform crosses zero many times over a short window.
    #[test]
    fn chaotic_drone_sounds_and_stays_bounded() {
        let mut world = build(SR, 1);
        let frames = (SR * 1.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "output should stay bounded"
        );
        let rms = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.02, "the drone should be audible, rms {rms}");
        let crossings = out.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            crossings > 100,
            "a chaotic drone should be broadband, got {crossings} crossings"
        );
    }
}
