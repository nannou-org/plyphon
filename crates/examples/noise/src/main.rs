//! "Metallic rain": random impulses ringing a resonator, over a quiet noise bed, via cpal.
//!
//! `Dust2.ar` fires sparse bipolar impulses at an average density; each excites `Ringz` (a resonator
//! from the filter set), so they land as short metallic pings at the resonant frequency. A quiet
//! `PinkNoise` bed adds "air". Both come from the noise family (`WhiteNoise`/`ClipNoise`/`GrayNoise`/
//! `PinkNoise`/`BrownNoise`/`Dust`/`Dust2`).
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// Average number of raindrops per second.
const DENSITY: f32 = 14.0;
/// The resonator's pitch (Hz).
const RING_FREQ: f32 = 1800.0;
/// How long each ping rings (seconds).
const RING_DECAY: f32 = 0.4;
/// Level of the pink-noise "air" bed.
const AIR: f32 = 0.05;
/// A gentle master gain.
const GAIN: f32 = 0.4;

/// A `BinaryOpUGen` applying operator `op` to two inputs.
fn binary(op: i16, a: InputRef, b: InputRef) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![a, b],
        num_outputs: 1,
        special_index: op,
    }
}

/// Build a `World` playing the metallic-rain texture.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mut units = vec![
        // 0: Dust2.ar(DENSITY) -> sparse bipolar impulses.
        UnitSpec::new("Dust2", Rate::Audio, vec![InputRef::Constant(DENSITY)], 1),
        // 1: Ringz(impulses, RING_FREQ, RING_DECAY) -> metallic pings.
        UnitSpec::new(
            "Ringz",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(RING_FREQ),
                InputRef::Constant(RING_DECAY),
            ],
            1,
        ),
        // 2: PinkNoise.ar -> air.
        UnitSpec::new("PinkNoise", Rate::Audio, vec![], 1),
        // 3: air * AIR.
        binary(
            2,
            InputRef::Unit { unit: 2, output: 0 },
            InputRef::Constant(AIR),
        ),
        // 4: pings + air.
        binary(
            0,
            InputRef::Unit { unit: 1, output: 0 },
            InputRef::Unit { unit: 3, output: 0 },
        ),
    ];
    // 5: Out.ar(0, [mix; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 4, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "noise".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("noise", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("metallic rain for 12s...");

    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 12);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    /// The texture should sound (finite, bounded, audible) and be dominated by the resonator's pitch:
    /// energy at RING_FREQ far exceeds a distant off-resonance bin.
    #[test]
    fn metallic_rain_rings_at_its_frequency() {
        let mut world = build(SR, 1);
        let frames = (SR * 4.0) as usize; // several seconds to catch enough drops
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "the pings should stay bounded"
        );
        assert!(rms(&out) > 0.005, "the rain should be audible");

        let on = goertzel(&out, RING_FREQ);
        let off = goertzel(&out, 300.0);
        assert!(
            on > 4.0 * off,
            "energy should concentrate at the ring frequency (on={on}, off={off})"
        );
    }

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
}
