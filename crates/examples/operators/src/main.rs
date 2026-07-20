//! A ring-modulated, soft-clipped bell tone built entirely from math operators, via cpal.
//!
//! The patch is a tour of the `BinaryOpUGen`/`UnaryOpUGen` operator set:
//! - `UnaryOpUGen(midicps)` turns MIDI note 48 (C3) into a base frequency in Hz.
//! - `UnaryOpUGen(midiratio)` turns an 11-semitone interval into a frequency ratio, and
//!   `BinaryOpUGen(*)` multiplies it against the base to get an inharmonic modulator frequency.
//! - `BinaryOpUGen(*)` ring-modulates the carrier and modulator sines, giving a metallic timbre.
//! - `BinaryOpUGen(*)` drives the signal hot, then `UnaryOpUGen(softclip)` saturates it back into
//!   range for warmth.
//!
//! The whole patch is in-engine (no control plane), like the sine example. It plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The base pitch as a MIDI note number (48 = C3).
const NOTE: f32 = 48.0;
/// The modulator interval above the base, in semitones (an inharmonic, bell-like ratio).
const MOD_INTERVAL: f32 = 11.0;
/// How hard the ring-modulated signal is driven before soft clipping.
const DRIVE: f32 = 3.0;
/// A gentle master gain.
const GAIN: f32 = 0.25;

// SuperCollider operator `special_index` selectors (see `plyphon_dsp::ops`).
const OP_MUL: i16 = 2;
const OP_MIDICPS: i16 = 17;
const OP_MIDIRATIO: i16 = 19;
const OP_SOFTCLIP: i16 = 43;

/// A control-rate `UnaryOpUGen` applying `op` to a single constant input.
fn unary_const(op: i16, input: f32) -> UnitSpec {
    UnitSpec {
        name: "UnaryOpUGen".to_string(),
        rate: Rate::Control,
        inputs: vec![InputRef::Constant(input)],
        num_outputs: 1,
        special_index: op,
    }
}

/// A `BinaryOpUGen` applying `op` to two unit/constant inputs at the given rate.
fn binary(op: i16, rate: Rate, a: InputRef, b: InputRef) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate,
        inputs: vec![a, b],
        num_outputs: 1,
        special_index: op,
    }
}

/// Build a `World` playing the ring-modulated, soft-clipped bell tone.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let def = SynthDef {
        name: "operators".to_string(),
        params: vec![],
        units: vec![
            // 0: base frequency = midicps(NOTE).
            unary_const(OP_MIDICPS, NOTE),
            // 1: interval ratio = midiratio(MOD_INTERVAL).
            unary_const(OP_MIDIRATIO, MOD_INTERVAL),
            // 2: modulator frequency = base * ratio.
            binary(
                OP_MUL,
                Rate::Control,
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
            ),
            // 3: carrier sine at the base frequency.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 4: modulator sine at the inharmonic frequency.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 5: ring modulation = carrier * modulator.
            binary(
                OP_MUL,
                Rate::Audio,
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Unit { unit: 4, output: 0 },
            ),
            // 6: drive the signal hot before saturation.
            binary(
                OP_MUL,
                Rate::Audio,
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Constant(DRIVE),
            ),
            // 7: soft-clip back into range.
            UnitSpec {
                name: "UnaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Unit { unit: 6, output: 0 }],
                num_outputs: 1,
                special_index: OP_SOFTCLIP,
            },
            // 8: out to every channel.
            UnitSpec::new(
                "Out",
                Rate::Audio,
                {
                    let mut ins = vec![InputRef::Constant(0.0)];
                    ins.extend((0..out_channels).map(|_| InputRef::Unit { unit: 7, output: 0 }));
                    ins
                },
                0,
            ),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("operators", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("ring-modulated, soft-clipped bell tone for 10s...");

    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 10);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    /// The operator patch should sound: non-silent, finite, and kept in range by the soft clipper.
    #[test]
    fn operator_patch_sounds_and_stays_bounded() {
        let mut world = build(SR, 1);
        let frames = (SR * 0.5) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        // softclip maps |x| -> at most ~1 (its (|x|-0.25)/x branch), so nothing runs away.
        assert!(
            out.iter().all(|&s| s.abs() <= 1.001),
            "soft clipping should bound the signal to about ±1"
        );
        assert!(rms(&out) > 0.05, "the tone should be clearly audible");
    }

    /// Ring modulation of two different frequencies produces sidebands, so the output is not a pure
    /// tone at either source frequency - it has broadband energy.
    #[test]
    fn ring_modulation_enriches_the_spectrum() {
        let mut world = build(SR, 1);
        let frames = (SR * 0.5) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        // A soft-clipped ring-mod tone crosses zero many times per cycle (rich harmonics); confirm it
        // is far from DC-like by counting sign changes.
        let crossings = out.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            crossings > 200,
            "a ring-modulated, saturated tone should be spectrally rich, got {crossings} crossings"
        );
    }
}
