//! A generative burble driven entirely by the low-frequency/dynamic noise family, via cpal.
//!
//! Three random modulators shape one saw voice: a stepped-then-ramped `LFNoise1` wanders the pitch,
//! a smooth `LFNoise2` sweeps the `RLPF` cutoff, and a smooth `LFDNoise3` shimmers the amplitude.
//! Nothing is periodic, so the patch never quite repeats - the characteristic use of `LFNoise*` as
//! modulation sources. Showcases `LFNoise0/1/2`, `LFClipNoise` and the dynamic `LFDNoise0/1/3`,
//! `LFDClipNoise`.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// How often the pitch picks a new random target (Hz).
const PITCH_RATE: f32 = 3.5;
/// Pitch centre and half-range (Hz); `LFNoise1` (~[-1, 1]) maps to `[MID-HALF, MID+HALF]`.
const PITCH_MID: f32 = 320.0;
const PITCH_HALF: f32 = 210.0;
/// How fast the smooth cutoff sweep wanders (Hz).
const CUTOFF_RATE: f32 = 2.0;
/// Cutoff centre and half-range (Hz).
const CUTOFF_MID: f32 = 1100.0;
const CUTOFF_HALF: f32 = 800.0;
/// How fast the amplitude shimmer wanders (Hz).
const TREM_RATE: f32 = 5.0;
/// A gentle master gain.
const GAIN: f32 = 0.2;

/// Build a `World` playing the wandering generative burble.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    // A control-rate MulAdd (`in * mul + add`).
    let mul_add = |src: u32, mul: f32, add: f32| UnitSpec {
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
    };

    let mut units = vec![
        // 0: LFNoise1.kr(PITCH_RATE) -> a stepped-then-ramped random contour in [-1, 1].
        UnitSpec::new(
            "LFNoise1",
            Rate::Control,
            vec![InputRef::Constant(PITCH_RATE)],
            1,
        ),
        // 1: map it to a wandering pitch.
        mul_add(0, PITCH_HALF, PITCH_MID),
        // 2: Saw.ar(pitch) -> a buzzy source that follows the wandering pitch.
        UnitSpec::new(
            "Saw",
            Rate::Audio,
            vec![InputRef::Unit { unit: 1, output: 0 }],
            1,
        ),
        // 3: LFNoise2.kr(CUTOFF_RATE) -> a smooth random contour for the cutoff.
        UnitSpec::new(
            "LFNoise2",
            Rate::Control,
            vec![InputRef::Constant(CUTOFF_RATE)],
            1,
        ),
        // 4: map it to a wandering cutoff.
        mul_add(3, CUTOFF_HALF, CUTOFF_MID),
        // 5: RLPF(saw, cutoff, rq=0.3) -> a resonant sweep.
        UnitSpec::new(
            "RLPF",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 4, output: 0 },
                InputRef::Constant(0.3),
            ],
            1,
        ),
        // 6: LFDNoise3.kr(TREM_RATE) -> a smooth random contour for the amplitude.
        UnitSpec::new(
            "LFDNoise3",
            Rate::Control,
            vec![InputRef::Constant(TREM_RATE)],
            1,
        ),
        // 7: map it to a unipolar tremolo gain in [0.2, 1.0].
        mul_add(6, 0.4, 0.6),
        // 8: apply the tremolo.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Unit { unit: 7, output: 0 },
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
    ];
    // 9: Out.ar(0, [voice; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 8, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "wandering".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("wandering", ROOT_GROUP_ID, AddAction::Tail);

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
    println!("a wandering generative burble for 14s...");

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

    /// The burble should sound, stay finite and bounded, carry pitched (saw) energy, and - because
    /// every modulator is aperiodic - vary in loudness across the render rather than sitting steady.
    #[test]
    fn wandering_sounds_and_varies() {
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
        assert!(rms > 0.02, "the burble should be audible, rms {rms}");

        // A saw voice crosses zero often (pitched, tens-to-hundreds of Hz), so there is real
        // waveform activity, not a DC blob.
        let crossings = out.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            crossings > 100,
            "should be pitched, got {crossings} crossings"
        );

        // The aperiodic modulation should make the loudness wander: compare windowed peaks.
        let win = SR as usize / 10; // 100 ms
        let peaks: Vec<f32> = out
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, &s| m.max(s.abs())))
            .collect();
        let loud = peaks.iter().cloned().fold(0.0f32, f32::max);
        let quiet = peaks.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            loud > 1.5 * quiet.max(1e-4),
            "the texture should wander in loudness (loud={loud}, quiet={quiet})"
        );
    }
}
