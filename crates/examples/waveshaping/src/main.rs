//! A wavefolder: a sine driven hard through `Fold`, with the drive swept by an LFO, via cpal.
//!
//! A pure `SinOsc` is amplified well past the fold bounds and passed through `Fold` (from the
//! range-shaping set `Clip`/`Wrap`/`Fold`/...). As the drive rises, the wave folds back on itself
//! more times, adding harmonics - the classic "West-coast" timbre. A slow `SinOsc.kr` LFO sweeps the
//! drive, so the tone brightens and darkens.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The tone's pitch (Hz).
const FREQ: f32 = 110.0;
/// How fast the drive sweeps (Hz).
const SWEEP_RATE: f32 = 0.25;
/// Drive range: the LFO maps to `[MIN_DRIVE, MAX_DRIVE]`.
const MIN_DRIVE: f32 = 1.0;
const MAX_DRIVE: f32 = 8.0;
/// The fold bound (the wave folds within `[-FOLD, FOLD]`).
const FOLD: f32 = 0.5;
/// A gentle master gain.
const GAIN: f32 = 0.25;

/// Build a `World` playing the swept wavefolder.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mid = 0.5 * (MAX_DRIVE + MIN_DRIVE);
    let half = 0.5 * (MAX_DRIVE - MIN_DRIVE);

    let mut units = vec![
        // 0: SinOsc.ar(FREQ) -> the tone.
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
            1,
        ),
        // 1: SinOsc.kr(SWEEP_RATE) -> drive LFO in [-1, 1].
        UnitSpec::new(
            "SinOsc",
            Rate::Control,
            vec![InputRef::Constant(SWEEP_RATE), InputRef::Constant(0.0)],
            1,
        ),
        // 2: MulAdd(LFO, half, mid) -> drive in [MIN_DRIVE, MAX_DRIVE].
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Control,
            inputs: vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(half),
                InputRef::Constant(mid),
            ],
            num_outputs: 1,
            special_index: 0,
        },
        // 3: tone * drive.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 2, output: 0 },
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
        // 4: Fold(driven, -FOLD, FOLD) -> the wavefolder.
        UnitSpec::new(
            "Fold",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(-FOLD),
                InputRef::Constant(FOLD),
            ],
            1,
        ),
    ];
    // 5: Out.ar(0, [folded; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 4, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "waveshaping".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("waveshaping", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("a swept wavefolder for 12s...");

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

    /// The wavefolder should stay within the fold bounds, sound, and brighten as the drive rises: the
    /// LFO (0.25 Hz -> 4 s period) peaks the drive near t=1 s and bottoms it near t=3 s, so the folded
    /// wave crosses zero far more often when driven hard.
    #[test]
    fn wavefolder_stays_bounded_and_brightens() {
        let mut world = build(SR, 1);
        let frames = (SR * 4.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() <= FOLD + 1e-4),
            "Fold must keep the wave within its bounds"
        );

        let s = SR as usize;
        let crossings = |w: &[f32]| w.windows(2).filter(|p| p[0] * p[1] < 0.0).count();
        let bright = crossings(&out[s / 2..3 * s / 2]); // drive high
        let dark = crossings(&out[5 * s / 2..7 * s / 2]); // drive low
        assert!(
            bright > dark,
            "harder drive should fold more (more zero-crossings): bright={bright}, dark={dark}"
        );
        assert!(bright > 0, "the wavefolder should be audible");
    }
}
