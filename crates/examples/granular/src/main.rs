//! A granular time-stretch with `Warp1`, via cpal.
//!
//! A short arpeggio (four notes of a minor-pentatonic climb) is synthesized once into a buffer, then
//! `Warp1` granulates it: a very slow `pointer` sweep walks the read position through the buffer while
//! `freqScale` holds the original pitch, so the phrase is stretched far longer than it was recorded -
//! the classic granular time-stretch. Each of the two output channels runs its own independent grain
//! cloud (with `windowRandRatio`-jittered window sizes), so the pad spreads across the stereo field on
//! its own. Showcases the granular family's buffer grains (`Warp1`, alongside `GrainBuf`/`TGrains` and
//! the triggered `GrainSin`/`GrainFM`/`GrainIn`).
//!
//! The buffer is synthesized in-process (no audio file), and the patch is otherwise in-engine (no
//! control plane), like the sine and moog examples; it plays in mono or stereo.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The notes of the recorded phrase (Hz): an A-minor-pentatonic climb A-C-E-A.
const PHRASE_HZ: [f32; 4] = [220.0, 261.63, 329.63, 440.0];
/// Each note's length in the buffer (seconds).
const NOTE_SECS: f32 = 0.6;
/// How fast the granular read pointer sweeps the buffer (Hz); very slow, so time is stretched.
const SWEEP_HZ: f32 = 0.06;
/// Grain length (seconds) and how many grains overlap - a long-ish window smears the notes together.
const WINDOW_SECS: f32 = 0.15;
const OVERLAPS: f32 = 4.0;
/// Per-grain window-length jitter (0-1); a little randomness decorrelates the two channels.
const WINDOW_RAND: f32 = 0.2;
/// A gentle master gain.
const GAIN: f32 = 0.5;

/// Synthesize the phrase into a mono buffer: each note is two partials under a raised-sine amplitude
/// envelope, so it fades in and out and the buffer ends near silence.
fn phrase_buffer(sample_rate: f32) -> Buffer {
    use std::f32::consts::{PI, TAU};
    let note_len = (sample_rate * NOTE_SECS) as usize;
    let frames = note_len * PHRASE_HZ.len();
    let samples: Vec<f32> = (0..frames)
        .map(|i| {
            let note = i / note_len;
            let phase = (i % note_len) as f32 / note_len as f32; // 0..1 within the note
            let t = i as f32 / sample_rate;
            let f = PHRASE_HZ[note];
            let env = (PI * phase).sin(); // 0 -> 1 -> 0 across the note
            let tone = (TAU * f * t).sin() * 0.6 + (TAU * 2.0 * f * t).sin() * 0.25;
            tone * env
        })
        .collect();
    Buffer::from_interleaved(samples, 1, sample_rate as f64)
}

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

/// Build a `World` playing the granular time-stretch.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    // The phrase lives in buffer 0; Warp1 reads it by bufnum.
    controller
        .buffer_set(0, Box::new(phrase_buffer(sample_rate)))
        .expect("buffer_set");

    let mut units = vec![
        // 0: LFSaw.kr(SWEEP_HZ) -> a slow [-1, 1] rising ramp (the read position sweep).
        UnitSpec::new(
            "LFSaw",
            Rate::Control,
            vec![InputRef::Constant(SWEEP_HZ), InputRef::Constant(0.0)],
            1,
        ),
        // 1: pointer = ramp mapped to [0, 1] across the buffer.
        mul_add(0, 0.5, 0.5),
        // 2: Warp1.ar(numChannels, bufnum, pointer, freqScale, windowSize, envbufnum, overlaps,
        //             windowRandRatio, interp) -> the stretched, spread grain cloud.
        UnitSpec::new(
            "Warp1",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),               // bufnum
                InputRef::Unit { unit: 1, output: 0 }, // pointer
                InputRef::Constant(1.0),               // freqScale (original pitch)
                InputRef::Constant(WINDOW_SECS),       // windowSize
                InputRef::Constant(-1.0),              // envbufnum (default sin^2 window)
                InputRef::Constant(OVERLAPS),          // overlaps
                InputRef::Constant(WINDOW_RAND),       // windowRandRatio
                InputRef::Constant(2.0),               // interp (linear)
            ],
            out_channels,
        ),
    ];
    // 3: Out.ar(0, Warp1's channels).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|ch| InputRef::Unit {
        unit: 2,
        output: ch as u32,
    }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "granular".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("granular", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("a granular time-stretch for 16s...");

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

    /// The stretched pad should sound, stay finite and bounded, and spread across the stereo field -
    /// the two independent grain clouds are audible on both channels yet not sample-identical.
    #[test]
    fn granular_sounds_and_spreads() {
        let mut world = build(SR, 2);
        let frames = (SR * 6.0) as usize;
        let mut out = vec![0.0f32; frames * 2];
        world.fill(&mut out, 2);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "output should stay bounded"
        );

        let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
        let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
        let rms = |c: &[f32]| (c.iter().map(|&s| s * s).sum::<f32>() / c.len() as f32).sqrt();
        assert!(rms(&ch0) > 0.01, "left channel should be audible");
        assert!(rms(&ch1) > 0.01, "right channel should be audible");
        assert!(
            ch0.iter().zip(&ch1).any(|(a, b)| (a - b).abs() > 1e-4),
            "the two grain clouds should differ (stereo spread)"
        );
    }
}
