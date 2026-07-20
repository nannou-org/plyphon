//! A morphing wavetable drone with `VOsc`, via cpal.
//!
//! A bank of wavetables is installed at buffers `0..NUM_TABLES`, each a band-limited sawtooth with one
//! more harmonic than the last - so the bank brightens from a pure sine (table 0) to a rich saw. A slow
//! LFO sweeps `VOsc`'s `bufpos` back and forth through the bank, and `VOsc` crossfades between the two
//! neighbouring tables, so the timbre morphs continuously over a fixed drone pitch. Showcases the
//! wavetable oscillators (`Osc`/`OscN`/`COsc`/`VOsc`/`VOsc3`) and scsynth's `(a, b)` wavetable format
//! (`to_wavetable`), fed here directly rather than through `/b_gen`.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or stereo.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
    to_wavetable,
};

/// Wavetables in the bank (installed at buffers `0..NUM_TABLES`).
const NUM_TABLES: usize = 8;
/// Logical samples per wavetable (a power of two, so the buffer is `2 * TABLE_SIZE` frames).
const TABLE_SIZE: usize = 1024;
/// The drone pitch (Hz).
const FREQ: f32 = 110.0;
/// How fast `bufpos` sweeps through the bank (Hz) - a slow morph.
const MORPH_HZ: f32 = 0.08;
/// A gentle master gain.
const GAIN: f32 = 0.28;

/// The `i`-th bank table: a peak-normalised band-limited sawtooth with `i + 1` harmonics (so table 0 is
/// a pure sine and the last is the brightest), packed into scsynth's `(a, b)` wavetable format.
fn bank_table(i: usize) -> Vec<f32> {
    let harmonics = i + 1;
    let mut samples: Vec<f32> = (0..TABLE_SIZE)
        .map(|n| {
            let phase = std::f32::consts::TAU * n as f32 / TABLE_SIZE as f32;
            (1..=harmonics)
                .map(|h| (h as f32 * phase).sin() / h as f32)
                .sum()
        })
        .collect();
    let peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 0.0 {
        for v in &mut samples {
            *v /= peak;
        }
    }
    to_wavetable(&samples)
}

/// Build a `World` playing the morphing wavetable drone (with the bank installed at buffers `0..N`).
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    // Install the wavetable bank at buffers 0..NUM_TABLES.
    for i in 0..NUM_TABLES {
        let _ = controller.buffer_set(
            i,
            Box::new(Buffer::from_interleaved(
                bank_table(i),
                1,
                sample_rate as f64,
            )),
        );
    }

    // `bufpos` sweeps [0, NUM_TABLES - 2] so the crossfade always has an upper neighbour (buffer
    // `floor(bufpos) + 1`). The LFO maps [-1, 1] onto that range.
    let span = (NUM_TABLES - 2) as f32 * 0.5;

    let mut units = vec![
        // 0: SinOsc.kr(MORPH_HZ) -> a slow [-1, 1] sweep.
        UnitSpec::new(
            "SinOsc",
            Rate::Control,
            vec![InputRef::Constant(MORPH_HZ), InputRef::Constant(0.0)],
            1,
        ),
        // 1: map it to a bank position in [0, NUM_TABLES - 2].
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Control,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(span),
                InputRef::Constant(span),
            ],
            num_outputs: 1,
            special_index: 0,
        },
        // 2: VOsc.ar(bufpos, FREQ, 0) -> the morphing voice.
        UnitSpec::new(
            "VOsc",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(FREQ),
                InputRef::Constant(0.0),
            ],
            1,
        ),
        // 3: RLPF(voice, 2500, 0.5) -> soften the brightest tables a touch.
        UnitSpec::new(
            "RLPF",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Constant(2500.0),
                InputRef::Constant(0.5),
            ],
            1,
        ),
        // 4: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(0.3),
            ],
            num_outputs: 1,
            special_index: 2,
        },
    ];
    // 5: Out.ar(0, [voice; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 4, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "wavetable".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("wavetable", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("a morphing wavetable drone for 16s...");

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

    /// The drone should sound, stay finite and bounded, hold its pitch, and brighten as `bufpos` sweeps
    /// from mid-bank toward the harmonic-rich tables (the LFO starts at 0 -> mid, then rises).
    #[test]
    fn wavetable_morph_sounds_and_brightens() {
        let mut world = build(SR, 1);
        let frames = (SR * 4.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "output should stay bounded"
        );
        let rms = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.01, "the drone should be audible, rms {rms}");

        // Holds its pitch at the fundamental.
        let fund = goertzel(&out, FREQ);
        assert!(
            fund > 5.0 * goertzel(&out, FREQ * 1.5),
            "should be pitched at {FREQ}"
        );

        // The upper harmonics grow as the bank position sweeps toward the brighter tables.
        let third = frames / 3;
        let high = |seg: &[f32]| {
            goertzel(seg, FREQ * 6.0) + goertzel(seg, FREQ * 7.0) + goertzel(seg, FREQ * 8.0)
        };
        let early = high(&out[..third]);
        let late = high(&out[2 * third..]);
        assert!(
            late > 2.0 * early.max(1e-6),
            "the timbre should brighten (early {early}, late {late})"
        );
    }
}
