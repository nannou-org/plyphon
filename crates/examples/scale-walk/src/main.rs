//! A scale-quantised melodic walk, via cpal.
//!
//! A smooth random `LFNoise1` wanders an index into a scale buffer; `Index` reads the nearest scale
//! degree, so the melody drifts continuously but always lands on an in-scale note (`midicps` turns the
//! MIDI note into a frequency, `Lag` glides between notes). Showcases the buffer-lookup selection
//! units (`Index`/`IndexL`/`WrapIndex`/`FoldIndex`) and `Select`.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// A two-octave C major pentatonic scale (MIDI note numbers) - the lookup table `Index` reads.
const SCALE: &[f32] = &[
    48.0, 50.0, 52.0, 55.0, 57.0, 60.0, 62.0, 64.0, 67.0, 69.0, 72.0,
];
/// How fast the index wanders (Hz).
const WANDER_HZ: f32 = 2.2;
/// A gentle master gain.
const GAIN: f32 = 0.22;
/// The `midicps` unary operator's `special_index`.
const MIDICPS: i16 = 17;

/// Build a `World` playing the scale-quantised walk (with the scale installed at buffer 0).
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    // Install the scale as a one-channel lookup buffer at index 0.
    let _ = controller.buffer_set(
        0,
        Box::new(Buffer::from_interleaved(
            SCALE.to_vec(),
            1,
            sample_rate as f64,
        )),
    );

    let len = SCALE.len() as f32;
    let half = len * 0.5;

    let mut units = vec![
        // 0: LFNoise1.kr(WANDER_HZ) -> a smooth random contour in [-1, 1].
        UnitSpec::new(
            "LFNoise1",
            Rate::Control,
            vec![InputRef::Constant(WANDER_HZ)],
            1,
        ),
        // 1: map it to a scale-buffer index in [0, len).
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Control,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(half),
                InputRef::Constant(half),
            ],
            num_outputs: 1,
            special_index: 0,
        },
        // 2: Index.kr(buf 0, index) -> the nearest scale degree (a MIDI note).
        UnitSpec::new(
            "Index",
            Rate::Control,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
            ],
            1,
        ),
        // 3: midicps(note) -> frequency.
        UnitSpec {
            name: "UnaryOpUGen".to_string(),
            rate: Rate::Control,
            inputs: vec![InputRef::Unit { unit: 2, output: 0 }],
            num_outputs: 1,
            special_index: MIDICPS,
        },
        // 4: Lag.kr(freq, 0.03) -> a short glide between notes.
        UnitSpec::new(
            "Lag",
            Rate::Control,
            vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(0.03),
            ],
            1,
        ),
        // 5: Saw.ar(freq) -> the voice.
        UnitSpec::new(
            "Saw",
            Rate::Audio,
            vec![InputRef::Unit { unit: 4, output: 0 }],
            1,
        ),
        // 6: RLPF(saw, 1600, 0.4) -> a little tone shaping.
        UnitSpec::new(
            "RLPF",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Constant(1600.0),
                InputRef::Constant(0.4),
            ],
            1,
        ),
        // 7: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 6, output: 0 },
                InputRef::Constant(0.35),
            ],
            num_outputs: 1,
            special_index: 2,
        },
    ];
    // 8: Out.ar(0, [voice; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 7, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "walk".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("walk", ROOT_GROUP_ID, AddAction::Tail);

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
    println!("a scale-quantised melodic walk for 14s...");

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

    /// The walk should sound, stay finite and bounded, be pitched (a saw voice), and change pitch
    /// over time (the wandering index visits different scale degrees).
    #[test]
    fn scale_walk_sounds_and_moves() {
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
        assert!(rms > 0.01, "the walk should be audible, rms {rms}");

        let crossings = |s: &[f32]| s.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(crossings(&out) > 200, "should be pitched");
        let third = frames / 3;
        let early = crossings(&out[..third]);
        let late = crossings(&out[2 * third..]);
        assert!(
            early.abs_diff(late) * 20 > early.max(late),
            "the pitch should wander (early {early}, late {late})"
        );
    }
}
