//! A hard-sync lead: a `SyncSaw` whose saw frequency is swept by an LFO over a fixed sync pitch, the
//! classic bright "sync sweep" sound, via cpal.
//!
//! `SyncSaw` runs a sawtooth (the swept `sawFreq`) hard-synced to a fixed `syncFreq`, so the pitch
//! stays put while the timbre sweeps from mellow to searing as the saw frequency climbs. Showcases
//! `SyncSaw` (one of the oscillators `LFTri`/`LFPar`/`LFCub`/`VarSaw`/`SyncSaw`/`FSinOsc`).
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The (fixed) synced pitch (Hz).
const SYNC_FREQ: f32 = 110.0;
/// How fast the saw frequency sweeps (Hz).
const SWEEP_RATE: f32 = 0.15;
/// Saw-frequency sweep bounds (Hz): the LFO maps to `[LOW, HIGH]`.
const SAW_LOW: f32 = 110.0;
const SAW_HIGH: f32 = 1100.0;
/// A gentle master gain.
const GAIN: f32 = 0.2;

/// Build a `World` playing the hard-sync lead.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mid = 0.5 * (SAW_HIGH + SAW_LOW);
    let half = 0.5 * (SAW_HIGH - SAW_LOW);

    let mut units = vec![
        // 0: SinOsc.kr(SWEEP_RATE) -> sweep LFO.
        UnitSpec::new(
            "SinOsc",
            Rate::Control,
            vec![InputRef::Constant(SWEEP_RATE), InputRef::Constant(0.0)],
            1,
        ),
        // 1: MulAdd(LFO, half, mid) -> sawFreq in [LOW, HIGH].
        UnitSpec {
            name: "MulAdd".to_string(),
            rate: Rate::Control,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(half),
                InputRef::Constant(mid),
            ],
            num_outputs: 1,
            special_index: 0,
        },
        // 2: SyncSaw(SYNC_FREQ, sawFreq) -> the hard-synced saw.
        UnitSpec::new(
            "SyncSaw",
            Rate::Audio,
            vec![
                InputRef::Constant(SYNC_FREQ),
                InputRef::Unit { unit: 1, output: 0 },
            ],
            1,
        ),
        // 3: tame the level.
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Constant(0.5),
            ],
            num_outputs: 1,
            special_index: 2, // multiply
        },
    ];
    // 4: Out.ar(0, [lead; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 3, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "hardsync".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("hardsync", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("a hard-sync sweep for 14s...");

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

    /// The lead should sound, stay bounded, and sweep its brightness: the LFO (0.15 Hz -> ~6.7 s
    /// period) peaks the saw frequency near t=1.7 s and bottoms it near t=5 s, so the waveform is far
    /// brighter (more zero-crossings) when the saw runs high.
    #[test]
    fn hard_sync_sweeps_its_brightness() {
        let mut world = build(SR, 1);
        let frames = (SR * 5.5) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 1.0),
            "output should stay bounded"
        );

        let s = SR as usize;
        let crossings = |w: &[f32]| w.windows(2).filter(|p| p[0] * p[1] < 0.0).count();
        let bright = crossings(&out[3 * s / 2..2 * s]); // saw ~1100 Hz
        let dark = crossings(&out[9 * s / 2..5 * s]); // saw ~110 Hz
        assert!(
            bright > 2 * dark,
            "the sync sweep should be brighter when the saw runs high (bright={bright}, dark={dark})"
        );
    }
}
