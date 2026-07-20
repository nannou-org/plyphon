//! A classic resonant low-pass sweep: a saw wave through an LFO-swept `RLPF`, via cpal.
//!
//! `Saw.ar(80)` gives a harmonically rich source; `RLPF` (a resonant two-pole low-pass) filters it
//! with a cutoff swept between 200 Hz and 4 kHz by a slow `SinOsc.kr` LFO (mapped with `MulAdd`), at
//! a moderate resonance. This is the archetypal subtractive-synth "filter sweep". `RLPF` is one of
//! the resonant biquads ported alongside `RHPF`, `BPF`, `BRF`, `Resonz` and `Ringz`.
//!
//! The whole patch is in-engine (no control plane), like the sine example, and plays in mono or
//! stereo.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The saw oscillator's pitch (Hz).
const SAW_FREQ: f32 = 80.0;
/// How fast the cutoff sweeps (Hz).
const SWEEP_RATE: f32 = 0.2;
/// Cutoff sweep bounds (Hz): the LFO maps to `[LOW, HIGH]`.
const CUTOFF_LOW: f32 = 200.0;
const CUTOFF_HIGH: f32 = 4000.0;
/// Reciprocal-Q of the filter (smaller = more resonant).
const RQ: f32 = 0.2;
/// A gentle master gain (kept low, as the resonant peak boosts the level).
const GAIN: f32 = 0.15;

/// Build a `World` playing the swept-filter saw.
fn build(sample_rate: f32, channels: usize) -> World {
    let out_channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    let mid = 0.5 * (CUTOFF_HIGH + CUTOFF_LOW);
    let half = 0.5 * (CUTOFF_HIGH - CUTOFF_LOW);

    let mut units = vec![
        // 0: SinOsc.kr(SWEEP_RATE) -> [-1, 1] LFO.
        UnitSpec::new(
            "SinOsc",
            Rate::Control,
            vec![InputRef::Constant(SWEEP_RATE), InputRef::Constant(0.0)],
            1,
        ),
        // 1: MulAdd(LFO, half, mid) -> cutoff in [LOW, HIGH].
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
        // 2: Saw.ar(SAW_FREQ) -> rich source.
        UnitSpec::new("Saw", Rate::Audio, vec![InputRef::Constant(SAW_FREQ)], 1),
        // 3: RLPF(saw, cutoff, RQ) -> resonant sweep.
        UnitSpec::new(
            "RLPF",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(RQ),
            ],
            1,
        ),
    ];
    // 4: Out.ar(0, [filtered; channels]).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    out_inputs.extend((0..out_channels).map(|_| InputRef::Unit { unit: 3, output: 0 }));
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));

    controller.add_synthdef(SynthDef {
        name: "filters".to_string(),
        params: vec![],
        units,
    });
    let _ = controller.synth_new("filters", ROOT_GROUP_ID, AddAction::Tail, &[]);

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
    println!("sweeping a resonant low-pass over a saw for 12s...");

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

    /// Magnitude of `samples` at `freq` (Hz) via the Goertzel algorithm.
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

    /// The swept filter should sound (non-silent, finite, bounded) and audibly open and close: the
    /// LFO (0.2 Hz -> 5 s period) peaks the cutoff near t=1.25 s and bottoms it near t=3.75 s. The saw
    /// has a harmonic at 2 kHz (80 Hz x 25), which passes wide open but is killed when the cutoff
    /// falls to ~200 Hz.
    #[test]
    fn resonant_sweep_opens_and_closes() {
        let mut world = build(SR, 1);
        let frames = (SR * 5.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 4.0),
            "the resonant peak should stay bounded"
        );
        assert!(rms(&out) > 0.02, "the filtered saw should be audible");

        let s = SR as usize;
        let open = goertzel(&out[s..3 * s / 2], 2000.0); // cutoff ~4 kHz
        let closed = goertzel(&out[7 * s / 2..4 * s], 2000.0); // cutoff ~200 Hz
        assert!(
            open > 4.0 * closed,
            "the 2 kHz harmonic should pass when open, not when closed (open={open}, closed={closed})"
        );
    }
}
