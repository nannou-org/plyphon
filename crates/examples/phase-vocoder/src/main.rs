//! A phase-vocoder spectral effect: a sawtooth **darkened in the frequency domain** by squaring each
//! partial's magnitude.
//!
//! `FFT` analyzes the saw into a packed-spectrum chain buffer; `PV_MagSquared` converts that frame to
//! polar form and squares every bin's magnitude (leaving the phases); `IFFT` converts back to complex
//! and resynthesizes. A saw's harmonics fall off as `1/n`, so squaring the magnitudes drops them to
//! `1/n^2` - the high harmonics fade faster than the low ones, and you hear the bright saw turn warm.
//! The control thread steps the fundamental through a short arpeggio so the effect plays a melody.
//!
//! Exercises the polar `PV_*` seam end to end: the `FFT` (complex) -> `PV_MagSquared` (polar) ->
//! `IFFT` (back to complex) coordinate round-trip via the buffer's tracked `coord`, the shared
//! `pv::to_polar`/`to_complex` conversions, and the packed-spectrum buffer convention. The first
//! *polar* unit, the counterpart to the `spectral` example's Cartesian `PV_MagMul`.

use plyphon::{
    AddAction, Buffer, Controller, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, World, engine,
};

/// Saw amplitude before analysis - small, since squaring magnitudes is a quadratic gain.
const AMP: f32 = 0.06;
/// Master gain to bring the darkened result back to a comfortable level.
const GAIN: f32 = 0.9;
/// FFT size for the analysis/resynthesis chain.
const FFT_SIZE: usize = 1024;
/// The fundamentals the control thread steps through (an A-minor-ish arpeggio, Hz).
const MELODY: [f32; 4] = [110.0, 130.81, 164.81, 220.0];

/// `Saw.ar(freq) * AMP -> FFT(0) -> PV_MagSquared -> IFFT(0) -> Out`. `freq` is a control parameter
/// the control thread sweeps; `unit` indices are wired by position.
fn pvoc_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 4, output: 0 });
    }
    SynthDef {
        name: "pvoc".to_string(),
        params: vec![Param::control("freq", MELODY[0])],
        units: vec![
            // 0: Saw.ar(freq).
            UnitSpec::new("Saw", Rate::Audio, vec![InputRef::Param(0)], 1),
            // 1: Saw * AMP.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(AMP),
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 2: FFT into chain buffer 0 (writes the complex spectrum).
            UnitSpec::new(
                "FFT",
                Rate::Control,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.5),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(FFT_SIZE as f32),
                ],
                1,
            ),
            // 3: PV_MagSquared - to polar, square each magnitude, keep phases (the darkening).
            UnitSpec::new(
                "PV_MagSquared",
                Rate::Control,
                vec![InputRef::Unit { unit: 2, output: 0 }],
                1,
            ),
            // 4: IFFT (back to complex, then resynthesize).
            UnitSpec::new(
                "IFFT",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(FFT_SIZE as f32),
                ],
                1,
            ),
            // 5: Out into every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// Build the engine, install the chain buffer (0), and start the synth. Returns the `Controller` (the
/// control thread steps `freq` through it), the `Nrt`, the `World`, and the synth's node id.
fn build(sample_rate: f64, channels: usize) -> (Controller, Nrt, World, i32) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate,
        output_channels: channels,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(
                vec![0.0; FFT_SIZE],
                1,
                sample_rate,
            )),
        )
        .expect("chain buffer");
    controller.add_synthdef(pvoc_def(channels));
    let node = controller
        .synth_new("pvoc", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (controller, nrt, world, node)
}

/// Control-loop cadence (ms) and how long each note holds.
const STEP_MS: u32 = 40;
const NOTE_MS: u32 = 360;

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!("playing a saw darkened by PV_MagSquared (FFT -> PV_MagSquared -> IFFT)...");

    let (stream, (mut controller, mut nrt, node)) =
        example_audio::play_with(GAIN, |sample_rate, channels| {
            let (controller, nrt, mut world, node) = build(sample_rate, channels);
            (
                move |out: &mut [f32], channels: usize| world.fill(out, channels),
                (controller, nrt, node),
            )
        });

    // Step the fundamental through the melody (control 0 of the synth), then service NRT cleanup.
    let mut tick: u32 = 0;
    example_audio::run_control(stream, 24_000, STEP_MS, move || {
        tick += 1;
        let note = (tick * STEP_MS / NOTE_MS) as usize % MELODY.len();
        let _ = controller.set_control(node, 0, MELODY[note]);
        nrt.process();
        while nrt.poll().is_some() {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    fn goertzel(samples: &[f32], freq: f32) -> f32 {
        let n = samples.len();
        let k = (0.5 + n as f32 * freq / SR as f32).floor();
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

    fn render_tail(world: &mut World, settle: usize, frames: usize) -> Vec<f32> {
        let mut blk = [0.0f32; 256];
        let mut done = 0;
        while done < settle {
            world.fill(&mut blk, 1);
            done += blk.len();
        }
        let mut out = Vec::with_capacity(frames);
        while out.len() < frames {
            world.fill(&mut blk, 1);
            out.extend_from_slice(&blk);
        }
        out.truncate(frames);
        out
    }

    /// `Saw.ar(freq) * AMP -> Out` - the dry reference, same source and level, no spectral processing.
    fn dry_saw_def() -> SynthDef {
        SynthDef {
            name: "dry".to_string(),
            params: vec![Param::control("freq", MELODY[0])],
            units: vec![
                UnitSpec::new("Saw", Rate::Audio, vec![InputRef::Param(0)], 1),
                UnitSpec {
                    name: "BinaryOpUGen".to_string(),
                    rate: Rate::Audio,
                    inputs: vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Constant(AMP),
                    ],
                    num_outputs: 1,
                    special_index: 2,
                },
                UnitSpec::new(
                    "Out",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.0),
                        InputRef::Unit { unit: 0, output: 0 },
                    ],
                    0,
                ),
            ],
        }
    }

    #[test]
    fn pv_magsquared_darkens_the_harmonic_balance() {
        // The processed saw's high harmonics fall off faster than the dry saw's: squaring the bin
        // magnitudes turns the saw's 1/n harmonic series into ~1/n^2. So the 5th-harmonic-to-
        // fundamental ratio is markedly smaller after PV_MagSquared than in the dry saw.
        let freq = MELODY[0];
        let (h1, h5) = (freq, 5.0 * freq);

        let (_c, _nrt, mut world, _node) = build(SR, 1);
        let wet = render_tail(&mut world, 8_192, 8_192);
        assert!(
            wet.iter().all(|s| s.is_finite()),
            "the polar round trip produced non-finite output"
        );
        assert!(wet.iter().any(|s| s.abs() > 1e-3), "pvoc chain was silent");
        assert!(
            wet.iter().all(|s| s.abs() <= 1.5),
            "pvoc output ran away (max {})",
            wet.iter().fold(0.0f32, |m, s| m.max(s.abs()))
        );

        // Dry reference saw at the same fundamental and level.
        let (mut dc, _dn, mut dworld) = engine(Options {
            sample_rate: SR,
            output_channels: 1,
            ..Options::default()
        });
        dc.add_synthdef(dry_saw_def());
        dc.synth_new("dry", ROOT_GROUP_ID, AddAction::Tail).unwrap();
        let dry = render_tail(&mut dworld, 8_192, 8_192);

        let wet_ratio = goertzel(&wet, h5) / goertzel(&wet, h1).max(1e-9);
        let dry_ratio = goertzel(&dry, h5) / goertzel(&dry, h1).max(1e-9);
        assert!(
            wet_ratio < 0.5 * dry_ratio,
            "PV_MagSquared should darken the spectrum (wet h5/h1 {wet_ratio:.4} vs dry {dry_ratio:.4})"
        );
    }
}
