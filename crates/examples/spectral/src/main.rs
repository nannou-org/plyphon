//! Spectral cross-synthesis: a sawtooth reshaped, in the frequency domain, by a pair of **moving
//! formants** - a vowel-like morph you can't get from a time-domain filter or FM.
//!
//! `FFT` analyzes the saw into a chain buffer; `PV_MagMul` multiplies that spectrum by a *magnitude
//! mask* held in a second buffer; `IFFT` resynthesizes the result. The mask is two band-pass bumps,
//! and the control thread redraws it every tick so the two bumps sweep across the spectrum in contrary
//! motion - so you hear the saw's harmonics light up and fade as the formants glide past them, the
//! spectral envelope animated directly. Nothing here is a time-domain filter.
//!
//! Exercises the FFT/spectral seam end to end (the shared `FftTables`, the per-unit `aux` analysis/
//! resynthesis rings, the packed-spectrum buffer convention, and `PV_MagMul`'s two-buffer access),
//! plus real-time control of the spectrum from the control thread.

use plyphon::{
    AddAction, Buffer, Controller, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

/// Peak amplitude of the saw before spectral shaping.
const AMP: f32 = 0.3;
/// Master gain (the narrow formants pass only part of the saw, so lift the result a little).
const GAIN: f32 = 1.6;
/// FFT size for the analysis/resynthesis chain.
const FFT_SIZE: usize = 1024;
/// Saw fundamental (Hz) - low, for a dense harmonic series to sculpt.
const FREQ: f32 = 110.0;
/// Half-width of each formant bump, in FFT bins.
const FORMANT_WIDTH: f32 = 7.0;
/// Formant 1 sweeps low -> high (bins); formant 2 sweeps high -> low (contrary motion).
const F1_LO: f32 = 5.0;
const F1_HI: f32 = 52.0;
const F2_LO: f32 = 16.0;
const F2_HI: f32 = 74.0;

/// A packed-spectrum (scsynth layout) magnitude mask: two band-pass bumps whose centres depend on the
/// loop phase `t` in `[0, 1)`. At `t = 0` formant 1 sits at [`F1_LO`] and formant 2 at [`F2_HI`]; by
/// `t = 0.5` they have swept to [`F1_HI`] and [`F2_LO`] (crossing over), then back.
fn formant_mask(t: f32) -> Buffer {
    // A smooth 0 -> 1 -> 0 sweep over the loop.
    let sweep = 0.5 - 0.5 * (core::f32::consts::TAU * t).cos();
    let c1 = F1_LO + (F1_HI - F1_LO) * sweep;
    let c2 = F2_HI - (F2_HI - F2_LO) * sweep;
    let mag = |k: f32| {
        let g1 = (-0.5 * ((k - c1) / FORMANT_WIDTH).powi(2)).exp();
        let g2 = (-0.5 * ((k - c2) / FORMANT_WIDTH).powi(2)).exp();
        (g1 + g2).min(1.0)
    };
    let mut data = vec![0.0f32; FFT_SIZE];
    // DC at index 0, Nyquist at index 1 (both real); bin `k` is the (re, im) pair at [2k, 2k+1], and a
    // real magnitude is simply re = mag, im = 0.
    data[0] = mag(0.0);
    data[1] = mag((FFT_SIZE / 2) as f32);
    for k in 1..FFT_SIZE / 2 {
        data[2 * k] = mag(k as f32);
    }
    Buffer::from_interleaved(data, 1, 0.0)
}

/// `Saw.ar(FREQ) * AMP -> FFT(0) -> PV_MagMul(0, mask 1) -> IFFT(0) -> Out`.
fn spectral_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 4, output: 0 });
    }
    SynthDef {
        name: "spectral".to_string(),
        params: vec![],
        units: vec![
            // 0: Saw.ar(FREQ).
            UnitSpec::new("Saw", Rate::Audio, vec![InputRef::Constant(FREQ)], 1),
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
            // 2: FFT into chain buffer 0.
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
            // 3: PV_MagMul(buffer 0, mask buffer 1) - the moving spectral formants.
            UnitSpec::new(
                "PV_MagMul",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            // 4: IFFT of the shaped spectrum.
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

/// Build the engine, install the chain buffer (0) and the initial mask (1), and start the synth.
/// Returns the `Controller` (the control thread keeps animating the mask through it) and the `Nrt`.
fn build(sample_rate: f64, channels: usize) -> (Controller, Nrt, World) {
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
    controller
        .buffer_set(1, Box::new(formant_mask(0.0)))
        .expect("mask buffer");
    controller.add_synthdef(spectral_def(channels));
    controller
        .synth_new("spectral", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (controller, nrt, world)
}

/// Control-loop cadence, and how long one full formant sweep takes.
const STEP_MS: u32 = 40;
const LOOP_MS: u32 = 6_000;

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!("playing two spectral formants sweeping across a saw (FFT -> PV_MagMul -> IFFT)...");

    let (stream, (mut controller, mut nrt)) =
        example_audio::play_with(GAIN, |sample_rate, channels| {
            let (controller, nrt, mut world) = build(sample_rate, channels);
            (
                move |out: &mut [f32], channels: usize| world.fill(out, channels),
                (controller, nrt),
            )
        });

    // Each tick, redraw the magnitude mask at the current loop phase so the formants sweep, then
    // service NRT cleanup (dropping the buffers the audio thread swapped out).
    let mut tick: u32 = 0;
    example_audio::run_control(stream, 24_000, STEP_MS, move || {
        tick += 1;
        let t = (tick as f32 * STEP_MS as f32 / LOOP_MS as f32).fract();
        let _ = controller.buffer_set(1, Box::new(formant_mask(t)));
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

    #[test]
    fn moving_formants_pass_their_band_and_cut_the_valley() {
        // With the t = 0 mask (formant 1 low, around bin 5 ~= 235 Hz), the low harmonics pass while a
        // mid harmonic in the valley between the two formants is cut. Render without the control-thread
        // sweep, so the mask stays at formant_mask(0).
        let (_c, _nrt, mut world) = build(SR, 1);
        let mut out = vec![0.0f32; 16_384];
        let mut tmp = [0.0f32; 512];
        let mut filled = 0;
        while filled < out.len() {
            world.fill(&mut tmp, 1);
            let n = tmp.len().min(out.len() - filled);
            out[filled..filled + n].copy_from_slice(&tmp[..n]);
            filled += n;
        }
        let tail = &out[8_192..];

        assert!(
            tail.iter().any(|s| s.abs() > 0.01),
            "spectral chain was silent"
        );
        assert!(
            tail.iter().all(|s| s.abs() <= 1.0),
            "spectral output left [-1, 1]"
        );
        // 3rd harmonic (330 Hz, bin ~7) sits in formant 1's pass-band; the 15th (1650 Hz, bin ~35) is
        // in the valley between the two formants and is strongly attenuated.
        let passed = goertzel(tail, 3.0 * FREQ);
        let cut = goertzel(tail, 15.0 * FREQ);
        assert!(
            passed > 5.0 * cut,
            "a pass-band harmonic should dominate a valley one (pass={passed:.4}, cut={cut:.4})"
        );
    }

    /// Render a `frames`-sample tail after letting the chain settle, and return it.
    fn settle_and_capture(world: &mut World, settle: usize, frames: usize) -> Vec<f32> {
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

    #[test]
    fn redrawing_the_mask_moves_the_spectral_peak() {
        // Animating the mask from the control thread really shifts the output spectrum: with formant 1
        // low (t = 0) the 3rd harmonic passes and the 10th is in the valley; after redrawing the mask
        // at t = 0.42 the formants have swept so the 10th now passes and the 3rd falls into a valley -
        // the dominance flips. This exercises the real-time `buffer_set` mask animation end to end.
        let (mut controller, _nrt, mut world) = build(SR, 1);
        let h3 = 3.0 * FREQ; // 330 Hz
        let h10 = 10.0 * FREQ; // 1100 Hz

        let early = settle_and_capture(&mut world, 8_192, 8_192);
        assert!(
            goertzel(&early, h3) > goertzel(&early, h10),
            "at t=0 the 3rd harmonic should beat the 10th"
        );

        // Sweep the mask; the command applies on the next render block.
        controller
            .buffer_set(1, Box::new(formant_mask(0.42)))
            .unwrap();
        let late = settle_and_capture(&mut world, 8_192, 8_192);
        assert!(
            goertzel(&late, h10) > goertzel(&late, h3),
            "after sweeping the mask the 10th harmonic should beat the 3rd"
        );
    }
}
