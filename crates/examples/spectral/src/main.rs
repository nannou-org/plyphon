//! A spectral low-pass effect: a sawtooth filtered in the frequency domain by the
//! `FFT -> PV_MagMul -> IFFT` chain.
//!
//! The saw's spectrum is analyzed into a chain buffer by `FFT`; `PV_MagMul` multiplies it by a static
//! *magnitude mask* (a second buffer, filled once on the control side: unity below a cutoff bin, zero
//! above) - a brick-wall low-pass entirely in the spectral domain; `IFFT` resynthesizes the result. So
//! you hear a buzzy saw with its upper harmonics removed, with no time-domain filter anywhere.
//!
//! Exercises the FFT/spectral seam end to end (the shared `FftTables`, the per-unit `aux` analysis/
//! resynthesis rings, the packed-spectrum buffer convention, and `PV_MagMul`'s two-buffer access).

use plyphon::{
    AddAction, Buffer, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

/// Peak amplitude of the saw before filtering.
const AMP: f32 = 0.2;
/// Master gain applied in the audio callback.
const GAIN: f32 = 1.0;
/// FFT size for the analysis/resynthesis chain.
const FFT_SIZE: usize = 1024;
/// Saw fundamental (Hz) - low enough for a dense harmonic series to filter.
const FREQ: f32 = 110.0;
/// Low-pass cutoff as an FFT bin index (`bin * sampleRate / FFT_SIZE` Hz).
const CUTOFF_BIN: usize = 36;

/// A packed-spectrum (scsynth layout) magnitude mask: unity magnitude below `CUTOFF_BIN`, zero above -
/// a brick-wall low-pass for `PV_MagMul`.
fn lowpass_mask() -> Buffer {
    let mut data = vec![0.0f32; FFT_SIZE];
    // DC passes; Nyquist (index 1) is cut. Bins `k` live at [2k, 2k+1] as (re, im); magnitude 1 is
    // simply re = 1, im = 0.
    data[0] = 1.0;
    for k in 1..FFT_SIZE / 2 {
        if k < CUTOFF_BIN {
            data[2 * k] = 1.0;
        }
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
            // 3: PV_MagMul(buffer 0, mask buffer 1) - the spectral low-pass.
            UnitSpec::new(
                "PV_MagMul",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            // 4: IFFT of the filtered spectrum.
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

/// Build the engine, install the chain buffer (0) and the mask (1), and start the synth.
fn build(sample_rate: f64, channels: usize) -> (Nrt, World) {
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
        .buffer_set(1, Box::new(lowpass_mask()))
        .expect("mask buffer");
    controller.add_synthdef(spectral_def(channels));
    controller
        .synth_new("spectral", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (nrt, world)
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!("playing a spectral low-pass (FFT -> PV_MagMul -> IFFT) on a 110 Hz saw (~10s)...");

    let (stream, mut nrt) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (nrt, mut world) = build(sample_rate, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            nrt,
        )
    });
    example_audio::run_control(stream, 10_000, 50, move || {
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
    fn low_pass_keeps_the_fundamental_and_cuts_high_harmonics() {
        let (_nrt, mut world) = build(SR, 1);
        // Render past the analysis latency.
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
        // The fundamental (110 Hz, well below the cutoff) survives; a high harmonic above the cutoff
        // (CUTOFF_BIN * bin ~= 1688 Hz; the 22nd harmonic at 2420 Hz is above it) is strongly
        // attenuated.
        let cutoff_hz = CUTOFF_BIN as f32 * SR as f32 / FFT_SIZE as f32;
        let high = 22.0 * FREQ; // 2420 Hz, above the cutoff
        assert!(high > cutoff_hz, "test harmonic should be above the cutoff");
        let fundamental = goertzel(tail, FREQ);
        let cut = goertzel(tail, high);
        assert!(
            fundamental > 5.0 * cut,
            "low-pass should keep {FREQ} Hz over {high} Hz (fund={fundamental:.4}, cut={cut:.4})"
        );
    }
}
