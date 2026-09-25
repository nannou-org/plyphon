//! The engine's FFT tables cover scsynth's whole size range, `SC_FFT_MINSIZE` (8) to
//! `SC_FFT_ABSOLUTE_MAXSIZE` (262144), with no plan built on first use.

#![cfg(feature = "fft")]

use plyphon_dsp::fft::{FftTables, WindowType, is_supported_size};

#[test]
fn every_scsynth_fft_size_is_supported() {
    for log2 in 3..=18 {
        assert!(is_supported_size(1 << log2), "2^{log2}");
    }
    for n in [4, 524_288, 100, 0] {
        assert!(!is_supported_size(n), "{n} is outside scsynth's range");
    }
}

#[test]
fn the_smallest_and_largest_sizes_round_trip() {
    let tables = FftTables::new();
    for n in [8usize, 262_144] {
        let signal: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 97) as f32 / 97.0 - 0.5)
            .collect();
        let mut time = signal.clone();
        let mut packed = vec![0.0; n];
        assert!(tables.forward(n, &mut time, &mut packed), "{n}: forward");
        let mut back = vec![0.0; n];
        assert!(tables.inverse(n, &packed, &mut back), "{n}: inverse");
        let err = signal
            .iter()
            .zip(&back)
            .fold(0.0f32, |e, (a, b)| e.max((a - b).abs()));
        assert!(err < 1e-4, "{n}: round-trip error {err}");
    }
}

#[test]
fn windows_are_stored_for_sine_and_hann_only() {
    let tables = FftTables::new();
    for n in [8usize, 262_144] {
        assert_eq!(tables.window(n, WindowType::Sine).len(), n);
        assert_eq!(tables.window(n, WindowType::Hann).len(), n);
        assert!(
            tables.window(n, WindowType::Rectangular).is_empty(),
            "the rectangular window is all ones and not stored"
        );
    }
}
