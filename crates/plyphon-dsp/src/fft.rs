//! Engine-owned FFT plans and analysis/resynthesis windows - the spectral analogue of
//! [`Wavetables`](crate::wavetable::Wavetables).
//!
//! [`FftTables`] is **always present** so the engine can thread `&FftTables` into every unit's
//! context unconditionally (no `#[cfg]` in the hot per-block assembly). When the `fft` feature is off
//! it is a zero-field struct and the [`forward`](FftTables::forward)/[`inverse`](FftTables::inverse)/
//! [`window`](FftTables::window) methods do not exist - and no spectral unit (which would call them)
//! is compiled either.
//!
//! Under `fft` it precomputes, off the audio thread, one real FFT/IFFT plan per supported power-of-two
//! size (64..=16384) plus the window tables. The transforms run **allocation-free** on the RT thread
//! via `realfft`'s `process_with_scratch`: the complex spectrum and the plan scratch live here behind
//! a `RefCell`, borrowed for the duration of one transform. The RT thread is
//! single-threaded and walks units sequentially, so the borrow never overlaps - `borrow_mut` is a
//! single O(1) flag check, no lock and no allocation.

#[cfg(feature = "fft")]
use alloc::sync::Arc;
#[cfg(feature = "fft")]
use alloc::vec::Vec;
#[cfg(feature = "fft")]
use core::cell::RefCell;

#[cfg(feature = "fft")]
use realfft::num_complex::Complex;
#[cfg(feature = "fft")]
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

#[cfg(feature = "fft")]
use crate::math;

/// Smallest supported FFT size, `2^6` = 64.
#[cfg(feature = "fft")]
const MIN_LOG2: u32 = 6;
/// Largest supported FFT size, `2^14` = 16384 (a wasm-memory- and RT-friendly cap).
#[cfg(feature = "fft")]
const MAX_LOG2: u32 = 14;
/// Number of supported sizes (64, 128, ..., 16384).
#[cfg(feature = "fft")]
const NUM_SIZES: usize = (MAX_LOG2 - MIN_LOG2 + 1) as usize;

/// An FFT analysis/resynthesis window - scsynth's `wintype`. The codes match scsynth: `-1`
/// rectangular, `0` sine (the default; applied at both ends it is a Hann window, which sums to unity
/// at 50% overlap), `1` Hann. Always available; the FFT units choose it at build time.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowType {
    /// `-1`: no window (a flat rectangle).
    Rectangular,
    /// `0`: a half-cycle sine window (scsynth's default).
    Sine,
    /// `1`: a Hann (raised-cosine) window.
    Hann,
}

impl WindowType {
    /// Decode scsynth's `wintype` input: `-1` rectangular, `1` Hann, anything else (incl. `0`) sine.
    pub fn from_code(code: f32) -> WindowType {
        match code as i32 {
            -1 => WindowType::Rectangular,
            1 => WindowType::Hann,
            _ => WindowType::Sine,
        }
    }

    /// Index into a per-size window table.
    #[cfg(feature = "fft")]
    fn index(self) -> usize {
        match self {
            WindowType::Rectangular => 0,
            WindowType::Sine => 1,
            WindowType::Hann => 2,
        }
    }
}

/// Whether `n` is a supported FFT size: a power of two in `[64, 16384]`. A spectral unit validates its
/// `fftsize` against this at build time so the audio thread always finds a plan.
#[cfg(feature = "fft")]
pub fn is_supported_size(n: usize) -> bool {
    n.is_power_of_two() && (MIN_LOG2..=MAX_LOG2).contains(&n.trailing_zeros())
}

/// The plan index for a supported size, else `None`.
#[cfg(feature = "fft")]
fn size_index(n: usize) -> Option<usize> {
    is_supported_size(n).then(|| (n.trailing_zeros() - MIN_LOG2) as usize)
}

/// Engine-owned FFT plans + windows. Empty unless built with `fft`.
pub struct FftTables {
    #[cfg(feature = "fft")]
    inner: FftInner,
}

/// The complex working memory a transform borrows for its duration (sized to the largest plan).
#[cfg(feature = "fft")]
struct Scratch {
    /// The half-complex spectrum (`N/2 + 1` bins); the forward writes it, the inverse reads it.
    spectrum: Vec<Complex<f32>>,
    /// realfft's per-call scratch.
    work: Vec<Complex<f32>>,
}

#[cfg(feature = "fft")]
struct FftInner {
    forward: [Arc<dyn RealToComplex<f32>>; NUM_SIZES],
    inverse: [Arc<dyn ComplexToReal<f32>>; NUM_SIZES],
    /// `windows[size_index][WindowType::index]`.
    windows: [[Vec<f32>; 3]; NUM_SIZES],
    scratch: RefCell<Scratch>,
}

#[cfg(feature = "fft")]
impl FftInner {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward: [Arc<dyn RealToComplex<f32>>; NUM_SIZES] =
            core::array::from_fn(|i| planner.plan_fft_forward(1usize << (i as u32 + MIN_LOG2)));
        let inverse: [Arc<dyn ComplexToReal<f32>>; NUM_SIZES] =
            core::array::from_fn(|i| planner.plan_fft_inverse(1usize << (i as u32 + MIN_LOG2)));
        let windows = core::array::from_fn(|i| {
            let n = 1usize << (i as u32 + MIN_LOG2);
            [
                make_window(WindowType::Rectangular, n),
                make_window(WindowType::Sine, n),
                make_window(WindowType::Hann, n),
            ]
        });
        // One scratch sized to the largest plan covers every size (each call uses a prefix).
        let work_len = forward
            .iter()
            .map(|p| p.get_scratch_len())
            .chain(inverse.iter().map(|p| p.get_scratch_len()))
            .max()
            .unwrap_or(0);
        let spectrum_len = (1usize << MAX_LOG2) / 2 + 1;
        let scratch = RefCell::new(Scratch {
            spectrum: vec![Complex::new(0.0, 0.0); spectrum_len],
            work: vec![Complex::new(0.0, 0.0); work_len],
        });
        FftInner {
            forward,
            inverse,
            windows,
            scratch,
        }
    }
}

impl FftTables {
    /// Build the FFT plans and windows (off the audio thread). Empty without the `fft` feature.
    pub fn new() -> Self {
        FftTables {
            #[cfg(feature = "fft")]
            inner: FftInner::new(),
        }
    }
}

impl Default for FftTables {
    fn default() -> Self {
        FftTables::new()
    }
}

#[cfg(feature = "fft")]
impl FftTables {
    /// Forward real FFT of `time_in` (`size` samples, consumed/overwritten) into `packed_out` (`size`
    /// floats, scsynth's packed layout `[DC, Nyquist, re1, im1, ...]`). Returns `false` (a no-op) if
    /// `size` is unsupported or the lengths are wrong. RT-safe: no allocation.
    pub fn forward(&self, size: usize, time_in: &mut [f32], packed_out: &mut [f32]) -> bool {
        let Some(idx) = size_index(size) else {
            return false;
        };
        if time_in.len() != size || packed_out.len() != size {
            return false;
        }
        let plan = &self.inner.forward[idx];
        let mut guard = self.inner.scratch.borrow_mut();
        let Scratch { spectrum, work } = &mut *guard;
        let spec = &mut spectrum[..size / 2 + 1];
        let work = &mut work[..plan.get_scratch_len()];
        if plan.process_with_scratch(time_in, spec, work).is_err() {
            return false;
        }
        pack(spec, packed_out);
        true
    }

    /// Inverse real FFT of the packed spectrum `packed_in` (`size` floats) into `time_out` (`size`
    /// samples), normalized by `1/size` (realfft's inverse is unnormalized). Returns `false` (a no-op)
    /// if `size` is unsupported or the lengths are wrong. RT-safe: no allocation.
    pub fn inverse(&self, size: usize, packed_in: &[f32], time_out: &mut [f32]) -> bool {
        let Some(idx) = size_index(size) else {
            return false;
        };
        if packed_in.len() != size || time_out.len() != size {
            return false;
        }
        let plan = &self.inner.inverse[idx];
        let mut guard = self.inner.scratch.borrow_mut();
        let Scratch { spectrum, work } = &mut *guard;
        let spec = &mut spectrum[..size / 2 + 1];
        unpack(packed_in, spec);
        let work = &mut work[..plan.get_scratch_len()];
        if plan.process_with_scratch(spec, time_out, work).is_err() {
            return false;
        }
        let norm = 1.0 / size as f32;
        for x in time_out.iter_mut() {
            *x *= norm;
        }
        true
    }

    /// The precomputed `wintype` window for `size` (an empty slice if `size` is unsupported).
    pub fn window(&self, size: usize, wintype: WindowType) -> &[f32] {
        match size_index(size) {
            Some(idx) => &self.inner.windows[idx][wintype.index()],
            None => &[],
        }
    }
}

/// Pack realfft's half-complex spectrum (`N/2 + 1` bins) into scsynth's flat layout
/// `[DC, Nyquist, re1, im1, ..., re(N/2-1), im(N/2-1)]` (`N` floats).
#[cfg(feature = "fft")]
fn pack(spec: &[Complex<f32>], out: &mut [f32]) {
    let half = spec.len() - 1; // N/2
    out[0] = spec[0].re; // DC (imag is 0)
    out[1] = spec[half].re; // Nyquist (imag is 0)
    for k in 1..half {
        out[2 * k] = spec[k].re;
        out[2 * k + 1] = spec[k].im;
    }
}

/// Inverse of [`pack`]: read scsynth's flat layout into realfft's half-complex spectrum (with zero
/// imaginary parts for DC and Nyquist, as the inverse transform expects).
#[cfg(feature = "fft")]
fn unpack(packed: &[f32], spec: &mut [Complex<f32>]) {
    let half = spec.len() - 1; // N/2
    spec[0] = Complex::new(packed[0], 0.0);
    spec[half] = Complex::new(packed[1], 0.0);
    for k in 1..half {
        spec[k] = Complex::new(packed[2 * k], packed[2 * k + 1]);
    }
}

/// One cycle of a `kind` window of length `n` (periodic form, so applying it at both analysis and
/// resynthesis sums to unity at 50% overlap).
#[cfg(feature = "fft")]
fn make_window(kind: WindowType, n: usize) -> Vec<f32> {
    use core::f64::consts::{PI, TAU};
    (0..n)
        .map(|i| match kind {
            WindowType::Rectangular => 1.0,
            WindowType::Sine => math::sin(PI * i as f64 / n as f64) as f32,
            WindowType::Hann => (0.5 - 0.5 * math::cos(TAU * i as f64 / n as f64)) as f32,
        })
        .collect()
}

#[cfg(all(test, feature = "fft"))]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    /// A naive DFT magnitude at bin `k` for a real signal, to check the packed forward transform.
    fn dft_bin(signal: &[f32], k: usize) -> (f32, f32) {
        let n = signal.len();
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &x) in signal.iter().enumerate() {
            let phase = -TAU * (k as f64) * (i as f64) / (n as f64);
            re += x as f64 * math::cos(phase);
            im += x as f64 * math::sin(phase);
        }
        (re as f32, im as f32)
    }

    #[test]
    fn forward_matches_a_naive_dft() {
        let tables = FftTables::new();
        for &size in &[64usize, 256, 1024] {
            // A cosine at bin 3 plus a DC offset: a known, easy-to-check spectrum.
            let mut time: Vec<f32> = (0..size)
                .map(|i| 0.25 + (TAU * 3.0 * i as f64 / size as f64).cos() as f32)
                .collect();
            let mut packed = vec![0.0f32; size];
            assert!(tables.forward(size, &mut time, &mut packed));

            // DC bin: packed[0]; bin 3: packed[6], packed[7].
            let (dc_re, _) = dft_bin(
                &(0..size)
                    .map(|i| 0.25 + (TAU * 3.0 * i as f64 / size as f64).cos() as f32)
                    .collect::<Vec<_>>(),
                0,
            );
            assert!(
                (packed[0] - dc_re).abs() < 1e-2,
                "DC: {} vs {dc_re}",
                packed[0]
            );
            let signal: Vec<f32> = (0..size)
                .map(|i| 0.25 + (TAU * 3.0 * i as f64 / size as f64).cos() as f32)
                .collect();
            let (re3, im3) = dft_bin(&signal, 3);
            assert!(
                (packed[6] - re3).abs() < 1e-1,
                "bin3 re: {} vs {re3}",
                packed[6]
            );
            assert!(
                (packed[7] - im3).abs() < 1e-1,
                "bin3 im: {} vs {im3}",
                packed[7]
            );
        }
    }

    #[test]
    fn forward_then_inverse_round_trips() {
        let tables = FftTables::new();
        let size = 512;
        let orig: Vec<f32> = (0..size)
            .map(|i| (TAU * 5.0 * i as f64 / size as f64).sin() as f32 * 0.5)
            .collect();
        let mut time = orig.clone();
        let mut packed = vec![0.0f32; size];
        let mut back = vec![0.0f32; size];
        assert!(tables.forward(size, &mut time, &mut packed));
        assert!(tables.inverse(size, &packed, &mut back));
        for (a, b) in orig.iter().zip(&back) {
            assert!((a - b).abs() < 1e-4, "round-trip drift: {a} vs {b}");
        }
    }

    #[test]
    fn rejects_unsupported_sizes() {
        let tables = FftTables::new();
        let mut t = vec![0.0f32; 100];
        let mut p = vec![0.0f32; 100];
        assert!(!tables.forward(100, &mut t, &mut p)); // not a power of two
        assert!(!is_supported_size(32)); // below the minimum
        assert!(!is_supported_size(32768)); // above the maximum
        assert!(is_supported_size(1024));
    }
}
