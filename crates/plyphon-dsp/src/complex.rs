//! Approximate polar/Cartesian conversion of one spectral bin - plyphon's port of scsynth's
//! `SC_Complex.h` lookup tables (`initTables`, `Complex::ToPolarApx`, `Polar::ToComplexApx`).
//!
//! Every scsynth `PV_*` unit, `IFFT`, `PackFFT` and `Unpack1FFT` converts a spectrum between its
//! Cartesian and polar forms with these table lookups rather than exact `hypot`/`atan2`/`sin`/`cos`,
//! so plyphon uses the same tables to produce the same bins.
//!
//! [`ComplexTables`] holds the tables. It is built off the audio thread, as part of the engine's
//! [`FftTables`](crate::fft::FftTables), and lent to the spectral units through their context.
//! Compiled only with the `fft` feature.

use alloc::vec::Vec;

use crate::math;

/// Entries in one cycle of the sine table (`kSineSize`).
const SINE_SIZE: usize = 8192;
/// Wraps a sine-table index into one cycle (`kSineMask`).
const SINE_MASK: i32 = SINE_SIZE as i32 - 1;
/// Sine-table entries per radian (`kSinePhaseScale`, a `double`).
const SINE_PHASE_SCALE: f64 = SINE_SIZE as f64 / TWOPI;
/// Entries in the phase and magnitude tables (`kPolarLUTSize`).
const POLAR_LUT_SIZE: usize = 2049;
/// Half the phase and magnitude tables, the entry for slope `0` (`kPolarLUTSize2`).
const POLAR_LUT_SIZE2: i32 = (POLAR_LUT_SIZE >> 1) as i32;

/// scsynth's `pi` (`std::acos(-1.)`), `pi2`, `pi32` and `twopi`, all `double`.
const PI: f64 = core::f64::consts::PI;
const PI2: f64 = PI * 0.5;
const PI32: f64 = PI * 1.5;
const TWOPI: f64 = PI * 2.0;

/// scsynth's `gSine`, `gPhaseLUT` and `gMagLUT`: the lookup tables behind
/// [`to_polar`](Self::to_polar) and [`to_complex`](Self::to_complex).
pub struct ComplexTables {
    /// One cycle of `sin`, with a guard entry: `sine[i] = sin(i * 2pi / 8192)`, `8193` entries.
    sine: Vec<f32>,
    /// `atan(slope)` for slopes from `-1` to `1` in steps of `1 / 1024`, `2049` entries.
    phase: Vec<f32>,
    /// `1 / cos(atan(slope))` for the same slopes: a bin's magnitude over its larger component.
    mag: Vec<f32>,
}

impl ComplexTables {
    /// Fill the tables as scsynth's `initTables` does (SC_Complex.h:59-77): double-precision `sin`,
    /// `atan` and `cos`, each result cast to `float`. Allocates, so call it off the audio thread.
    pub fn new() -> ComplexTables {
        let sine_index_to_phase = TWOPI / SINE_SIZE as f64;
        let sine = (0..=SINE_SIZE)
            .map(|i| math::sin(i as f64 * sine_index_to_phase) as f32)
            .collect();
        let r_polar_lut_size2 = 1.0 / f64::from(POLAR_LUT_SIZE2);
        let angles = (0..POLAR_LUT_SIZE as i32)
            .map(|i| math::atan((i - POLAR_LUT_SIZE2) as f64 * r_polar_lut_size2));
        let phase = angles.clone().map(|angle| angle as f32).collect();
        // scsynth's `1.f / cos(angle)` divides in double: the `float` numerator is promoted.
        let mag = angles
            .map(|angle| (1.0 / math::cos(angle)) as f32)
            .collect();
        ComplexTables { sine, phase, mag }
    }

    /// Cartesian `(real, imag)` to polar `(mag, phase)` by table lookup - scsynth's
    /// `Complex::ToPolarApx`. The phase lies in `[-pi/4, 7pi/4]`, not `(-pi, pi]`, and `(0, 0)` maps
    /// to `(0, 0)`.
    pub fn to_polar(&self, real: f32, imag: f32) -> (f32, f32) {
        let absreal = real.abs();
        let absimag = imag.abs();
        if absreal > absimag {
            let (mag, phase) = self.lookup(imag / real, absreal);
            if real > 0.0 {
                (mag, phase)
            } else {
                (mag, (PI + f64::from(phase)) as f32)
            }
        } else if absimag > 0.0 {
            let (mag, phase) = self.lookup(real / imag, absimag);
            if imag > 0.0 {
                (mag, (PI2 - f64::from(phase)) as f32)
            } else {
                (mag, (PI32 - f64::from(phase)) as f32)
            }
        } else {
            (0.0, 0.0)
        }
    }

    /// The magnitude and first-octant phase for `slope` (the smaller component over the larger, in
    /// `[-1, 1]`) and the larger component's absolute value `abs`. scsynth computes the index in
    /// `float` and truncates it to `int32`. A NaN slope (two infinite components, or a NaN one beside
    /// a non-zero one), where scsynth's cast is undefined, reads entry `0`.
    fn lookup(&self, slope: f32, abs: f32) -> (f32, f32) {
        let half = POLAR_LUT_SIZE2 as f32;
        let index = ((half + half * slope) as i32).clamp(0, POLAR_LUT_SIZE as i32 - 1) as usize;
        (self.mag[index] * abs, self.phase[index])
    }

    /// Polar `(mag, phase)` to Cartesian `(real, imag)` by table lookup - scsynth's
    /// `Polar::ToComplexApx`. The phase is scaled to table entries in `double`, truncated to `int32`
    /// and wrapped into one cycle (a negative phase wraps through two's complement, as in scsynth);
    /// the cosine is the sine a quarter cycle on. A phase whose scaled index does not fit an
    /// `int32` (beyond about `1.6e6` radians, or NaN), where scsynth's cast is undefined, saturates
    /// here (NaN reads entry `0`).
    pub fn to_complex(&self, mag: f32, phase: f32) -> (f32, f32) {
        let sinindex = (SINE_PHASE_SCALE * f64::from(phase)) as i32 & SINE_MASK;
        let cosindex = (sinindex + (SINE_SIZE as i32 >> 2)) & SINE_MASK;
        (
            mag * self.sine[cosindex as usize],
            mag * self.sine[sinindex as usize],
        )
    }
}

impl Default for ComplexTables {
    fn default() -> Self {
        ComplexTables::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table entry matches scsynth's `initTables` bit for bit. The expected value is an FNV-1a
    /// hash over the entries' bit patterns (`sine`, then `phase`, then `mag`), from a C++ harness
    /// that includes scsynth's `SC_Complex.h`; it is the same on macOS and glibc Linux.
    #[test]
    fn tables_match_scsynth() {
        let t = ComplexTables::new();
        assert_eq!(t.sine.len(), SINE_SIZE + 1);
        assert_eq!(t.phase.len(), POLAR_LUT_SIZE);
        assert_eq!(t.mag.len(), POLAR_LUT_SIZE);
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for v in t.sine.iter().chain(&t.phase).chain(&t.mag) {
            for byte in v.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(hash, 0x245d_a66b_7671_b5dc);
    }
}
