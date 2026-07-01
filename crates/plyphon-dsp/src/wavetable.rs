//! Shared wavetables owned by the engine.
//!
//! SuperCollider keeps its sine table in process-global statics reached through the plugin
//! `InterfaceTable`. plyphon instead owns the tables in a [`Wavetables`] value held by the engine
//! and lends them to units by argument while they process, so there is no global
//! mutable state and multiple engines can coexist.

use alloc::vec::Vec;
use core::f64::consts::TAU;

use crate::math;

/// Number of samples in one cycle of the sine table (matching scsynth's default).
pub const SINE_SIZE: usize = 16384;

/// The wavetables shared by oscillator units.
///
/// Tables carry one guard sample (a copy of index 0) past the end so linear interpolation can read
/// `table[i + 1]` without bounds juggling.
#[derive(Clone, Debug)]
pub struct Wavetables {
    /// One cycle of a sine, `SINE_SIZE + 1` samples (`sine[i] == sin(TAU * i / SINE_SIZE)`).
    sine: Vec<f32>,
}

impl Wavetables {
    /// Build the default wavetable bank.
    pub fn new() -> Self {
        let mut sine = Vec::with_capacity(SINE_SIZE + 1);
        for i in 0..=SINE_SIZE {
            let phase = (i as f64) / (SINE_SIZE as f64) * TAU;
            sine.push(math::sin(phase) as f32);
        }
        Wavetables { sine }
    }

    /// One cycle of a sine with a trailing guard sample (`SINE_SIZE + 1` samples).
    pub fn sine(&self) -> &[f32] {
        &self.sine
    }
}

impl Default for Wavetables {
    fn default() -> Self {
        Self::new()
    }
}

/// Linearly interpolate `table` (a one-cycle table with a trailing guard sample) at normalised
/// `phase` in cycles. Only the fractional part of `phase` is used.
#[inline]
pub fn lookup_cycle(table: &[f32], phase: f32) -> f32 {
    let n = table.len() - 1; // last entry is the guard sample
    let frac_phase = phase - math::floor(phase); // wrap into [0, 1)
    let pos = frac_phase * n as f32;
    let i = pos as usize; // 0..=n-1 (frac_phase < 1)
    let frac = pos - i as f32;
    let a = table[i];
    let b = table[i + 1];
    a + frac * (b - a)
}

/// Pack one cycle of plain samples into scsynth's *wavetable format* - the layout the interpolating
/// wavetable oscillators (`Osc`/`COsc`/`VOsc`) read. Each logical sample `s[i]` becomes an `(a, b)`
/// pair `(2·s[i] - s[i+1], s[i+1] - s[i])`, so `N` samples yield `2N` floats and a read at fractional
/// position `i + frac` is the single multiply-add `a + b·(1 + frac)` (see [`lookup_wavetable`]).
///
/// This is exactly scsynth's `add_wpartial` transform (`OscUGens.cpp`), so `/b_gen …  wavetable`
/// produces byte-identical tables. `samples` is treated as periodic: the last pair wraps `s[N-1]` to
/// `s[0]`.
pub fn to_wavetable(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    let mut wt = Vec::with_capacity(2 * n);
    for i in 0..n {
        let cur = samples[i];
        let next = samples[(i + 1) % n];
        wt.push(2.0 * cur - next); // a
        wt.push(next - cur); // b
    }
    wt
}

/// Read the `(a, b)` pair at whole index `i` blended by `frac` (in `[0, 1)`). scsynth stores `b` as a
/// slope and reads it against `1 + frac` (its `PhaseFrac1` bias), which reconstructs the plain linear
/// blend `s[i] + (s[i+1] - s[i])·frac`.
#[inline]
fn interp_pair(wt: &[f32], i: usize, frac: f32) -> f32 {
    wt[2 * i] + wt[2 * i + 1] * (1.0 + frac)
}

/// Interpolate a *wavetable-format* table (`(a, b)` pairs, see [`to_wavetable`]) at normalised `phase`
/// in cycles. `wt` holds `2N` floats for `N` logical samples; only the fractional part of `phase` is
/// used (the table is one periodic cycle). Reconstructs the same linear interpolation as
/// [`lookup_cycle`] but with the pre-differenced coefficients scsynth stores, so the per-sample cost
/// is one multiply-add. An empty table reads `0.0`.
#[inline]
pub fn lookup_wavetable(wt: &[f32], phase: f32) -> f32 {
    let n = wt.len() / 2; // logical samples
    if n == 0 {
        return 0.0;
    }
    let frac_phase = phase - math::floor(phase); // wrap into [0, 1)
    let pos = frac_phase * n as f32;
    let i = (pos as usize).min(n - 1); // 0..=n-1 (frac_phase < 1)
    interp_pair(wt, i, pos - i as f32)
}

/// Waveshape `x` (nominally in `[-1, 1]`) through a *wavetable-format* transfer function `wt` (e.g. a
/// Chebyshev table from `/b_gen cheby … wavetable`), as scsynth's `Shaper` does: `x` maps linearly
/// across the whole `N`-sample table (`x = -1` → start, `x = 0` → middle, `x = +1` → end) and is read
/// with linear interpolation. An empty table reads `0.0`.
#[inline]
pub fn shape_wavetable(wt: &[f32], x: f32) -> f32 {
    let n = wt.len() / 2; // logical samples
    if n == 0 {
        return 0.0;
    }
    let offset = n as f32 * 0.5;
    // Clamp so the whole index stays in `0..=n-1` (scsynth's `fmaxindex = N - 0.001`).
    let findex = (offset * (1.0 + x)).clamp(0.0, n as f32 - 0.001);
    let i = findex as usize;
    interp_pair(wt, i, findex - i as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn wavetable_recovers_samples_at_integer_positions() {
        let samples = [0.0f32, 1.0, 0.0, -1.0];
        let wt = to_wavetable(&samples);
        assert_eq!(wt.len(), 8);
        for (i, &s) in samples.iter().enumerate() {
            let phase = i as f32 / samples.len() as f32;
            assert!(close(lookup_wavetable(&wt, phase), s), "sample {i}");
        }
    }

    #[test]
    fn wavetable_interpolates_and_wraps_like_plain() {
        let samples = [0.0f32, 1.0, 0.0, -1.0];
        let wt = to_wavetable(&samples);
        // Midway between s[0]=0 and s[1]=1 is 0.5.
        assert!(close(lookup_wavetable(&wt, 0.125), 0.5));
        // The last pair wraps s[3]=-1 back to s[0]=0; midway is -0.5.
        assert!(close(lookup_wavetable(&wt, 0.875), -0.5));
        // Phase wraps into [0, 1): 1.125 reads the same as 0.125.
        assert!(close(lookup_wavetable(&wt, 1.125), 0.5));
    }

    #[test]
    fn shape_wavetable_applies_the_transfer_function() {
        // A linear-ramp transfer function f(u) = u: shaping x returns ~x across the interior range
        // (x = 0 reads the table's middle, x = -1 the start, x = +1 the end).
        let n = 64;
        let samples: Vec<f32> = (0..n).map(|i| -1.0 + 2.0 * i as f32 / n as f32).collect();
        let wt = to_wavetable(&samples);
        for &x in &[-0.9f32, -0.5, -0.1, 0.3, 0.7] {
            assert!(
                (shape_wavetable(&wt, x) - x).abs() < 1e-3,
                "shape({x}) = {}",
                shape_wavetable(&wt, x)
            );
        }
        assert!(
            (shape_wavetable(&wt, 0.0) - 0.0).abs() < 1e-3,
            "x=0 is the midpoint"
        );
    }

    #[test]
    fn wavetable_matches_plain_lookup_cycle() {
        // A wavetable read and a guard-sample plain read of the same cycle agree everywhere.
        let mut samples = [0.0f32; 32];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = math::sin(TAU * i as f64 / 32.0) as f32;
        }
        let wt = to_wavetable(&samples);
        let mut plain = samples.to_vec();
        plain.push(samples[0]); // guard sample for lookup_cycle
        for k in 0..200 {
            let phase = k as f32 / 200.0;
            assert!(close(
                lookup_wavetable(&wt, phase),
                lookup_cycle(&plain, phase)
            ));
        }
    }
}
