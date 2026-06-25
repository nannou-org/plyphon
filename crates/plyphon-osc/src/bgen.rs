//! Buffer generators for `/b_gen` (scsynth's `sine1`/`sine2`/`sine3`/`cheby`), plus the `normalize`
//! flag. Each is a pure function that fills a caller-provided sample slice (the dispatcher allocates a
//! fresh buffer of the target's frame count and installs it via the `/b_alloc` swap path), so there
//! is no engine round-trip and no I/O. `no_std`-safe: trig comes from [`plyphon_dsp::math`].
//!
//! The `copy` generator is *not* here - it reads a live RT-owned buffer, so it is an engine command.

use core::f32::consts::TAU;

use plyphon_dsp::math::sin;

/// `sine1`: `amps[k]` is the amplitude of the `(k+1)`-th harmonic over one period spanning `out`.
pub fn sine1(out: &mut [f32], amps: &[f32]) {
    fill(out, |phase| {
        amps.iter()
            .enumerate()
            .map(|(k, &amp)| amp * sin((k as f32 + 1.0) * phase))
            .sum()
    });
}

/// `sine2`: `(freq, amp)` pairs - `freq` in cycles over the table.
pub fn sine2(out: &mut [f32], pairs: &[(f32, f32)]) {
    fill(out, |phase| {
        pairs
            .iter()
            .map(|&(freq, amp)| amp * sin(freq * phase))
            .sum()
    });
}

/// `sine3`: `(freq, amp, phase)` triples - `freq` in cycles over the table, `phase` in radians.
pub fn sine3(out: &mut [f32], triples: &[(f32, f32, f32)]) {
    fill(out, |phase| {
        triples
            .iter()
            .map(|&(freq, amp, ph)| amp * sin(freq * phase + ph))
            .sum()
    });
}

/// `cheby`: a waveshaper transfer function `sum_k coeffs[k] * T_{k+1}(x)` over `x` in `[-1, 1]`
/// mapped across the table (`coeffs[0]` multiplies `T1`). Evaluated by the Chebyshev recurrence.
pub fn cheby(out: &mut [f32], coeffs: &[f32]) {
    let n = out.len();
    for (i, sample) in out.iter_mut().enumerate() {
        let x = if n <= 1 {
            -1.0
        } else {
            2.0 * i as f32 / (n as f32 - 1.0) - 1.0
        };
        let (mut t_prev, mut t_cur) = (1.0f32, x); // T0, T1
        let mut acc = 0.0;
        for &c in coeffs {
            acc += c * t_cur;
            let t_next = 2.0 * x * t_cur - t_prev;
            t_prev = t_cur;
            t_cur = t_next;
        }
        *sample = acc;
    }
}

/// Scale `out` so its peak magnitude is exactly 1.0 (scsynth's `normalize` flag). No-op if silent.
pub fn normalize(out: &mut [f32]) {
    let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if peak > 0.0 {
        let scale = 1.0 / peak;
        for v in out.iter_mut() {
            *v *= scale;
        }
    }
}

/// Fill `out` from `f`, called with the base phase `2*pi*i/N` of each sample.
fn fill(out: &mut [f32], f: impl Fn(f32) -> f32) {
    let n = out.len();
    if n == 0 {
        return;
    }
    for (i, sample) in out.iter_mut().enumerate() {
        *sample = f(TAU * i as f32 / n as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn sine1_single_partial_is_a_pure_sine() {
        let mut out = [0.0f32; 16];
        sine1(&mut out, &[1.0]);
        for (i, &s) in out.iter().enumerate() {
            assert!(close(s, (TAU * i as f32 / 16.0).sin()));
        }
    }

    #[test]
    fn sine1_second_partial() {
        let mut out = [0.0f32; 16];
        sine1(&mut out, &[0.0, 1.0]);
        for (i, &s) in out.iter().enumerate() {
            assert!(close(s, (TAU * 2.0 * i as f32 / 16.0).sin()));
        }
    }

    #[test]
    fn normalize_makes_peak_one() {
        let mut out = [0.0, 0.25, -0.5, 0.1];
        normalize(&mut out);
        let peak = out.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(close(peak, 1.0));
        // The shape is preserved (the old peak -0.5 maps to -1.0).
        assert!(close(out[2], -1.0));
    }

    #[test]
    fn cheby_t1_is_identity_ramp() {
        let mut out = [0.0f32; 8];
        cheby(&mut out, &[1.0]); // T1(x) = x over [-1, 1]
        for (i, &s) in out.iter().enumerate() {
            let x = 2.0 * i as f32 / 7.0 - 1.0;
            assert!(close(s, x));
        }
    }

    #[test]
    fn cheby_t2_is_2x2_minus_1() {
        let mut out = [0.0f32; 8];
        cheby(&mut out, &[0.0, 1.0]); // T2(x) = 2x^2 - 1
        for (i, &s) in out.iter().enumerate() {
            let x = 2.0 * i as f32 / 7.0 - 1.0;
            assert!(close(s, 2.0 * x * x - 1.0));
        }
    }
}
