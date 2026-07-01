//! SuperCollider's scalar math-operator kernels (`f32`), shared across UGens.
//!
//! These mirror scsynth's `SC_InlineUnaryOp.h` / `SC_InlineBinaryOp.h` and the operator calc
//! functions in `BinaryOpUGens.cpp` / `UnaryOpUGens.cpp` exactly, so a `BinaryOpUGen` /
//! `UnaryOpUGen` (and the later `Clip`/`Wrap`/`Fold`/`LinExp`/... units that share these bounds
//! operations) produce sample-identical results to scsynth. They operate on `f32` because that is
//! the rate scsynth evaluates its audio operator path at.
//!
//! Kernels that depend on the graph RNG (`rand`/`rand2`/`linrand`/`coin`/`rrand`/`exprand`) are
//! intentionally absent: those ride the dedicated noise/rand UGens' per-unit RNG seam, not the
//! stateless operator shells.

use crate::math;

/// `sqrt(2) - 1`, the diagonal-distance coefficient used by [`hypotx`].
const SQRT2_M1: f32 = core::f32::consts::SQRT_2 - 1.0;

// ---------------------------------------------------------------------------------------------
// Unary kernels.
// ---------------------------------------------------------------------------------------------

/// Logical not: `1` when `x <= 0`, else `0` (scsynth `sc_not`).
pub fn not(x: f32) -> f32 {
    if x > 0.0 { 0.0 } else { 1.0 }
}

/// Ones' complement of `x` truncated to an integer (scsynth `sc_bitNot`).
pub fn bit_not(x: f32) -> f32 {
    !(x as i32) as f32
}

/// `-1`/`+1`/`0` for negative/positive/zero `x` (scsynth `sc_sign`).
pub fn sign(x: f32) -> f32 {
    if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Signed square root: `sqrt(x)` extended so that `sqrt(x) == -sqrt(-x)` for `x < 0`
/// (scsynth `sc_sqrt`).
pub fn signed_sqrt(x: f32) -> f32 {
    if x < 0.0 {
        -math::sqrt(-x)
    } else {
        math::sqrt(x)
    }
}

/// Convert a MIDI note number to frequency in Hz (scsynth `sc_midicps`).
pub fn midicps(note: f32) -> f32 {
    440.0 * math::powf(2.0, (note - 69.0) / 12.0)
}

/// Convert a frequency in Hz to a MIDI note number (scsynth `sc_cpsmidi`).
pub fn cpsmidi(freq: f32) -> f32 {
    math::log2(freq / 440.0) * 12.0 + 69.0
}

/// Convert an interval in MIDI notes to a frequency ratio (scsynth `sc_midiratio`).
pub fn midiratio(midi: f32) -> f32 {
    math::powf(2.0, midi / 12.0)
}

/// Convert a frequency ratio to an interval in MIDI notes (scsynth `sc_ratiomidi`).
pub fn ratiomidi(ratio: f32) -> f32 {
    12.0 * math::log2(ratio)
}

/// Convert decimal octaves to frequency in Hz (scsynth `sc_octcps`).
pub fn octcps(note: f32) -> f32 {
    440.0 * math::powf(2.0, note - 4.75)
}

/// Convert frequency in Hz to decimal octaves (scsynth `sc_cpsoct`).
pub fn cpsoct(freq: f32) -> f32 {
    math::log2(freq / 440.0) + 4.75
}

/// Convert linear amplitude to decibels (scsynth `sc_ampdb`).
pub fn ampdb(amp: f32) -> f32 {
    math::log10(amp) * 20.0
}

/// Convert decibels to linear amplitude (scsynth `sc_dbamp`).
pub fn dbamp(db: f32) -> f32 {
    math::powf(10.0, db * 0.05)
}

/// A nonlinear soft distortion `x / (1 + |x|)` (scsynth `sc_distort`).
pub fn distort(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// Distortion with a perfectly linear region from `-0.5` to `+0.5` (scsynth `sc_softclip`).
pub fn softclip(x: f32) -> f32 {
    let absx = x.abs();
    if absx <= 0.5 { x } else { (absx - 0.25) / x }
}

/// A rectangular window value: `1` for `0 <= x <= 1`, else `0` (scsynth `sc_rectwindow`).
pub fn rect_window(x: f32) -> f32 {
    if !(0.0..=1.0).contains(&x) { 0.0 } else { 1.0 }
}

/// A Hann window value over `0..=1`, else `0` (scsynth `sc_hanwindow`).
pub fn han_window(x: f32) -> f32 {
    if !(0.0..=1.0).contains(&x) {
        0.0
    } else {
        0.5 - 0.5 * math::cos(x * core::f32::consts::TAU)
    }
}

/// A Welch window value over `0..=1`, else `0` (scsynth `sc_welwindow`).
pub fn wel_window(x: f32) -> f32 {
    if !(0.0..=1.0).contains(&x) {
        0.0
    } else {
        math::sin(x * core::f32::consts::PI)
    }
}

/// A triangle window value over `0..=1`, else `0` (scsynth `sc_triwindow`).
pub fn tri_window(x: f32) -> f32 {
    if !(0.0..=1.0).contains(&x) {
        0.0
    } else if x < 0.5 {
        2.0 * x
    } else {
        -2.0 * x + 2.0
    }
}

/// Clamp `x` to a ramp on `0..=1` (scsynth `sc_ramp`).
pub fn ramp(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Map `x` onto an S-curve, clamped to `0..=1` (scsynth `sc_scurve`).
pub fn scurve(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * (3.0 - 2.0 * x)
    }
}

// ---------------------------------------------------------------------------------------------
// Binary kernels.
// ---------------------------------------------------------------------------------------------

/// Floored modulo with scsynth's fast paths and `b == 0 -> 0` guard (scsynth `sc_mod`).
pub fn modulo(mut x: f32, hi: f32) -> f32 {
    let lo = 0.0;
    if x >= hi {
        x -= hi;
        if x < hi {
            return x;
        }
    } else if x < lo {
        x += hi;
        if x >= lo {
            return x;
        }
    } else {
        return x;
    }
    if hi == lo {
        return lo;
    }
    x - hi * math::floor(x / hi)
}

/// Wrap `x` into `[lo, hi)` (scsynth `sc_wrap`).
pub fn wrap(mut x: f32, lo: f32, hi: f32) -> f32 {
    let range;
    if x >= hi {
        range = hi - lo;
        x -= range;
        if x < hi {
            return x;
        }
    } else if x < lo {
        range = hi - lo;
        x += range;
        if x >= lo {
            return x;
        }
    } else {
        return x;
    }
    if hi == lo {
        return lo;
    }
    x - range * math::floor((x - lo) / range)
}

/// Fold `x` into `[lo, hi]` (scsynth `sc_fold`).
pub fn fold(mut x: f32, lo: f32, hi: f32) -> f32 {
    let x0 = x - lo;
    if x >= hi {
        x = hi + hi - x;
        if x >= lo {
            return x;
        }
    } else if x < lo {
        x = lo + lo - x;
        if x < hi {
            return x;
        }
    } else {
        return x;
    }
    if hi == lo {
        return lo;
    }
    let range = hi - lo;
    let range2 = range + range;
    let mut c = x0 - range2 * math::floor(x0 / range2);
    if c >= range {
        c = range2 - c;
    }
    c + lo
}

/// Wrap integer `x` into `[lo, hi]` *inclusive* (scsynth's integer `sc_wrap`). A degenerate
/// `hi < lo` collapses to `lo`.
pub fn iwrap(x: i32, lo: i32, hi: i32) -> i32 {
    if hi < lo {
        return lo;
    }
    let range = hi - lo + 1;
    lo + (x - lo).rem_euclid(range)
}

/// Fold integer `x` into `[lo, hi]` *inclusive* with a triangle wave (scsynth's integer `sc_fold`). A
/// degenerate `hi <= lo` collapses to `lo`.
pub fn ifold(x: i32, lo: i32, hi: i32) -> i32 {
    if hi <= lo {
        return lo;
    }
    let b = hi - lo;
    let two_b = b + b;
    let mut c = (x - lo).rem_euclid(two_b);
    if c > b {
        c = two_b - c;
    }
    c + lo
}

/// Clamp `x` to `[lo, hi]` (scsynth `sc_clip`). Unlike [`f32::clamp`] this never panics when
/// `lo > hi`, matching scsynth's `max(min(x, hi), lo)`.
pub fn clip(x: f32, lo: f32, hi: f32) -> f32 {
    x.min(hi).max(lo)
}

/// Round `x` to the nearest multiple of `quant` (scsynth `sc_round`).
pub fn round(x: f32, quant: f32) -> f32 {
    if quant == 0.0 {
        x
    } else {
        math::floor(x / quant + 0.5) * quant
    }
}

/// Round `x` up to a multiple of `quant` (scsynth `sc_roundUp`).
pub fn round_up(x: f32, quant: f32) -> f32 {
    if quant == 0.0 {
        x
    } else {
        math::ceil(x / quant) * quant
    }
}

/// Truncate `x` down to a multiple of `quant` (scsynth `sc_trunc`).
pub fn trunc(x: f32, quant: f32) -> f32 {
    if quant == 0.0 {
        x
    } else {
        math::floor(x / quant) * quant
    }
}

/// Signed exponentiation `a^b`, extended so `pow(a, b) == -pow(-a, b)` for `a < 0`
/// (scsynth `sc_pow`).
pub fn pow(a: f32, b: f32) -> f32 {
    if a >= 0.0 {
        math::powf(a, b)
    } else {
        -math::powf(-a, b)
    }
}

/// The "taxicab" hypotenuse `|x| + |y| - (sqrt(2) - 1) * min(|x|, |y|)` (scsynth `sc_hypotx`).
pub fn hypotx(x: f32, y: f32) -> f32 {
    let x = x.abs();
    let y = y.abs();
    x + y - SQRT2_M1 * x.min(y)
}

/// Greatest common divisor of the truncated integer parts (scsynth `sc_gcd`).
pub fn gcd(u: f32, v: f32) -> f32 {
    gcd_i64(math::trunc(u) as i64, math::trunc(v) as i64) as f32
}

/// Least common multiple of the truncated integer parts (scsynth `sc_lcm`).
pub fn lcm(u: f32, v: f32) -> f32 {
    let a = math::trunc(u) as i64;
    let b = math::trunc(v) as i64;
    if a == 0 || b == 0 {
        0.0
    } else {
        ((a * b) / gcd_i64(a, b)) as f32
    }
}

/// Integer GCD following scsynth's sign convention (negative only when both inputs are `<= 0`).
fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let negative = a <= 0 && b <= 0;
    a = a.abs();
    b = b.abs();
    if a == 1 || b == 1 {
        return if negative { -1 } else { 1 };
    }
    if a < b {
        core::mem::swap(&mut a, &mut b);
    }
    while b > 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    if negative { -a } else { a }
}

/// Bitwise AND of the truncated integer parts (scsynth `sc_andt`).
pub fn bit_and(a: f32, b: f32) -> f32 {
    ((a as i32) & (b as i32)) as f32
}

/// Bitwise OR of the truncated integer parts (scsynth `sc_ort`).
pub fn bit_or(a: f32, b: f32) -> f32 {
    ((a as i32) | (b as i32)) as f32
}

/// Bitwise XOR of the truncated integer parts (scsynth `sc_xort`).
pub fn bit_xor(a: f32, b: f32) -> f32 {
    ((a as i32) ^ (b as i32)) as f32
}

/// Left shift `a << b` of the truncated integer parts (scsynth `sc_lst`).
pub fn shift_left(a: f32, b: f32) -> f32 {
    (a as i32).wrapping_shl(b as i32 as u32) as f32
}

/// Right shift `a >> b` of the truncated integer parts (scsynth `sc_rst`).
pub fn shift_right(a: f32, b: f32) -> f32 {
    (a as i32).wrapping_shr(b as i32 as u32) as f32
}

/// Ring modulation plus the first source: `a*b + a` (scsynth `sc_ring1`).
pub fn ring1(a: f32, b: f32) -> f32 {
    a * b + a
}

/// Ring modulation plus both sources: `a*b + a + b` (scsynth `sc_ring2`).
pub fn ring2(a: f32, b: f32) -> f32 {
    a * b + a + b
}

/// Ring modulation variant `a*a*b` (scsynth `sc_ring3`).
pub fn ring3(a: f32, b: f32) -> f32 {
    a * a * b
}

/// Ring modulation variant `a*a*b - a*b*b` (scsynth `sc_ring4`).
pub fn ring4(a: f32, b: f32) -> f32 {
    a * a * b - a * b * b
}

/// Difference of squares `a*a - b*b` (scsynth `sc_difsqr`).
pub fn difsqr(a: f32, b: f32) -> f32 {
    a * a - b * b
}

/// Sum of squares `a*a + b*b` (scsynth `sc_sumsqr`).
pub fn sumsqr(a: f32, b: f32) -> f32 {
    a * a + b * b
}

/// Square of the sum `(a + b)^2` (scsynth `sc_sqrsum`).
pub fn sqrsum(a: f32, b: f32) -> f32 {
    let z = a + b;
    z * z
}

/// Square of the difference `(a - b)^2` (scsynth `sc_sqrdif`).
pub fn sqrdif(a: f32, b: f32) -> f32 {
    let z = a - b;
    z * z
}

/// Absolute difference `|a - b|` (scsynth `sc_absdif`).
pub fn absdif(a: f32, b: f32) -> f32 {
    (a - b).abs()
}

/// Thresholding: `0` when `a < b`, else `a` (scsynth `sc_thresh`).
pub fn thresh(a: f32, b: f32) -> f32 {
    if a < b { 0.0 } else { a }
}

/// Two-quadrant multiply: `a * 0.5 * (b + |b|)` - `0` when `b <= 0`, else `a*b`
/// (scsynth `sc_amclip`).
pub fn amclip(a: f32, b: f32) -> f32 {
    a * 0.5 * (b + b.abs())
}

/// Scale the negative part of `a`: `a` when `a >= 0`, else `a*b` (scsynth's `scaleneg` calc).
pub fn scaleneg(a: f32, b: f32) -> f32 {
    if a >= 0.0 { a } else { a * b }
}

/// Bilateral clip of `a` to `±b` (scsynth's `clip2` calc).
pub fn clip2(a: f32, b: f32) -> f32 {
    if a > b {
        b
    } else if a < -b {
        -b
    } else {
        a
    }
}

/// Residual of clipping `a` to `±b` (scsynth's `excess` calc).
pub fn excess(a: f32, b: f32) -> f32 {
    if a > b {
        a - b
    } else if a < -b {
        a + b
    } else {
        0.0
    }
}

/// Bilateral fold of `a` to `±b` (scsynth `sc_fold2`).
pub fn fold2(a: f32, b: f32) -> f32 {
    fold(a, -b, b)
}

/// Bilateral wrap of `a` to `±b` (scsynth `sc_wrap2`).
pub fn wrap2(a: f32, b: f32) -> f32 {
    wrap(a, -b, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts `a` and `b` are within `1e-5` (relative for large magnitudes).
    fn close(a: f32, b: f32) {
        let tol = 1e-5 * b.abs().max(1.0);
        assert!((a - b).abs() <= tol, "expected {b}, got {a}");
    }

    #[test]
    fn pitch_conversions_round_trip() {
        // A4 = MIDI 69 = 440 Hz; an octave is 12 semitones / a 2x ratio.
        close(midicps(69.0), 440.0);
        close(midicps(81.0), 880.0);
        close(cpsmidi(440.0), 69.0);
        close(cpsmidi(880.0), 81.0);
        close(midiratio(12.0), 2.0);
        close(ratiomidi(2.0), 12.0);
        close(octcps(cpsoct(440.0)), 440.0);
        // -6 dB roughly halves power: dbamp/ampdb invert each other.
        close(dbamp(ampdb(0.25)), 0.25);
        close(ampdb(1.0), 0.0);
    }

    #[test]
    fn signed_sqrt_is_odd() {
        close(signed_sqrt(4.0), 2.0);
        close(signed_sqrt(-4.0), -2.0);
        close(signed_sqrt(0.0), 0.0);
    }

    #[test]
    fn modulo_matches_floored_semantics() {
        close(modulo(5.0, 3.0), 2.0);
        close(modulo(-1.0, 3.0), 2.0); // floored: result takes the sign of the divisor
        close(modulo(3.0, 3.0), 0.0);
        assert_eq!(modulo(1.0, 0.0), 0.0); // divisor 0 yields 0, not NaN
    }

    #[test]
    fn wrap_and_fold_stay_in_range() {
        close(wrap(1.2, -1.0, 1.0), -0.8);
        close(wrap(-1.2, -1.0, 1.0), 0.8);
        close(fold(1.2, -1.0, 1.0), 0.8);
        close(fold(-1.2, -1.0, 1.0), -0.8);
        // wrap2/fold2 are the ±b specialisations.
        close(wrap2(1.2, 1.0), -0.8);
        close(fold2(1.2, 1.0), 0.8);
    }

    #[test]
    fn clip_and_clip2_do_not_panic_on_inverted_bounds() {
        close(clip(5.0, 0.0, 1.0), 1.0);
        close(clip(-5.0, 0.0, 1.0), 0.0);
        close(clip2(5.0, 2.0), 2.0);
        close(clip2(-5.0, 2.0), -2.0);
        close(clip2(0.5, 2.0), 0.5);
        // A negative bound must not panic (unlike f32::clamp).
        let _ = clip2(0.0, -1.0);
        let _ = clip(0.0, 1.0, -1.0);
    }

    #[test]
    fn rounding_quantises() {
        close(round(0.7, 0.5), 0.5);
        close(round(0.8, 0.5), 1.0);
        close(round_up(0.1, 0.5), 0.5);
        close(trunc(0.9, 0.5), 0.5);
        // quant 0 is the identity.
        close(round(0.73, 0.0), 0.73);
    }

    #[test]
    fn pow_is_odd_for_negative_base() {
        close(pow(2.0, 3.0), 8.0);
        close(pow(-2.0, 3.0), -8.0); // sc_pow: -pow(2,3)
    }

    #[test]
    fn gcd_lcm_follow_sc_sign_convention() {
        close(gcd(12.0, 8.0), 4.0);
        close(lcm(4.0, 6.0), 12.0);
        close(gcd(-12.0, -8.0), -4.0); // negative only when both inputs <= 0
        close(gcd(-12.0, 8.0), 4.0);
        close(lcm(0.0, 5.0), 0.0);
    }

    #[test]
    fn ring_and_square_combinators() {
        close(ring1(2.0, 3.0), 8.0); // 2*3 + 2
        close(ring2(2.0, 3.0), 11.0); // 2*3 + 2 + 3
        close(ring3(2.0, 3.0), 12.0); // 2*2*3
        close(ring4(2.0, 3.0), -6.0); // 4*3 - 2*9 = 12 - 18
        close(difsqr(3.0, 2.0), 5.0);
        close(sumsqr(3.0, 2.0), 13.0);
        close(sqrsum(3.0, 2.0), 25.0);
        close(sqrdif(3.0, 2.0), 1.0);
        close(absdif(2.0, 5.0), 3.0);
    }

    #[test]
    fn windows_are_zero_outside_unit_interval() {
        assert_eq!(rect_window(-0.1), 0.0);
        assert_eq!(rect_window(1.1), 0.0);
        close(rect_window(0.5), 1.0);
        close(han_window(0.5), 1.0); // peak at the centre
        assert_eq!(han_window(-0.1), 0.0);
        close(wel_window(0.5), 1.0);
        close(tri_window(0.5), 1.0);
        close(tri_window(0.25), 0.5);
    }

    #[test]
    fn shaping_and_bits() {
        close(scurve(0.5), 0.5);
        assert_eq!(scurve(-1.0), 0.0);
        assert_eq!(scurve(2.0), 1.0);
        close(ramp(0.5), 0.5);
        assert_eq!(ramp(-1.0), 0.0);
        close(amclip(2.0, 3.0), 6.0); // b > 0 -> a*b
        close(amclip(2.0, -3.0), 0.0); // b <= 0 -> 0
        close(scaleneg(2.0, 3.0), 2.0); // a >= 0 -> a
        close(scaleneg(-2.0, 3.0), -6.0); // a < 0 -> a*b
        close(bit_and(6.0, 3.0), 2.0);
        close(bit_or(4.0, 1.0), 5.0);
        close(bit_xor(6.0, 3.0), 5.0);
        close(shift_left(1.0, 3.0), 8.0);
        close(shift_right(8.0, 2.0), 2.0);
    }
}
