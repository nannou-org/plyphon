//! Sub-sample interpolation kernels - plyphon's ports of scsynth's interpolation helpers.
//!
//! These reconstruct a continuous signal between discrete samples. [`lininterp`] is the 2-point
//! linear blend and [`cubicinterp`] is scsynth's 4-point cubic (both from `SC_SndBuf.h`), used by the
//! linear/cubic-interpolating delays, buffer readers, oscillators, and by `LFDNoise1`/`LFDNoise3`.

/// scsynth's `lininterp`: the linear blend `a + x * (b - a)`, evaluated at fractional position `x` in
/// `[0, 1]` between `a` (at `x = 0`) and `b` (at `x = 1`).
pub fn lininterp(x: f32, a: f32, b: f32) -> f32 {
    a + x * (b - a)
}

/// scsynth's `cubicinterp`: a 4-point cubic through `y0..y3`, evaluated at fractional position `x`
/// in `[0, 1]` between the middle two points `y1` and `y2` (so `x = 0` gives `y1`, `x = 1` gives
/// `y2`). Matches the Catmull-Rom coefficients in scsynth exactly.
pub fn cubicinterp(x: f32, y0: f32, y1: f32, y2: f32, y3: f32) -> f32 {
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * x + c2) * x + c1) * x + c0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lininterp_blends_the_endpoints() {
        assert!((lininterp(0.0, 2.0, 5.0) - 2.0).abs() < 1e-6);
        assert!((lininterp(1.0, 2.0, 5.0) - 5.0).abs() < 1e-6);
        assert!((lininterp(0.5, 2.0, 5.0) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn cubicinterp_hits_the_endpoints() {
        // At x = 0 the cubic passes through y1, at x = 1 through y2 (regardless of the outer points).
        assert!((cubicinterp(0.0, -1.0, 2.0, 5.0, -3.0) - 2.0).abs() < 1e-6);
        assert!((cubicinterp(1.0, -1.0, 2.0, 5.0, -3.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cubicinterp_is_linear_on_a_ramp() {
        // Four collinear points make the cubic reduce to the straight line through them, so the
        // midpoint reads the average of y1 and y2.
        let y = |t: f32| 3.0 * t + 1.0;
        let (y0, y1, y2, y3) = (y(0.0), y(1.0), y(2.0), y(3.0));
        assert!((cubicinterp(0.5, y0, y1, y2, y3) - y(1.5)).abs() < 1e-5);
    }
}
