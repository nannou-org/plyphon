//! Transcendental float math that works with or without `std`.
//!
//! On stable Rust the `f32`/`f64` transcendental methods (`sin`, `exp`, `powf`, ...) and even
//! `sqrt`/`floor`/`ceil` are provided by `std`; `core` gates them behind the unstable
//! `core_float_math` feature, so a `no_std` build cannot write `x.sin()`. These free fns route to
//! the inherent methods under the `std` feature and to [`libm`] without it, so DSP code reads the
//! same in both builds - `math::sin(x)` in place of `x.sin()`.
//!
//! The simpler float ops the engine uses (`abs`, `min`, `max`, `clamp`, `recip`, `signum`) are
//! already stable in `core`, so those stay as plain method calls and are not routed through here.

/// A float width the engine evaluates transcendental functions on (`f32` or `f64`).
///
/// Sealed: DSP code calls the free fns in this module rather than these methods, and no type
/// outside the crate can implement it. It exists only so each free fn serves both widths.
pub trait Real: sealed::Sealed + Copy {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn tanh(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn rem_euclid(self, rhs: Self) -> Self;
}

/// Sine of `x` (radians).
pub fn sin<F: Real>(x: F) -> F {
    x.sin()
}

/// Cosine of `x` (radians).
pub fn cos<F: Real>(x: F) -> F {
    x.cos()
}

/// Tangent of `x` (radians).
pub fn tan<F: Real>(x: F) -> F {
    x.tan()
}

/// Hyperbolic tangent of `x`.
pub fn tanh<F: Real>(x: F) -> F {
    x.tanh()
}

/// `e` raised to the power `x`.
pub fn exp<F: Real>(x: F) -> F {
    x.exp()
}

/// Natural logarithm of `x`.
pub fn ln<F: Real>(x: F) -> F {
    x.ln()
}

/// Square root of `x`.
pub fn sqrt<F: Real>(x: F) -> F {
    x.sqrt()
}

/// Largest integer not greater than `x`.
pub fn floor<F: Real>(x: F) -> F {
    x.floor()
}

/// Smallest integer not less than `x`.
pub fn ceil<F: Real>(x: F) -> F {
    x.ceil()
}

/// `x` raised to the power `n`.
pub fn powf<F: Real>(x: F, n: F) -> F {
    x.powf(n)
}

/// Least nonnegative remainder of `x` modulo `rhs`.
pub fn rem_euclid<F: Real>(x: F, rhs: F) -> F {
    x.rem_euclid(rhs)
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

// Under `std` each method forwards to the inherent `f32`/`f64` method (method-call syntax resolves
// to the inherent method, which outranks the trait method, so there is no recursion).
#[cfg(feature = "std")]
mod imp {
    use super::Real;

    impl Real for f32 {
        fn sin(self) -> Self {
            self.sin()
        }
        fn cos(self) -> Self {
            self.cos()
        }
        fn tan(self) -> Self {
            self.tan()
        }
        fn tanh(self) -> Self {
            self.tanh()
        }
        fn exp(self) -> Self {
            self.exp()
        }
        fn ln(self) -> Self {
            self.ln()
        }
        fn sqrt(self) -> Self {
            self.sqrt()
        }
        fn floor(self) -> Self {
            self.floor()
        }
        fn ceil(self) -> Self {
            self.ceil()
        }
        fn powf(self, n: Self) -> Self {
            self.powf(n)
        }
        fn rem_euclid(self, rhs: Self) -> Self {
            self.rem_euclid(rhs)
        }
    }

    impl Real for f64 {
        fn sin(self) -> Self {
            self.sin()
        }
        fn cos(self) -> Self {
            self.cos()
        }
        fn tan(self) -> Self {
            self.tan()
        }
        fn tanh(self) -> Self {
            self.tanh()
        }
        fn exp(self) -> Self {
            self.exp()
        }
        fn ln(self) -> Self {
            self.ln()
        }
        fn sqrt(self) -> Self {
            self.sqrt()
        }
        fn floor(self) -> Self {
            self.floor()
        }
        fn ceil(self) -> Self {
            self.ceil()
        }
        fn powf(self, n: Self) -> Self {
            self.powf(n)
        }
        fn rem_euclid(self, rhs: Self) -> Self {
            self.rem_euclid(rhs)
        }
    }
}

// Without `std` the inherent methods do not exist, so route to `libm` (`*f` suffix for `f32`).
#[cfg(not(feature = "std"))]
mod imp {
    use super::Real;

    impl Real for f32 {
        fn sin(self) -> Self {
            libm::sinf(self)
        }
        fn cos(self) -> Self {
            libm::cosf(self)
        }
        fn tan(self) -> Self {
            libm::tanf(self)
        }
        fn tanh(self) -> Self {
            libm::tanhf(self)
        }
        fn exp(self) -> Self {
            libm::expf(self)
        }
        fn ln(self) -> Self {
            libm::logf(self)
        }
        fn sqrt(self) -> Self {
            libm::sqrtf(self)
        }
        fn floor(self) -> Self {
            libm::floorf(self)
        }
        fn ceil(self) -> Self {
            libm::ceilf(self)
        }
        fn powf(self, n: Self) -> Self {
            libm::powf(self, n)
        }
        fn rem_euclid(self, rhs: Self) -> Self {
            let r = self % rhs;
            if r < 0.0 { r + rhs.abs() } else { r }
        }
    }

    impl Real for f64 {
        fn sin(self) -> Self {
            libm::sin(self)
        }
        fn cos(self) -> Self {
            libm::cos(self)
        }
        fn tan(self) -> Self {
            libm::tan(self)
        }
        fn tanh(self) -> Self {
            libm::tanh(self)
        }
        fn exp(self) -> Self {
            libm::exp(self)
        }
        fn ln(self) -> Self {
            libm::log(self)
        }
        fn sqrt(self) -> Self {
            libm::sqrt(self)
        }
        fn floor(self) -> Self {
            libm::floor(self)
        }
        fn ceil(self) -> Self {
            libm::ceil(self)
        }
        fn powf(self, n: Self) -> Self {
            libm::pow(self, n)
        }
        fn rem_euclid(self, rhs: Self) -> Self {
            let r = self % rhs;
            if r < 0.0 { r + rhs.abs() } else { r }
        }
    }
}
