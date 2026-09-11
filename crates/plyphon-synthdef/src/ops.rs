//! The math API: the [`UGenBuilder`] finalize trait, operator overloading, and named math methods,
//! emitting `BinaryOpUGen`/`UnaryOpUGen` (and `MulAdd` for `mul_add`).
//!
//! Everything goes through [`SynthDefBuilder::add`], so operators multichannel-expand like any
//! Signal, and the result rate is the max rate of the actual operands ([`RateMode::MaxOfInputs`]).

use crate::builder::{Channel, RateMode, Signal, SynthDefBuilder, UGenInput};

pub fn binary_op<'g>(selector: i16, a: UGenInput<'g>, b: UGenInput<'g>) -> Signal<'g> {
    let g: &'g SynthDefBuilder = a
        .builder()
        .or_else(|| b.builder())
        .expect("operator requires at least one signal operand");
    g.add("BinaryOpUGen", RateMode::MaxOfInputs, &[a, b], 1, selector)
}

pub fn unary_op<'g>(selector: i16, a: UGenInput<'g>) -> Signal<'g> {
    let g: &'g SynthDefBuilder = a.builder().expect("operator requires a signal operand");
    g.add("UnaryOpUGen", RateMode::MaxOfInputs, &[a], 1, selector)
}

fn mul_add_node<'g>(input: UGenInput<'g>, mul: UGenInput<'g>, add: UGenInput<'g>) -> Signal<'g> {
    let g: &'g SynthDefBuilder = input.builder().expect("mul_add requires a signal input");
    g.add("MulAdd", RateMode::MaxOfInputs, &[input, mul, add], 1, 0)
}

// --- UGenBuilder + named methods -------------------------------------------------------------------

/// Generate the whole named-math surface from one selector list: the [`UGenBuilder`] trait (whose
/// default methods give every UGen *builder* the full API from a single one-line impl) plus the
/// mirror inherent impls on `Channel` (by value, it's `Copy`) and `Signal` (by `&self`, so call
/// sites don't consume it). Inherent methods shadow the trait defaults, which keeps `sig.abs()`
/// non-consuming on an already-finalized `Signal`.
macro_rules! math_api {
    (
        unary { $($(#[$ud:meta])* $uname:ident => $usel:expr;)* }
        binary { $($(#[$bd:meta])* $bname:ident => $bsel:expr;)* }
    ) => {
        /// The finalize trait for UGen *builders*, which emit nothing into the def until they
        /// are used as an input or `.signal()` is called.
        ///
        /// The default methods give implementors the full named math API (each one finalizes
        /// `self`, then emits the op unit). Already-finalized values (`Channel`, `Signal`) get the
        /// same methods as inherent, non-consuming impls instead of via this trait.
        pub trait UGenBuilder<'g>: Sized + Into<UGenInput<'g>> {
            /// Finalize into the def (emitting the unit(s)) and return the resulting signal. Sinks such as
            /// `Out` produce no signal and so do not implement this trait; they have an inherent
            /// `emit()` terminal instead.
            fn signal(self) -> Signal<'g>;

            /// `self * mul + add`, as a single `MulAdd` unit (per channel).
            fn mul_add(self, mul: impl Into<UGenInput<'g>>, add: impl Into<UGenInput<'g>>) -> Signal<'g> {
                mul_add_node(self.into(), mul.into(), add.into())
            }

            $($(#[$ud])* fn $uname(self) -> Signal<'g> { unary_op($usel, self.into()) })*
            $($(#[$bd])* fn $bname(self, rhs: impl Into<UGenInput<'g>>) -> Signal<'g> {
                binary_op($bsel, self.into(), rhs.into())
            })*
        }

        impl<'g> Channel<'g> {
            /// `self * mul + add`, as a single `MulAdd` unit.
            pub fn mul_add(self, mul: impl Into<UGenInput<'g>>, add: impl Into<UGenInput<'g>>) -> Signal<'g> {
                mul_add_node(self.into(), mul.into(), add.into())
            }

            $($(#[$ud])* pub fn $uname(self) -> Signal<'g> { unary_op($usel, self.into()) })*
            $($(#[$bd])* pub fn $bname(self, rhs: impl Into<UGenInput<'g>>) -> Signal<'g> {
                binary_op($bsel, self.into(), rhs.into())
            })*
        }

        impl<'g> Signal<'g> {
            /// `self * mul + add`, as a single `MulAdd` unit (per channel).
            pub fn mul_add(&self, mul: impl Into<UGenInput<'g>>, add: impl Into<UGenInput<'g>>) -> Signal<'g> {
                mul_add_node(self.into(), mul.into(), add.into())
            }

            $($(#[$ud])* pub fn $uname(&self) -> Signal<'g> { unary_op($usel, self.into()) })*
            $($(#[$bd])* pub fn $bname(&self, rhs: impl Into<UGenInput<'g>>) -> Signal<'g> {
                binary_op($bsel, self.into(), rhs.into())
            })*
        }
    };
}

// The numbers are scsynth `special_index` selectors, with SC's `op*` names from
// `SpecialSelectorsOperatorsAndClasses.h` alongside; the source of truth is the engine's own tables in
// `plyphon-unit/src/unit/binary_op.rs` and `unary_op.rs`.
//
// Every engine op except the four with no use in a def: `asFloat`/`asInt` (identity type
// conversions), `thru` (identity) and `silence` (constant 0). `neg` and `not` are the `-` and `!`
// operators rather than methods (an inherent by-value `not` would trip clippy's
// `should_implement_trait`); `%` is `modulo`.
math_api! {
    unary {
        /// Bitwise not, treating `a` as an integer.
        bit_not => 4; // opBitNot
        /// `|a|`
        abs => 5; // opAbs
        /// Round up.
        ceil => 8; // opCeil
        /// Round down.
        floor => 9; // opFloor
        /// Fractional part: `a - floor(a)`.
        frac => 10; // opFrac
        /// `-1`, `0` or `1`.
        sign => 11; // opSign
        /// `a * a`
        squared => 12; // opSquared
        /// `a * a * a`
        cubed => 13; // opCubed
        /// Signed square root.
        sqrt => 14; // opSqrt
        /// `e^a`
        exp => 15; // opExp
        /// `1 / a`
        recip => 16; // opRecip
        /// MIDI note number to frequency (Hz).
        midi_cps => 17; // opMIDICPS
        /// Frequency (Hz) to MIDI note number.
        cps_midi => 18; // opCPSMIDI
        /// MIDI interval (semitones) to frequency ratio.
        midi_ratio => 19; // opMIDIRatio
        /// Frequency ratio to MIDI interval (semitones).
        ratio_midi => 20; // opRatioMIDI
        /// Decibels to linear amplitude.
        db_amp => 21; // opDbAmp
        /// Linear amplitude to decibels.
        amp_db => 22; // opAmpDb
        /// Decimal octaves to frequency (Hz).
        oct_cps => 23; // opOctCPS
        /// Frequency (Hz) to decimal octaves.
        cps_oct => 24; // opCPSOct
        /// Natural logarithm.
        log => 25; // opLog
        /// Base-2 logarithm.
        log2 => 26; // opLog2
        /// Base-10 logarithm (of `|a|`).
        log10 => 27; // opLog10
        /// Sine.
        sin => 28; // opSin
        /// Cosine.
        cos => 29; // opCos
        /// Tangent.
        tan => 30; // opTan
        /// Arcsine.
        asin => 31; // opArcSin
        /// Arccosine.
        acos => 32; // opArcCos
        /// Arctangent.
        atan => 33; // opArcTan
        /// Hyperbolic sine.
        sinh => 34; // opSinH
        /// Hyperbolic cosine.
        cosh => 35; // opCosH
        /// Hyperbolic tangent.
        tanh => 36; // opTanH
        /// Nonlinear distortion `a / (1 + |a|)`.
        distort => 42; // opDistort
        /// Nonlinear distortion, linear below +/-0.5.
        soft_clip => 43; // opSoftClip
        /// Rectangular window over `0..1`: `1` inside, `0` outside.
        rect_window => 48; // opRectWindow
        /// Hann (raised-cosine) window over `0..1`, `0` outside.
        han_window => 49; // opHanWindow
        /// Welch (parabolic) window over `0..1`, `0` outside.
        welch_window => 50; // opWelchWindow
        /// Triangular window over `0..1`, `0` outside.
        tri_window => 51; // opTriWindow
        /// Linear ramp: `0` below `0`, `1` above `1`, `a` in between.
        ramp => 52; // opRamp
        /// S-curve over `0..1`: `a^2 * (3 - 2a)`, clamped outside.
        s_curve => 53; // opSCurve
    }
    binary {
        /// Integer division: `floor(a / b)` (sclang's `div`; `/` is float division).
        idiv => 3; // opIDiv
        /// Floating-point modulo (`mod` is a keyword); also the `%` operator.
        modulo => 5; // opMod
        /// `1` where `a == b`, else `0`.
        eq => 6; // opEQ
        /// `1` where `a != b`, else `0`.
        ne => 7; // opNE
        /// `1` where `a < b`, else `0`.
        lt => 8; // opLT
        /// `1` where `a > b`, else `0`.
        gt => 9; // opGT
        /// `1` where `a <= b`, else `0`.
        le => 10; // opLE
        /// `1` where `a >= b`, else `0`.
        ge => 11; // opGE
        /// The smaller of the two signals.
        min => 12; // opMin
        /// The larger of the two signals.
        max => 13; // opMax
        /// Bitwise and, treating both as integers.
        bit_and => 14; // opBitAnd
        /// Bitwise or, treating both as integers.
        bit_or => 15; // opBitOr
        /// Bitwise exclusive or, treating both as integers.
        bit_xor => 16; // opBitXor
        /// Least common multiple, treating both as integers.
        lcm => 17; // opLCM
        /// Greatest common divisor, treating both as integers.
        gcd => 18; // opGCD
        /// Round to a multiple of `rhs`.
        round => 19; // opRound
        /// Round up to a multiple of `rhs`.
        round_up => 20; // opRoundUp
        /// Truncate to a multiple of `rhs`.
        trunc => 21; // opTrunc
        /// Arctangent of `self / rhs`.
        atan2 => 22; // opAtan2
        /// `sqrt(a^2 + b^2)`
        hypot => 23; // opHypot
        /// Fast approximation of `hypot` (sclang's `hypotApx`).
        hypot_apx => 24; // opHypotx
        /// `a^b`
        pow => 25; // opPow
        /// Bit shift left, treating both as integers.
        shift_left => 26; // opShiftLeft
        /// Bit shift right, treating both as integers.
        shift_right => 27; // opShiftRight
        /// Ring modulation plus carrier: `a * b + a`.
        ring1 => 30; // opRing1
        /// `a * b + a + b`
        ring2 => 31; // opRing2
        /// `a * a * b`
        ring3 => 32; // opRing3
        /// `a * a * b - a * b * b`
        ring4 => 33; // opRing4
        /// `a^2 - b^2`
        dif_sqr => 34; // opDifSqr
        /// `a^2 + b^2`
        sum_sqr => 35; // opSumSqr
        /// `(a + b)^2`
        sqr_sum => 36; // opSqrSum
        /// `(a - b)^2`
        sqr_dif => 37; // opSqrDif
        /// `|a - b|`
        abs_dif => 38; // opAbsDif
        /// `a` where `a >= b`, else `0`.
        thresh => 39; // opThresh
        /// Two-quadrant multiply: `a * b` where `b > 0`, else `0`.
        am_clip => 40; // opAMClip
        /// `a * b` where `a < 0`, else `a`.
        scale_neg => 41; // opScaleNeg
        /// Bilateral clip: `a` clipped to `+/-b`.
        clip2 => 42; // opClip2
        /// What `clip2` removes: `a - clip2(a, b)`.
        excess => 43; // opExcess
        /// Bilateral fold.
        fold2 => 44; // opFold2
        /// Bilateral wrap.
        wrap2 => 45; // opWrap2
        /// `a`, ignoring `rhs` - but keeping it in the def, so it still runs.
        first_arg => 46; // opFirstArg
    }
}

// NOTE: `Channel`/`Signal` deliberately do NOT implement `UGenBuilder` - they already are finalized,
// and a by-value trait method would win method resolution over the inherent `&self` methods
// (by-value candidates are probed before autoref), making `sig.midi_cps()` consume the handle.

// --- Overload operators in std::ops -------------------------------------------------------------

/// For each binary operator: a blanket-RHS impl per signal-valued LHS (`Channel`, `Signal`, `&Signal`),
/// plus concrete impls for scalar LHS (`f32`/`i32` on the left, e.g. `440.0 - sig`).
/// The scalar-LHS impls satisfy the orphan rule because the local type appears in the trait's
/// arguments with no generic type parameter before it.
macro_rules! impl_binary_operator {
    ($Trait:ident, $method:ident, $selector:expr) => {
        impl<'g, R: Into<UGenInput<'g>>> core::ops::$Trait<R> for Channel<'g> {
            type Output = Signal<'g>;
            fn $method(self, rhs: R) -> Signal<'g> {
                binary_op($selector, self.into(), rhs.into())
            }
        }
        impl<'g, R: Into<UGenInput<'g>>> core::ops::$Trait<R> for Signal<'g> {
            type Output = Signal<'g>;
            fn $method(self, rhs: R) -> Signal<'g> {
                binary_op($selector, self.into(), rhs.into())
            }
        }
        impl<'a, 'g, R: Into<UGenInput<'g>>> core::ops::$Trait<R> for &'a Signal<'g> {
            type Output = Signal<'g>;
            fn $method(self, rhs: R) -> Signal<'g> {
                binary_op($selector, self.into(), rhs.into())
            }
        }
        impl_binary_operator!(@scalar_lhs $Trait, $method, $selector, f32);
        impl_binary_operator!(@scalar_lhs $Trait, $method, $selector, i32);
    };
    (@scalar_lhs $Trait:ident, $method:ident, $selector:expr, $scalar:ty) => {
        impl<'g> core::ops::$Trait<Channel<'g>> for $scalar {
            type Output = Signal<'g>;
            fn $method(self, rhs: Channel<'g>) -> Signal<'g> {
                binary_op($selector, self.into(), rhs.into())
            }
        }
        impl<'g> core::ops::$Trait<Signal<'g>> for $scalar {
            type Output = Signal<'g>;
            fn $method(self, rhs: Signal<'g>) -> Signal<'g> {
                binary_op($selector, self.into(), rhs.into())
            }
        }
        impl<'a, 'g> core::ops::$Trait<&'a Signal<'g>> for $scalar {
            type Output = Signal<'g>;
            fn $method(self, rhs: &'a Signal<'g>) -> Signal<'g> {
                binary_op($selector, self.into(), rhs.into())
            }
        }
    };
}

impl_binary_operator!(Add, add, 0); // opAdd
impl_binary_operator!(Sub, sub, 1); // opSub
impl_binary_operator!(Mul, mul, 2); // opMul
impl_binary_operator!(Div, div, 4); // opFDiv
impl_binary_operator!(Rem, rem, 5); // opMod

/// The unary operators: `-` (opNeg) and `!` (opNot: `1` where `a == 0`, else `0`), on `Channel`,
/// `Signal` and `&Signal`.
macro_rules! impl_unary_operator {
    ($Trait:ident, $method:ident, $selector:expr) => {
        impl<'g> core::ops::$Trait for Channel<'g> {
            type Output = Signal<'g>;
            fn $method(self) -> Signal<'g> {
                unary_op($selector, self.into())
            }
        }
        impl<'g> core::ops::$Trait for Signal<'g> {
            type Output = Signal<'g>;
            fn $method(self) -> Signal<'g> {
                unary_op($selector, self.into())
            }
        }
        impl<'g> core::ops::$Trait for &Signal<'g> {
            type Output = Signal<'g>;
            fn $method(self) -> Signal<'g> {
                unary_op($selector, self.into())
            }
        }
    };
}

impl_unary_operator!(Neg, neg, 0); // opNeg
impl_unary_operator!(Not, not, 1); // opNot

/// Stamp the `std::ops` operators (with blanket, `Into<UGenInput>` RHS) onto a UGen builder type.
/// Scalar-on-left (`0.5 * builder`) is not provided - the orphan rule forbids a blanket there, so
/// write `builder * 0.5` or finalize with `.signal()` first.
///
/// Exported so downstream crates can give their custom UGen builders the same operators (the
/// [`ugen!`](crate::ugen) macro invokes it automatically).
#[macro_export]
macro_rules! impl_builder_ops {
    ($Builder:ident) => {
        $crate::impl_builder_ops!(@op $Builder, Add, add, 0); // opAdd
        $crate::impl_builder_ops!(@op $Builder, Sub, sub, 1); // opSub
        $crate::impl_builder_ops!(@op $Builder, Mul, mul, 2); // opMul
        $crate::impl_builder_ops!(@op $Builder, Div, div, 4); // opFDiv
        $crate::impl_builder_ops!(@op $Builder, Rem, rem, 5); // opMod
        $crate::impl_builder_ops!(@unop $Builder, Neg, neg, 0); // opNeg
        $crate::impl_builder_ops!(@unop $Builder, Not, not, 1); // opNot
    };
    (@op $Builder:ident, $Trait:ident, $method:ident, $selector:expr) => {
        impl<'g, R: Into<$crate::UGenInput<'g>>> core::ops::$Trait<R> for $Builder<'g> {
            type Output = $crate::Signal<'g>;
            fn $method(self, rhs: R) -> $crate::Signal<'g> {
                $crate::__private::binary_op($selector, self.into(), rhs.into())
            }
        }
    };
    (@unop $Builder:ident, $Trait:ident, $method:ident, $selector:expr) => {
        impl<'g> core::ops::$Trait for $Builder<'g> {
            type Output = $crate::Signal<'g>;
            fn $method(self) -> $crate::Signal<'g> {
                $crate::__private::unary_op($selector, self.into())
            }
        }
    };
}
