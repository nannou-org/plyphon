//! Demand-rate `BinaryOpUGen`/`UnaryOpUGen` - plyphon's port of scsynth's `*_d` operator kernels
//! (`BinaryOpUGens.cpp`, `UnaryOpUGens.cpp`).
//!
//! An operator UGen may be compiled at demand rate, in which case it becomes a demand *source*: it
//! pulls its operands and yields the operator applied to them, so `Dseq([1, 2]) + Dseries(10, 10)`
//! is itself a sequence. The operator kernels are the very same tables the audio- and control-rate
//! operators use (`binary_op`/`unary_op`), wrapped in the pull protocol below, so an operator
//! behaves identically at every rate and a corpus that starts using a new operator index needs no
//! change here.
//!
//! Every operator the reference's demand switch lists (`ChooseDemandFunc` in each file) has the
//! same formula as its calc-rate kernel, so the shared tables serve both. An index the switch does
//! not list falls through as the reference's does: to `add` for binary, to pass-through for unary.
//!
//! The random operators (binary `rrand`/`exprand`, unary `rand`, `rand2`, `linrand`, `bilinrand`,
//! `sum3rand` and `coin`) draw from the synth's random stream, which a demand unit cannot reach
//! here, so they are rejected at build rather than silently losing their randomness.
//!
//! An audio-rate operand reads the block's *first* sample, where the reference reads the sample its
//! pull is offset to. A demand pull carries no sample offset in plyphon, so this is the engine-wide
//! behaviour of every demand-rate operand read, not something these two units decide.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::binary_op::binary_op;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use crate::unit::unary_op::unary_op;

/// The first operand input, shared by both operators.
const A: usize = 0;
/// The binary operator's second operand input.
const B: usize = 1;

/// The demand-rate kernel for binary operator `index`: the shared calc-rate kernel, or `add` for an
/// index the reference's demand switch does not list (its `default: func = &add_d`).
fn binary_kernel(index: i16) -> fn(f32, f32) -> f32 {
    binary_op(index).unwrap_or(|a, b| a + b)
}

/// The demand-rate kernel for unary operator `index`: the shared calc-rate kernel, or pass-through
/// for an index the reference's demand switch does not list (its `default: func = &thru_d`).
fn unary_kernel(index: i16) -> fn(f32) -> f32 {
    unary_op(index).unwrap_or(|a| a)
}

/// `BinaryOpUGen.dr(a, b)`: `a <op> b` over two pulled operands, with `<op>` selected by the
/// SynthDef's `special_index`.
///
/// Both operands are pulled on every produce - there is no short-circuit, so a sequence feeding one
/// side stays in step with the other even when its sibling is exhausted - and the result is
/// [`f32::NAN`] (exhausted) if either operand is `NaN`. A reset propagates to both operands.
///
/// The constructor does not prime: the unit's first pull is its first operand pull (the reference's
/// binary constructor writes `0` into its output and pulls nothing). This is asymmetric with
/// [`DemandUnaryOp`], and the asymmetry is the reference's.
///
/// `NaN` never reaches the operator: the protocol above handles it first.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DemandBinaryOp {
    /// The SuperCollider binary operator index, resolved to a kernel on each produce.
    op: u32,
}

impl DemandUnit for DemandBinaryOp {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        ctx.reset(A);
        ctx.reset(B);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let a = ctx.demand(A);
        let b = ctx.demand(B);
        if a.is_nan() || b.is_nan() {
            return f32::NAN;
        }
        binary_kernel(self.op as i16)(a, b)
    }
}

/// Constructor for [`DemandBinaryOp`].
pub struct DemandBinaryOpCtor;

impl DemandUnitDef for DemandBinaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() != 2 {
            return Err(BuildError::WrongInputCount);
        }
        // `rrand` and `exprand` need the synth's random stream.
        if matches!(ctx.special_index, 47 | 48) {
            return Err(BuildError::UnsupportedOp(ctx.special_index));
        }
        Ok(demand_unit_spec(DemandBinaryOp {
            op: ctx.special_index as u32,
        }))
    }
}

/// `UnaryOpUGen.dr(a)`: `<op>(a)` over one pulled operand, with `<op>` selected by the SynthDef's
/// `special_index`.
///
/// The result is [`f32::NAN`] (exhausted) when the operand is `NaN`; a reset propagates to the
/// operand.
///
/// The kernels come from the shared table, so a transcendental operator (`sin` and friends) is
/// evaluated by the same float math the calc-rate operators use, and carries the same last-ULP
/// differences from the reference's own math library.
///
/// The reference's constructor runs the unit's calc function once, which consumes one element of
/// the operand before any consumer pulls. plyphon compiles off the audio thread, where there is no
/// operand to pull and no `init` hook on a demand unit, so the first produce performs that
/// constructor pull and then the pull it returns - two pulls, one value. A reset does not re-arm
/// the prime: the reference's constructor runs once per instantiation. Deferring the prime is
/// observable only when two unary operators share one source - the reference consumes both prime
/// elements at construction, this port one per operator's first pull.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DemandUnaryOp {
    /// The SuperCollider unary operator index, resolved to a kernel on each produce.
    op: u32,
    /// `0` until the first produce has performed the constructor's operand pull.
    primed: u32,
}

impl DemandUnit for DemandUnaryOp {
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        // A reset rewinds the source, which also undoes the reference's construction-time
        // prime pull - so the prime is spent here too, and the next produce reads the
        // source's first element rather than its second.
        self.primed = 1;
        ctx.reset(A);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        if self.primed == 0 {
            self.primed = 1;
            let _ = ctx.demand(A);
        }
        let a = ctx.demand(A);
        if a.is_nan() {
            return f32::NAN;
        }
        unary_kernel(self.op as i16)(a)
    }
}

/// Constructor for [`DemandUnaryOp`].
pub struct DemandUnaryOpCtor;

impl DemandUnitDef for DemandUnaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() != 1 {
            return Err(BuildError::WrongInputCount);
        }
        // `rand`, `rand2`, `linrand`, `bilinrand`, `sum3rand` and `coin` need the synth's random
        // stream.
        if matches!(ctx.special_index, 37..=41 | 44) {
            return Err(BuildError::UnsupportedOp(ctx.special_index));
        }
        Ok(demand_unit_spec(DemandUnaryOp {
            op: ctx.special_index as u32,
            primed: 0,
        }))
    }
}
