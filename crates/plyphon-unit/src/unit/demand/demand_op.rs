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
//! `sum3rand` and `coin`) draw from the synth's random stream ([`DemandCtx::rgen`]), as the
//! reference's `*_d` random kernels draw from `mParent->mRGen`.
//!
//! An audio-rate operand reads the block's *first* sample, where the reference reads the sample its
//! pull is offset to. A demand pull carries no sample offset in plyphon, so this is the engine-wide
//! behaviour of every demand-rate operand read, not something these two units decide.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::binary_op::binary_op;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use crate::unit::unary_op::{is_random, random_unary, unary_op};

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
/// The constructor does not prime: the reference's binary constructor writes `0` into its output
/// and pulls nothing. This is asymmetric with [`DemandUnaryOp`], and the asymmetry is the
/// reference's.
///
/// `NaN` never reaches the operator: the protocol above handles it first.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DemandBinaryOp {
    /// The SuperCollider binary operator index, resolved to a kernel on each produce.
    op: u32,
}

impl DemandUnit for DemandBinaryOp {
    fn init(&mut self, _ctx: &mut DemandCtx<'_>) {}

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
        match self.op {
            // `rrand_d`: a uniform draw between the operands, ordered low to high.
            47 => {
                let u = ctx.rgen().next_unipolar();
                if b > a {
                    a + u * (b - a)
                } else {
                    b + u * (a - b)
                }
            }
            // `exprand_d`: `RGen::exprandrng` between the operands, ordered low to high.
            48 => {
                let (lo, hi) = if b > a { (a, b) } else { (b, a) };
                ctx.rgen().next_exprand(lo as f64, hi as f64) as f32
            }
            op => binary_kernel(op as i16)(a, b),
        }
    }
}

/// Constructor for [`DemandBinaryOp`].
pub struct DemandBinaryOpCtor;

impl DemandUnitDef for DemandBinaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() != 2 {
            return Err(BuildError::WrongInputCount);
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
/// the operand before any consumer pulls; the constructor here does the same.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DemandUnaryOp {
    /// The SuperCollider unary operator index, resolved to a kernel on each produce.
    op: u32,
}

impl DemandUnit for DemandUnaryOp {
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        let _ = self.produce(ctx);
    }

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        ctx.reset(A);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let op = self.op as i16;
        let a = ctx.demand(A);
        // `coin_d` draws before it checks the operand; the other random kernels check first.
        if op == 44 {
            let coin = random_unary(op, ctx.rgen(), a);
            return if a.is_nan() { f32::NAN } else { coin };
        }
        if a.is_nan() {
            return f32::NAN;
        }
        if is_random(op) {
            return random_unary(op, ctx.rgen(), a);
        }
        unary_kernel(op)(a)
    }
}

/// Constructor for [`DemandUnaryOp`].
pub struct DemandUnaryOpCtor;

impl DemandUnitDef for DemandUnaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() != 1 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(demand_unit_spec(DemandUnaryOp {
            op: ctx.special_index as u32,
        }))
    }
}
