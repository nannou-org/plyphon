//! `BinaryOpUGen` - applies a binary math operator (chosen by `special_index`) to two inputs.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;
use plyphon_dsp::{math, ops};

/// `a <op> b`, where `<op>` is selected by the SynthDef's `special_index` (matching SuperCollider's
/// binary operator indices). Each input may be audio- or control-rate; the output is audio-rate.
///
/// The operator is stored as its `special_index` selector (re-resolved to a fn once per block) rather
/// than a fn pointer, so the state is [`Pod`] and lives in the rt-pool; `*_audio` are `0`/`1` flags.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BinaryOp {
    op: u32,
    a_audio: u32,
    b_audio: u32,
}

impl BinaryOp {
    /// Run the block with `op` inlined into the loop. Monomorphised per closure at the `process`
    /// dispatch below, so the hot operators compile to straight-line (vectorizable) loops instead
    /// of a per-sample indirect call.
    #[inline(always)]
    fn run(&self, ctx: &mut ProcessCtx<'_>, op: impl Fn(f32, f32) -> f32 + Copy) {
        let out = ctx.outs.audio(0);
        match (self.a_audio != 0, self.b_audio != 0) {
            (true, true) => {
                let a = ctx.ins.audio(0);
                let b = ctx.ins.audio(1);
                for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                    *o = op(x, y);
                }
            }
            (true, false) => {
                let a = ctx.ins.audio(0);
                let y = ctx.ins.control(1);
                for (o, &x) in out.iter_mut().zip(a) {
                    *o = op(x, y);
                }
            }
            (false, true) => {
                let x = ctx.ins.control(0);
                let b = ctx.ins.audio(1);
                for (o, &y) in out.iter_mut().zip(b) {
                    *o = op(x, y);
                }
            }
            (false, false) => out.fill(op(ctx.ins.control(0), ctx.ins.control(1))),
        }
    }
}

impl Unit for BinaryOp {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The hot arithmetic operators - typically the most numerous units in a real graph -
        // dispatch to monomorphised loops; the rest fall back to the table's fn pointer, whose
        // per-sample indirect call LLVM cannot inline or vectorize.
        match self.op {
            0 => self.run(ctx, |a, b| a + b),
            1 => self.run(ctx, |a, b| a - b),
            2 => self.run(ctx, |a, b| a * b),
            4 => self.run(ctx, |a, b| a / b),
            12 => self.run(ctx, |a, b| a.min(b)),
            13 => self.run(ctx, |a, b| a.max(b)),
            op => {
                let Some(f) = binary_op(op as i16) else {
                    return DoneAction::Nothing;
                };
                self.run(ctx, f);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`BinaryOp`].
pub struct BinaryOpCtor;

impl UnitDef for BinaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 2 {
            return Err(BuildError::WrongInputCount);
        }
        let a_audio = (ctx.input_rates[0] == Rate::Audio) as u32;
        let b_audio = (ctx.input_rates[1] == Rate::Audio) as u32;
        // The RNG-driven operators draw fresh randomness per frame, so they build into a
        // stateful variant instead of the pure-fn table.
        if matches!(ctx.special_index, 47 | 48) {
            return Ok(unit_spec(RandBinaryOp {
                exponential: (ctx.special_index == 48) as u32,
                a_audio,
                b_audio,
            }));
        }
        // Validate now so a bad operator fails at build, not silently at runtime.
        binary_op(ctx.special_index).ok_or(BuildError::UnsupportedOp(ctx.special_index))?;
        Ok(unit_spec(BinaryOp {
            op: ctx.special_index as u32,
            a_audio,
            b_audio,
        }))
    }
}

/// The RNG-driven binary operators, `rrand` (special index 47) and `exprand` (48): a fresh random
/// draw per output frame, scaled between the two inputs (ordered low-to-high), from the synth's
/// shared random stream - scsynth's `rrand_1`/`exprand_1`, which read the graph's `RGen`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RandBinaryOp {
    /// `0` = uniform (`rrand`), `1` = equal-probability-per-octave (`exprand`).
    exponential: u32,
    a_audio: u32,
    b_audio: u32,
}

impl Unit for RandBinaryOp {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ProcessCtx {
            ins, outs, rgen, ..
        } = ctx;
        let a_ctrl = ins.control(0);
        let b_ctrl = ins.control(1);
        let a_sig = (self.a_audio != 0).then(|| ins.audio(0));
        let b_sig = (self.b_audio != 0).then(|| ins.audio(1));
        let exponential = self.exponential != 0;
        for (i, o) in outs.audio(0).iter_mut().enumerate() {
            let xa = a_sig.map_or(a_ctrl, |s| s[i]);
            let xb = b_sig.map_or(b_ctrl, |s| s[i]);
            let (lo, hi) = if xb > xa { (xa, xb) } else { (xb, xa) };
            *o = if exponential {
                math::exp(math::ln(hi / lo) * rgen.next_unipolar()) * lo
            } else {
                lo + rgen.next_unipolar() * (hi - lo)
            };
        }
        DoneAction::Nothing
    }
}

/// Map a SuperCollider binary operator index to its function (see SC's `opAdd`/`opMul`/... enum in
/// `SpecialSelectorsOperatorsAndClasses.h`; kernels match the `*_1` calc functions in
/// `BinaryOpUGens.cpp`). The RNG-driven ops (`opRandRange` 47, `opExpRandRange` 48) build into the
/// stateful [`RandBinaryOp`] instead; the unimplemented-at-audio-rate ops (`opUnsignedShift`,
/// `opFill`) are absent.
fn binary_op(index: i16) -> Option<fn(f32, f32) -> f32> {
    Some(match index {
        0 => |a, b| a + b,                           // opAdd
        1 => |a, b| a - b,                           // opSub
        2 => |a, b| a * b,                           // opMul
        3 => |a, b| math::floor(a / b),              // opIDiv
        4 => |a, b| a / b,                           // opFDiv
        5 => ops::modulo,                            // opMod
        6 => |a, b| if a == b { 1.0 } else { 0.0 },  // opEQ
        7 => |a, b| if a != b { 1.0 } else { 0.0 },  // opNE
        8 => |a, b| if a < b { 1.0 } else { 0.0 },   // opLT
        9 => |a, b| if a > b { 1.0 } else { 0.0 },   // opGT
        10 => |a, b| if a <= b { 1.0 } else { 0.0 }, // opLE
        11 => |a, b| if a >= b { 1.0 } else { 0.0 }, // opGE
        12 => |a, b| a.min(b),                       // opMin
        13 => |a, b| a.max(b),                       // opMax
        14 => ops::bit_and,                          // opBitAnd
        15 => ops::bit_or,                           // opBitOr
        16 => ops::bit_xor,                          // opBitXor
        17 => ops::lcm,                              // opLCM
        18 => ops::gcd,                              // opGCD
        19 => ops::round,                            // opRound
        20 => ops::round_up,                         // opRoundUp
        21 => ops::trunc,                            // opTrunc
        22 => |a, b| math::atan2(a, b),              // opAtan2
        23 => |a, b| math::hypot(a, b),              // opHypot
        24 => ops::hypotx,                           // opHypotx
        25 => ops::pow,                              // opPow
        26 => ops::shift_left,                       // opShiftLeft
        27 => ops::shift_right,                      // opShiftRight
        30 => ops::ring1,                            // opRing1
        31 => ops::ring2,                            // opRing2
        32 => ops::ring3,                            // opRing3
        33 => ops::ring4,                            // opRing4
        34 => ops::difsqr,                           // opDifSqr
        35 => ops::sumsqr,                           // opSumSqr
        36 => ops::sqrsum,                           // opSqrSum
        37 => ops::sqrdif,                           // opSqrDif
        38 => ops::absdif,                           // opAbsDif
        39 => ops::thresh,                           // opThresh
        40 => ops::amclip,                           // opAMClip
        41 => ops::scaleneg,                         // opScaleNeg
        42 => ops::clip2,                            // opClip2
        43 => ops::excess,                           // opExcess
        44 => ops::fold2,                            // opFold2
        45 => ops::wrap2,                            // opWrap2
        46 => |a, _| a,                              // opFirstArg
        _ => return None,
    })
}
