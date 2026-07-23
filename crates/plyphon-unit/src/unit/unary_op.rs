//! `UnaryOpUGen` - applies a unary math operator (chosen by `special_index`) to one input.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;
use plyphon_dsp::{math, ops};

/// `<op>(a)`, where `<op>` is selected by the SynthDef's `special_index` (matching SuperCollider's
/// unary operator indices). The input may be audio- or control-rate; the output is audio-rate.
///
/// The operator is stored as its `special_index` selector (re-resolved to a fn once per block) rather
/// than a fn pointer, so the state is [`Pod`] and lives in the rt-pool; `a_audio` is a `0`/`1` flag.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UnaryOp {
    op: u32,
    a_audio: u32,
}

impl UnaryOp {
    /// Run the block with `op` inlined into the loop. Monomorphised per closure at the `process`
    /// dispatch below, so the hot operators compile to straight-line (vectorizable) loops instead
    /// of a per-sample indirect call.
    #[inline(always)]
    fn run(&self, ctx: &mut ProcessCtx<'_>, op: impl Fn(f32) -> f32 + Copy) {
        let out = ctx.outs.audio(0);
        if self.a_audio != 0 {
            let a = ctx.ins.audio(0);
            for (o, &x) in out.iter_mut().zip(a) {
                *o = op(x);
            }
        } else {
            out.fill(op(ctx.ins.control(0)));
        }
    }
}

impl Unit for UnaryOp {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The hot operators dispatch to monomorphised loops; the rest fall back to the table's fn
        // pointer, whose per-sample indirect call LLVM cannot inline or vectorize.
        match self.op {
            0 => self.run(ctx, |a| -a),
            5 => self.run(ctx, |a| a.abs()),
            12 => self.run(ctx, |a| a * a),
            13 => self.run(ctx, |a| a * a * a),
            op => {
                let Some(f) = unary_op(op as i16) else {
                    return DoneAction::Nothing;
                };
                self.run(ctx, f);
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`UnaryOp`].
pub struct UnaryOpCtor;

impl UnitDef for UnaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 1 {
            return Err(BuildError::WrongInputCount);
        }
        // Validate now so a bad operator fails at build, not silently at runtime.
        unary_op(ctx.special_index).ok_or(BuildError::UnsupportedOp(ctx.special_index))?;
        Ok(unit_spec(UnaryOp {
            op: ctx.special_index as u32,
            a_audio: (ctx.input_rates[0] == Rate::Audio) as u32,
        }))
    }
}

/// Map a SuperCollider unary operator index to its function (see SC's `opNeg`/`opAbs`/... enum in
/// `SpecialSelectorsOperatorsAndClasses.h`; kernels match the calc functions in
/// `UnaryOpUGens.cpp`). The RNG-driven ops (`opRand`/`opRand2`/`opLinRand`/`opBiLinRand`/
/// `opSum3Rand`/`opCoin`) and the remaining non-signal ops (`opIsNil`/...) are absent;
/// `opAsFloat`/`opAsInt` pass through, matching scsynth's `thru` default for both.
fn unary_op(index: i16) -> Option<fn(f32) -> f32> {
    Some(match index {
        0 => |a| -a,                    // opNeg
        1 => ops::not,                  // opNot
        4 => ops::bit_not,              // opBitNot
        5 => |a| a.abs(),               // opAbs
        6 => |a| a,                     // opAsFloat (already a float; identity)
        7 => |a| a,                     // opAsInt (scsynth has no case for it: `thru`, identity)
        8 => math::ceil,                // opCeil
        9 => math::floor,               // opFloor
        10 => |a| a - math::floor(a),   // opFrac
        11 => ops::sign,                // opSign
        12 => |a| a * a,                // opSquared
        13 => |a| a * a * a,            // opCubed
        14 => ops::signed_sqrt,         // opSqrt
        15 => math::exp,                // opExp
        16 => |a| 1.0 / a,              // opRecip
        17 => ops::midicps,             // opMIDICPS
        18 => ops::cpsmidi,             // opCPSMIDI
        19 => ops::midiratio,           // opMIDIRatio
        20 => ops::ratiomidi,           // opRatioMIDI
        21 => ops::dbamp,               // opDbAmp
        22 => ops::ampdb,               // opAmpDb
        23 => ops::octcps,              // opOctCPS
        24 => ops::cpsoct,              // opCPSOct
        25 => math::ln,                 // opLog (natural)
        26 => math::log2,               // opLog2
        27 => |a| math::log10(a.abs()), // opLog10
        28 => math::sin,                // opSin
        29 => math::cos,                // opCos
        30 => math::tan,                // opTan
        31 => math::asin,               // opArcSin
        32 => math::acos,               // opArcCos
        33 => math::atan,               // opArcTan
        34 => math::sinh,               // opSinH
        35 => math::cosh,               // opCosH
        36 => math::tanh,               // opTanH
        42 => ops::distort,             // opDistort
        43 => ops::softclip,            // opSoftClip
        46 => |_| 0.0,                  // opSilence
        47 => |a| a,                    // opThru
        48 => ops::rect_window,         // opRectWindow
        49 => ops::han_window,          // opHanWindow
        50 => ops::wel_window,          // opWelchWindow
        51 => ops::tri_window,          // opTriWindow
        52 => ops::ramp,                // opRamp
        53 => ops::scurve,              // opSCurve
        _ => return None,
    })
}
