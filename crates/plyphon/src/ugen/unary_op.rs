//! `UnaryOpUGen` - applies a unary math operator (chosen by `special_index`) to one input.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenDef};
use crate::ugen::{BuiltUgen, DoneAction, ProcessCtx, Ugen, ugen_spec};

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

impl Ugen for UnaryOp {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let Some(op) = unary_op(self.op as i16) else {
            return DoneAction::Nothing;
        };
        let out = ctx.outs.audio(0);
        if self.a_audio != 0 {
            let a = ctx.ins.audio(0);
            for (o, &x) in out.iter_mut().zip(a) {
                *o = op(x);
            }
        } else {
            out.fill(op(ctx.ins.control(0)));
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`UnaryOp`].
pub struct UnaryOpCtor;

impl UgenDef for UnaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError> {
        if ctx.input_rates.len() != 1 {
            return Err(BuildError::WrongInputCount);
        }
        // Validate now so a bad operator fails at build, not silently at runtime.
        unary_op(ctx.special_index).ok_or(BuildError::UnsupportedOp(ctx.special_index))?;
        Ok(ugen_spec(UnaryOp {
            op: ctx.special_index as u32,
            a_audio: (ctx.input_rates[0] == Rate::Audio) as u32,
        }))
    }
}

/// Map a SuperCollider unary operator index to its function (see SC's `opNeg`/`opAbs`/... enum).
fn unary_op(index: i16) -> Option<fn(f32) -> f32> {
    Some(match index {
        0 => |a| -a,
        5 => |a| a.abs(),
        8 => |a| a.ceil(),
        9 => |a| a.floor(),
        10 => |a| a - a.floor(), // frac
        11 => |a| {
            if a > 0.0 {
                1.0
            } else if a < 0.0 {
                -1.0
            } else {
                0.0
            }
        }, // sign
        12 => |a| a * a,         // squared
        13 => |a| a * a * a,     // cubed
        14 => |a| a.sqrt(),
        15 => |a| a.exp(),
        16 => |a| 1.0 / a, // reciprocal
        25 => |a| a.ln(),  // log (natural)
        28 => |a| a.sin(),
        29 => |a| a.cos(),
        30 => |a| a.tan(),
        36 => |a| a.tanh(),
        42 => |a| a / (1.0 + a.abs()), // distort
        _ => return None,
    })
}
