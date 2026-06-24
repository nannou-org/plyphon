//! `BinaryOpUGen` - applies a binary math operator (chosen by `special_index`) to two inputs.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenDef};
use crate::ugen::{BuiltUgen, DoneAction, ProcessCtx, Ugen, ugen_spec};

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

impl Ugen for BinaryOp {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let Some(op) = binary_op(self.op as i16) else {
            return DoneAction::Nothing;
        };
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
        DoneAction::Nothing
    }
}

/// Constructor for [`BinaryOp`].
pub struct BinaryOpCtor;

impl UgenDef for BinaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUgen, BuildError> {
        if ctx.input_rates.len() != 2 {
            return Err(BuildError::WrongInputCount);
        }
        // Validate now so a bad operator fails at build, not silently at runtime.
        binary_op(ctx.special_index).ok_or(BuildError::UnsupportedOp(ctx.special_index))?;
        Ok(ugen_spec(BinaryOp {
            op: ctx.special_index as u32,
            a_audio: (ctx.input_rates[0] == Rate::Audio) as u32,
            b_audio: (ctx.input_rates[1] == Rate::Audio) as u32,
        }))
    }
}

/// Map a SuperCollider binary operator index to its function (see SC's `opAdd`/`opMul`/... enum).
fn binary_op(index: i16) -> Option<fn(f32, f32) -> f32> {
    Some(match index {
        0 => |a, b| a + b,
        1 => |a, b| a - b,
        2 => |a, b| a * b,
        4 => |a, b| a / b,
        5 => |a, b| a - b * (a / b).floor(), // mod (floored, as in SC)
        8 => |a, b| if a < b { 1.0 } else { 0.0 },
        9 => |a, b| if a > b { 1.0 } else { 0.0 },
        10 => |a, b| if a <= b { 1.0 } else { 0.0 },
        11 => |a, b| if a >= b { 1.0 } else { 0.0 },
        12 => |a, b| a.min(b),
        13 => |a, b| a.max(b),
        25 => |a, b| a.powf(b),
        30 => |a, b| a * b + a,                  // ring1
        31 => |a, b| a * b + a + b,              // ring2
        34 => |a, b| a * a - b * b,              // difsqr
        35 => |a, b| a * a + b * b,              // sumsqr
        38 => |a, b| (a - b).abs(),              // absdif
        42 => |a, b| a.clamp(-b.abs(), b.abs()), // clip2
        _ => return None,
    })
}
