//! `BinaryOpUGen` - applies a binary math operator (chosen by `special_index`) to two inputs.

use crate::bus::AudioBus;
use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{Inputs, Outputs, ProcessContext, Ugen};

/// `a <op> b`, where `<op>` is selected by the SynthDef's `special_index` (matching SuperCollider's
/// binary operator indices). Each input may be audio- or control-rate; the output is audio-rate.
pub struct BinaryOp {
    op: fn(f32, f32) -> f32,
    a_audio: bool,
    b_audio: bool,
}

impl Ugen for BinaryOp {
    fn process(
        &mut self,
        _ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        _out_bus: &mut AudioBus,
    ) {
        let op = self.op;
        let out = outs.audio(0);
        match (self.a_audio, self.b_audio) {
            (true, true) => {
                let a = ins.audio(0);
                let b = ins.audio(1);
                for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                    *o = op(x, y);
                }
            }
            (true, false) => {
                let a = ins.audio(0);
                let y = ins.control(1);
                for (o, &x) in out.iter_mut().zip(a) {
                    *o = op(x, y);
                }
            }
            (false, true) => {
                let x = ins.control(0);
                let b = ins.audio(1);
                for (o, &y) in out.iter_mut().zip(b) {
                    *o = op(x, y);
                }
            }
            (false, false) => out.fill(op(ins.control(0), ins.control(1))),
        }
    }
}

/// Constructor for [`BinaryOp`].
pub struct BinaryOpCtor;

impl UgenCtor for BinaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        if ctx.input_rates.len() != 2 {
            return Err(BuildError::WrongInputCount);
        }
        let op =
            binary_op(ctx.special_index).ok_or(BuildError::UnsupportedOp(ctx.special_index))?;
        Ok(Box::new(BinaryOp {
            op,
            a_audio: ctx.input_rates[0] == Rate::Audio,
            b_audio: ctx.input_rates[1] == Rate::Audio,
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
