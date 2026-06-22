//! `UnaryOpUGen` - applies a unary math operator (chosen by `special_index`) to one input.

use crate::bus::AudioBus;
use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{Inputs, Outputs, ProcessContext, Ugen};

/// `<op>(a)`, where `<op>` is selected by the SynthDef's `special_index` (matching SuperCollider's
/// unary operator indices). The input may be audio- or control-rate; the output is audio-rate.
pub struct UnaryOp {
    op: fn(f32) -> f32,
    a_audio: bool,
}

impl Ugen for UnaryOp {
    fn process(
        &mut self,
        _ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        _out_bus: &mut AudioBus,
    ) {
        let op = self.op;
        let out = outs.audio(0);
        if self.a_audio {
            let a = ins.audio(0);
            for (o, &x) in out.iter_mut().zip(a) {
                *o = op(x);
            }
        } else {
            out.fill(op(ins.control(0)));
        }
    }
}

/// Constructor for [`UnaryOp`].
pub struct UnaryOpCtor;

impl UgenCtor for UnaryOpCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        if ctx.input_rates.len() != 1 {
            return Err(BuildError::WrongInputCount);
        }
        let op = unary_op(ctx.special_index).ok_or(BuildError::UnsupportedOp(ctx.special_index))?;
        Ok(Box::new(UnaryOp {
            op,
            a_audio: ctx.input_rates[0] == Rate::Audio,
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
