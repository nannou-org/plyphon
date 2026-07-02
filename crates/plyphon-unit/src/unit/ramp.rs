//! Linear-ramp smoothers - plyphon's ports of scsynth's `Ramp` and `VarLag`.
//!
//! Unlike the exponential [`Lag`](crate::unit::util::Lag) family, these sample their input at
//! intervals and ramp *linearly* toward each new value. `Ramp` re-samples every `lagTime` seconds;
//! `VarLag` starts a fresh ramp whenever the (block-rate) input changes, rescaling the in-flight ramp
//! when `lagTime` changes. Both keep an integer sample `counter` and a per-sample `slope`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::sample_channel;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};

/// `Ramp.ar/kr(in, lagTime)`: a linear-interpolating sample-and-hold. Every `lagTime` seconds it reads
/// the input and ramps linearly toward it over the next interval, so a stepped control becomes a
/// piecewise-linear glide.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Ramp {
    level: f64,
    slope: f64,
    counter: i32,
    _pad: u32,
}

impl Ramp {
    const IN: usize = 0;
    const PERIOD: usize = 1;
}

impl Unit for Ramp {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.level = ctx.ins.control(Self::IN) as f64;
        self.counter = 1;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let period = ins.control(Self::PERIOD) as f64;
        let sr = ctx.own.sample_rate;
        let out = ctx.outs.audio(0);
        let block = out.len();
        let mut level = self.level;
        let mut slope = self.slope;
        let mut counter = self.counter.max(1);
        let mut i = 0;
        while i < block {
            let n = (block - i).min(counter as usize);
            for o in &mut out[i..i + n] {
                *o = level as f32;
                level += slope;
            }
            i += n;
            counter -= n as i32;
            if counter <= 0 {
                counter = ((period * sr) as i32).max(1);
                // scsynth reads the input at the segment boundary (`*in` after advancing); clamp the
                // last segment to the final in-block sample rather than reading past the block.
                let target = sample_channel(&ins, Self::IN, i.min(block - 1)) as f64;
                slope = (target - level) / counter as f64;
            }
        }
        self.level = level;
        self.slope = slope;
        self.counter = counter;
        DoneAction::Nothing
    }
}

/// Constructor for [`Ramp`].
pub struct RampCtor;

impl UnitDef for RampCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Ramp {
            level: 0.0,
            slope: 0.0,
            counter: 1,
            _pad: 0,
        }))
    }
}

/// `VarLag.ar/kr(in, time, start)`: a linear lag that ramps from `start` toward `in` over `time`
/// seconds, restarting the ramp whenever the (block-rate) input changes and rescaling the in-flight
/// ramp when `time` changes. (scsynth's server `VarLag` is only the compiled form for a linear warp;
/// other warps expand to `EnvGen` in the language.)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VarLag {
    level: f64,
    slope: f64,
    counter: i32,
    in_prev: f32,
    lag_prev: f32,
    _pad: u32,
}

impl VarLag {
    const IN: usize = 0;
    const TIME: usize = 1;
    const START: usize = 2;
}

impl Unit for VarLag {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let in0 = ctx.ins.control(Self::IN);
        let lag = ctx.ins.control(Self::TIME);
        self.level = ctx.ins.control(Self::START) as f64;
        self.counter = ((lag * ctx.own.sample_rate as f32) as i32).max(1);
        self.slope = (in0 as f64 - self.level) / self.counter as f64;
        self.in_prev = in0;
        self.lag_prev = lag;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // Change detection is block-rate: scsynth reads only the first input sample.
        let in0 = ctx.ins.control(Self::IN);
        let lag = ctx.ins.control(Self::TIME);
        let sr = ctx.own.sample_rate;
        let mut level = self.level;
        let mut slope = self.slope;
        let mut counter = self.counter;
        if in0 != self.in_prev {
            counter = ((lag * sr as f32) as i32).max(1);
            slope = (in0 as f64 - level) / counter as f64;
            self.in_prev = in0;
            self.lag_prev = lag;
        } else if lag != self.lag_prev {
            if counter != 0 {
                let scale = (lag / self.lag_prev) as f64;
                counter = ((counter as f64 * scale) as i32).max(1);
                slope /= scale;
            }
            self.lag_prev = lag;
        }
        let target = self.in_prev as f64;
        for o in ctx.outs.audio(0).iter_mut() {
            *o = level as f32;
            if counter > 0 {
                level += slope;
                counter -= 1;
            } else {
                level = target;
            }
        }
        self.level = level;
        self.slope = slope;
        self.counter = counter;
        DoneAction::Nothing
    }
}

/// Constructor for [`VarLag`].
pub struct VarLagCtor;

impl UnitDef for VarLagCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(VarLag {
            level: 0.0,
            slope: 0.0,
            counter: 1,
            in_prev: 0.0,
            lag_prev: 0.0,
            _pad: 0,
        }))
    }
}
