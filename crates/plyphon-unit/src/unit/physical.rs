//! Physical-model oscillators - plyphon's ports of scsynth's `Spring`, `Ball` and `TBall`
//! (`PhysicalModelingUGens.cpp`).
//!
//! `Spring` integrates a damped mass-spring driven by an input force. `Ball` models a ball bouncing
//! on a (moving) floor under gravity; `TBall` is the same physics but outputs the collision velocity
//! as a trigger. `Ball`/`TBall` add a tiny per-unit RNG dither to reduce sampling jitter, so they
//! embed a Taus88 [`Rng`] and reseed it like the noise units.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rng::Rng;

/// `Spring.ar(in, spring, damping)`: a damped mass on a spring driven by the input force; outputs the
/// spring force.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Spring {
    pos: f32,
    vel: f32,
}

impl Unit for Spring {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let c = ctx.own.sample_dur as f32;
        let rc = ctx.own.sample_rate as f32;
        let spring = ctx.ins.control(1) * c;
        let damping = 1.0 - ctx.ins.control(2);
        let (mut pos, mut vel) = (self.pos, self.vel);
        let input = ctx.ins.audio(0);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(input) {
            let force = x * c - pos * spring;
            vel = (force + vel) * damping;
            pos += vel;
            *o = force * rc;
        }
        self.pos = pos;
        self.vel = vel;
        DoneAction::Nothing
    }
}

/// Constructor for [`Spring`].
pub struct SpringCtor;

impl UnitDef for SpringCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Spring { pos: 0.0, vel: 0.0 }))
    }
}

/// `Ball.ar(in, gravity, damping, friction)`: a ball bouncing on the floor position `in` under
/// gravity; outputs the ball's height.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Ball {
    pos: f32,
    vel: f32,
    prev: f32,
    rng: Rng,
}

impl Unit for Ball {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn init(&mut self, ctx: &InitCtx<'_>) {
        let floor = ctx.ins.control(0);
        self.pos = floor;
        self.prev = floor;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let c = ctx.own.sample_dur as f32;
        let g_in = ctx.ins.control(1);
        let damping = 1.0 - ctx.ins.control(2);
        let friction = ctx.ins.control(3);
        let maxvel = c * 1000.0;
        let inter = c * 1000.0;
        let g = c * g_in;
        let k = friction * g_in; // stickiness proportional to gravity
        let (mut pos, mut vel, mut prev) = (self.pos, self.vel, self.prev);
        let input = ctx.ins.audio(0);
        for (o, &floor) in ctx.outs.audio(0).iter_mut().zip(input) {
            vel -= g;
            pos += vel;
            let dist = pos - floor;
            let floorvel = (floor - prev).clamp(-maxvel, maxvel);
            let vel_diff = floorvel - vel;
            if dist.abs() < k {
                if dist.abs() < k * 0.005 {
                    vel = 0.0;
                    pos = floor + g;
                } else {
                    vel += vel_diff * inter;
                    pos += (floor - pos) * inter;
                }
            } else if dist <= 0.0 {
                pos = floor - dist;
                vel = vel_diff * damping;
                vel += self.rng.next_unipolar() * 0.00005 * g_in; // dither
            }
            prev = floor;
            *o = pos;
        }
        self.pos = pos;
        self.vel = vel;
        self.prev = prev;
        DoneAction::Nothing
    }
}

/// Constructor for [`Ball`].
pub struct BallCtor;

impl UnitDef for BallCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Ball {
            pos: 0.0,
            vel: 0.0,
            prev: 0.0,
            rng: Rng::new(ctx.seed),
        }))
    }
}

/// `TBall.ar(in, gravity, damping, friction)`: the same bouncing-ball physics as [`Ball`], but its
/// output is the collision velocity at each bounce (`0` between bounces) - a physically-modelled
/// bounce trigger.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TBall {
    pos: f64,
    prev: f64,
    vel: f32,
    rng: Rng,
}

impl Unit for TBall {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn init(&mut self, ctx: &InitCtx<'_>) {
        let floor = ctx.ins.control(0) as f64;
        self.pos = floor;
        self.prev = floor;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let c = ctx.own.sample_dur as f32;
        let g_in = ctx.ins.control(1);
        let damping = 1.0 - ctx.ins.control(2);
        let friction = ctx.ins.control(3);
        let maxvel = (c * 1000.0) as f64;
        let inter = (c * 10000.0) as f64;
        let g = (c * g_in) as f64;
        let k = (friction * g_in) as f64;
        let (mut pos, mut prev) = (self.pos, self.prev);
        let mut vel = self.vel as f64;
        let input = ctx.ins.audio(0);
        for (o, &floor) in ctx.outs.audio(0).iter_mut().zip(input) {
            let floor = floor as f64;
            let mut outval = 0.0;
            vel -= g;
            pos += vel;
            let dist = pos - floor;
            let floorvel = (floor - prev).clamp(-maxvel, maxvel);
            let vel_diff = floorvel - vel;
            if dist.abs() < k {
                if dist.abs() < k * 0.005 {
                    vel = 0.0;
                    pos = floor + g;
                } else {
                    vel += vel_diff * inter;
                    pos += (floor - pos) * inter;
                }
            } else if dist <= 0.0 {
                pos = floor - dist;
                vel = (floorvel - vel) * damping as f64;
                outval = vel;
                vel += self.rng.next_unipolar() as f64 * 0.001 * g_in as f64; // dither
            }
            prev = floor;
            *o = outval as f32;
        }
        self.pos = pos;
        self.prev = prev;
        self.vel = vel as f32;
        DoneAction::Nothing
    }
}

/// Constructor for [`TBall`].
pub struct TBallCtor;

impl UnitDef for TBallCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(TBall {
            pos: 0.0,
            prev: 0.0,
            vel: 0.0,
            rng: Rng::new(ctx.seed),
        }))
    }
}
