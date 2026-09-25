//! `DemandEnvGen` - an envelope generator whose segments are demanded, plyphon's port of scsynth's
//! `DemandEnvGen` (`DemandUGens.cpp`).

use core::f64::consts::PI;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{DemandAccess, DemandWorld, demand_next, demand_reset};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// `DemandEnvGen.kr/ar(level, dur, shape, curve, gate, reset, levelScale, levelBias, timeScale,
/// doneAction)`: an envelope that demands each segment's end level, duration, shape and curve when
/// the previous segment ends. Inputs are in that order.
///
/// When the synth is constructed the first `level` is demanded as the start level (`0` if it is
/// `NaN`) and one sample is computed. Each segment then demands `dur` (scaled by `timeScale`), then
/// `shape` and `curve` (`curve` first when the gate is audio-rate), then its end `level` (scaled by
/// `levelScale` and offset by `levelBias`); a `NaN` shape or curve keeps the previous one. A segment
/// of at most one sample, or with `dur * 0.5` under a sample, is linear. Shapes follow `Env`: step,
/// linear, exponential, sine, welch, curve (linear when `|curve| < 0.001`), squared and cubed; any
/// other shape holds the level.
///
/// A `NaN` duration stops the envelope for the rest of the calc and starts a segment that never
/// ends, so its pending release never comes due. A `NaN` end level steps to the previous end level
/// and ends the segment at once: the next running sample releases. A release stops the envelope and
/// applies `doneAction`. A rising `reset` (from `<= 0` to `> 0`) resets the `level`,
/// `dur` and `shape` inputs (and `curve`, unless the gate is audio-rate), demands one `level` -
/// discarded when `reset <= 1`, jumped to when `reset > 1` - and starts a new segment.
///
/// `gate` is read at the end of each calc (each sample when it is audio-rate): `>= 1` keeps the
/// envelope running, a value in `(0, 1)` runs it and schedules a release, and `<= 0` holds the
/// current level. The value read decides whether the next calc starts running; a stop or release
/// within a calc lasts until it ends. With a control-rate gate the envelope runs a block at a
/// time: `reset` is read once per block and compared with the previous block's value on every
/// sample, so a rising reset repeats on each sample of its block. With an audio-rate gate every
/// sample reads `gate` and `reset`; a `reset` that is not audio-rate there (scsynth reads past its
/// one-value buffer) reads its one value on every sample.
///
/// Inputs read as plain values (`gate`, `reset`, the scales, `doneAction`) read a demand-rate source
/// as `0`, where scsynth reads the source's last output. With an audio-rate gate, a demanded input
/// wired to an audio-rate signal reads that sample; a nested demand source's own audio-rate inputs
/// read their first sample. scsynth leaves the stored shape and curve uninitialised until the first
/// calc ends, which a `NaN` first shape or curve reads; plyphon starts them at `0`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DemandEnvGen {
    /// Sine/welch/curve offset.
    a2: f64,
    /// Sine/welch recurrence coefficient, or the curve's scaled distance to its asymptote.
    b1: f64,
    /// Sine/welch/squared/cubed state.
    y1: f64,
    /// Sine/welch/squared/cubed state.
    y2: f64,
    /// Per-sample increment (linear, squared, cubed) or factor (exponential, curve).
    grow: f64,
    /// The current level.
    level: f64,
    /// The current segment's end level.
    end_level: f64,
    /// The current curve.
    curve: f64,
    /// Samples left in the current segment.
    phase: f32,
    /// The previous `reset` value, for rising-edge detection.
    prev_reset: f32,
    /// The current shape.
    shape: i32,
    /// Whether a release is scheduled for the end of the current segment.
    release: u32,
    /// Whether the next calc starts with the envelope running.
    running: u32,
    /// Whether `gate` is audio-rate (scsynth's `DemandEnvGen_next_a`).
    gate_audio: u32,
}

impl DemandEnvGen {
    const LEVEL: usize = 0;
    const DUR: usize = 1;
    const SHAPE: usize = 2;
    const CURVE: usize = 3;
    const GATE: usize = 4;
    const RESET: usize = 5;
    const LEVEL_SCALE: usize = 6;
    const LEVEL_BIAS: usize = 7;
    const TIME_SCALE: usize = 8;
    const DONE_ACTION: usize = 9;
    const NUM_INPUTS: usize = 10;

    const STEP: i32 = 0;
    const LINEAR: i32 = 1;
    const EXPONENTIAL: i32 = 2;
    const SINE: i32 = 3;
    const WELCH: i32 = 4;
    const CURVE_SHAPE: i32 = 5;
    const SQUARED: i32 = 6;
    const CUBED: i32 = 7;

    /// Set up a new segment from its demanded duration and end level: the segment's sample count and
    /// shape, the end level (or a release when it is `NaN`), and the shape's parameters. `level` and
    /// `shape` are the calc's working copies; `self.shape`/`self.end_level` hold the stored values.
    #[allow(clippy::too_many_arguments)]
    fn begin_segment(
        &mut self,
        ins: &Inputs<'_>,
        dur: f32,
        end: f32,
        curve: f64,
        sample_dur: f64,
        phase: &mut f32,
        shape: &mut i32,
        release: &mut bool,
        level: &mut f64,
    ) {
        let count = if *phase <= 1.0 {
            *shape = Self::LINEAR;
            1.0f32
        } else {
            *phase
        };
        if ((dur * 0.5) as f64) < sample_dur {
            *shape = Self::LINEAR;
        }
        let end_level = if (end as f64).is_nan() {
            *release = true;
            *phase = 0.0;
            *shape = Self::STEP;
            self.end_level
        } else {
            let e = end as f64 * ins.control(Self::LEVEL_SCALE) as f64
                + ins.control(Self::LEVEL_BIAS) as f64;
            self.end_level = e;
            e
        };
        let count = count as f64;
        match *shape {
            Self::STEP => *level = end_level,
            Self::LINEAR => self.grow = (end_level - *level) / count,
            Self::EXPONENTIAL => self.grow = math::powf(end_level / *level, 1.0 / count),
            Self::SINE => {
                let w = PI / count;
                self.a2 = (end_level + *level) * 0.5;
                self.b1 = 2.0 * math::cos(w);
                self.y1 = (end_level - *level) * 0.5;
                self.y2 = self.y1 * math::sin(PI * 0.5 - w);
                *level = self.a2 - self.y1;
            }
            Self::WELCH => {
                let w = (PI * 0.5) / count;
                self.b1 = 2.0 * math::cos(w);
                if end_level >= *level {
                    self.a2 = *level;
                    self.y1 = 0.0;
                    self.y2 = -math::sin(w) * (end_level - *level);
                } else {
                    self.a2 = end_level;
                    self.y1 = *level - end_level;
                    self.y2 = math::cos(w) * (*level - end_level);
                }
                *level = self.a2 + self.y1;
            }
            Self::CURVE_SHAPE => {
                if curve.abs() < 0.001 {
                    self.shape = Self::LINEAR;
                    *shape = Self::LINEAR;
                    self.grow = (end_level - *level) / count;
                } else {
                    let a1 = (end_level - *level) / (1.0 - math::exp(curve));
                    self.a2 = *level + a1;
                    self.b1 = a1;
                    self.grow = math::exp(curve / math::ceil(count));
                }
            }
            Self::SQUARED => {
                self.y1 = math::sqrt(*level);
                self.y2 = math::sqrt(end_level);
                self.grow = (self.y2 - self.y1) / count;
            }
            Self::CUBED => {
                self.y1 = math::powf(*level, 0.33333333);
                self.y2 = math::powf(end_level, 0.33333333);
                self.grow = (self.y2 - self.y1) / count;
            }
            _ => {}
        }
    }

    /// Advance the running envelope one sample along `shape`.
    fn step(&mut self, shape: i32, level: &mut f64) {
        match shape {
            Self::LINEAR => *level += self.grow,
            Self::EXPONENTIAL => *level *= self.grow,
            Self::SINE => {
                let y0 = self.b1 * self.y1 - self.y2;
                *level = self.a2 - y0;
                self.y2 = self.y1;
                self.y1 = y0;
            }
            Self::WELCH => {
                let y0 = self.b1 * self.y1 - self.y2;
                *level = self.a2 + y0;
                self.y2 = self.y1;
                self.y1 = y0;
            }
            Self::CURVE_SHAPE => {
                self.b1 *= self.grow;
                *level = self.a2 - self.b1;
            }
            Self::SQUARED => {
                self.y1 += self.grow;
                *level = self.y1 * self.y1;
            }
            Self::CUBED => {
                self.y1 += self.grow;
                *level = self.y1 * self.y1 * self.y1;
            }
            // Step, sustain and any other shape hold the level.
            _ => {}
        }
    }

    /// Read `gate` for the next calc (scsynth writes `m_running` directly): `>= 1` runs, `(0, 1)`
    /// runs and schedules a release, `<= 0` holds.
    fn gate(&mut self, zgate: f32, release: &mut bool) {
        if zgate >= 1.0 {
            self.running = 1;
        } else if zgate > 0.0 {
            self.running = 1;
            *release = true;
        } else {
            self.running = 0;
        }
    }

    /// Demand input `k` for sample `i` of an audio-rate-gated calc (scsynth's `DEMANDINPUT_A(k, i +
    /// 1)`): an audio-rate input reads sample `i`, any other input is demanded.
    fn demand_at(
        ins: &Inputs<'_>,
        demand: &mut DemandAccess<'_>,
        world: &mut DemandWorld<'_, '_>,
        k: usize,
        i: usize,
    ) -> f32 {
        if ins.rate(k) == Rate::Audio {
            ins.audio(k)[i]
        } else {
            demand_next(ins, demand, world, k)
        }
    }

    /// Sample `i` of a per-sample input (scsynth's `ZXP` over the input), which is audio-rate when
    /// scsynth reads it this way.
    fn sample(ins: &Inputs<'_>, k: usize, i: usize) -> f32 {
        if ins.rate(k) == Rate::Audio {
            ins.audio(k)[i]
        } else {
            ins.control(k)
        }
    }

    /// scsynth's `DemandEnvGen_next_k`: a control-rate gate, over the calc's samples.
    fn next_k(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            buf_counter: ctx.buf_counter,
            rgen: &mut *ctx.rgen,
        };
        let ins = &ctx.ins;
        let demand = &mut ctx.demand;
        let sample_rate = ctx.own.sample_rate;
        let sample_dur = ctx.own.sample_dur;
        let zreset = ins.control(Self::RESET);
        let mut level = self.level;
        let mut phase = self.phase;
        let mut curve = self.curve;
        let mut release = self.release != 0;
        let mut running = self.running != 0;
        let mut shape = self.shape;
        let out = ctx.outs.audio(0);
        for o in out.iter_mut() {
            if zreset > 0.0 && self.prev_reset <= 0.0 {
                demand_reset(ins, demand, &mut world, Self::LEVEL);
                demand_reset(ins, demand, &mut world, Self::DUR);
                demand_reset(ins, demand, &mut world, Self::SHAPE);
                demand_reset(ins, demand, &mut world, Self::CURVE);
                // A reset of at most 1 discards the first level; above 1 it jumps to it.
                let first = demand_next(ins, demand, &mut world, Self::LEVEL);
                if zreset > 1.0 {
                    level = first as f64;
                }
                release = false;
                running = true;
                phase = 0.0;
            }
            if phase <= 0.0 && running {
                if release {
                    running = false;
                    release = false;
                    done = done.max(DoneAction::from_code(ins.control(Self::DONE_ACTION)));
                } else {
                    let dur = demand_next(ins, demand, &mut world, Self::DUR);
                    if dur.is_nan() {
                        release = true;
                        running = false;
                        phase = f32::MAX;
                    } else {
                        phase = ((dur * ins.control(Self::TIME_SCALE)) as f64 * sample_rate
                            + phase as f64) as f32;
                    }
                    let fshape = demand_next(ins, demand, &mut world, Self::SHAPE);
                    shape = if fshape.is_nan() {
                        self.shape
                    } else {
                        fshape as i32
                    };
                    curve = demand_next(ins, demand, &mut world, Self::CURVE) as f64;
                    if curve.is_nan() {
                        curve = self.curve;
                    }
                    let end = demand_next(ins, demand, &mut world, Self::LEVEL);
                    self.begin_segment(
                        ins,
                        dur,
                        end,
                        curve,
                        sample_dur,
                        &mut phase,
                        &mut shape,
                        &mut release,
                        &mut level,
                    );
                }
            }
            if running {
                self.step(shape, &mut level);
                phase -= 1.0;
            }
            *o = level as f32;
        }
        self.gate(ins.control(Self::GATE), &mut release);
        self.level = level;
        self.curve = curve;
        self.shape = shape;
        self.prev_reset = zreset;
        self.release = release as u32;
        self.phase = phase;
        done
    }

    /// scsynth's `DemandEnvGen_next_a`: an audio-rate gate, read (with `reset`) on every sample.
    fn next_a(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            buf_counter: ctx.buf_counter,
            rgen: &mut *ctx.rgen,
        };
        let ins = &ctx.ins;
        let demand = &mut ctx.demand;
        let sample_rate = ctx.own.sample_rate;
        let sample_dur = ctx.own.sample_dur;
        let mut prev_reset = self.prev_reset;
        let mut level = self.level;
        let mut phase = self.phase;
        let mut curve = self.curve;
        let mut release = self.release != 0;
        let mut running = self.running != 0;
        let mut shape = self.shape;
        let out = ctx.outs.audio(0);
        for (i, o) in out.iter_mut().enumerate() {
            let zreset = Self::sample(ins, Self::RESET, i);
            if zreset > 0.0 && prev_reset <= 0.0 {
                demand_reset(ins, demand, &mut world, Self::LEVEL);
                // A reset of at most 1 discards the first level; above 1 it jumps to it.
                let first = Self::demand_at(ins, demand, &mut world, Self::LEVEL, i);
                if zreset > 1.0 {
                    level = first as f64;
                }
                demand_reset(ins, demand, &mut world, Self::DUR);
                demand_reset(ins, demand, &mut world, Self::SHAPE);
                release = false;
                running = true;
                phase = 0.0;
            }
            prev_reset = zreset;
            if phase <= 0.0 && running {
                if release {
                    running = false;
                    release = false;
                    done = done.max(DoneAction::from_code(ins.control(Self::DONE_ACTION)));
                } else {
                    let dur = Self::demand_at(ins, demand, &mut world, Self::DUR, i);
                    if dur.is_nan() {
                        release = true;
                        running = false;
                        phase = f32::MAX;
                    } else {
                        phase = ((dur * ins.control(Self::TIME_SCALE)) as f64 * sample_rate
                            + phase as f64) as f32;
                    }
                    curve = Self::demand_at(ins, demand, &mut world, Self::CURVE, i) as f64;
                    if curve.is_nan() {
                        curve = self.curve;
                    }
                    let fshape = Self::demand_at(ins, demand, &mut world, Self::SHAPE, i);
                    shape = if fshape.is_nan() {
                        self.shape
                    } else {
                        fshape as i32
                    };
                    let end = Self::demand_at(ins, demand, &mut world, Self::LEVEL, i);
                    self.begin_segment(
                        ins,
                        dur,
                        end,
                        curve,
                        sample_dur,
                        &mut phase,
                        &mut shape,
                        &mut release,
                        &mut level,
                    );
                }
            }
            if running {
                self.step(shape, &mut level);
                phase -= 1.0;
            }
            *o = level as f32;
            self.gate(Self::sample(ins, Self::GATE, i), &mut release);
        }
        self.level = level;
        self.curve = curve;
        self.shape = shape;
        self.prev_reset = prev_reset;
        self.release = release as u32;
        self.phase = phase;
        done
    }
}

impl Unit for DemandEnvGen {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor demands the start level, then computes one sample with the control-rate
        // calc whatever the gate's rate.
        let first = {
            let mut world = DemandWorld {
                buffers: &mut *ctx.buffers,
                local_bufs: &mut ctx.local_bufs,
                node_id: ctx.node_id,
                node_msgs: &mut ctx.node_msgs,
                buf_counter: ctx.buf_counter,
                rgen: &mut *ctx.rgen,
            };
            demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::LEVEL) as f64
        };
        self.level = if first.is_nan() { 0.0 } else { first };
        self.end_level = self.level;
        self.release = 0;
        self.prev_reset = 0.0;
        self.phase = 0.0;
        self.running = (ctx.ins.control(Self::GATE) > 0.0) as u32;
        self.next_k(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.gate_audio != 0 {
            self.next_a(ctx)
        } else {
            self.next_k(ctx)
        }
    }
}

/// Constructor for [`DemandEnvGen`].
pub struct DemandEnvGenCtor;

impl UnitDef for DemandEnvGenCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != DemandEnvGen::NUM_INPUTS {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DemandEnvGen {
            gate_audio: (ctx.input_rates[DemandEnvGen::GATE] == Rate::Audio) as u32,
            ..DemandEnvGen::zeroed()
        }))
    }
}
