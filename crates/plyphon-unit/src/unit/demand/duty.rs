//! `Duty` - a demand-driven sequencer, plyphon's port of scsynth's `Duty`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{DemandAccess, DemandWorld, demand_next, demand_reset};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// Apply the reset specialization shared by `Duty` and `TDuty`.
fn reset_at(
    count: &mut f32,
    prev_reset: &mut f32,
    ins: &Inputs<'_>,
    demand: &mut DemandAccess<'_>,
    world: &mut DemandWorld<'_, '_>,
    frame_rate: f32,
    sample: usize,
) {
    const DUR: usize = 0;
    const RESET: usize = 1;
    const LEVEL: usize = 3;

    if ins.rate(RESET) == Rate::Demand {
        if *prev_reset <= 0.0 {
            demand_reset(ins, demand, world, LEVEL);
            demand_reset(ins, demand, world, DUR);
            *count = 0.0;
            *prev_reset += demand_next(ins, demand, world, RESET, sample + 1) * frame_rate;
        } else {
            *prev_reset -= 1.0;
        }
        return;
    }

    let reset = if ins.rate(RESET) == Rate::Audio {
        ins.audio(RESET)[sample]
    } else {
        ins.control(RESET)
    };
    if reset > 0.0 && *prev_reset <= 0.0 {
        demand_reset(ins, demand, world, LEVEL);
        demand_reset(ins, demand, world, DUR);
        *count = 0.0;
    }
    *prev_reset = reset;
}

/// `Duty.kr/ar(dur, reset, level, doneAction)`: a self-clocking sequencer. It counts down `dur`
/// seconds (demanded from the `dur` input), then demands the next `level` and holds it for the next
/// `dur`. `dur` and `level` are typically demand sources (e.g. `Dseq`), so `Duty` drives a sequence
/// entirely on the audio thread with no control-plane messages. An exhausted (`NaN`) duration
/// fires `doneAction` once and freezes the unit on its held level (scsynth's `NaN` count) until a
/// rising `reset` revives it; an exhausted (`NaN`) level holds the previous value and fires
/// `doneAction` too. A rising `reset` resets the `dur`/`level` sources and restarts the count.
///
/// The compiled input order is `[dur, reset, doneAction, level]`: the `.ar`/`.kr` methods take
/// `(dur, reset, level, doneAction)` but pass `doneAction` before `level` to the UGen, so `level`
/// (a demand source) is the last input.
///
/// This is a normal (pushed) [`Unit`]; the pulling of `dur`/`level` is what makes it demand-driven.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Duty {
    /// Frames remaining until the next demand (fractional remainder preserved for sample accuracy).
    count: f32,
    /// The currently held output value.
    level: f32,
    /// Previous `reset` value, for rising-edge detection.
    prev_reset: f32,
    /// `0`/`1`: control-rate (one value per block, counts in control frames) vs audio-rate (a full
    /// block, counts in samples).
    audio: u32,
}

impl Duty {
    const DUR: usize = 0;
    const RESET: usize = 1;
    const DONE: usize = 2;
    const LEVEL: usize = 3;

    /// Demand the next duration (in frames) and level when the count elapses. Returns the done action
    /// to apply if the duration source is exhausted. Takes `ins`/`demand`/`world` as disjoint borrows
    /// so the caller can hold its output scratch at the same time (audio rate writes per sample).
    fn refill(
        &mut self,
        ins: &Inputs<'_>,
        demand: &mut DemandAccess<'_>,
        world: &mut DemandWorld<'_, '_>,
        frame_rate: f32,
        offset: usize,
    ) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let dur = demand_next(ins, demand, world, Self::DUR, offset);
        self.count += dur * frame_rate;
        if self.count.is_nan() {
            // An exhausted dur stream poisons the count like scsynth's `count = dur*sr + count`:
            // `count <= 0` is never true again, so the unit freezes on its held level and
            // `doneAction` fires exactly once. Only a reset (`count = 0`) revives it.
            done = DoneAction::from_code(ins.control(Self::DONE));
        }
        // The level is still pulled (and output) on the exhausting refill, as in scsynth.
        let level = demand_next(ins, demand, world, Self::LEVEL, offset);
        if level.is_nan() {
            // An exhausted level stream holds the previous value and *also* fires `doneAction`
            // (scsynth's `if (sc_isnan(x)) { x = prevout; DoneAction(...); }`).
            done = done.max(DoneAction::from_code(ins.control(Self::DONE)));
        } else {
            self.level = level;
        }
        done
    }
}

impl Unit for Duty {
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            rgen: &mut *ctx.rgen,
        };
        let frame_rate = ctx.own.sample_rate;
        self.prev_reset = if ctx.ins.rate(Self::RESET) == Rate::Demand {
            (demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::RESET, 1) as f64 * frame_rate)
                as f32
        } else {
            0.0
        };
        self.count = (demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::DUR, 1) as f64
            * frame_rate) as f32;
        self.level = demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::LEVEL, 1);
        *ctx.outs.control(0) = self.level;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut done = DoneAction::Nothing;
        // The demand sources' world reach, built once from disjoint `ctx` fields (buffers/node_msgs);
        // the per-sample output borrow (`ctx.outs`) and the pull borrows (`ctx.ins`/`ctx.demand`) are
        // all separate fields, so they coexist.
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            rgen: &mut *ctx.rgen,
        };
        let frame_rate = ctx.own.sample_rate as f32;

        if self.audio != 0 {
            let out = ctx.outs.audio(0);
            for (sample, o) in out.iter_mut().enumerate() {
                reset_at(
                    &mut self.count,
                    &mut self.prev_reset,
                    &ctx.ins,
                    &mut ctx.demand,
                    &mut world,
                    frame_rate,
                    sample,
                );
                if self.count <= 0.0 {
                    done = done.max(self.refill(
                        &ctx.ins,
                        &mut ctx.demand,
                        &mut world,
                        frame_rate,
                        sample + 1,
                    ));
                }
                *o = self.level;
                self.count -= 1.0;
            }
        } else {
            // Control rate: one value per block, counting down in control frames.
            reset_at(
                &mut self.count,
                &mut self.prev_reset,
                &ctx.ins,
                &mut ctx.demand,
                &mut world,
                frame_rate,
                0,
            );
            if self.count <= 0.0 {
                done = done.max(self.refill(&ctx.ins, &mut ctx.demand, &mut world, frame_rate, 1));
            }
            *ctx.outs.control(0) = self.level;
            self.count -= 1.0;
        }
        done
    }
}

/// Constructor for [`Duty`].
pub struct DutyCtor;

impl UnitDef for DutyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Duty {
            count: 0.0,
            level: 0.0,
            prev_reset: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `TDuty.kr/ar(dur, reset, level, doneAction, gapFirst)`: a self-clocking *trigger* sequencer.
/// Like [`Duty`] it counts down `dur` seconds demanded from its `dur` source, but at each boundary
/// it emits the demanded `level` for a single frame (a one-frame impulse) and `0` in between,
/// rather than holding the level. `gapFirst = 0` fires the first impulse immediately; a non-zero
/// `gapFirst` waits one demanded duration before it. An exhausted (`NaN`) duration fires
/// `doneAction` once and freezes the unit at `0` until a rising `reset` revives it; a `NaN` level
/// emits `0`. A rising `reset` resets the sources and restarts the count.
///
/// Input order matches [`Duty`] with `gapFirst` appended: `[dur, reset, doneAction, level,
/// gapFirst]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TDuty {
    /// Frames remaining until the next demand (fractional remainder preserved for sample accuracy).
    count: f32,
    /// Previous `reset` value, for rising-edge detection.
    prev_reset: f32,
    /// `0`/`1`: control-rate (one value per block) vs audio-rate (a full block).
    audio: u32,
}

impl TDuty {
    const DUR: usize = 0;
    const RESET: usize = 1;
    const DONE: usize = 2;
    const LEVEL: usize = 3;
    const GAP_FIRST: usize = 4;

    /// Demand the next duration (in frames) and level at a boundary, emitting the level as a
    /// one-frame impulse. Returns the done action if the duration source is exhausted.
    fn fire(
        &mut self,
        ins: &Inputs<'_>,
        demand: &mut DemandAccess<'_>,
        world: &mut DemandWorld<'_, '_>,
        frame_rate: f32,
        offset: usize,
    ) -> (f32, DoneAction) {
        let mut done = DoneAction::Nothing;
        let dur = demand_next(ins, demand, world, Self::DUR, offset);
        self.count += dur * frame_rate;
        if self.count.is_nan() {
            // As in [`Duty::refill`], an exhausted dur stream poisons the count so the unit
            // freezes (emitting `0`) after firing `doneAction` once; a rising reset revives it.
            // Unlike `Duty`, scsynth's `TDuty_Ctor` polls nothing up front, so the very first
            // boundary fires `doneAction` too.
            done = DoneAction::from_code(ins.control(Self::DONE));
        }
        let level = demand_next(ins, demand, world, Self::LEVEL, offset);
        (if level.is_nan() { 0.0 } else { level }, done)
    }
}

impl Unit for TDuty {
    fn construct(&mut self, ctx: &mut ProcessCtx<'_>) {
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            rgen: &mut *ctx.rgen,
        };
        let frame_rate = ctx.own.sample_rate;
        self.prev_reset = if ctx.ins.rate(Self::RESET) == Rate::Demand {
            (demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::RESET, 1) as f64 * frame_rate)
                as f32
        } else {
            0.0
        };
        self.count = if ctx.ins.control(Self::GAP_FIRST) != 0.0 {
            (demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::DUR, 1) as f64 * frame_rate)
                as f32
        } else {
            0.0
        };
        *ctx.outs.control(0) = 0.0;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            rgen: &mut *ctx.rgen,
        };
        let frame_rate = ctx.own.sample_rate as f32;

        if self.audio != 0 {
            let out = ctx.outs.audio(0);
            for (sample, o) in out.iter_mut().enumerate() {
                reset_at(
                    &mut self.count,
                    &mut self.prev_reset,
                    &ctx.ins,
                    &mut ctx.demand,
                    &mut world,
                    frame_rate,
                    sample,
                );
                *o = if self.count <= 0.0 {
                    let (level, action) = self.fire(
                        &ctx.ins,
                        &mut ctx.demand,
                        &mut world,
                        frame_rate,
                        sample + 1,
                    );
                    done = done.max(action);
                    level
                } else {
                    0.0
                };
                self.count -= 1.0;
            }
        } else {
            reset_at(
                &mut self.count,
                &mut self.prev_reset,
                &ctx.ins,
                &mut ctx.demand,
                &mut world,
                frame_rate,
                0,
            );
            *ctx.outs.control(0) = if self.count <= 0.0 {
                let (level, action) =
                    self.fire(&ctx.ins, &mut ctx.demand, &mut world, frame_rate, 1);
                done = done.max(action);
                level
            } else {
                0.0
            };
            self.count -= 1.0;
        }
        done
    }
}

/// Constructor for [`TDuty`].
pub struct TDutyCtor;

impl UnitDef for TDutyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(TDuty {
            count: 0.0,
            prev_reset: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
