//! `Duty` - a demand-driven sequencer, plyphon's port of scsynth's `Duty`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{DemandAccess, DemandWorld, demand_next, demand_reset};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// `Duty.kr/ar(dur, reset, level, doneAction)`: a self-clocking sequencer. It counts down `dur`
/// seconds (demanded from the `dur` input), then demands the next `level` and holds it for the next
/// `dur`. `dur` and `level` are typically demand sources (e.g. `Dseq`), so `Duty` drives a sequence
/// entirely on the audio thread with no control-plane messages. An exhausted (`NaN`) duration
/// fires `doneAction` once and freezes the unit on its held level (scsynth's `NaN` count) until a
/// rising `reset` revives it; a `NaN` level holds the previous value. A rising `reset` resets the
/// `dur`/`level` sources and restarts the count.
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
    count: f64,
    /// The currently held output value.
    level: f32,
    /// Previous `reset` value, for rising-edge detection.
    prev_reset: f32,
    /// `0`/`1`: control-rate (one value per block, counts in control frames) vs audio-rate (a full
    /// block, counts in samples).
    audio: u32,
    /// `0` until the first refill has run. The first refill stands in for scsynth's ctor-time
    /// `DEMANDINPUT` poll, which cannot fire `doneAction`, so a dur stream that is empty from the
    /// very start freezes silently.
    primed: u32,
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
        frame_rate: f64,
    ) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let dur = demand_next(ins, demand, world, Self::DUR);
        if dur.is_nan() {
            // An exhausted dur stream poisons the count like scsynth's `count = dur*sr + count`:
            // `count <= 0` is never true again, so the unit freezes on its held level and
            // `doneAction` fires exactly once. Only a rising reset (`count = 0`) revives it. On
            // the first refill (scsynth's ctor poll) it freezes without firing.
            self.count = f64::NAN;
            if self.primed != 0 {
                done = DoneAction::from_code(ins.control(Self::DONE));
            }
        } else {
            self.count += dur as f64 * frame_rate;
        }
        // The level is still pulled (and output) on the exhausting refill, as in scsynth.
        let level = demand_next(ins, demand, world, Self::LEVEL);
        if !level.is_nan() {
            self.level = level;
        }
        self.primed = 1;
        done
    }
}

impl Unit for Duty {
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
        };

        // A rising reset restarts the count and resets the demand sources.
        let reset = ctx.ins.control(Self::RESET);
        if reset > 0.0 && self.prev_reset <= 0.0 {
            demand_reset(&ctx.ins, &mut ctx.demand, &mut world, Self::LEVEL);
            demand_reset(&ctx.ins, &mut ctx.demand, &mut world, Self::DUR);
            self.count = 0.0;
        }
        self.prev_reset = reset;

        if self.audio != 0 {
            let frame_rate = ctx.audio.sample_rate;
            let out = ctx.outs.audio(0);
            for o in out.iter_mut() {
                if self.count <= 0.0 {
                    done = done.max(self.refill(&ctx.ins, &mut ctx.demand, &mut world, frame_rate));
                }
                *o = self.level;
                self.count -= 1.0;
            }
        } else {
            // Control rate: one value per block, counting down in control frames.
            let frame_rate = ctx.control.sample_rate;
            if self.count <= 0.0 {
                done = done.max(self.refill(&ctx.ins, &mut ctx.demand, &mut world, frame_rate));
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
            primed: 0,
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
    count: f64,
    /// Previous `reset` value, for rising-edge detection.
    prev_reset: f32,
    /// `0`/`1`: control-rate (one value per block) vs audio-rate (a full block).
    audio: u32,
    /// Non-zero if the first impulse waits one demanded duration (scsynth's `gapFirst`).
    gap_first: u32,
    /// `0` until the first block establishes the (optional) initial gap.
    warmed: u32,
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
        frame_rate: f64,
    ) -> (f32, DoneAction) {
        let mut done = DoneAction::Nothing;
        let dur = demand_next(ins, demand, world, Self::DUR);
        if dur.is_nan() {
            // As in [`Duty::refill`], an exhausted dur stream poisons the count so the unit
            // freezes (emitting `0`) after firing `doneAction` once; a rising reset revives it.
            // Unlike `Duty`, scsynth's `TDuty_Ctor` polls nothing up front, so the very first
            // boundary fires `doneAction` too.
            self.count = f64::NAN;
            done = DoneAction::from_code(ins.control(Self::DONE));
        } else {
            self.count += dur as f64 * frame_rate;
        }
        let level = demand_next(ins, demand, world, Self::LEVEL);
        (if level.is_nan() { 0.0 } else { level }, done)
    }
}

impl Unit for TDuty {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
        };
        let frame_rate = if self.audio != 0 {
            ctx.audio.sample_rate
        } else {
            ctx.control.sample_rate
        };

        // A `gapFirst` synth demands one duration up front so the first impulse is delayed by it.
        // A dur stream already exhausted here freezes the unit silently - scsynth's ctor-time
        // `m_count = DEMANDINPUT(dur) * sr` going `NaN` before any calc can fire `doneAction`.
        if self.warmed == 0 {
            if self.gap_first != 0 {
                let dur = demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::DUR);
                self.count = if dur.is_nan() {
                    f64::NAN
                } else {
                    dur as f64 * frame_rate
                };
            }
            self.warmed = 1;
        }

        let reset = ctx.ins.control(Self::RESET);
        if reset > 0.0 && self.prev_reset <= 0.0 {
            demand_reset(&ctx.ins, &mut ctx.demand, &mut world, Self::LEVEL);
            demand_reset(&ctx.ins, &mut ctx.demand, &mut world, Self::DUR);
            self.count = 0.0;
        }
        self.prev_reset = reset;

        if self.audio != 0 {
            let out = ctx.outs.audio(0);
            for o in out.iter_mut() {
                *o = if self.count <= 0.0 {
                    let (level, action) =
                        self.fire(&ctx.ins, &mut ctx.demand, &mut world, frame_rate);
                    done = done.max(action);
                    level
                } else {
                    0.0
                };
                self.count -= 1.0;
            }
        } else {
            *ctx.outs.control(0) = if self.count <= 0.0 {
                let (level, action) = self.fire(&ctx.ins, &mut ctx.demand, &mut world, frame_rate);
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

/// Constructor for [`TDuty`]: bakes the `gapFirst` flag from its constant input.
pub struct TDutyCtor;

impl UnitDef for TDutyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let gap_first = ctx.const_input(TDuty::GAP_FIRST).unwrap_or(0.0) != 0.0;
        Ok(unit_spec(TDuty {
            count: 0.0,
            prev_reset: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
            gap_first: gap_first as u32,
            warmed: 0,
        }))
    }
}
