//! `Duty` and `TDuty` - demand-driven sequencers, plyphon's port of scsynth's `Duty` and `TDuty`
//! (`DemandUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{DemandAccess, DemandWorld, demand_next, demand_reset};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

/// The `reset` input of `Duty`/`TDuty` read once per block (scsynth's `_dk` calc, a control- or
/// scalar-rate reset).
const RESET_CONTROL: u32 = 0;
/// The `reset` input read per sample (scsynth's `_da` calc, an audio-rate reset).
const RESET_AUDIO: u32 = 1;
/// The `reset` input pulled as a stream of durations between resets (scsynth's `_dd` calc, a
/// demand-rate reset).
const RESET_DEMAND: u32 = 2;

/// The calc scsynth's `Duty_Ctor`/`TDuty_Ctor` select from the `reset` input's rate.
fn reset_mode(ctx: &BuildContext<'_>, reset: usize) -> u32 {
    match ctx.input_rates.get(reset) {
        Some(Rate::Audio) => RESET_AUDIO,
        Some(Rate::Demand) => RESET_DEMAND,
        _ => RESET_CONTROL,
    }
}

/// The rate a unit's durations count in: samples at audio rate, control blocks at control rate.
fn frame_rate(ctx: &ProcessCtx<'_>, audio: bool) -> f64 {
    if audio {
        ctx.audio.sample_rate
    } else {
        ctx.control.sample_rate
    }
}

/// The reset stage shared by `Duty` and `TDuty`, run at the top of each frame as scsynth's three
/// calcs do. A control-rate reset reads the block's value, an audio-rate one the frame's sample; a
/// rising edge resets the `level` and `dur` sources and restarts the count. A demand-rate reset
/// counts down its own duration (`prev_reset`, in frames) and, when it elapses, resets the sources,
/// restarts the count and pulls the next reset duration.
#[allow(clippy::too_many_arguments)]
fn reset_stage(
    mode: u32,
    frame: usize,
    count: &mut f32,
    prev_reset: &mut f32,
    ins: &Inputs<'_>,
    demand: &mut DemandAccess<'_>,
    world: &mut DemandWorld<'_, '_>,
    (reset, level, dur): (usize, usize, usize),
    sr: f32,
) {
    if mode == RESET_DEMAND {
        if *prev_reset <= 0.0 {
            demand_reset(ins, demand, world, level);
            demand_reset(ins, demand, world, dur);
            *count = 0.0;
            *prev_reset += demand_next(ins, demand, world, reset) * sr;
        } else {
            *prev_reset -= 1.0;
        }
    } else {
        let z = if mode == RESET_AUDIO {
            ins.audio(reset)[frame]
        } else {
            ins.control(reset)
        };
        if z > 0.0 && *prev_reset <= 0.0 {
            demand_reset(ins, demand, world, level);
            demand_reset(ins, demand, world, dur);
            *count = 0.0;
        }
        *prev_reset = z;
    }
}

/// The constructor's `m_prevreset`: a demand-rate reset pulls its first duration (in frames, the
/// product taken in double as scsynth's `DEMANDINPUT(duty_reset) * SAMPLERATE`); any other reset
/// starts at 0.
fn initial_reset(
    mode: u32,
    ins: &Inputs<'_>,
    demand: &mut DemandAccess<'_>,
    world: &mut DemandWorld<'_, '_>,
    reset: usize,
    frame_rate: f64,
) -> f32 {
    if mode == RESET_DEMAND {
        (demand_next(ins, demand, world, reset) as f64 * frame_rate) as f32
    } else {
        0.0
    }
}

/// `Duty.kr/ar(dur, reset, level, doneAction)`: a self-clocking sequencer. It counts down `dur`
/// seconds (demanded from the `dur` input), then demands the next `level` and holds it for the next
/// `dur`. `dur` and `level` are typically demand sources (e.g. `Dseq`), so `Duty` drives a sequence
/// entirely on the audio thread with no control-plane messages. An exhausted (`NaN`) duration
/// fires `doneAction` once and freezes the unit on its held level (scsynth's `NaN` count) until a
/// reset revives it; an exhausted (`NaN`) level holds the previous value and fires `doneAction`
/// too.
///
/// A reset resets the `dur`/`level` sources and restarts the count. How `reset` is read follows its
/// rate, as scsynth's `Duty_Ctor` selects `Duty_next_dk`/`_da`/`_dd`: a control-rate reset fires on a
/// rising edge of its block value, an audio-rate reset on a rising edge at any sample, and a
/// demand-rate reset is a stream of durations in seconds, the unit resetting each time one elapses.
///
/// The compiled input order is `[dur, reset, doneAction, level]`: the `.ar`/`.kr` methods take
/// `(dur, reset, level, doneAction)` but pass `doneAction` before `level` to the UGen, so `level`
/// (a demand source) is the last input.
///
/// This is a normal (pushed) [`Unit`]; the pulling of `dur`/`level` is what makes it demand-driven.
/// The counts are single-precision, as scsynth's `m_count`/`m_prevreset` are.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Duty {
    /// Frames remaining until the next demand (fractional remainder preserved for sample accuracy).
    count: f32,
    /// The currently held output value.
    level: f32,
    /// The previous `reset` value for edge detection, or with a demand-rate reset the frames
    /// remaining until the next reset (scsynth's `m_prevreset` serves both).
    prev_reset: f32,
    /// `0`/`1`: control-rate (one value per block, counts in control frames) vs audio-rate (a full
    /// block, counts in samples).
    audio: u32,
    /// How `reset` is read: [`RESET_CONTROL`], [`RESET_AUDIO`] or [`RESET_DEMAND`].
    reset_mode: u32,
}

impl Duty {
    const DUR: usize = 0;
    const RESET: usize = 1;
    const DONE: usize = 2;
    const LEVEL: usize = 3;

    /// Demand the next duration (in frames) and level when the count elapses, as scsynth's
    /// `count = DEMANDINPUT_A(duty_dur) * sr + count`. Returns the done action to apply if either
    /// source is exhausted. Takes `ins`/`demand`/`world` as disjoint borrows so the caller can hold
    /// its output scratch at the same time (audio rate writes per sample).
    fn refill(
        &mut self,
        ins: &Inputs<'_>,
        demand: &mut DemandAccess<'_>,
        world: &mut DemandWorld<'_, '_>,
        sr: f32,
    ) -> DoneAction {
        let mut done = DoneAction::Nothing;
        // An exhausted dur stream poisons the count: `count <= 0` is never true again, so the unit
        // freezes on its held level and `doneAction` fires exactly once. Only a reset (`count = 0`)
        // revives it.
        self.count += demand_next(ins, demand, world, Self::DUR) * sr;
        if self.count.is_nan() {
            done = DoneAction::from_code(ins.control(Self::DONE));
        }
        // The level is still pulled (and output) on the exhausting refill, as in scsynth.
        let level = demand_next(ins, demand, world, Self::LEVEL);
        if level.is_nan() {
            // An exhausted level stream holds the previous value and *also* fires `doneAction`
            // (scsynth's `if (sc_isnan(x)) { x = prevout; DoneAction(...); }`).
            done = done.max(DoneAction::from_code(ins.control(Self::DONE)));
        } else {
            self.level = level;
        }
        done
    }

    /// One frame of scsynth's calc: the reset stage, a refill when the count has elapsed, then the
    /// held level. Returns the frame's output and any done action.
    fn frame(
        &mut self,
        frame: usize,
        ins: &Inputs<'_>,
        demand: &mut DemandAccess<'_>,
        world: &mut DemandWorld<'_, '_>,
        sr: f32,
    ) -> (f32, DoneAction) {
        reset_stage(
            self.reset_mode,
            frame,
            &mut self.count,
            &mut self.prev_reset,
            ins,
            demand,
            world,
            (Self::RESET, Self::LEVEL, Self::DUR),
            sr,
        );
        let mut done = DoneAction::Nothing;
        if self.count <= 0.0 {
            done = self.refill(ins, demand, world, sr);
        }
        self.count -= 1.0;
        (self.level, done)
    }
}

impl Unit for Duty {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `Duty_Ctor`: a demand-rate reset pulls its first duration, then the first duration and
        // level are demanded and the level written, without running the calc. These pulls fire no
        // `doneAction`: an exhausted duration freezes the unit (a `NaN` count) and an exhausted
        // level is written as it is.
        let frame_rate = frame_rate(ctx, self.audio != 0);
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            buf_counter: ctx.buf_counter,
            rgen: &mut *ctx.rgen,
            fft: ctx.fft,
        };
        self.prev_reset = initial_reset(
            self.reset_mode,
            &ctx.ins,
            &mut ctx.demand,
            &mut world,
            Self::RESET,
            frame_rate,
        );
        let dur = demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::DUR);
        self.count = (dur as f64 * frame_rate) as f32;
        self.level = demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::LEVEL);
        *ctx.outs.control(0) = self.level;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let sr = frame_rate(ctx, self.audio != 0) as f32;
        // The demand sources' world reach, built once from disjoint `ctx` fields (buffers/node_msgs);
        // the per-sample output borrow (`ctx.outs`) and the pull borrows (`ctx.ins`/`ctx.demand`) are
        // all separate fields, so they coexist.
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            buf_counter: ctx.buf_counter,
            rgen: &mut *ctx.rgen,
            fft: ctx.fft,
        };
        if self.audio != 0 {
            let out = ctx.outs.audio(0);
            for (i, o) in out.iter_mut().enumerate() {
                let (x, action) = self.frame(i, &ctx.ins, &mut ctx.demand, &mut world, sr);
                *o = x;
                done = done.max(action);
            }
        } else {
            let (x, action) = self.frame(0, &ctx.ins, &mut ctx.demand, &mut world, sr);
            *ctx.outs.control(0) = x;
            done = action;
        }
        done
    }
}

/// Constructor for [`Duty`]: selects how `reset` is read from its rate.
pub struct DutyCtor;

impl UnitDef for DutyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(Duty {
            count: 0.0,
            level: 0.0,
            prev_reset: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
            reset_mode: reset_mode(ctx, Duty::RESET),
        }))
    }
}

/// `TDuty.kr/ar(dur, reset, level, doneAction, gapFirst)`: a self-clocking *trigger* sequencer.
/// Like [`Duty`] it counts down `dur` seconds demanded from its `dur` source, but at each boundary
/// it emits the demanded `level` for a single frame (a one-frame impulse) and `0` in between,
/// rather than holding the level. `gapFirst = 0` fires the first impulse immediately; a non-zero
/// `gapFirst` waits one demanded duration before it. An exhausted (`NaN`) duration fires
/// `doneAction` once and freezes the unit at `0` until a reset revives it; a `NaN` level emits `0`.
/// `reset` is read by its rate exactly as [`Duty`]'s is (scsynth's `TDuty_next_dk`/`_da`/`_dd`).
///
/// Input order matches [`Duty`] with `gapFirst` appended: `[dur, reset, doneAction, level,
/// gapFirst]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TDuty {
    /// Frames remaining until the next demand (fractional remainder preserved for sample accuracy).
    count: f32,
    /// The previous `reset` value for edge detection, or with a demand-rate reset the frames
    /// remaining until the next reset.
    prev_reset: f32,
    /// `0`/`1`: control-rate (one value per block) vs audio-rate (a full block).
    audio: u32,
    /// Non-zero if the first impulse waits one demanded duration (scsynth's `gapFirst`).
    gap_first: u32,
    /// How `reset` is read: [`RESET_CONTROL`], [`RESET_AUDIO`] or [`RESET_DEMAND`].
    reset_mode: u32,
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
        sr: f32,
    ) -> (f32, DoneAction) {
        let mut done = DoneAction::Nothing;
        // As in [`Duty::refill`], an exhausted dur stream poisons the count so the unit freezes
        // (emitting `0`) after firing `doneAction` once; a reset revives it. Unlike `Duty`,
        // scsynth's `TDuty_Ctor` polls nothing up front, so the very first boundary fires
        // `doneAction` too.
        self.count += demand_next(ins, demand, world, Self::DUR) * sr;
        if self.count.is_nan() {
            done = DoneAction::from_code(ins.control(Self::DONE));
        }
        let level = demand_next(ins, demand, world, Self::LEVEL);
        (if level.is_nan() { 0.0 } else { level }, done)
    }

    /// One frame of scsynth's calc: the reset stage, then an impulse when the count has elapsed.
    fn frame(
        &mut self,
        frame: usize,
        ins: &Inputs<'_>,
        demand: &mut DemandAccess<'_>,
        world: &mut DemandWorld<'_, '_>,
        sr: f32,
    ) -> (f32, DoneAction) {
        reset_stage(
            self.reset_mode,
            frame,
            &mut self.count,
            &mut self.prev_reset,
            ins,
            demand,
            world,
            (Self::RESET, Self::LEVEL, Self::DUR),
            sr,
        );
        let out = if self.count <= 0.0 {
            self.fire(ins, demand, world, sr)
        } else {
            (0.0, DoneAction::Nothing)
        };
        self.count -= 1.0;
        out
    }
}

impl Unit for TDuty {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `TDuty_Ctor` writes 0 without running the calc. A demand-rate reset pulls its first
        // duration; then a `gapFirst` synth demands one duration up front, delaying the first
        // impulse by it. A duration already exhausted here freezes the unit (a `NaN` count) without
        // firing `doneAction`.
        let frame_rate = frame_rate(ctx, self.audio != 0);
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            buf_counter: ctx.buf_counter,
            rgen: &mut *ctx.rgen,
            fft: ctx.fft,
        };
        self.prev_reset = initial_reset(
            self.reset_mode,
            &ctx.ins,
            &mut ctx.demand,
            &mut world,
            Self::RESET,
            frame_rate,
        );
        if self.gap_first != 0 {
            let dur = demand_next(&ctx.ins, &mut ctx.demand, &mut world, Self::DUR);
            self.count = (dur as f64 * frame_rate) as f32;
        }
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let mut done = DoneAction::Nothing;
        let sr = frame_rate(ctx, self.audio != 0) as f32;
        let mut world = DemandWorld {
            buffers: &mut *ctx.buffers,
            local_bufs: &mut ctx.local_bufs,
            node_id: ctx.node_id,
            node_msgs: &mut ctx.node_msgs,
            buf_counter: ctx.buf_counter,
            rgen: &mut *ctx.rgen,
            fft: ctx.fft,
        };
        if self.audio != 0 {
            let out = ctx.outs.audio(0);
            for (i, o) in out.iter_mut().enumerate() {
                let (x, action) = self.frame(i, &ctx.ins, &mut ctx.demand, &mut world, sr);
                *o = x;
                done = done.max(action);
            }
        } else {
            let (x, action) = self.frame(0, &ctx.ins, &mut ctx.demand, &mut world, sr);
            *ctx.outs.control(0) = x;
            done = action;
        }
        done
    }
}

/// Constructor for [`TDuty`]: bakes the `gapFirst` flag from its constant input and selects how
/// `reset` is read from its rate.
pub struct TDutyCtor;

impl UnitDef for TDutyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let gap_first = ctx.const_input(TDuty::GAP_FIRST).unwrap_or(0.0) != 0.0;
        Ok(unit_spec(TDuty {
            count: 0.0,
            prev_reset: 0.0,
            audio: (ctx.rate == Rate::Audio) as u32,
            gap_first: gap_first as u32,
            reset_mode: reset_mode(ctx, TDuty::RESET),
        }))
    }
}
