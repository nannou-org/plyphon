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
/// entirely on the audio thread with no control-plane messages. A `NaN` duration triggers
/// `doneAction`; a `NaN` level holds the previous value. A rising `reset` resets the `dur`/`level`
/// sources and restarts the count.
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
    _pad: u32,
}

impl Duty {
    const DUR: usize = 0;
    const RESET: usize = 1;
    const LEVEL: usize = 2;
    const DONE: usize = 3;

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
            done = DoneAction::from_code(ins.control(Self::DONE));
        } else {
            self.count += dur as f64 * frame_rate;
        }
        let level = demand_next(ins, demand, world, Self::LEVEL);
        if !level.is_nan() {
            self.level = level;
        }
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
            _pad: 0,
        }))
    }
}
