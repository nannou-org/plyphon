//! `Linen` - a gated linear attack/sustain/release envelope, plyphon's port of scsynth's `Linen`
//! (`LFUGens.cpp`).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

/// Where a [`Linen`] is in its envelope - scsynth's `m_stage` values.
mod stage {
    /// Ramping from the current level to `susLevel` over `attackTime`.
    pub const ATTACK: i32 = 0;
    /// Holding the sustain level until the gate falls.
    pub const SUSTAIN: i32 = 1;
    /// Ramping to zero over the release time.
    pub const RELEASE: i32 = 2;
    /// The release just ended: output zero, report done and fire the done action.
    pub const FINISH: i32 = 3;
    /// Idle at zero, waiting for a rising gate.
    pub const IDLE: i32 = 4;
}

/// `Linen.kr(gate, attackTime, susLevel, releaseTime, doneAction)`: on a rising `gate` it ramps
/// linearly from its current level to `susLevel` over `attackTime` seconds and holds there; when
/// `gate` falls to `<= 0` it ramps to zero over `releaseTime`, then marks itself done and fires
/// `doneAction`. A gate `<= -1` releases over `-gate - 1` seconds instead, and a gate `<= -1` at
/// the start releases straight away.
///
/// A direct port of scsynth's `Linen_next_k`: the level is advanced once per calc (so once per
/// control block), and the ramps are counted in samples of the unit's own rate, at least one.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Linen {
    /// The current level (scsynth's `m_level`).
    level: f64,
    /// The per-calc level increment of the current ramp.
    slope: f64,
    /// Calcs left in the current ramp.
    counter: i32,
    /// One of the [`stage`] values.
    stage: i32,
    /// The gate of the previous calc, for rising-edge detection.
    prev_gate: f32,
    _pad: u32,
}

impl Linen {
    const GATE: usize = 0;
    const ATTACK_TIME: usize = 1;
    const SUS_LEVEL: usize = 2;
    const RELEASE_TIME: usize = 3;
    const DONE_ACTION: usize = 4;
}

/// A ramp duration in calcs: `(int)(time * SAMPLERATE)`, at least one.
fn ramp_calcs(time: f32, sample_rate: f64) -> i32 {
    ((time as f64 * sample_rate) as i32).max(1)
}

impl Unit for Linen {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.level = 0.0;
        self.stage = stage::IDLE;
        self.prev_gate = 0.0;
        if ctx.ins.control(Self::GATE) <= -1.0 {
            // Early release.
            self.stage = stage::SUSTAIN;
        }
        self.process(ctx)
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let gate = ctx.ins.control(Self::GATE);
        let sample_rate = ctx.own.sample_rate;
        let mut done = DoneAction::Nothing;

        if self.prev_gate <= 0.0 && gate > 0.0 {
            ctx.done.clear_done();
            self.stage = stage::ATTACK;
            let attack_time = ctx.ins.control(Self::ATTACK_TIME);
            let sus_level = ctx.ins.control(Self::SUS_LEVEL);
            let counter = ramp_calcs(attack_time, sample_rate);
            self.slope = (sus_level as f64 - self.level) / counter as f64;
            self.counter = counter;
        }

        let out = match self.stage {
            stage::ATTACK | stage::RELEASE => {
                let out = self.level as f32;
                self.level += self.slope;
                self.counter -= 1;
                if self.counter == 0 {
                    self.stage += 1;
                }
                out
            }
            stage::SUSTAIN => {
                let out = self.level as f32;
                if gate <= -1.0 {
                    // Cutoff: release over `-gate - 1` seconds.
                    self.stage = stage::RELEASE;
                    let counter = ramp_calcs(-gate - 1.0, sample_rate);
                    self.slope = -self.level / counter as f64;
                    self.counter = counter;
                } else if gate <= 0.0 {
                    self.stage = stage::RELEASE;
                    let counter = ramp_calcs(ctx.ins.control(Self::RELEASE_TIME), sample_rate);
                    self.slope = -self.level / counter as f64;
                    self.counter = counter;
                }
                out
            }
            stage::FINISH => {
                ctx.done.mark_done();
                self.stage += 1;
                done = DoneAction::from_code(ctx.ins.control(Self::DONE_ACTION));
                0.0
            }
            _ => 0.0,
        };
        // scsynth writes only the first sample of the output (`*OUT(0)`).
        *ctx.outs.control(0) = out;
        self.prev_gate = gate;
        done
    }
}

/// Constructor for [`Linen`].
pub struct LinenCtor;

impl UnitDef for LinenCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 5 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Linen {
            level: 0.0,
            slope: 0.0,
            counter: 0,
            stage: stage::IDLE,
            prev_gate: 0.0,
            _pad: 0,
        }))
    }
}
