//! `SinOsc` - a wavetable sine oscillator, plyphon's port of scsynth's `SinOsc`.

use core::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;
use plyphon_dsp::wavetable::lookup_cycle;

/// Calc-variant tags, chosen from the frequency input's rate at build time (scsynth picks one of
/// `SinOsc_next_i{k,a}{k,a}`; we branch on the freq rate once per block, not per sample). Stored as a
/// `u32` so the state is [`Pod`] and lives in the rt-pool.
mod calc {
    /// Frequency is constant or control-rate (one value per block).
    pub const FREQ_CONTROL: u32 = 0;
    /// Frequency is audio-rate (one value per sample).
    pub const FREQ_AUDIO: u32 = 1;
}

/// `SinOsc.ar(freq, phase)`: a sine read from the shared wavetable via a normalised phase
/// accumulator. `phase` is a phase offset in radians.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SinOsc {
    /// Normalised phase accumulator in cycles, kept in `[0, 1)`.
    phase: f32,
    /// Which calc variant (see [`calc`]), chosen from the freq input rate at build time.
    calc: u32,
}

impl SinOsc {
    const FREQ: usize = 0;
    const PHASE: usize = 1;
}

impl Unit for SinOsc {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let table = ctx.wavetables.sine();
        let sample_dur = ctx.audio.sample_dur as f32;
        // Phase offset in cycles (radians / 2pi). Constant/control rate for now.
        let phase_offset = ctx.ins.control(Self::PHASE) / TAU;
        match self.calc {
            calc::FREQ_AUDIO => {
                let freq = ctx.ins.audio(Self::FREQ);
                for (o, &f) in ctx.outs.audio(0).iter_mut().zip(freq) {
                    *o = lookup_cycle(table, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + f * sample_dur);
                }
            }
            _ => {
                let inc = ctx.ins.control(Self::FREQ) * sample_dur;
                for o in ctx.outs.audio(0).iter_mut() {
                    *o = lookup_cycle(table, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + inc);
                }
            }
        }
        DoneAction::Nothing
    }
}

/// Wrap a phase in cycles into `[0, 1)`.
#[inline]
fn wrap_unit(x: f32) -> f32 {
    x - x.floor()
}

/// Constructor for [`SinOsc`].
pub struct SinOscCtor;

impl UnitDef for SinOscCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        let calc = match ctx.input_rates.first().copied() {
            Some(Rate::Audio) => calc::FREQ_AUDIO,
            _ => calc::FREQ_CONTROL,
        };
        Ok(unit_spec(SinOsc { phase: 0.0, calc }))
    }
}
