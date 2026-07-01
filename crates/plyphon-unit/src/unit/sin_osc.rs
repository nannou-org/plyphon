//! `SinOsc` (a wavetable sine oscillator) and `FSinOsc` (a fast sine from a resonator) - plyphon's
//! ports of scsynth's `SinOsc` and `FSinOsc`.

use core::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
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
    x - math::floor(x)
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

/// `FSinOsc.ar(freq, iphase)`: a fast sine oscillator built from a two-pole resonator recurrence
/// (`y = 2*cos(w)*y1 - y2`) rather than a table lookup. Cheap, but it can drift in amplitude if
/// `freq` is swept far. `freq` is read at control rate; the coefficient is recomputed when it changes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FSinOsc {
    b1: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    _pad: u32,
}

impl Unit for FSinOsc {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let freq = ctx.ins.control(0);
        let iphase = ctx.ins.control(1) as f64;
        let w = freq as f64 * TAU as f64 * ctx.audio.sample_dur;
        self.b1 = 2.0 * math::cos(w);
        // Seed the recurrence two samples back so it oscillates as `sin(n*w + iphase)`.
        self.y1 = math::sin(iphase - w);
        self.y2 = math::sin(iphase - 2.0 * w);
        self.freq = freq;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(0);
        if freq != self.freq {
            let w = freq as f64 * TAU as f64 * ctx.audio.sample_dur;
            self.b1 = 2.0 * math::cos(w);
            self.freq = freq;
        }
        let b1 = self.b1;
        let (mut y1, mut y2) = (self.y1, self.y2);
        for o in ctx.outs.audio(0).iter_mut() {
            let y0 = b1 * y1 - y2;
            *o = y0 as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = y1;
        self.y2 = y2;
        DoneAction::Nothing
    }
}

/// Constructor for [`FSinOsc`].
pub struct FSinOscCtor;

impl UnitDef for FSinOscCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(FSinOsc {
            b1: 0.0,
            y1: 0.0,
            y2: 0.0,
            freq: 0.0,
            _pad: 0,
        }))
    }
}
