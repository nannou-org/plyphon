//! `SinOsc` - a wavetable sine oscillator, plyphon's port of scsynth's `SinOsc`.

use core::f32::consts::TAU;

use crate::bus::AudioBus;
use crate::error::BuildError;
use crate::rate::Rate;
use crate::ugen::registry::{BuildContext, UgenCtor};
use crate::ugen::{DoneAction, Inputs, Outputs, ProcessContext, Ugen};
use crate::wavetable::lookup_cycle;

/// Which calc variant to use, chosen from the frequency input's rate at build time (scsynth picks
/// one of `SinOsc_next_i{k,a}{k,a}`; we branch on the freq rate once per block, not per sample).
#[derive(Copy, Clone, Debug)]
enum Calc {
    /// Frequency is constant or control-rate (one value per block).
    FreqControl,
    /// Frequency is audio-rate (one value per sample).
    FreqAudio,
}

/// `SinOsc.ar(freq, phase)`: a sine read from the shared wavetable via a normalised phase
/// accumulator. `phase` is a phase offset in radians.
pub struct SinOsc {
    calc: Calc,
    /// Normalised phase accumulator in cycles, kept in `[0, 1)`.
    phase: f32,
}

impl SinOsc {
    const FREQ: usize = 0;
    const PHASE: usize = 1;
}

impl Ugen for SinOsc {
    fn process(
        &mut self,
        ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        _out_bus: &mut AudioBus,
    ) -> DoneAction {
        let table = ctx.wavetables.sine();
        let sample_dur = ctx.audio.sample_dur as f32;
        // Phase offset in cycles (radians / 2pi). Constant/control rate for now.
        let phase_offset = ins.control(Self::PHASE) / TAU;
        let out = outs.audio(0);
        match self.calc {
            Calc::FreqControl => {
                let inc = ins.control(Self::FREQ) * sample_dur;
                for o in out.iter_mut() {
                    *o = lookup_cycle(table, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + inc);
                }
            }
            Calc::FreqAudio => {
                let freq = ins.audio(Self::FREQ);
                for (o, &f) in out.iter_mut().zip(freq) {
                    *o = lookup_cycle(table, self.phase + phase_offset);
                    self.phase = wrap_unit(self.phase + f * sample_dur);
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

impl UgenCtor for SinOscCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<Box<dyn Ugen>, BuildError> {
        let calc = match ctx.input_rates.first().copied() {
            Some(Rate::Audio) => Calc::FreqAudio,
            _ => Calc::FreqControl,
        };
        Ok(Box::new(SinOsc { calc, phase: 0.0 }))
    }
}
