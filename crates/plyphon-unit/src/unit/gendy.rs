//! `Gendy1` - plyphon's port of Xenakis's dynamic stochastic synthesis (scsynth's
//! `GendynUGens.cpp`).
//!
//! The oscillator walks a set of breakpoints (control points). Each has an amplitude and a
//! duration; both drift by a bounded random step drawn from one of seven distributions every time
//! the oscillator reaches the point, and the waveform is the linear interpolation between
//! successive points. The random steps come from the synth's shared random stream, so two
//! instances of the same def decorrelate and one instance replays exactly under the same seed.
//!
//! The breakpoint arrays live in the unit's [`aux`](crate::unit::Aux) memory, sized at compile
//! time from the constant `initCPs` input (scsynth `RTAlloc`s them at construction).

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::math;
use plyphon_dsp::rng::Rng;

/// Xenakis's random-walk step distributions, selected by the integer `ampdist`/`durdist` inputs.
/// `a` is the distribution parameter (clamped to `[0.0001, 1]`), `f` a uniform `[0, 1)` draw; the
/// result is a step in roughly `[-1, 1]`. A direct port of scsynth's `Gendyn_distribution`.
fn distribution(which: i32, a: f32, f: f32) -> f32 {
    let a = a.clamp(0.0001, 1.0);
    match which {
        // LINEAR: the uniform draw itself, mapped to bipolar.
        0 => 2.0 * f - 1.0,
        // CAUCHY.
        1 => {
            let c = math::atan(10.0 * a);
            (1.0 / a) * math::tan(c * (2.0 * f - 1.0)) * 0.1
        }
        // LOGIST.
        2 => {
            let c = 0.5 + (0.499 * a);
            let c = math::ln((1.0 - c) / c);
            let f = ((f - 0.5) * 0.998 * a) + 0.5;
            math::ln((1.0 - f) / f) / c
        }
        // HYPERBCOS.
        3 => {
            let c = math::tan(1.5692255 * a);
            let temp = math::tan(1.5692255 * a * f) / c;
            let temp = math::ln(temp * 0.999 + 0.001) * (-0.1447648);
            2.0 * temp - 1.0
        }
        // ARCSINE.
        4 => {
            let c = math::sin(core::f32::consts::FRAC_PI_2 * a);
            math::sin(core::f32::consts::PI * (f - 0.5) * a) / c
        }
        // EXPON.
        5 => {
            let c = math::ln(1.0 - (0.999 * a));
            let temp = math::ln(1.0 - (f * 0.999 * a)) / c;
            2.0 * temp - 1.0
        }
        // SINUS: the parameter alone (the driving oscillator is not modelled here).
        6 => 2.0 * a - 1.0,
        _ => 2.0 * f - 1.0,
    }
}

/// Fold `amp` back into `[-1, 1]` by reflection (scsynth's amplitude wrap).
fn fold_amp(mut amp: f32) -> f32 {
    if !(-1.0..=1.0).contains(&amp) {
        if amp < 0.0 {
            amp += 4.0;
        }
        amp %= 4.0;
        if (1.0..3.0).contains(&amp) {
            amp = 2.0 - amp;
        } else if amp > 1.0 {
            amp -= 4.0;
        }
    }
    amp
}

/// Fold `dur` back toward `[0, 1]` by reflection (scsynth's duration wrap).
fn fold_dur(mut dur: f32) -> f32 {
    if !(0.0..=1.0).contains(&dur) {
        if dur < 0.0 {
            dur += 2.0;
        }
        dur %= 2.0;
        dur = 2.0 - dur;
    }
    dur
}

/// `Gendy1(ampdist, durdist, adparam, ddparam, minfreq, maxfreq, ampscale, durscale, initCPs,
/// knum)`: an audio-rate dynamic-stochastic oscillator.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Gendy1 {
    /// Interpolation phase between the current and next breakpoint; wraps at `1.0`. Starts at `1.0`
    /// so the first sample immediately computes a breakpoint.
    phase: f64,
    /// Amplitude of the breakpoint the oscillator is leaving.
    amp: f32,
    /// Amplitude of the breakpoint it is heading toward.
    next_amp: f32,
    /// Phase increment per sample, from the current breakpoint's duration and the frequency range.
    speed: f32,
    /// Index of the current breakpoint within the memory arrays.
    index: u32,
    /// Number of breakpoints the memory arrays hold (the built `initCPs`).
    memory_size: u32,
    /// `0` until the first block seeds the breakpoint arrays from the shared stream.
    seeded: u32,
}

impl Gendy1 {
    const AMPDIST: usize = 0;
    const DURDIST: usize = 1;
    const ADPARAM: usize = 2;
    const DDPARAM: usize = 3;
    const MINFREQ: usize = 4;
    const MAXFREQ: usize = 5;
    const AMPSCALE: usize = 6;
    const DURSCALE: usize = 7;
    const INIT_CPS: usize = 8;
    const KNUM: usize = 9;

    /// Seed the breakpoint arrays: amplitudes uniform in `[-1, 1)`, durations in `[0, 1)`.
    fn seed(rng: &mut Rng, amp_mem: &mut [f32], dur_mem: &mut [f32]) {
        for a in amp_mem.iter_mut() {
            *a = 2.0 * rng.next_unipolar() - 1.0;
        }
        for d in dur_mem.iter_mut() {
            *d = rng.next_unipolar();
        }
    }
}

impl Unit for Gendy1 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let memory_size = self.memory_size as usize;
        let which_amp = ctx.ins.control(Self::AMPDIST) as i32;
        let which_dur = ctx.ins.control(Self::DURDIST) as i32;
        let aamp = ctx.ins.control(Self::ADPARAM);
        let adur = ctx.ins.control(Self::DDPARAM);
        let minfreq = ctx.ins.control(Self::MINFREQ);
        let maxfreq = ctx.ins.control(Self::MAXFREQ);
        let scaleamp = ctx.ins.control(Self::AMPSCALE);
        let scaledur = ctx.ins.control(Self::DURSCALE);
        // `knum` limits how many of the allocated breakpoints are active; out of range means all.
        let knum = ctx.ins.control(Self::KNUM) as i32;
        let num = if knum < 1 || knum as usize > memory_size {
            memory_size
        } else {
            knum as usize
        };
        let freq_mul = ctx.own.sample_dur as f32;

        let (amp_mem, dur_mem) = ctx.aux.f32_mut().split_at_mut(memory_size);
        if self.seeded == 0 {
            Gendy1::seed(ctx.rgen, amp_mem, dur_mem);
            self.seeded = 1;
        }

        let mut phase = self.phase;
        let mut amp = self.amp;
        let mut next_amp = self.next_amp;
        let mut speed = self.speed;
        let mut index = self.index as usize;
        for slot in ctx.outs.audio(0).iter_mut() {
            if phase >= 1.0 {
                phase -= 1.0;
                index = (index + 1) % num;
                amp = next_amp;
                next_amp = fold_amp(
                    amp_mem[index]
                        + scaleamp * distribution(which_amp, aamp, ctx.rgen.next_unipolar()),
                );
                amp_mem[index] = next_amp;
                let rate = fold_dur(
                    dur_mem[index]
                        + scaledur * distribution(which_dur, adur, ctx.rgen.next_unipolar()),
                );
                dur_mem[index] = rate;
                speed = (minfreq + (maxfreq - minfreq) * rate) * freq_mul * num as f32;
            }
            *slot = ((1.0 - phase) as f32 * amp) + (phase as f32 * next_amp);
            phase += speed as f64;
        }

        self.phase = phase;
        self.amp = amp;
        self.next_amp = next_amp;
        self.speed = speed;
        self.index = index as u32;
        DoneAction::Nothing
    }
}

/// Constructor for [`Gendy1`]: sizes the breakpoint arrays from the constant `initCPs` input.
pub struct Gendy1Ctor;

impl UnitDef for Gendy1Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Gendy1::KNUM {
            return Err(BuildError::WrongInputCount);
        }
        let memory_size = ctx
            .const_input(Gendy1::INIT_CPS)
            .map(|cps| (cps as i32).max(1) as usize)
            .unwrap_or(12);
        Ok(unit_spec_aux(
            Gendy1 {
                phase: 1.0,
                amp: 0.0,
                next_amp: 0.0,
                speed: 100.0,
                index: 0,
                memory_size: memory_size as u32,
                seeded: 0,
            },
            2 * memory_size * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}
