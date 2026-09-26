//! `Vibrato` - a randomised vibrato for a frequency input, plyphon's port of scsynth's `Vibrato`
//! (`LFUGens.cpp`).
//!
//! Each vibrato cycle is two parabolic half-waves, an upper one scaled by `scaleA` and a lower one
//! by `scaleB`. At every cycle boundary the rate and both depths are drawn afresh, varied by
//! `rateVariation`/`depthVariation` with `frand2` draws from the synth's random stream
//! ([`ProcessCtx::rgen`]). A `delay` passes the input through unmodulated, and an `onset` then
//! fades the depth in linearly.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::sig;
use crate::unit::{BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::ops;
use plyphon_dsp::rng::Rng;

/// `Vibrato.ar/kr(freq, rate, depth, delay, onset, rateVariation, depthVariation, iphase, trig)`:
/// modulates `freq` by a vibrato of `rate` Hz and `depth` (a ratio: `0.02` is ±2%), after `delay`
/// seconds and a linear `onset` fade-in, randomising the rate and depth of every cycle by
/// `rateVariation` and `depthVariation`. `iphase` sets the starting phase (`0..1`), and a rising
/// `trig` restarts the vibrato - phase, delay and onset - with fresh draws.
///
/// A direct port of scsynth's `Vibrato_Ctor` and `Vibrato_next`. The phase runs over `[-1, 3)` in
/// steps of `4 * rate / SAMPLERATE`, in double precision; the rest is single precision, as in
/// scsynth.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vibrato {
    /// The vibrato phase, over `[-1, 3)` (scsynth's `mPhase`).
    phase: f64,
    /// The onset's per-sample level increment, `1 / (1 + attack)`.
    attack_slope: f64,
    /// The onset's current depth scale.
    attack_level: f64,
    /// `4 * SAMPLEDUR`, turning a rate in Hz into a phase increment.
    freq_mul: f32,
    /// The upper half-wave's depth.
    scale_a: f32,
    /// The lower half-wave's depth.
    scale_b: f32,
    /// The current phase increment (scsynth's `mFreq`).
    freq: f32,
    /// Samples of `delay` left.
    delay: i32,
    /// Samples of `onset` left.
    attack: i32,
    /// The previous `trig` value, for rising-edge detection.
    trig: f32,
    _pad: u32,
}

impl Vibrato {
    const IN: usize = 0;
    const RATE: usize = 1;
    const DEPTH: usize = 2;
    const DELAY: usize = 3;
    const ONSET: usize = 4;
    const RATE_VARIATION: usize = 5;
    const DEPTH_VARIATION: usize = 6;
    const IPHASE: usize = 7;
    const TRIG: usize = 8;

    /// `4.0 * sc_wrap(iphase, 0, 1) - 1.0`: the phase `iphase` starts a cycle at.
    fn start_phase(ins: &Inputs<'_>) -> f64 {
        4.0 * ops::wrap(ins.control(Self::IPHASE), 0.0, 1.0) as f64 - 1.0
    }

    /// The three per-cycle draws, in scsynth's order: the phase increment, then `scaleA`, then
    /// `scaleB`, each `x * (1 + variation * frand2())`.
    fn draw(&self, ins: &Inputs<'_>, rgen: &mut Rng) -> (f32, f32, f32) {
        let rate = ins.control(Self::RATE) * self.freq_mul;
        let depth = ins.control(Self::DEPTH);
        let rate_variation = ins.control(Self::RATE_VARIATION);
        let depth_variation = ins.control(Self::DEPTH_VARIATION);
        let freq = rate * (1.0 + rate_variation * rgen.next_bipolar());
        let scale_a = depth * (1.0 + depth_variation * rgen.next_bipolar());
        let scale_b = depth * (1.0 + depth_variation * rgen.next_bipolar());
        (freq, scale_a, scale_b)
    }

    /// (Re)start the vibrato: the start phase, fresh draws, and the delay and onset counts
    /// (`(int)(seconds * SAMPLERATE)`).
    fn start(&mut self, ins: &Inputs<'_>, rgen: &mut Rng, sample_rate: f64) {
        self.phase = Self::start_phase(ins);
        let (freq, scale_a, scale_b) = self.draw(ins, rgen);
        self.freq = freq;
        self.scale_a = scale_a;
        self.scale_b = scale_b;
        self.delay = (ins.control(Self::DELAY) as f64 * sample_rate) as i32;
        self.attack = (ins.control(Self::ONSET) as f64 * sample_rate) as i32;
        self.attack_slope = 1.0 / 1i32.wrapping_add(self.attack) as f64;
        self.attack_level = self.attack_slope;
    }
}

/// Which of `Vibrato_next`'s three sections a block continues into.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Section {
    Attack,
    Normal,
    Done,
}

impl Unit for Vibrato {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `Vibrato_Ctor`: seed the state (drawing from the stream), run the calc for one sample,
        // then put back the phase, rate, depths and trigger - but not the delay and onset counts,
        // which keep the constructor sample's step.
        self.freq_mul = (4.0 * ctx.own.sample_dur) as f32;
        self.start(&ctx.ins, ctx.rgen, ctx.own.sample_rate);
        self.trig = 0.0;
        let (phase, freq, scale_a, scale_b) = (self.phase, self.freq, self.scale_a, self.scale_b);
        let action = self.process(ctx);
        self.phase = phase;
        self.freq = freq;
        self.scale_a = scale_a;
        self.scale_b = scale_b;
        self.trig = 0.0;
        action
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ProcessCtx {
            ins,
            outs,
            rgen,
            own,
            ..
        } = ctx;
        let ins = *ins;
        let cur_trig = ins.control(Self::TRIG);
        if self.trig <= 0.0 && cur_trig > 0.0 {
            self.freq_mul = (4.0 * own.sample_dur) as f32;
            self.start(&ins, rgen, own.sample_rate);
        }
        self.trig = cur_trig;

        let input = sig(&ins, Self::IN);
        let out = outs.audio(0);
        let len = out.len();
        let mut ffreq = self.freq as f64;
        let mut phase = self.phase;
        let mut scale_a = self.scale_a;
        let mut scale_b = self.scale_b;
        // `inNumSamples` as scsynth counts it down, and the next sample to write.
        let mut remaining = len as i32;
        let mut i = 0;

        let mut section = if self.delay > 0 {
            let remain = remaining.min(self.delay);
            self.delay -= remain;
            remaining -= remain;
            for _ in 0..remain {
                out[i] = input.at(i);
                i += 1;
            }
            if self.delay <= 0 && remaining > 0 {
                if self.attack > 0 {
                    Section::Attack
                } else {
                    Section::Normal
                }
            } else {
                Section::Done
            }
        } else if self.attack != 0 {
            Section::Attack
        } else {
            Section::Normal
        };

        if section == Section::Attack {
            let remain = remaining.min(self.attack);
            self.attack -= remain;
            remaining -= remain;
            let attack_slope = self.attack_slope;
            let mut attack_level = self.attack_level;
            // A negative count (a negative `onset`) runs no samples, as scsynth's `LOOP` does.
            for _ in 0..remain.max(0) {
                let x = input.at(i);
                let level = attack_level as f32;
                out[i] = if phase < 1.0 {
                    let z = phase as f32;
                    x * (1.0 + level * scale_a * (1.0 - z * z))
                } else if phase < 3.0 {
                    let z = (phase - 2.0) as f32;
                    x * (1.0 + level * scale_b * (z * z - 1.0))
                } else {
                    phase -= 4.0;
                    let z = phase as f32;
                    let (f, a, b) = self.draw(&ins, rgen);
                    ffreq = f as f64;
                    scale_a = a;
                    scale_b = b;
                    x * (1.0 + level * scale_a * (1.0 - z * z))
                };
                phase += ffreq;
                attack_level += attack_slope;
                i += 1;
            }
            self.attack_level = attack_level;
            section = if self.attack <= 0 && remaining > 0 {
                Section::Normal
            } else {
                Section::Done
            };
        }

        if section == Section::Normal {
            // scsynth runs `remaining` samples here. Only after a negative onset can that exceed
            // the block, where scsynth writes past its buffers; plyphon stops at the block's end.
            for _ in 0..(remaining as usize).min(len - i) {
                let x = input.at(i);
                out[i] = if phase < 1.0 {
                    let z = phase as f32;
                    x * (1.0 + scale_a * (1.0 - z * z))
                } else if phase < 3.0 {
                    let z = (phase - 2.0) as f32;
                    x * (1.0 + scale_b * (z * z - 1.0))
                } else {
                    phase -= 4.0;
                    let z = phase as f32;
                    let (f, a, b) = self.draw(&ins, rgen);
                    ffreq = f as f64;
                    scale_a = a;
                    scale_b = b;
                    x * (1.0 + scale_a * (1.0 - z * z))
                };
                phase += ffreq;
                i += 1;
            }
        }

        self.phase = phase;
        self.freq = ffreq as f32;
        self.scale_a = scale_a;
        self.scale_b = scale_b;
        DoneAction::Nothing
    }
}

/// Constructor for [`Vibrato`].
pub struct VibratoCtor;

impl UnitDef for VibratoCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 9 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Vibrato {
            phase: 0.0,
            attack_slope: 0.0,
            attack_level: 0.0,
            freq_mul: 0.0,
            scale_a: 0.0,
            scale_b: 0.0,
            freq: 0.0,
            delay: 0,
            attack: 0,
            trig: 0.0,
            _pad: 0,
        }))
    }
}
