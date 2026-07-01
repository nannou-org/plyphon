//! `PitchShift` - plyphon's port of scsynth's granular pitch shifter (`DelayUGens.cpp`).
//!
//! A time-domain pitch shifter: the input is written into a delay line ([aux memory](crate::unit::Aux),
//! sized at build time from the constant `windowSize`) and read back by **four** overlapping
//! triangular-windowed grains, each 90 degrees out of phase. Each grain's read head drifts against the
//! write head at a rate set by `pitchRatio` (so the grain replays the recent past faster or slower =
//! transposed), and every quarter-window a fresh grain is spawned round-robin, crossfading over the one
//! it replaces. `pitchDispersion`/`timeDispersion` jitter each grain's pitch and start position from a
//! per-unit [`Rng`]. The four windowed reads are summed and halved.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::interp::lininterp;
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::rng::Rng;

const IN: usize = 0;
const WINSIZE: usize = 1;
const PITCHRATIO: usize = 2;
const PITCHDISP: usize = 3;
const TIMEDISP: usize = 4;

/// `PitchShift.ar(in, windowSize, pitchRatio, pitchDispersion, timeDispersion)`: a granular pitch
/// shifter. `pitchRatio` transposes (2 = up an octave), `windowSize` the grain length (a longer window
/// smears transients less but blurs pitch), and the dispersions add per-grain pitch/time jitter.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PitchShift {
    /// Per-unit RNG for the pitch/time dispersion (scsynth's graph `RGen`).
    rng: Rng,
    /// Each grain's fractional read distance behind the write head (`dsamp1..4`).
    dsamp: [f32; 4],
    /// Each grain's per-sample change in read distance (`dsamp*_slope`) = `1 - pitchRatio`.
    dsamp_slope: [f32; 4],
    /// Each grain's triangular-window amplitude (`ramp1..4`).
    ramp: [f32; 4],
    /// Each grain's per-sample window change (`ramp*_slope`), `+/- slope`.
    ramp_slope: [f32; 4],
    /// The window ramp rate `2 / framesize`.
    slope: f32,
    /// `len - 1`, the wrap mask for the power-of-two delay line.
    mask: u32,
    /// The window length in samples, a multiple of 4 (`framesize`).
    framesize: u32,
    /// Samples until the next grain spawns (`counter`); reset to `framesize / 4` each spawn.
    counter: i32,
    /// Which of the four grains was spawned last (`stage`, 0..3).
    stage: u32,
    /// Monotonic write phase (`m_iwrphase`), masked into the line.
    iwrphase: u32,
    /// `1` until the first block has zeroed the (dirty) delay line.
    first: u32,
}

impl Unit for PitchShift {
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let ins = ctx.ins;
        let in_audio = (ins.rate(IN) == Rate::Audio).then(|| ins.audio(IN));
        let in_ctrl = ins.control(IN);
        let pchratio = ins.control(PITCHRATIO);
        let winsize = ins.control(WINSIZE);
        let pchdisp = ins.control(PITCHDISP);
        let timedisp = ins.control(TIMEDISP).max(0.0).min(winsize) * sr;

        let out = ctx.outs.audio(0);
        let buf = ctx.aux.f32_mut();
        if buf.is_empty() {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        // The line is dirty aux memory; zero it once so unwritten taps read 0 (scsynth's cold-start
        // `_z` guard, achieved here by clearing the whole line on the first block).
        if self.first != 0 {
            buf.fill(0.0);
            self.first = 0;
        }

        let mask = self.mask;
        let framesize = self.framesize;
        let slope = self.slope;
        let mut iwrphase = self.iwrphase;
        let mut counter = self.counter;
        let mut stage = self.stage;
        let mut dsamp = self.dsamp;
        let mut dsamp_slope = self.dsamp_slope;
        let mut ramp = self.ramp;
        let mut ramp_slope = self.ramp_slope;

        for (i, o) in out.iter_mut().enumerate() {
            if counter <= 0 {
                // Spawn the next grain, round-robin, crossfading over the grain two stages back.
                counter = (framesize >> 2) as i32;
                stage = (stage + 1) & 3;
                let mut disppchratio = pchratio;
                if pchdisp != 0.0 {
                    disppchratio += pchdisp * self.rng.next_bipolar();
                }
                disppchratio = disppchratio.clamp(0.0, 4.0);
                let pchratio1 = disppchratio - 1.0;
                let samp_slope = -pchratio1;
                let mut startpos = if pchratio1 < 0.0 {
                    2.0
                } else {
                    framesize as f32 * pchratio1 + 2.0
                };
                startpos += timedisp * self.rng.next_unipolar();
                let s = stage as usize;
                dsamp_slope[s] = samp_slope;
                dsamp[s] = startpos;
                ramp[s] = 0.0;
                ramp_slope[s] = slope;
                // The grain two ahead (== the one fading out) reverses its ramp.
                ramp_slope[(s + 2) & 3] = -slope;
            }

            iwrphase = (iwrphase + 1) & mask;
            let mut value = 0.0f32;
            for k in 0..4 {
                dsamp[k] += dsamp_slope[k];
                let idsamp = dsamp[k] as i64;
                let frac = dsamp[k] - idsamp as f32;
                let irdphase = ((iwrphase as i64 - idsamp) as u32) & mask;
                let irdphaseb = irdphase.wrapping_sub(1) & mask;
                let d1 = buf[irdphase as usize];
                let d2 = buf[irdphaseb as usize];
                value += lininterp(frac, d1, d2) * ramp[k];
                ramp[k] += ramp_slope[k];
            }
            buf[iwrphase as usize] = in_audio.map_or(in_ctrl, |sig| sig[i]);
            *o = value * 0.5;
            counter -= 1;
        }

        self.iwrphase = iwrphase;
        self.counter = counter;
        self.stage = stage;
        self.dsamp = dsamp;
        self.dsamp_slope = dsamp_slope;
        self.ramp = ramp;
        self.ramp_slope = ramp_slope;
        DoneAction::Nothing
    }
}

/// Constructor for [`PitchShift`]. Sizes the delay line from the constant `windowSize`.
pub struct PitchShiftCtor;

impl UnitDef for PitchShiftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 5 {
            return Err(BuildError::WrongInputCount);
        }
        let sr = ctx.audio.sample_rate;
        let block = ctx.audio.block_size;
        // `windowSize` sizes the line, so it must be a compile-time constant; clamp to scsynth's
        // 3-sample minimum (below which its own delay maths misbehaves).
        let winsize = ctx
            .const_input(WINSIZE)
            .ok_or(BuildError::AuxRequiresConstant { input: WINSIZE })?
            .max(3.0 / sr as f32);
        // The line holds three windows plus a little headroom, rounded up to a power of two.
        let base = math::ceil(winsize as f64 * sr * 3.0 + 3.0) as i64;
        let len = ((base + block as i64).max(1) as u64).next_power_of_two() as u32;
        let framesize = (((winsize as f64 * sr) as i64 + 2) & !3) as u32;
        let slope = 2.0 / framesize as f32;
        let aux_bytes = len as usize * core::mem::size_of::<f32>();
        Ok(unit_spec_aux(
            PitchShift {
                rng: Rng::new(0),
                dsamp: [2.0; 4],
                dsamp_slope: [0.0; 4],
                ramp: [0.5, 1.0, 0.5, 0.0],
                ramp_slope: [-slope, -slope, slope, slope],
                slope,
                mask: len - 1,
                framesize,
                counter: (framesize >> 2) as i32,
                stage: 3,
                iwrphase: 0,
                first: 1,
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}
