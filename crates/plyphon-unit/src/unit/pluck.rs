//! `Pluck` - plyphon's port of scsynth's Karplus-Strong plucked string (`DelayUGens.cpp`).
//!
//! `Pluck` is a cubic comb delay (like `CombC`, on per-instance [aux memory](crate::unit::Aux)) with a
//! Karplus-Strong twist: a rising `trig` lets the excitation `in` (usually a noise burst) into the
//! delay line for exactly one delay period, and the feedback path runs the delayed value through a
//! one-zero lowpass (`(1 - |coef|)*value + coef*lastsamp`) - the string damping that makes successive
//! periods progressively duller. It reuses the delay family's read kernel, `sc_CalcFeedback` coefficient
//! and cold-start guard.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::delay::{Interp, calc_feedback, clamp_delay, line_len, read_delayed};
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::rate::Rate;

const IN: usize = 0;
const TRIG: usize = 1;
const MAXDELAY: usize = 2;
const DELAY: usize = 3;
const DECAY: usize = 4;
const COEF: usize = 5;

/// `Pluck.ar(in, trig, maxdelaytime, delaytime, decaytime, coef)`: a plucked string. `in` excites the
/// string (gated in for one delay period on each rising `trig`); `delaytime` sets the pitch period,
/// `decaytime` the ring time, and `coef` the one-zero damping in the feedback loop.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Pluck {
    /// Current delay in samples (`m_dsamp`), possibly fractional and mid-slope.
    dsamp: f32,
    /// The `delaytime`/`decaytime`/`coef` seen last block, to detect a change and slope.
    delaytime: f32,
    decaytime: f32,
    coef: f32,
    /// The recirculation coefficient last block (`m_feedbk`).
    feedbk: f32,
    /// The one-zero lowpass memory (`m_lastsamp`), the unit's output the previous sample.
    lastsamp: f32,
    /// The `trig` value last sample (`m_prevtrig`), for rising-edge detection.
    prevtrig: f32,
    /// Samples of excitation still to admit (`m_inputsamps`); set to one delay period on a trigger.
    inputsamps: u32,
    /// Delay-line length in samples (`m_idelaylen`), a power of two.
    len: u32,
    /// `len - 1`, the wrap mask (`m_mask`).
    mask: u32,
    /// Monotonic write phase (`m_iwrphase`).
    iwrphase: u32,
    /// Samples written so far, saturating at `len` (`m_numoutput`).
    numoutput: u32,
    /// `1` if `trig` is audio-rate (per-sample edge detection), else control (edge at block start).
    trig_audio: u32,
    /// `1` if `coef` is audio-rate (read per sample), else control (slope on change).
    coef_audio: u32,
}

impl Unit for Pluck {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let dt = ctx.ins.control(DELAY);
        let decay = ctx.ins.control(DECAY);
        let min = Interp::Cubic.min_delay(true);
        self.delaytime = dt;
        self.decaytime = decay;
        self.coef = ctx.ins.control(COEF);
        self.dsamp = clamp_delay(dt * ctx.audio.sample_rate as f32, min, self.len as f32);
        self.feedbk = calc_feedback(dt, decay);
    }

    #[allow(clippy::needless_range_loop)]
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let min = Interp::Cubic.min_delay(true);
        let max = self.len as f32;
        let mask = self.mask;
        let mut iwrphase = self.iwrphase;
        let warm = self.numoutput >= self.len;

        let ins = ctx.ins;
        let in_audio = (ins.rate(IN) == Rate::Audio).then(|| ins.audio(IN));
        let in_ctrl = ins.control(IN);
        let trig_audio = self.trig_audio != 0;
        let trig_slice = if trig_audio { ins.audio(TRIG) } else { &[] };
        let trig_ctrl = ins.control(TRIG);
        let delaytime = ins.control(DELAY);
        let decaytime = ins.control(DECAY);
        let coef_audio = self.coef_audio != 0;
        let coef_slice = if coef_audio { ins.audio(COEF) } else { &[] };
        let coef_ctrl = ins.control(COEF);

        let out = ctx.outs.audio(0);
        let buf = ctx.aux.f32_mut();
        if buf.is_empty() {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        let n = out.len();

        // Slope the delay tap, feedback and (control-rate) coef when any of delay/decay/coef changed
        // (scsynth's combined `if` guard), else hold them steady.
        let coef_changed = !coef_audio && coef_ctrl != self.coef;
        let changed = delaytime != self.delaytime || decaytime != self.decaytime || coef_changed;
        let next_dsamp = clamp_delay(delaytime * sr, min, max);
        let next_feedbk = calc_feedback(delaytime, decaytime);
        let (dsamp_slope, feedbk_slope, coef_slope) = if changed {
            let cs = if coef_audio {
                0.0
            } else {
                (coef_ctrl - self.coef) / n as f32
            };
            (
                (next_dsamp - self.dsamp) / n as f32,
                (next_feedbk - self.feedbk) / n as f32,
                cs,
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        let mut dsamp = self.dsamp;
        let mut feedbk = self.feedbk;
        let mut curcoef = self.coef;
        let mut lastsamp = self.lastsamp;
        let mut prevtrig = self.prevtrig;
        let mut inputsamps = self.inputsamps;
        let trig_period = || (delaytime * sr + 0.5).max(0.0) as u32;

        // A control-rate trigger fires at most once, at the block start.
        if !trig_audio {
            if prevtrig <= 0.0 && trig_ctrl > 0.0 {
                inputsamps = trig_period();
            }
            prevtrig = trig_ctrl;
        }

        for i in 0..n {
            if trig_audio {
                let t = trig_slice[i];
                if prevtrig <= 0.0 && t > 0.0 {
                    inputsamps = trig_period();
                }
                prevtrig = t;
            }
            dsamp += dsamp_slope;
            let idsamp = dsamp as i64;
            let frac = dsamp - idsamp as f32;
            let thisin = if inputsamps > 0 {
                inputsamps -= 1;
                in_audio.map_or(in_ctrl, |s| s[i])
            } else {
                0.0
            };
            let value = read_delayed(buf, iwrphase, mask, idsamp, frac, Interp::Cubic, warm);
            let thiscoef = if coef_audio { coef_slice[i] } else { curcoef };
            let onepole = (1.0 - thiscoef.abs()) * value + thiscoef * lastsamp;
            buf[(iwrphase & mask) as usize] = thisin + feedbk * onepole;
            out[i] = onepole;
            lastsamp = onepole;
            feedbk += feedbk_slope;
            curcoef += coef_slope;
            iwrphase = iwrphase.wrapping_add(1);
        }

        self.dsamp = dsamp;
        self.feedbk = feedbk;
        self.coef = coef_ctrl;
        self.delaytime = delaytime;
        self.decaytime = decaytime;
        self.lastsamp = zap(lastsamp as f64) as f32;
        self.prevtrig = prevtrig;
        self.inputsamps = inputsamps;
        self.iwrphase = iwrphase;
        if !warm {
            self.numoutput = self.numoutput.saturating_add(n as u32).min(self.len);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Pluck`]. Sizes the aux delay line from the constant `maxdelaytime`.
pub struct PluckCtor;

impl UnitDef for PluckCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 6 {
            return Err(BuildError::WrongInputCount);
        }
        let max_delay = ctx
            .const_input(MAXDELAY)
            .ok_or(BuildError::AuxRequiresConstant { input: MAXDELAY })?;
        let len = line_len(max_delay, ctx.audio.sample_rate, ctx.audio.block_size);
        let aux_bytes = len as usize * core::mem::size_of::<f32>();
        Ok(unit_spec_aux(
            Pluck {
                dsamp: 0.0,
                delaytime: 0.0,
                decaytime: 0.0,
                coef: 0.0,
                feedbk: 0.0,
                lastsamp: 0.0,
                prevtrig: 0.0,
                inputsamps: 0,
                len,
                mask: len - 1,
                iwrphase: 0,
                numoutput: 0,
                trig_audio: matches!(ctx.input_rates[TRIG], Rate::Audio) as u32,
                coef_audio: matches!(ctx.input_rates[COEF], Rate::Audio) as u32,
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}
