//! Resonant EQ / formant filters - plyphon's ports of scsynth's `Formlet` and `MidEQ`.
//!
//! `Formlet` is an FOF-like formant filter: two `Ringz`-style resonators (an attack and a decay) at
//! the same frequency, subtracted so the impulse response swells in and rings out. `MidEQ` is a
//! parametric peaking/notching EQ (a boost or cut of `db` around `freq`). Both derive their `f64`
//! coefficients from the (control-rate) parameters once per block, recomputing only on a change
//! (plyphon's block-rate convention; scsynth `CALCSLOPE`-interpolates), and flush their feedback state
//! with the shared `zap`.

use core::f64::consts::LN_10;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::decay::decay_coef;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// The two-pole resonator coefficients `(b1, b2)` for pole radius `r` at normalised angular frequency
/// `ffreq` (radians/sample) - the `Ringz`/`Formlet` recurrence `y0 = x + b1*y1 + b2*y2`.
fn resonator_coefs(r: f64, ffreq: f64) -> (f64, f64) {
    let two_r = 2.0 * r;
    let r2 = r * r;
    let cost = (two_r * math::cos(ffreq)) / (1.0 + r2);
    (two_r * cost, -r2)
}

/// `Formlet.ar(in, freq, attacktime, decaytime)`: a resonant formant filter - a decay resonator at
/// `freq` minus an attack resonator at `freq`, so the response rings up over `attacktime` and out over
/// `decaytime`. A short `attacktime` gives a sharp onset.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Formlet {
    y01: f64,
    y02: f64,
    b01: f64,
    b02: f64,
    y11: f64,
    y12: f64,
    b11: f64,
    b12: f64,
    freq: f32,
    attack: f32,
    decay: f32,
    _pad: u32,
}

impl Formlet {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const ATTACK: usize = 2;
    const DECAY: usize = 3;
}

impl Unit for Formlet {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let attack = ctx.ins.control(Self::ATTACK);
        let decay = ctx.ins.control(Self::DECAY);
        if freq != self.freq || attack != self.attack || decay != self.decay {
            let sr = ctx.audio.sample_rate;
            let ffreq = freq as f64 * ctx.audio.radians_per_sample;
            let (b01, b02) = resonator_coefs(decay_coef(decay, sr), ffreq);
            let (b11, b12) = resonator_coefs(decay_coef(attack, sr), ffreq);
            self.b01 = b01;
            self.b02 = b02;
            self.b11 = b11;
            self.b12 = b12;
            self.freq = freq;
            self.attack = attack;
            self.decay = decay;
        }
        let (b01, b02, b11, b12) = (self.b01, self.b02, self.b11, self.b12);
        let (mut y01, mut y02, mut y11, mut y12) = (self.y01, self.y02, self.y11, self.y12);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let ain = x as f64;
            let y00 = ain + b01 * y01 + b02 * y02;
            let y10 = ain + b11 * y11 + b12 * y12;
            *o = (0.25 * ((y00 - y02) - (y10 - y12))) as f32;
            y02 = y01;
            y01 = y00;
            y12 = y11;
            y11 = y10;
        }
        self.y01 = zap(y01);
        self.y02 = zap(y02);
        self.y11 = zap(y11);
        self.y12 = zap(y12);
        DoneAction::Nothing
    }
}

/// Constructor for [`Formlet`].
pub struct FormletCtor;

impl UnitDef for FormletCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Formlet {
            y01: 0.0,
            y02: 0.0,
            b01: 0.0,
            b02: 0.0,
            y11: 0.0,
            y12: 0.0,
            b11: 0.0,
            b12: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            attack: f32::NAN,
            decay: f32::NAN,
            _pad: 0,
        }))
    }
}

/// `MidEQ.ar(in, freq, rq, db)`: a parametric peaking EQ - boosts (or, for negative `db`, cuts) a band
/// of width `rq` (reciprocal Q) centred on `freq` by `db` decibels, leaving the rest of the spectrum
/// flat. `db = 0` passes the signal through unchanged.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MidEQ {
    y1: f64,
    y2: f64,
    a0: f64,
    b1: f64,
    b2: f64,
    freq: f32,
    bw: f32,
    db: f32,
    _pad: u32,
}

impl MidEQ {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const RQ: usize = 2;
    const DB: usize = 3;
}

impl Unit for MidEQ {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(Self::FREQ);
        let bw = ctx.ins.control(Self::RQ);
        let db = ctx.ins.control(Self::DB);
        if freq != self.freq || bw != self.bw || db != self.db {
            let amp = math::exp(db as f64 * (LN_10 / 20.0)) - 1.0; // sc_dbamp(db) - 1
            let pfreq = freq as f64 * ctx.audio.radians_per_sample;
            let pbw = bw as f64 * pfreq * 0.5;
            let c = 1.0 / math::tan(pbw);
            let d = 2.0 * math::cos(pfreq);
            let a0 = 1.0 / (1.0 + c);
            self.b1 = c * d * a0;
            self.b2 = (1.0 - c) * a0;
            self.a0 = a0 * amp;
            self.freq = freq;
            self.bw = bw;
            self.db = db;
        }
        let (a0, b1, b2) = (self.a0, self.b1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let zin = x as f64;
            let y0 = zin + b1 * y1 + b2 * y2;
            *o = (zin + a0 * (y0 - y2)) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// Constructor for [`MidEQ`].
pub struct MidEQCtor;

impl UnitDef for MidEQCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(MidEQ {
            y1: 0.0,
            y2: 0.0,
            a0: 0.0,
            b1: 0.0,
            b2: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            bw: f32::NAN,
            db: f32::NAN,
            _pad: 0,
        }))
    }
}
