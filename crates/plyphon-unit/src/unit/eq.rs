//! Resonant EQ / formant filters - plyphon's ports of scsynth's `Formlet` and `MidEQ`, plus the
//! BEQSuite biquads `BLowPass`/`BHiPass`/`BBandPass`/`BPeakEQ`/`BLowShelf`/`BHiShelf`.
//!
//! `Formlet` is an FOF-like formant filter: two `Ringz`-style resonators (an attack and a decay) at
//! the same frequency, subtracted so the impulse response swells in and rings out. `MidEQ` is a
//! parametric peaking/notching EQ (a boost or cut of `db` around `freq`). The BEQSuite units share
//! one biquad kernel ([`Beq`]) and differ only in how [`BeqKind`] derives the coefficients - the
//! RBJ audio-EQ-cookbook formulas, matching scsynth's per-class math exactly.
//!
//! All units derive their `f64` coefficients from their parameters once per block, recomputing
//! only on a change, and flush their feedback state with the shared `zap`. This is plyphon's
//! block-rate convention, and it diverges from scsynth in two ways: for control-rate parameters
//! scsynth `CALCSLOPE`-interpolates the recomputed coefficients across the transition block (so
//! only that one block differs - the steady-state responses are identical), and for *audio-rate*
//! parameters scsynth's `_aa` variants recompute the coefficients per sample, where plyphon reads
//! the modulator at the block's first sample and holds it, like every other filter in the crate.

use core::f64::consts::{LN_2, LN_10};

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
            let sr = ctx.own.sample_rate;
            let ffreq = freq as f64 * ctx.own.radians_per_sample;
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
            let pfreq = freq as f64 * ctx.own.radians_per_sample;
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

/// Which BEQSuite response to compute. The build-time domain; stored in [`Beq`] as a `u32` tag
/// (via `BeqKind::to_tag`) so the state is [`Pod`] and lives in the rt-pool.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BeqKind {
    /// 12 dB/octave low-pass: `BLowPass.ar(in, freq, rq)`.
    LowPass,
    /// 12 dB/octave high-pass: `BHiPass.ar(in, freq, rq)`.
    HighPass,
    /// Band-pass: `BBandPass.ar(in, freq, bw)`, with `bw` the bandwidth in octaves.
    BandPass,
    /// Peaking EQ (boost/cut of `db` around `freq`): `BPeakEQ.ar(in, freq, rq, db)`.
    PeakEQ,
    /// Low shelf (boost/cut of `db` below `freq`): `BLowShelf.ar(in, freq, rs, db)`, with `rs` the
    /// reciprocal of the shelf slope.
    LowShelf,
    /// High shelf (boost/cut of `db` above `freq`): `BHiShelf.ar(in, freq, rs, db)`.
    HighShelf,
}

impl BeqKind {
    /// Encode as the `u32` tag stored in [`Beq`].
    fn to_tag(self) -> u32 {
        match self {
            BeqKind::LowPass => 0,
            BeqKind::HighPass => 1,
            BeqKind::BandPass => 2,
            BeqKind::PeakEQ => 3,
            BeqKind::LowShelf => 4,
            BeqKind::HighShelf => 5,
        }
    }

    /// Decode the `u32` tag stored in [`Beq`] (any unknown tag is low-pass).
    fn from_tag(tag: u32) -> BeqKind {
        match tag {
            1 => BeqKind::HighPass,
            2 => BeqKind::BandPass,
            3 => BeqKind::PeakEQ,
            4 => BeqKind::LowShelf,
            5 => BeqKind::HighShelf,
            _ => BeqKind::LowPass,
        }
    }

    /// Whether this response takes a fourth `db` gain input (the peak and shelf filters do; the
    /// pass filters are fully described by `freq` and their width input).
    fn has_db(self) -> bool {
        matches!(
            self,
            BeqKind::PeakEQ | BeqKind::LowShelf | BeqKind::HighShelf
        )
    }

    /// The number of inputs this response requires (`in`, `freq`, the width, and `db` if any).
    fn num_inputs(self) -> usize {
        if self.has_db() { 4 } else { 3 }
    }

    /// The RBJ biquad coefficients `(a0, a1, a2, b1, b2)` for this response at normalised angular
    /// frequency `w0` (radians/sample). `width` is the kind's second parameter (`rq`, `bw` in
    /// octaves, or `rs`); `db` is the peak/shelf gain and is ignored by the pass filters. Feedback
    /// terms use scsynth's added-`b*` sign convention, so the recurrence is
    /// `y0 = x + b1*y1 + b2*y2; out = a0*y0 + a1*y1 + a2*y2`.
    fn coefs(self, w0: f64, width: f64, db: f64) -> (f64, f64, f64, f64, f64) {
        let cosw0 = math::cos(w0);
        let sinw0 = math::sin(w0);
        match self {
            BeqKind::LowPass => {
                let i = 1.0 - cosw0;
                let alpha = sinw0 * 0.5 * width;
                let b0rz = 1.0 / (1.0 + alpha);
                let a0 = i * 0.5 * b0rz;
                (a0, i * b0rz, a0, cosw0 * 2.0 * b0rz, (1.0 - alpha) * -b0rz)
            }
            BeqKind::HighPass => {
                let i = 1.0 + cosw0;
                let alpha = sinw0 * 0.5 * width;
                let b0rz = 1.0 / (1.0 + alpha);
                let a0 = i * 0.5 * b0rz;
                (a0, -i * b0rz, a0, cosw0 * 2.0 * b0rz, (1.0 - alpha) * -b0rz)
            }
            BeqKind::BandPass => {
                // ln(2)/2 * bw * w0/sin(w0) maps the octave bandwidth onto the resonance.
                let alpha = sinw0 * math::sinh((0.5 * LN_2) * width * w0 / sinw0);
                let b0rz = 1.0 / (1.0 + alpha);
                let a0 = alpha * b0rz;
                (a0, 0.0, -a0, cosw0 * 2.0 * b0rz, (1.0 - alpha) * -b0rz)
            }
            BeqKind::PeakEQ => {
                let amp = math::exp(db * (LN_10 * 0.025)); // 10^(db/40)
                let alpha = sinw0 * 0.5 * width;
                let b0rz = 1.0 / (1.0 + alpha / amp);
                let b1 = 2.0 * b0rz * cosw0;
                (
                    (1.0 + alpha * amp) * b0rz,
                    -b1,
                    (1.0 - alpha * amp) * b0rz,
                    b1,
                    (1.0 - alpha / amp) * -b0rz,
                )
            }
            BeqKind::LowShelf => {
                let amp = math::exp(db * (LN_10 * 0.025)); // 10^(db/40)
                let alpha = sinw0 * 0.5 * math::sqrt((amp + 1.0 / amp) * (width - 1.0) + 2.0);
                let i = (amp + 1.0) * cosw0;
                let j = (amp - 1.0) * cosw0;
                let k = 2.0 * math::sqrt(amp) * alpha;
                let b0rz = 1.0 / ((amp + 1.0) + j + k);
                (
                    amp * ((amp + 1.0) - j + k) * b0rz,
                    2.0 * amp * ((amp - 1.0) - i) * b0rz,
                    amp * ((amp + 1.0) - j - k) * b0rz,
                    2.0 * ((amp - 1.0) + i) * b0rz,
                    ((amp + 1.0) + j - k) * -b0rz,
                )
            }
            BeqKind::HighShelf => {
                let amp = math::exp(db * (LN_10 * 0.025)); // 10^(db/40)
                let alpha = sinw0 * 0.5 * math::sqrt((amp + 1.0 / amp) * (width - 1.0) + 2.0);
                let i = (amp + 1.0) * cosw0;
                let j = (amp - 1.0) * cosw0;
                let k = 2.0 * math::sqrt(amp) * alpha;
                let b0rz = 1.0 / ((amp + 1.0) - j + k);
                (
                    amp * ((amp + 1.0) + j + k) * b0rz,
                    -2.0 * amp * ((amp - 1.0) + i) * b0rz,
                    amp * ((amp + 1.0) + j - k) * b0rz,
                    -2.0 * ((amp - 1.0) - i) * b0rz,
                    ((amp + 1.0) - j - k) * -b0rz,
                )
            }
        }
    }
}

/// A BEQSuite biquad: `BLowPass.ar(in, freq, rq)` and friends (see [`BeqKind`]).
///
/// `Pod` state for the rt-pool: `f64` history/coefficients first, then the cached parameters and
/// the [`BeqKind`] tag (`repr(C)` lays this out with no implicit padding).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Beq {
    y1: f64,
    y2: f64,
    a0: f64,
    a1: f64,
    a2: f64,
    b1: f64,
    b2: f64,
    freq: f32,
    width: f32,
    db: f32,
    kind: u32,
}

impl Beq {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const WIDTH: usize = 2;
    const DB: usize = 3;
}

impl Unit for Beq {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let kind = BeqKind::from_tag(self.kind);
        let freq = ctx.ins.control(Self::FREQ);
        let width = ctx.ins.control(Self::WIDTH);
        let db = if kind.has_db() {
            ctx.ins.control(Self::DB)
        } else {
            0.0
        };
        if freq != self.freq || width != self.width || db != self.db {
            let w0 = freq as f64 * ctx.own.radians_per_sample;
            let (a0, a1, a2, b1, b2) = kind.coefs(w0, width as f64, db as f64);
            self.a0 = a0;
            self.a1 = a1;
            self.a2 = a2;
            self.b1 = b1;
            self.b2 = b2;
            self.freq = freq;
            self.width = width;
            self.db = db;
        }
        let (a0, a1, a2, b1, b2) = (self.a0, self.a1, self.a2, self.b1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            let y0 = x as f64 + b1 * y1 + b2 * y2;
            *o = (a0 * y0 + a1 * y1 + a2 * y2) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// Constructor for [`Beq`], parameterized by the response [`BeqKind`].
pub struct BeqCtor(pub BeqKind);

impl UnitDef for BeqCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < self.0.num_inputs() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Beq {
            y1: 0.0,
            y2: 0.0,
            a0: 0.0,
            a1: 0.0,
            a2: 0.0,
            b1: 0.0,
            b2: 0.0,
            freq: f32::NAN, // force coefficient computation on the first block
            width: f32::NAN,
            db: f32::NAN,
            kind: self.0.to_tag(),
        }))
    }
}
