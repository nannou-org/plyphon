//! Resonant biquad filters - plyphon's ports of scsynth's `RLPF`, `RHPF`, `BPF`, `BRF`, `Resonz` and
//! `Ringz`.
//!
//! Each is a two-pole/two-zero section carrying two samples of feedback history, with coefficients
//! derived from a centre frequency and a second parameter (reciprocal-Q, bandwidth, or ring time).
//! Following the [`Butter`](crate::unit::filter::Butter) convention, coefficients are recomputed when
//! the (control-rate) parameters change and held constant across the block. State is `f64`, flushed
//! with `zap` (scsynth's `zapgremlins`).

use core::f64::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::decay::decay_coef;
use crate::unit::filter::zap;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `RLPF.ar(in, freq, rq)`: a resonant low-pass with reciprocal-Q `rq`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RLPF {
    a0: f64,
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    reson: f32,
}

impl Unit for RLPF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(1);
        let reson = ctx.ins.control(2);
        if freq != self.freq || reson != self.reson {
            let qres = (reson as f64).max(0.001);
            let pfreq = freq as f64 * TAU / ctx.own.sample_rate;
            let d = math::tan(pfreq * qres * 0.5);
            let c = (1.0 - d) / (1.0 + d);
            let cosf = math::cos(pfreq);
            self.b1 = (1.0 + c) * cosf;
            self.b2 = -c;
            self.a0 = (1.0 + c - self.b1) * 0.25;
            self.freq = freq;
            self.reson = reson;
        }
        let (a0, b1, b2) = (self.a0, self.b1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let y0 = a0 * x as f64 + b1 * y1 + b2 * y2;
            *o = (y0 + 2.0 * y1 + y2) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// `RHPF.ar(in, freq, rq)`: a resonant high-pass with reciprocal-Q `rq`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct RHPF {
    a0: f64,
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    reson: f32,
}

impl Unit for RHPF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(1);
        let reson = ctx.ins.control(2);
        if freq != self.freq || reson != self.reson {
            let qres = (reson as f64).max(0.001);
            let pfreq = freq as f64 * TAU / ctx.own.sample_rate;
            let d = math::tan(pfreq * qres * 0.5);
            let c = (1.0 - d) / (1.0 + d);
            let cosf = math::cos(pfreq);
            self.b1 = (1.0 + c) * cosf;
            self.b2 = -c;
            self.a0 = (1.0 + c + self.b1) * 0.25;
            self.freq = freq;
            self.reson = reson;
        }
        let (a0, b1, b2) = (self.a0, self.b1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let y0 = a0 * x as f64 + b1 * y1 + b2 * y2;
            *o = (y0 - 2.0 * y1 + y2) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// `BPF.ar(in, freq, bw)`: a band-pass with bandwidth `bw` in octaves (as a fraction of `freq`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BPF {
    a0: f64,
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    bw: f32,
}

impl Unit for BPF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(1);
        let bw = ctx.ins.control(2);
        if freq != self.freq || bw != self.bw {
            let pfreq = freq as f64 * TAU / ctx.own.sample_rate;
            let pbw = bw as f64 * pfreq * 0.5;
            let c = 1.0 / math::tan(pbw);
            let d = 2.0 * math::cos(pfreq);
            self.a0 = 1.0 / (1.0 + c);
            self.b1 = c * d * self.a0;
            self.b2 = (1.0 - c) * self.a0;
            self.freq = freq;
            self.bw = bw;
        }
        let (a0, b1, b2) = (self.a0, self.b1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let y0 = x as f64 + b1 * y1 + b2 * y2;
            *o = (a0 * (y0 - y2)) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// `BRF.ar(in, freq, bw)`: a band-reject (notch) with bandwidth `bw` in octaves.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BRF {
    a0: f64,
    a1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    bw: f32,
}

impl Unit for BRF {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(1);
        let bw = ctx.ins.control(2);
        if freq != self.freq || bw != self.bw {
            let pfreq = freq as f64 * TAU / ctx.own.sample_rate;
            let pbw = bw as f64 * pfreq * 0.5;
            let c = math::tan(pbw);
            let d = 2.0 * math::cos(pfreq);
            self.a0 = 1.0 / (1.0 + c);
            self.a1 = -d * self.a0;
            self.b2 = (1.0 - c) * self.a0;
            self.freq = freq;
            self.bw = bw;
        }
        let (a0, a1, b2) = (self.a0, self.a1, self.b2);
        let (mut y1, mut y2) = (self.y1, self.y2);
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
            let ay = a1 * y1;
            let y0 = x as f64 - ay - b2 * y2;
            *o = (a0 * (y0 + y2) + ay) as f32;
            y2 = y1;
            y1 = y0;
        }
        self.y1 = zap(y1);
        self.y2 = zap(y2);
        DoneAction::Nothing
    }
}

/// `Resonz.ar(in, freq, bwr)`: a resonator with bandwidth-ratio `bwr` (bandwidth as a fraction of
/// `freq`), normalised to unity peak gain.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Resonz {
    a0: f64,
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    rq: f32,
}

impl Unit for Resonz {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(1);
        let rq = ctx.ins.control(2);
        if freq != self.freq || rq != self.rq {
            let ffreq = freq as f64 * TAU / ctx.own.sample_rate;
            let r = 1.0 - ffreq * rq as f64 * 0.5;
            let two_r = 2.0 * r;
            let r2 = r * r;
            let cost = (two_r * math::cos(ffreq)) / (1.0 + r2);
            self.b1 = two_r * cost;
            self.b2 = -r2;
            self.a0 = (1.0 - r2) * 0.5;
            self.freq = freq;
            self.rq = rq;
        }
        resonz_kernel(ctx, self.a0, self.b1, self.b2, &mut self.y1, &mut self.y2);
        DoneAction::Nothing
    }
}

/// `Ringz.ar(in, freq, decayTime)`: a resonator whose ring decays to `-60 dB` over `decayTime`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Ringz {
    b1: f64,
    b2: f64,
    y1: f64,
    y2: f64,
    freq: f32,
    decay_time: f32,
}

impl Unit for Ringz {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(1);
        let decay_time = ctx.ins.control(2);
        if freq != self.freq || decay_time != self.decay_time {
            let ffreq = freq as f64 * TAU / ctx.own.sample_rate;
            let r = decay_coef(decay_time, ctx.own.sample_rate);
            let two_r = 2.0 * r;
            let r2 = r * r;
            let cost = (two_r * math::cos(ffreq)) / (1.0 + r2);
            self.b1 = two_r * cost;
            self.b2 = -r2;
            self.freq = freq;
            self.decay_time = decay_time;
        }
        // `Ringz`'s feed-forward gain is the constant 0.5 scsynth hard-codes.
        resonz_kernel(ctx, 0.5, self.b1, self.b2, &mut self.y1, &mut self.y2);
        DoneAction::Nothing
    }
}

/// The shared `Resonz`/`Ringz` recurrence: `y0 = in + b1*y1 + b2*y2; out = a0*(y0 - y2)`.
fn resonz_kernel(
    ctx: &mut ProcessCtx<'_>,
    a0: f64,
    b1: f64,
    b2: f64,
    sy1: &mut f64,
    sy2: &mut f64,
) {
    let (mut y1, mut y2) = (*sy1, *sy2);
    for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(0)) {
        let y0 = x as f64 + b1 * y1 + b2 * y2;
        *o = (a0 * (y0 - y2)) as f32;
        y2 = y1;
        y1 = y0;
    }
    *sy1 = zap(y1);
    *sy2 = zap(y2);
}

macro_rules! biquad_ctor {
    ($ctor:ident, $unit:ident { $($field:ident: $init:expr),* $(,)? }) => {
        #[doc = concat!("Constructor for [`", stringify!($unit), "`].")]
        pub struct $ctor;

        impl UnitDef for $ctor {
            fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
                if ctx.input_rates.len() < 3 {
                    return Err(BuildError::WrongInputCount);
                }
                Ok(unit_spec($unit { $($field: $init),* }))
            }
        }
    };
}

// `freq`/parameter seeded with NaN so the coefficients are computed on the first block.
biquad_ctor!(
    RLPFCtor,
    RLPF {
        a0: 0.0,
        b1: 0.0,
        b2: 0.0,
        y1: 0.0,
        y2: 0.0,
        freq: f32::NAN,
        reson: f32::NAN
    }
);
biquad_ctor!(
    RHPFCtor,
    RHPF {
        a0: 0.0,
        b1: 0.0,
        b2: 0.0,
        y1: 0.0,
        y2: 0.0,
        freq: f32::NAN,
        reson: f32::NAN
    }
);
biquad_ctor!(
    BPFCtor,
    BPF {
        a0: 0.0,
        b1: 0.0,
        b2: 0.0,
        y1: 0.0,
        y2: 0.0,
        freq: f32::NAN,
        bw: f32::NAN
    }
);
biquad_ctor!(
    BRFCtor,
    BRF {
        a0: 0.0,
        a1: 0.0,
        b2: 0.0,
        y1: 0.0,
        y2: 0.0,
        freq: f32::NAN,
        bw: f32::NAN
    }
);
biquad_ctor!(
    ResonzCtor,
    Resonz {
        a0: 0.0,
        b1: 0.0,
        b2: 0.0,
        y1: 0.0,
        y2: 0.0,
        freq: f32::NAN,
        rq: f32::NAN
    }
);
biquad_ctor!(
    RingzCtor,
    Ringz {
        b1: 0.0,
        b2: 0.0,
        y1: 0.0,
        y2: 0.0,
        freq: f32::NAN,
        decay_time: f32::NAN
    }
);
