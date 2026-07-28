//! Translated sc3-plugins envelope and Moog-family processing units.
//!
//! The implementations preserve the pinned server arithmetic and coefficient cadence, with
//! deterministic finite-input recovery at the host boundary.
//!
//! `EnvDetect` is translated from SLUGens:
//! Copyright (c) 2002 James McCartney. All rights reserved.
//! UGens by Nick Collins, released under the GNU General Public License, version 2 or later.
//!
//! `DFM1` is translated from the DFM-1 Digital Filter Module:
//! Copyright (C) 2010, 2011 Tony Hardie-Bick and Jonny Stutters.
//! Distributed under the GNU General Public License, version 2 or later.
//!
//! `MoogLadder` is translated from BhobUGens:
//! filter plugins by bhob rainey; based on work by Victor Lazzarini and Antti Huovilainen.
//! SuperCollider copyright (c) 2002 James McCartney. All rights reserved.
//! Distributed under the GNU General Public License, version 2 or later.
//!
//! `MoogVCF` is translated from JoshUGens:
//! Copyright 2005 Josh Parmenter and copyright (c) 2002 James McCartney.
//! Distributed under the GNU General Public License, version 2 or later.
//!
//! The `MoogVCF` exponential approximation retains Paul Mineiro's 2011 implementation. Copyright
//! (C) 2011 Paul Mineiro. Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that source redistributions retain the copyright notice,
//! conditions, and disclaimer; binary redistributions reproduce them in accompanying materials;
//! and neither Paul Mineiro's name nor contributors' names endorse derived products without prior
//! written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//! WITHOUT EXPRESS OR IMPLIED WARRANTIES, INCLUDING MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//! PURPOSE. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//! INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, HOWEVER CAUSED.

use core::f32::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, Inputs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// Reject malformed fixed-shape calc units before they reach the audio thread.
fn validate_calc(
    ctx: &BuildContext<'_>,
    inputs: usize,
    outputs: usize,
    node_rates: &[Rate],
) -> Result<(), BuildError> {
    if ctx.input_rates.len() != inputs {
        return Err(BuildError::WrongInputCount);
    }
    if ctx.num_outputs != outputs {
        return Err(BuildError::WrongOutputCount {
            expected: outputs,
            actual: ctx.num_outputs,
        });
    }
    if ctx.special_index != 0 {
        return Err(BuildError::UnsupportedOp(ctx.special_index));
    }
    if !node_rates.contains(&ctx.rate) {
        return Err(BuildError::UnsupportedUnitRate);
    }
    if ctx
        .input_rates
        .iter()
        .any(|rate| *rate == Rate::Demand || (ctx.rate == Rate::Control && *rate == Rate::Audio))
    {
        return Err(BuildError::UnsupportedUnitRate);
    }
    Ok(())
}

/// Read a signal inlet at `sample`, expanding scalar/control values and replacing non-finite input
/// samples with zero before they can enter recursive state.
#[inline]
fn signal(ins: &Inputs<'_>, inlet: usize, sample: usize) -> f32 {
    let value = if ins.rate(inlet) == Rate::Audio {
        ins.audio(inlet).get(sample).copied().unwrap_or(0.0)
    } else {
        ins.control(inlet)
    };
    if value.is_finite() { value } else { 0.0 }
}

/// Read a block-sampled control, retaining the previous finite bounded value across invalid blocks.
#[inline]
fn block_control(ins: &Inputs<'_>, inlet: usize, remembered: &mut f32, min: f32, max: f32) -> f32 {
    let value = ins.control(inlet);
    if value.is_finite() {
        *remembered = value.clamp(min, max);
    }
    *remembered
}

/// Read an audio-specialized control at one sample, retaining its last finite bounded value.
#[inline]
fn sample_control(
    ins: &Inputs<'_>,
    inlet: usize,
    sample: usize,
    remembered: &mut f32,
    min: f32,
    max: f32,
) -> f32 {
    let value = if ins.rate(inlet) == Rate::Audio {
        ins.audio(inlet).get(sample).copied().unwrap_or(*remembered)
    } else {
        ins.control(inlet)
    };
    if value.is_finite() {
        *remembered = value.clamp(min, max);
    }
    *remembered
}

/// `EnvDetect.ar(in, attack, release)`: asymmetric absolute-value envelope follower.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EnvDetect {
    envelope: f32,
    attack: f32,
    release: f32,
}

impl EnvDetect {
    const IN: usize = 0;
    const ATTACK: usize = 1;
    const RELEASE: usize = 2;
}

impl Unit for EnvDetect {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let attack = block_control(&ctx.ins, Self::ATTACK, &mut self.attack, 0.0, f32::MAX);
        let release = block_control(&ctx.ins, Self::RELEASE, &mut self.release, 0.0, f32::MAX);
        let ga = time_coefficient(attack, ctx.own.sample_rate as f32);
        let gr = time_coefficient(release, ctx.own.sample_rate as f32);
        let mut envelope = self.envelope;
        let input = ctx.ins.audio(Self::IN);
        for (out, &input) in ctx.outs.audio(0).iter_mut().zip(input) {
            let next = if input.is_finite() { input.abs() } else { 0.0 };
            let coefficient = if envelope < next { ga } else { gr };
            envelope *= coefficient;
            envelope += (1.0 - coefficient) * next;
            if !envelope.is_finite() {
                envelope = 0.0;
            }
            *out = envelope;
        }
        self.envelope = envelope;
        DoneAction::Nothing
    }
}

/// Convert attack/release seconds to the source one-pole coefficient.
fn time_coefficient(time: f32, sample_rate: f32) -> f32 {
    if time == 0.0 {
        0.0
    } else {
        math::exp(-1.0f64 / (sample_rate as f64 * time as f64)) as f32
    }
}

/// Constructor for [`EnvDetect`].
pub struct EnvDetectCtor;

impl UnitDef for EnvDetectCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_calc(ctx, 3, 1, &[Rate::Audio])?;
        if ctx.input_rates[0] != Rate::Audio {
            return Err(BuildError::UnsupportedUnitRate);
        }
        Ok(unit_spec(EnvDetect {
            envelope: 0.0,
            attack: 100.0,
            release: 0.0,
        }))
    }
}

/// `DFM1.ar(in, freq, res, inputgain, type, noiselevel)`: nonlinear two-stage analog-model filter.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Dfm1 {
    l: f64,
    h: f64,
    a: f64,
    b: f64,
    r: f64,
    s: f64,
    za: f64,
    zb: f64,
    zh: f64,
    zr: f64,
    zy: f64,
    noise_x: u32,
    noise_y: u32,
    noise_z: u32,
    noise_w: u32,
    frequency: f32,
    resonance: f32,
    input_gain: f32,
    filter_type: f32,
    noise_level: f32,
    _pad: u32,
}

impl Dfm1 {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const RES: usize = 2;
    const INPUT_GAIN: usize = 3;
    const TYPE: usize = 4;
    const NOISE_LEVEL: usize = 5;
}

impl Unit for Dfm1 {
    fn reseed(&mut self, seed: u64) {
        let base = seed as u32 ^ (seed >> 32) as u32;
        self.noise_x = base;
        self.noise_y = base.wrapping_add(1);
        self.noise_z = base.wrapping_add(2);
        self.noise_w = base.wrapping_add(3);
        if self.noise_x | self.noise_y | self.noise_z | self.noise_w == 0 {
            self.noise_x = 1;
        }
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let nyquist = (ctx.own.sample_rate as f32 * 0.5).max(1.0);
        let frequency = block_control(&ctx.ins, Self::FREQ, &mut self.frequency, 1.0, nyquist);
        let resonance = block_control(&ctx.ins, Self::RES, &mut self.resonance, 0.0, 10.0);
        let raw_input_gain = ctx.ins.control(Self::INPUT_GAIN);
        if raw_input_gain.is_finite() {
            self.input_gain = raw_input_gain;
        }
        let raw_type = ctx.ins.control(Self::TYPE);
        if raw_type.is_finite() {
            self.filter_type = raw_type;
        }
        let noise_level = block_control(
            &ctx.ins,
            Self::NOISE_LEVEL,
            &mut self.noise_level,
            0.0,
            f32::MAX,
        );

        let [at, bt, resonance_correction] =
            dfm1_coefficients(frequency as f64, ctx.own.sample_rate);
        let rt = resonance_correction * resonance as f64;
        let mut lt = self.input_gain as f64 * DFM1_PRE_GAIN;
        let mut ht = lt;
        if self.filter_type < 0.5 {
            ht = 0.0;
        } else {
            lt = 0.0;
        }
        lt *= 1.0 - at;
        let st = if (noise_level as f64) < DFM1_MIN_NOISE {
            DFM1_MIN_NOISE
        } else {
            noise_level as f64 * DFM1_NOISE_GAIN
        };

        let count = ctx.outs.audio(0).len().max(1) as f64;
        let li = dfm1_increment((lt - self.l) / count);
        let hi = dfm1_increment((ht - self.h) / count);
        let ai = dfm1_increment((at - self.a) / count);
        let bi = dfm1_increment((bt - self.b) / count);
        let ri = dfm1_increment((rt - self.r) / count);
        let si = dfm1_increment((st - self.s) / count);

        let mut l = self.l;
        let mut h = self.h;
        let mut a = self.a;
        let mut b = self.b;
        let mut r = self.r;
        let mut s = self.s;
        let mut za = self.za;
        let mut zb = self.zb;
        let mut zh = self.zh;
        let mut zr = self.zr;
        let mut zy = self.zy;
        let mut noise = [self.noise_x, self.noise_y, self.noise_z, self.noise_w];
        let ins = &ctx.ins;
        let out = ctx.outs.audio(0);

        for (sample_index, output) in out.iter_mut().enumerate() {
            l += li;
            h += hi;
            a += ai;
            b += bi;
            r += ri;
            s += si;
            let input = signal(ins, Self::IN, sample_index) as f64;

            za = input * l + r * (zy - zr) + a * za + s * dfm1_normal(&mut noise);
            zb = za * (1.0 - b) + (input - zh) * h + b * zb + s * dfm1_normal(&mut noise);
            zh = input;
            zr = zy;
            zb = zb.clamp(-1.0, 1.0);
            let nx = DFM1_N * zb;
            zy = zb - 2.0 * nx * nx * nx;
            zy += s * dfm1_normal(&mut noise);
            let value = zy * DFM1_POST_GAIN;
            *output = if value.is_finite() { value as f32 } else { 0.0 };
            if !za.is_finite()
                || !zb.is_finite()
                || !zh.is_finite()
                || !zr.is_finite()
                || !zy.is_finite()
            {
                za = 0.0;
                zb = 0.0;
                zh = 0.0;
                zr = 0.0;
                zy = 0.0;
            }
        }

        self.l = lt;
        self.h = ht;
        self.a = at;
        self.b = bt;
        self.r = rt;
        self.s = st;
        self.za = za;
        self.zb = zb;
        self.zh = zh;
        self.zr = zr;
        self.zy = zy;
        [self.noise_x, self.noise_y, self.noise_z, self.noise_w] = noise;
        DoneAction::Nothing
    }
}

const DFM1_MIN_NOISE: f64 = 0.00001;
const DFM1_POST_GAIN: f64 = 0.95 * 3.0 / 2.0;
const DFM1_PRE_GAIN: f64 = (1.0 / 0.95) * 3.0 / 2.0;
const DFM1_NOISE_GAIN: f64 = 0.05;
#[allow(clippy::excessive_precision)]
const DFM1_N: f64 = 0.55032120814910446;

fn dfm1_increment(value: f64) -> f64 {
    if value.abs() < 0.000001 { 0.0 } else { value }
}

fn dfm1_coefficients(frequency: f64, sample_rate: f64) -> [f64; 3] {
    let rate = sample_rate.max(1.0);
    let normalized = (frequency / rate).clamp(1.0 / 131_072.0, 0.5 - 1.0 / 128.0 - 0.001);
    let mut octave = 0usize;
    let mut octave_base = 1.0 / 131_072.0;
    let mut next_base = octave_base * 2.0;
    while octave < 16 && normalized >= next_base {
        octave_base = next_base;
        next_base *= 2.0;
        octave += 1;
    }
    let step = 32.0 * (normalized - octave_base) / octave_base;
    let index = math::floor(step) as usize;
    let fraction = step - index as f64;
    let address = 3 * (index + octave * 32);
    let xa = DFM1_LUT[address];
    let xb = DFM1_LUT[address + 1];
    let xr = DFM1_LUT[address + 2];
    let ya = DFM1_LUT[address + 3];
    let yb = DFM1_LUT[address + 4];
    let yr = DFM1_LUT[address + 5];
    [
        xa + (ya - xa) * fraction,
        xb + (yb - xb) * fraction,
        xr + (yr - xr) * fraction,
    ]
}

fn dfm1_xor128(state: &mut [u32; 4]) -> i32 {
    let t = state[0] ^ state[0].wrapping_shl(15);
    state[0] = state[1];
    state[1] = state[2];
    state[2] = state[3];
    state[3] = (state[3] ^ state[3].wrapping_shr(21)) ^ (t ^ t.wrapping_shr(4));
    state[3] as i32
}

fn dfm1_uniform(state: &mut [u32; 4]) -> f64 {
    let value = 0.5 + dfm1_xor128(state) as f64 * (1.0 / 4_294_967_296.0);
    value.max(f64::MIN_POSITIVE)
}

fn dfm1_normal(state: &mut [u32; 4]) -> f64 {
    let hz = dfm1_xor128(state);
    let iz = (hz & 127) as usize;
    if (hz as f32).abs() < DFM1_KN[iz] as f32 {
        hz as f64 * DFM1_WN[iz]
    } else {
        dfm1_normal_tail(hz, iz, state)
    }
}

fn dfm1_normal_tail(mut hz: i32, mut iz: usize, state: &mut [u32; 4]) -> f64 {
    const R: f64 = 3.442619855899;
    loop {
        let x = hz as f64 * DFM1_WN[iz];
        if iz == 0 {
            let (tail, y) = loop {
                let tail = -math::ln(dfm1_uniform(state)) * 0.2904764516147;
                let y = -math::ln(dfm1_uniform(state));
                if y + y >= tail * tail {
                    break (tail, y);
                }
            };
            let _ = y;
            return if hz > 0 { R + tail } else { -R - tail };
        }
        if DFM1_FN[iz] + dfm1_uniform(state) * (DFM1_FN[iz - 1] - DFM1_FN[iz])
            < math::exp(-0.5 * x * x)
        {
            return x;
        }
        hz = dfm1_xor128(state);
        iz = (hz & 127) as usize;
        if (hz as f32).abs() < DFM1_KN[iz] as f32 {
            return hz as f64 * DFM1_WN[iz];
        }
    }
}

/// Constructor for [`Dfm1`].
pub struct Dfm1Ctor;

impl UnitDef for Dfm1Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_calc(ctx, 6, 1, &[Rate::Audio])?;
        let base = ctx.seed as u32 ^ (ctx.seed >> 32) as u32;
        Ok(unit_spec(Dfm1 {
            l: 0.0,
            h: 0.0,
            a: 0.0,
            b: 0.0,
            r: 0.0,
            s: 0.0,
            za: 0.0,
            zb: 0.0,
            zh: 0.0,
            zr: 0.0,
            zy: 0.0,
            noise_x: base,
            noise_y: base.wrapping_add(1),
            noise_z: base.wrapping_add(2),
            noise_w: base.wrapping_add(3),
            frequency: 1000.0,
            resonance: 0.1,
            input_gain: 1.0,
            filter_type: 0.0,
            noise_level: 0.0003,
            _pad: 0,
        }))
    }
}

/// `MoogLadder.ar/kr(in, ffreq, res)`: Huovilainen/Lazzarini nonlinear four-pole ladder.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MoogLadder {
    fco: f32,
    k2v: f32,
    res: f32,
    xnm1: f32,
    y1nm1: f32,
    y2nm1: f32,
    y3nm1: f32,
    y4nm1: f32,
    y1n: f32,
    y2n: f32,
    y3n: f32,
    y4n: f32,
    y1n1: f32,
    last_cutoff: f32,
    last_res: f32,
}

impl MoogLadder {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const RES: usize = 2;
}

impl Unit for MoogLadder {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let nyquist = ctx.own.sample_rate as f32 * 0.5;
        let cutoff = initial_control(ctx.ins.control(Self::FREQ), 440.0, 0.0, nyquist);
        let res = initial_control(ctx.ins.control(Self::RES), 0.0, 0.0, 1.0);
        self.fco = cutoff;
        self.k2v = moog_ladder_ctor_coefficient(cutoff, ctx.own.sample_dur as f32);
        self.res = res;
        self.last_cutoff = cutoff;
        self.last_res = res;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let cutoff_audio = ctx.ins.rate(Self::FREQ) == Rate::Audio;
        let res_audio = ctx.ins.rate(Self::RES) == Rate::Audio;
        let nyquist = ctx.own.sample_rate as f32 * 0.5;
        let count = ctx.outs.audio(0).len().max(1) as f32;
        let mut fco = self.fco;
        let mut k2v = self.k2v;
        let mut res = self.res;
        let mut cutoff_slope = 0.0;
        let mut res_slope = 0.0;

        if !cutoff_audio {
            let next = block_control(&ctx.ins, Self::FREQ, &mut self.last_cutoff, 0.0, nyquist);
            if next != fco {
                let target = moog_ladder_coefficient(next, ctx.own.sample_dur as f32, 1.25);
                cutoff_slope = (target - k2v) / count;
            }
        }
        if !res_audio {
            let next = block_control(&ctx.ins, Self::RES, &mut self.last_res, 0.0, 1.0);
            res_slope = (next - res) / count;
        }

        let mut xnm1 = self.xnm1;
        let mut y1nm1 = self.y1nm1;
        let mut y2nm1 = self.y2nm1;
        let mut y3nm1 = self.y3nm1;
        let mut y4nm1 = self.y4nm1;
        let mut y1n = self.y1n;
        let mut y2n = self.y2n;
        let mut y3n = self.y3n;
        let mut y4n = self.y4n;
        let mut y1n1 = self.y1n1;
        let ins = &ctx.ins;
        let out = ctx.outs.audio(0);

        for (sample_index, output) in out.iter_mut().enumerate() {
            if cutoff_audio {
                let cutoff = sample_control(
                    ins,
                    Self::FREQ,
                    sample_index,
                    &mut self.last_cutoff,
                    0.0,
                    nyquist,
                );
                if cutoff != fco {
                    let scale = if res_audio { 1.220_703_1 } else { 1.25 };
                    k2v = moog_ladder_coefficient(cutoff, ctx.own.sample_dur as f32, scale);
                    fco = cutoff;
                }
            }
            let sample_res = if res_audio {
                let next =
                    sample_control(ins, Self::RES, sample_index, &mut self.last_res, 0.0, 1.0);
                res = next;
                next
            } else {
                res
            };
            let input = signal(ins, Self::IN, sample_index);
            moog_ladder_stage(
                input, sample_res, k2v, &mut xnm1, &mut y1nm1, &mut y2nm1, &mut y3nm1, &mut y4nm1,
                &mut y1n, &mut y2n, &mut y3n, &mut y4n, &mut y1n1,
            );
            moog_ladder_stage(
                input, sample_res, k2v, &mut xnm1, &mut y1nm1, &mut y2nm1, &mut y3nm1, &mut y4nm1,
                &mut y1n, &mut y2n, &mut y3n, &mut y4n, &mut y1n1,
            );
            *output = if y4n.is_finite() { y4n } else { 0.0 };
            k2v += cutoff_slope;
            res += res_slope;
        }

        if !cutoff_audio {
            fco = self.last_cutoff;
        }
        if !res_audio {
            res = self.last_res;
        }
        self.fco = fco;
        self.k2v = finite_or_zero(k2v);
        self.res = res;
        self.xnm1 = zap_f32(xnm1);
        self.y1nm1 = zap_f32(y1nm1);
        self.y2nm1 = zap_f32(y2nm1);
        self.y3nm1 = zap_f32(y3nm1);
        self.y4nm1 = zap_f32(y4nm1);
        self.y1n = zap_f32(y1n);
        self.y2n = zap_f32(y2n);
        self.y3n = zap_f32(y3n);
        self.y4n = zap_f32(y4n);
        self.y1n1 = zap_f32(y1n1);
        DoneAction::Nothing
    }
}

fn initial_control(value: f32, fallback: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback.clamp(min, max)
    }
}

fn moog_ladder_ctor_coefficient(cutoff: f32, sample_dur: f32) -> f32 {
    let fc = cutoff * sample_dur;
    let f = fc * 0.5;
    let fcr = 1.8730 * fc * fc * fc + 0.4955 * fc * fc - 0.6490 * fc + 0.9988;
    1.25 * (1.0 - math::exp(-TAU * fcr * f))
}

fn moog_ladder_coefficient(cutoff: f32, sample_dur: f32, scale: f32) -> f32 {
    let fc = cutoff * sample_dur;
    let f = fc * 0.5;
    let fcr = 1.8730 * (fc * fc * fc + 0.4955 * (fc * fc)) - 0.6490 * fc + 0.9988;
    scale * (1.0 - math::exp(-TAU * fcr * f))
}

#[allow(clippy::too_many_arguments)]
fn moog_ladder_stage(
    input: f32,
    res: f32,
    k2v: f32,
    xnm1: &mut f32,
    y1nm1: &mut f32,
    y2nm1: &mut f32,
    y3nm1: &mut f32,
    y4nm1: &mut f32,
    y1n: &mut f32,
    y2n: &mut f32,
    y3n: &mut f32,
    y4n: &mut f32,
    y1n1: &mut f32,
) {
    const I2V: f32 = 0.70466;
    let xn = input - 8.0 * res * *y4n;
    let xn1 = xn * I2V;
    let y2nm11 = *y2nm1 * I2V;
    let y3nm11 = *y3nm1 * I2V;
    let y4nm11 = *y4nm1 * I2V;
    *y1n = *xnm1 + k2v * (xn1 / (1.0 + xn1.abs()) - *y1n1 / (1.0 + y1n1.abs()));
    *y1n1 = *y1n * I2V;
    *y2n = *y2nm1 + k2v * (*y1n1 / (1.0 + y1n1.abs()) - y2nm11 / (1.0 + y2nm11.abs()));
    let y2n1 = *y2n * I2V;
    *y3n = *y3nm1 + k2v * (y2n1 / (1.0 + y2n1.abs()) - y3nm11 / (1.0 + y3nm11.abs()));
    let y3n1 = *y3n * I2V;
    *y4n = *y4nm1 + k2v * (y3n1 / (1.0 + y3n1.abs()) - y4nm11 / (1.0 + y4nm11.abs()));
    *y4n = (*y4n + *y4nm1) * 0.5;
    *xnm1 = xn;
    *y1nm1 = *y1n;
    *y2nm1 = *y2n;
    *y3nm1 = *y3n;
    *y4nm1 = *y4n;
}

/// Constructor for [`MoogLadder`].
pub struct MoogLadderCtor;

impl UnitDef for MoogLadderCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_calc(ctx, 3, 1, &[Rate::Audio, Rate::Control])?;
        Ok(unit_spec(MoogLadder {
            fco: 440.0,
            k2v: 0.0,
            res: 0.0,
            xnm1: 0.0,
            y1nm1: 0.0,
            y2nm1: 0.0,
            y3nm1: 0.0,
            y4nm1: 0.0,
            y1n: 0.0,
            y2n: 0.0,
            y3n: 0.0,
            y4n: 0.0,
            y1n1: 0.0,
            last_cutoff: 440.0,
            last_res: 0.0,
        }))
    }
}

/// `MoogVCF.ar(in, fco, res)`: four cascaded bilinear one-poles with nonlinear feedback limiting.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MoogVcf {
    fco: f32,
    res: f32,
    xnm1: f32,
    y1nm1: f32,
    y2nm1: f32,
    y3nm1: f32,
    y1n: f32,
    y2n: f32,
    y3n: f32,
    y4n: f32,
    kp: f32,
    pp1d2: f32,
    k: f32,
    last_cutoff: f32,
    last_res: f32,
}

impl MoogVcf {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const RES: usize = 2;
}

impl Unit for MoogVcf {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let nyquist = ctx.own.sample_rate as f32 * 0.5;
        let cutoff = initial_control(ctx.ins.control(Self::FREQ), 440.0, 0.0, nyquist);
        let res = initial_control(ctx.ins.control(Self::RES), 0.0, 0.0, 1.0);
        self.fco = cutoff * 2.0 * ctx.own.sample_dur as f32;
        self.res = res;
        [self.kp, self.pp1d2, self.k] = moog_vcf_parameters(self.fco, res);
        self.last_cutoff = cutoff;
        self.last_res = res;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let cutoff_rate = ctx.ins.rate(Self::FREQ);
        let res_rate = ctx.ins.rate(Self::RES);
        let cutoff_audio = cutoff_rate == Rate::Audio;
        // A scalar cutoff selects the source's block/block fallback even when resonance is wired at
        // audio rate; that variant samples resonance once and ramps it across the callback.
        let res_audio = res_rate == Rate::Audio && cutoff_rate != Rate::Scalar;
        let scalar_pair = cutoff_rate == Rate::Scalar && res_rate == Rate::Scalar;
        let nyquist = ctx.own.sample_rate as f32 * 0.5;
        let normalized_scale = 2.0 * ctx.own.sample_dur as f32;
        let count = ctx.outs.audio(0).len().max(1) as f32;
        let mut fco = self.fco;
        let mut res = self.res;
        let mut fco_slope = 0.0;
        let mut res_slope = 0.0;
        if !cutoff_audio {
            let cutoff = block_control(&ctx.ins, Self::FREQ, &mut self.last_cutoff, 0.0, nyquist);
            fco_slope = (cutoff * normalized_scale - fco) / count;
        }
        if !res_audio {
            let next = block_control(&ctx.ins, Self::RES, &mut self.last_res, 0.0, 1.0);
            if !(cutoff_rate == Rate::Control && res_rate == Rate::Scalar) {
                res_slope = (next - res) / count;
            }
        }

        let mut xnm1 = self.xnm1;
        let mut y1nm1 = self.y1nm1;
        let mut y2nm1 = self.y2nm1;
        let mut y3nm1 = self.y3nm1;
        let mut y1n = self.y1n;
        let mut y2n = self.y2n;
        let mut y3n = self.y3n;
        let mut y4n = self.y4n;
        let ins = &ctx.ins;
        let out = ctx.outs.audio(0);
        for (sample_index, output) in out.iter_mut().enumerate() {
            let sample_fco = if cutoff_audio {
                sample_control(
                    ins,
                    Self::FREQ,
                    sample_index,
                    &mut self.last_cutoff,
                    0.0,
                    nyquist,
                ) * normalized_scale
            } else {
                fco
            };
            let sample_res = if res_audio {
                sample_control(ins, Self::RES, sample_index, &mut self.last_res, 0.0, 1.0)
            } else {
                res
            };
            let [kp, pp1d2, k] = if scalar_pair {
                [self.kp, self.pp1d2, self.k]
            } else {
                moog_vcf_parameters(sample_fco, sample_res)
            };
            let mut xn = signal(ins, Self::IN, sample_index);
            xn -= k * y4n;
            y1n = xn * pp1d2 + xnm1 * pp1d2 - kp * y1n;
            y2n = y1n * pp1d2 + y1nm1 * pp1d2 - kp * y2n;
            y3n = y2n * pp1d2 + y2nm1 * pp1d2 - kp * y3n;
            y4n = y3n * pp1d2 + y3nm1 * pp1d2 - kp * y4n;
            y4n = moog_vcf_limit(y4n);
            xnm1 = xn;
            y1nm1 = y1n;
            y2nm1 = y2n;
            y3nm1 = y3n;
            *output = if y4n.is_finite() { y4n } else { 0.0 };
            fco += fco_slope;
            res += res_slope;
        }
        fco = self.last_cutoff * normalized_scale;
        if res_audio || res_slope != 0.0 {
            res = self.last_res;
        }
        self.fco = fco;
        self.res = res;
        self.xnm1 = zap_f32(xnm1);
        self.y1nm1 = zap_f32(y1nm1);
        self.y2nm1 = zap_f32(y2nm1);
        self.y3nm1 = zap_f32(y3nm1);
        self.y1n = zap_f32(y1n);
        self.y2n = zap_f32(y2n);
        self.y3n = zap_f32(y3n);
        self.y4n = zap_f32(y4n);
        DoneAction::Nothing
    }
}

fn moog_vcf_parameters(normalized_cutoff: f32, resonance: f32) -> [f32; 3] {
    let cutoff = normalized_cutoff.min(1.0);
    let kp = 3.6 * cutoff - (1.6 * cutoff) * cutoff - 1.0;
    let pp1d2 = (kp + 1.0) * 0.5;
    let scale = moog_vcf_fast_exp((1.0 - pp1d2) * 1.386249);
    [kp, pp1d2, resonance * scale]
}

#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn moog_vcf_fast_exp(value: f32) -> f32 {
    let p = 1.442695040 * value;
    let offset = if p < 0.0 { 1.0 } else { 0.0 };
    let clipped = p.max(-126.0);
    let whole = clipped as i32 as f32;
    let z = clipped - whole + offset;
    let bits = ((1u32 << 23) as f32
        * (clipped + 121.274055 + 27.728024 / (4.8425255 - z) - 1.4901291 * z))
        as u32;
    f32::from_bits(bits)
}

fn moog_vcf_limit(value: f32) -> f32 {
    let value = value.clamp(-core::f32::consts::SQRT_2, core::f32::consts::SQRT_2);
    value - value * value * (value * (1.0 / 6.0))
}

/// Constructor for [`MoogVcf`].
pub struct MoogVcfCtor;

impl UnitDef for MoogVcfCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_calc(ctx, 3, 1, &[Rate::Audio])?;
        Ok(unit_spec(MoogVcf {
            fco: 0.0,
            res: 0.0,
            xnm1: 0.0,
            y1nm1: 0.0,
            y2nm1: 0.0,
            y3nm1: 0.0,
            y1n: 0.0,
            y2n: 0.0,
            y3n: 0.0,
            y4n: 0.0,
            kp: 0.0,
            pp1d2: 0.0,
            k: 0.0,
            last_cutoff: 440.0,
            last_res: 0.0,
        }))
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn zap_f32(value: f32) -> f32 {
    let magnitude = value.abs();
    if magnitude > 1e-15 && magnitude < 1e15 {
        value
    } else {
        0.0
    }
}

#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const DFM1_LUT: [f64; 1536] = [
    9.99973611184550000e-01, 9.99912919551707100e-01, 1.30301311049131200e+00,
    9.99972786565934400e-01, 9.99910198478017100e-01, 1.30301257325016030e+00,
    9.99971961903567300e-01, 9.99907477265117800e-01, 1.30301203598286030e+00,
    9.99971137239655700e-01, 9.99904756052283000e-01, 1.30301149871292640e+00,
    9.99970312575524800e-01, 9.99902034843886100e-01, 1.30301096144352520e+00,
    9.99969487910093500e-01, 9.99899313636360100e-01, 1.30301042417364330e+00,
    9.99968663244606900e-01, 9.99896592433812500e-01, 1.30300988690454340e+00,
    9.99967838578037500e-01, 9.99893871232854100e-01, 1.30300934963539580e+00,
    9.99967013910486100e-01, 9.99891150033816800e-01, 1.30300881236453910e+00,
    9.99966189242048000e-01, 9.99888428837015200e-01, 1.30300827509508380e+00,
    9.99965364572813800e-01, 9.99885707642747900e-01, 1.30300773782445510e+00,
    9.99964539902869400e-01, 9.99882986451298900e-01, 1.30300720055431160e+00,
    9.99963715233267700e-01, 9.99880265266142000e-01, 1.30300666328409300e+00,
    9.99962890561218700e-01, 9.99877544078071600e-01, 1.30300612601248940e+00,
    9.99962065889621200e-01, 9.99874822896652900e-01, 1.30300558874206730e+00,
    9.99961241215807000e-01, 9.99872101713080900e-01, 1.30300505147056530e+00,
    9.99960416542543800e-01, 9.99869380536488400e-01, 1.30300451419879870e+00,
    9.99959591867274300e-01, 9.99866659358437900e-01, 1.30300397692695210e+00,
    9.99958767191797200e-01, 9.99863938184863700e-01, 1.30300343965540890e+00,
    9.99957942515340000e-01, 9.99861217013217000e-01, 1.30300290238334200e+00,
    9.99957117838792600e-01, 9.99858495846434500e-01, 1.30300236511203440e+00,
    9.99956293161410700e-01, 9.99855774682059700e-01, 1.30300182783921440e+00,
    9.99955468483262800e-01, 9.99853053520319300e-01, 1.30300129056723010e+00,
    9.99954643804415000e-01, 9.99850332361431300e-01, 1.30300075329585230e+00,
    9.99953819124175900e-01, 9.99847611203113600e-01, 1.30300021602270900e+00,
    9.99952994442620500e-01, 9.99844890045614600e-01, 1.30299967874918300e+00,
    9.99952169762006500e-01, 9.99842168896383100e-01, 1.30299914147677830e+00,
    9.99951345078787000e-01, 9.99839447743716600e-01, 1.30299860420315980e+00,
    9.99950520396571000e-01, 9.99836726599522200e-01, 1.30299806693015100e+00,
    9.99949695711930700e-01, 9.99834005452490600e-01, 1.30299752965640710e+00,
    9.99948871026992000e-01, 9.99831284309636100e-01, 1.30299699238267960e+00,
    9.99948046341804700e-01, 9.99828563171122700e-01, 1.30299645510862100e+00,
    9.99947221655758900e-01, 9.99825842034938400e-01, 1.30299591783491350e+00,
    9.99945572328680300e-01, 9.99820399926579200e-01, 1.30299484331751310e+00,
    9.99943922953122800e-01, 9.99814957678905800e-01, 1.30299376876996100e+00,
    9.99942273573416000e-01, 9.99809515438188200e-01, 1.30299269422093470e+00,
    9.99940624189884200e-01, 9.99804073205495800e-01, 1.30299161967028290e+00,
    9.99938974804579400e-01, 9.99798630987601000e-01, 1.30299054512099890e+00,
    9.99937325414909400e-01, 9.99793188775949500e-01, 1.30298947056971540e+00,
    9.99935676022830700e-01, 9.99787746576997400e-01, 1.30298839601841850e+00,
    9.99934026627513900e-01, 9.99782304388007400e-01, 1.30298732146551480e+00,
    9.99932377229217500e-01, 9.99776862209833600e-01, 1.30298624691255770e+00,
    9.99930727827423000e-01, 9.99771420040765000e-01, 1.30298517235910370e+00,
    9.99929078422631900e-01, 9.99765977882455900e-01, 1.30298409780447640e+00,
    9.99927429015078900e-01, 9.99760535735680800e-01, 1.30298302324935760e+00,
    9.99925779604039300e-01, 9.99755093598049100e-01, 1.30298194869371530e+00,
    9.99924130189748600e-01, 9.99749651470337900e-01, 1.30298087413713780e+00,
    9.99922480772432000e-01, 9.99744209353289700e-01, 1.30297979957983420e+00,
    9.99920831352304400e-01, 9.99738767247614200e-01, 1.30297872502211880e+00,
    9.99919181928704300e-01, 9.99733325151128800e-01, 1.30297765046352840e+00,
    9.99917532502697100e-01, 9.99727883067348400e-01, 1.30297657590463540e+00,
    9.99915883072391400e-01, 9.99722440990033000e-01, 1.30297550134461470e+00,
    9.99914233639642600e-01, 9.99716998925304400e-01, 1.30297442678451980e+00,
    9.99912584203434000e-01, 9.99711556869807800e-01, 1.30297335222319700e+00,
    9.99910934764749100e-01, 9.99706114826788400e-01, 1.30297227766120870e+00,
    9.99909285322033000e-01, 9.99700672791116400e-01, 1.30297120309864930e+00,
    9.99907635876810400e-01, 9.99695230767821500e-01, 1.30297012853527350e+00,
    9.99905986428510400e-01, 9.99689788755021000e-01, 1.30296905397144310e+00,
    9.99904336976947000e-01, 9.99684346752101500e-01, 1.30296797940675880e+00,
    9.99902687521941200e-01, 9.99678904758471900e-01, 1.30296690484151180e+00,
    9.99901038064023200e-01, 9.99673462775881400e-01, 1.30296583027547850e+00,
    9.99899388603013600e-01, 9.99668020803739300e-01, 1.30296475570913300e+00,
    9.99897739139249600e-01, 9.99662578843156700e-01, 1.30296368114185200e+00,
    9.99896089671719400e-01, 9.99657136890796400e-01, 1.30296260657397500e+00,
    9.99894440201253600e-01, 9.99651694949398400e-01, 1.30296153200516400e+00,
    9.99891141444042100e-01, 9.99640811734670600e-01, 1.30295938299150030e+00,
    9.99887842487511600e-01, 9.99629927944940500e-01, 1.30295723385341320e+00,
    9.99884543518015400e-01, 9.99619044195025900e-01, 1.30295508471237280e+00,
    9.99881244536053900e-01, 9.99608160486578100e-01, 1.30295293556865700e+00,
    9.99877945541516500e-01, 9.99597276819233400e-01, 1.30295078642224180e+00,
    9.99874646534866100e-01, 9.99586393194518000e-01, 1.30294863727286980e+00,
    9.99871347515162600e-01, 9.99575509609332600e-01, 1.30294648812107400e+00,
    9.99868048483260400e-01, 9.99564626066495900e-01, 1.30294433896682100e+00,
    9.99864749439055200e-01, 9.99553742565664300e-01, 1.30294218980960340e+00,
    9.99861450381684100e-01, 9.99542859103991300e-01, 1.30294004064919100e+00,
    9.99858151312191500e-01, 9.99531975684923800e-01, 1.30293789148652680e+00,
    9.99854852230358700e-01, 9.99521092307740800e-01, 1.30293574232112990e+00,
    9.99851553135740900e-01, 9.99510208970974800e-01, 1.30293359315281740e+00,
    9.99848254028608800e-01, 9.99499325675520000e-01, 1.30293144398217100e+00,
    9.99844954908995300e-01, 9.99488442421485200e-01, 1.30292929480844300e+00,
    9.99841655776932000e-01, 9.99477559208975400e-01, 1.30292714563220110e+00,
    9.99838356632449700e-01, 9.99466676038093100e-01, 1.30292499645336800e+00,
    9.99835057475260000e-01, 9.99455792907887000e-01, 1.30292284727145310e+00,
    9.99831758305710900e-01, 9.99444909819505600e-01, 1.30292069808716930e+00,
    9.99828459123321000e-01, 9.99434026771362000e-01, 1.30291854889993700e+00,
    9.99825159928828100e-01, 9.99423143765890400e-01, 1.30291639971006010e+00,
    9.99821860721765600e-01, 9.99412260801551700e-01, 1.30291425051767000e+00,
    9.99818561501973300e-01, 9.99401377877818500e-01, 1.30291210132212120e+00,
    9.99815262269675300e-01, 9.99390494995430500e-01, 1.30290995212411630e+00,
    9.99811963024902500e-01, 9.99379612154490600e-01, 1.30290780292338940e+00,
    9.99808663767685300e-01, 9.99368729355099000e-01, 1.30290565372003190e+00,
    9.99805364497874400e-01, 9.99357846596764400e-01, 1.30290350451376850e+00,
    9.99802065215677900e-01, 9.99346963880173500e-01, 1.30290135530497640e+00,
    9.99798765920692500e-01, 9.99336081203996400e-01, 1.30289920609313570e+00,
    9.99795466613548200e-01, 9.99325198570312300e-01, 1.30289705687910230e+00,
    9.99792167293685000e-01, 9.99314315977274000e-01, 1.30289490766200420e+00,
    9.99788867961301300e-01, 9.99303433425537100e-01, 1.30289275844229070e+00,
    9.99782270031230800e-01, 9.99281670993145200e-01, 1.30288846049761140e+00,
    9.99775671302410300e-01, 9.99259906256557900e-01, 1.30288416205438250e+00,
    9.99769072523316900e-01, 9.99238141684554900e-01, 1.30287986360025390e+00,
    9.99762473694331700e-01, 9.99216377278397600e-01, 1.30287556513540470e+00,
    9.99755874815162100e-01, 9.99194613037126300e-01, 1.30287126665956230e+00,
    9.99749275885748600e-01, 9.99172848960548100e-01, 1.30286696817275780e+00,
    9.99742676906588900e-01, 9.99151085050310600e-01, 1.30286266967531800e+00,
    9.99736077877152700e-01, 9.99129321304668600e-01, 1.30285837116681000e+00,
    9.99729478797594000e-01, 9.99107557724135400e-01, 1.30285407264754280e+00,
    9.99722879668190500e-01, 9.99085794309631100e-01, 1.30284977411762170e+00,
    9.99716280488464100e-01, 9.99064031059583000e-01, 1.30284547557647420e+00,
    9.99709681258691100e-01, 9.99042267974908500e-01, 1.30284117702454980e+00,
    9.99703081978961500e-01, 9.99020505055907900e-01, 1.30283687846188300e+00,
    9.99696482649016900e-01, 9.98998742301734100e-01, 1.30283257988830250e+00,
    9.99689883269010900e-01, 9.98976979712898600e-01, 1.30282828130392800e+00,
    9.99683283838927900e-01, 9.98955217289354200e-01, 1.30282398270855240e+00,
    9.99676684358865300e-01, 9.98933455031426800e-01, 1.30281968410246060e+00,
    9.99670084828707600e-01, 9.98911692938740400e-01, 1.30281538548549300e+00,
    9.99663485248347800e-01, 9.98889931010946900e-01, 1.30281108685756800e+00,
    9.99656885617840000e-01, 9.98868169248229900e-01, 1.30280678821883600e+00,
    9.99650285937540200e-01, 9.98846407651766400e-01, 1.30280248956927710e+00,
    9.99643686206911200e-01, 9.98824646219791400e-01, 1.30279819090890720e+00,
    9.99637086426258500e-01, 9.98802884953315600e-01, 1.30279389223759500e+00,
    9.99630486595501600e-01, 9.98781123852079300e-01, 1.30278959355543030e+00,
    9.99623886714613400e-01, 9.98759362915997700e-01, 1.30278529486240080e+00,
    9.99617286783708400e-01, 9.98737602145452000e-01, 1.30278099615852860e+00,
    9.99610686802631900e-01, 9.98715841539938000e-01, 1.30277669744384260e+00,
    9.99604086771590300e-01, 9.98694081100140500e-01, 1.30277239871828270e+00,
    9.99597486690311500e-01, 9.98672320825166900e-01, 1.30276809998183960e+00,
    9.99590886558961000e-01, 9.98650560715567800e-01, 1.30276380123448530e+00,
    9.99584286377702800e-01, 9.98628800771888300e-01, 1.30275950247644310e+00,
    9.99577686146082300e-01, 9.98607040992634900e-01, 1.30275520370739750e+00,
    9.99564488622571900e-01, 9.98563532116906600e-01, 1.30274660814915400e+00,
    9.99551287904108400e-01, 9.98520014029949000e-01, 1.30273801059729370e+00,
    9.99538086983769200e-01, 9.98476496599362800e-01, 1.30272941300090280e+00,
    9.99524885863075600e-01, 9.98432979830201200e-01, 1.30272081536110230e+00,
    9.99511684541998900e-01, 9.98389463722406900e-01, 1.30271221767802170e+00,
    9.99498483020435800e-01, 9.98345948275677200e-01, 1.30270361995146430e+00,
    9.99485281298456600e-01, 9.98302433490281200e-01, 1.30269502218142840e+00,
    9.99472079376225100e-01, 9.98258919366796300e-01, 1.30268642436815300e+00,
    9.99458877253338500e-01, 9.98215405903932700e-01, 1.30267782651129930e+00,
    9.99445674930177300e-01, 9.98171893102981700e-01, 1.30266922861121380e+00,
    9.99432472406418700e-01, 9.98128380962916800e-01, 1.30266063066754370e+00,
    9.99419269682301200e-01, 9.98084869484561400e-01, 1.30265203268062320e+00,
    9.99406066757785900e-01, 9.98041358667825200e-01, 1.30264343465025200e+00,
    9.99392863632778900e-01, 9.97997848512436200e-01, 1.30263483657651460e+00,
    9.99379660307334200e-01, 9.97954339018609900e-01, 1.30262623845941740e+00,
    9.99366456781284000e-01, 9.97910830185830700e-01, 1.30261764029882900e+00,
    9.99353253054850700e-01, 9.97867322014868700e-01, 1.30260904209491280e+00,
    9.99340049127905400e-01, 9.97823814505337400e-01, 1.30260044384755760e+00,
    9.99326845000511000e-01, 9.97780307657481300e-01, 1.30259184555689880e+00,
    9.99313640672579800e-01, 9.97736801471048300e-01, 1.30258324722280400e+00,
    9.99300436144132200e-01, 9.97693295946144100e-01, 1.30257464884529470e+00,
    9.99287231415243000e-01, 9.97649791083052200e-01, 1.30256605042450270e+00,
    9.99274026485749600e-01, 9.97606286881274400e-01, 1.30255745196026700e+00,
    9.99260821355856300e-01, 9.97562783341520900e-01, 1.30254885345267100e+00,
    9.99247616025349300e-01, 9.97519280463126000e-01, 1.30254025490175150e+00,
    9.99234410494258200e-01, 9.97475778246223800e-01, 1.30253165630733330e+00,
    9.99221204762686200e-01, 9.97432276691191600e-01, 1.30252305766960700e+00,
    9.99207998830565000e-01, 9.97388775797843000e-01, 1.30251445898853900e+00,
    9.99194792697945900e-01, 9.97345275566383400e-01, 1.30250586026408040e+00,
    9.99181586364716900e-01, 9.97301775996482300e-01, 1.30249726149625960e+00,
    9.99168379830883900e-01, 9.97258277088196000e-01, 1.30248866268503250e+00,
    9.99155173096585400e-01, 9.97214778842018600e-01, 1.30248006383051960e+00,
    9.99128771393024800e-01, 9.97127825063639400e-01, 1.30246287404318380e+00,
    9.99102356907417800e-01, 9.97040834476159000e-01, 1.30244567628250600e+00,
    9.99075941608295300e-01, 9.96953846499333600e-01, 1.30242847834113260e+00,
    9.99049525506666800e-01, 9.96866861169722900e-01, 1.30241128022634770e+00,
    9.99023108602626000e-01, 9.96779878487934800e-01, 1.30239408193819560e+00,
    9.98996690896165700e-01, 9.96692898454246200e-01, 1.30237688347674680e+00,
    9.98970272387072700e-01, 9.96605921068255300e-01, 1.30235968484187460e+00,
    9.98943853075436400e-01, 9.96518946330557000e-01, 1.30234248603375960e+00,
    9.98917432961158100e-01, 9.96431974241126400e-01, 1.30232528705234140e+00,
    9.98891012044162200e-01, 9.96345004800015000e-01, 1.30230808789759260e+00,
    9.98864590324447200e-01, 9.96258038007517900e-01, 1.30229088856957300e+00,
    9.98838167801934100e-01, 9.96171073863675900e-01, 1.30227368906828780e+00,
    9.98811744476537300e-01, 9.96084112368507000e-01, 1.30225648939372070e+00,
    9.98785320348329300e-01, 9.95997153522550700e-01, 1.30223928954594030e+00,
    9.98758895417139500e-01, 9.95910197325545400e-01, 1.30222208952491950e+00,
    9.98732469683004200e-01, 9.95823243777911800e-01, 1.30220488933072880e+00,
    9.98706043145820500e-01, 9.95736292879610500e-01, 1.30218768896331240e+00,
    9.98679615805544800e-01, 9.95649344630799500e-01, 1.30217048842272190e+00,
    9.98653187662156200e-01, 9.95562399031709500e-01, 1.30215328770892970e+00,
    9.98626758715608000e-01, 9.95475456082488100e-01, 1.30213608682202420e+00,
    9.98600328965797300e-01, 9.95388515783096800e-01, 1.30211888576194550e+00,
    9.98573898412707200e-01, 9.95301578133780600e-01, 1.30210168452875560e+00,
    9.98547467056322700e-01, 9.95214643134790400e-01, 1.30208448312244580e+00,
    9.98521034896552000e-01, 9.95127710786125000e-01, 1.30206728154306300e+00,
    9.98494601933339900e-01, 9.95040781087903400e-01, 1.30205007979055200e+00,
    9.98468168166681400e-01, 9.94953854040408900e-01, 1.30203287786500320e+00,
    9.98441733596511100e-01, 9.94866929643727800e-01, 1.30201567576638800e+00,
    9.98415298222722000e-01, 9.94780007897808500e-01, 1.30199847349471100e+00,
    9.98388862045366100e-01, 9.94693088803122300e-01, 1.30198127105001850e+00,
    9.98362425064320900e-01, 9.94606172359567200e-01, 1.30196406843231420e+00,
    9.98335987279573800e-01, 9.94519258567402400e-01, 1.30194686564161040e+00,
    9.98309548691066700e-01, 9.94432347426736900e-01, 1.30192966267792150e+00,
    9.98256718632182600e-01, 9.94258695893545500e-01, 1.30189528845773460e+00,
    9.98203837428590300e-01, 9.94084897415052100e-01, 1.30186088235971660e+00,
    9.98150952919595100e-01, 9.93911109250355800e-01, 1.30182647551177300e+00,
    9.98098065194435700e-01, 9.93737331695170000e-01, 1.30179206797237310e+00,
    9.98045174252842900e-01, 9.93563564751015700e-01, 1.30175765974169890e+00,
    9.97992280094473400e-01, 9.93389808419169700e-01, 1.30172325081989930e+00,
    9.97939382718916400e-01, 9.93216062700687600e-01, 1.30168884120706570e+00,
    9.97886482125783600e-01, 9.93042327596699600e-01, 1.30165443090328600e+00,
    9.97833578314691800e-01, 9.92868603108352800e-01, 1.30162001990872360e+00,
    9.97780671285215300e-01, 9.92694889236655600e-01, 1.30158560822346180e+00,
    9.97727761036999000e-01, 9.92521185982847300e-01, 1.30155119584760630e+00,
    9.97674847569647100e-01, 9.92347493348035300e-01, 1.30151678278129400e+00,
    9.97621930882741900e-01, 9.92173811333254600e-01, 1.30148236902460700e+00,
    9.97569010975918100e-01, 9.92000139939712900e-01, 1.30144795457766700e+00,
    9.97516087848782200e-01, 9.91826479168525800e-01, 1.30141353944061720e+00,
    9.97463161500935600e-01, 9.91652829020792200e-01, 1.30137912361353970e+00,
    9.97410231931982300e-01, 9.91479189497620300e-01, 1.30134470709656910e+00,
    9.97357299141496300e-01, 9.91305560600019600e-01, 1.30131028988977440e+00,
    9.97304363129160400e-01, 9.91131942329357200e-01, 1.30127587199331680e+00,
    9.97251423894524300e-01, 9.90958334686564200e-01, 1.30124145340727050e+00,
    9.97198481437225600e-01, 9.90784737672860600e-01, 1.30120703413179050e+00,
    9.97145535756862800e-01, 9.90611151289337500e-01, 1.30117261416697020e+00,
    9.97092586853015900e-01, 9.90437575537026500e-01, 1.30113819351289070e+00,
    9.97039634725348400e-01, 9.90264010417232700e-01, 1.30110377216972920e+00,
    9.96986679373400000e-01, 9.90090455930856600e-01, 1.30106935013753680e+00,
    9.96933720796822200e-01, 9.89916912079164500e-01, 1.30103492741644320e+00,
    9.96880758995221000e-01, 9.89743378863275200e-01, 1.30100050400659080e+00,
    9.96827793968199000e-01, 9.89569856284295400e-01, 1.30096607990808100e+00,
    9.96774825715328500e-01, 9.89396344343233800e-01, 1.30093165512099660e+00,
    9.96721854236254800e-01, 9.89222843041337700e-01, 1.30089722964547640e+00,
    9.96668879530577600e-01, 9.89049352379706600e-01, 1.30086280348164360e+00,
    9.96615901597890300e-01, 9.88875872359418400e-01, 1.30082837662958230e+00,
    9.96510134666229800e-01, 9.88529594432026900e-01, 1.30075964991566060e+00,
    9.96404163024114000e-01, 9.88182731055582700e-01, 1.30069079581541120e+00,
    9.96298177750633600e-01, 9.87835907915101100e-01, 1.30062193849904600e+00,
    9.96192179558646900e-01, 9.87489127363469900e-01, 1.30055307843269690e+00,
    9.96086168447792100e-01, 9.87142389418795000e-01, 1.30048421561910100e+00,
    9.95980144414921200e-01, 9.86795694090070700e-01, 1.30041535005915980e+00,
    9.95874107456887800e-01, 9.86449041386300600e-01, 1.30034648175379820e+00,
    9.95768057570501400e-01, 9.86102431316352000e-01, 1.30027761070391650e+00,
    9.95661994752623300e-01, 9.85755863889265300e-01, 1.30020873691043400e+00,
    9.95555919000095900e-01, 9.85409339114025200e-01, 1.30013986037426380e+00,
    9.95449830309733900e-01, 9.85062856999531900e-01, 1.30007098109630030e+00,
    9.95343728678377000e-01, 9.84716417554773500e-01, 1.30000209907748030e+00,
    9.95237614102863700e-01, 9.84370020788739500e-01, 1.29993321431871100e+00,
    9.95131486580029500e-01, 9.84023666710414200e-01, 1.29986432682091820e+00,
    9.95025346106682800e-01, 9.83677355328700700e-01, 1.29979543658499150e+00,
    9.94919192679664100e-01, 9.83331086652611500e-01, 1.29972654361186500e+00,
    9.94813026295800700e-01, 9.82984860691122500e-01, 1.29965764790245400e+00,
    9.94706846951908900e-01, 9.82638677453179100e-01, 1.29958874945767920e+00,
    9.94600654644828600e-01, 9.82292536947809200e-01, 1.29951984827845930e+00,
    9.94494449371355800e-01, 9.81946439183903500e-01, 1.29945094436571250e+00,
    9.94388231128318300e-01, 9.81600384170462100e-01, 1.29938203772034900e+00,
    9.94281999912550700e-01, 9.81254371916512900e-01, 1.29931312834330460e+00,
    9.94175755720855700e-01, 9.80908402430985500e-01, 1.29924421623549710e+00,
    9.94069498550039700e-01, 9.80562475722827300e-01, 1.29917530139783690e+00,
    9.93963228396933600e-01, 9.80216591801071600e-01, 1.29910638383126060e+00,
    9.93856945258324700e-01, 9.79870750674614600e-01, 1.29903746353667660e+00,
    9.93750649131043400e-01, 9.79524952352499600e-01, 1.29896854051502620e+00,
    9.93644340011890800e-01, 9.79179196843679800e-01, 1.29889961476722160e+00,
    9.93538017897676500e-01, 9.78833484157141100e-01, 1.29883068629419360e+00,
    9.93431682785202300e-01, 9.78487814301851600e-01, 1.29876175509686930e+00,
    9.93325334671272800e-01, 9.78142187286791900e-01, 1.29869282117617260e+00,
    9.93218973552676900e-01, 9.77796603120898700e-01, 1.29862388453302050e+00,
    9.93007010766083100e-01, 9.77108156157434200e-01, 1.29848652049932060e+00,
    9.92794228081488500e-01, 9.76417386068298000e-01, 1.29834864793419040e+00,
    9.92581387580731300e-01, 9.75726768858368500e-01, 1.29821076077715200e+00,
    9.92368494954446700e-01, 9.75036323161366800e-01, 1.29807286273984410e+00,
    9.92155550221139000e-01, 9.74346049192482700e-01, 1.29793495385834900e+00,
    9.91942553355481000e-01, 9.73655947024738200e-01, 1.29779703414039730e+00,
    9.91729504331766200e-01, 9.72966016730016600e-01, 1.29765910359347680e+00,
    9.91516403124255700e-01, 9.72276258380188800e-01, 1.29752116222509860e+00,
    9.91303249707179200e-01, 9.71586672047116200e-01, 1.29738321004277050e+00,
    9.91090044054759300e-01, 9.70897257802730200e-01, 1.29724524705403280e+00,
    9.90876786141167700e-01, 9.70208015718889100e-01, 1.29710727326641700e+00,
    9.90663475940568400e-01, 9.69518945867520200e-01, 1.29696928868747500e+00,
    9.90450113427089000e-01, 9.68830048320524900e-01, 1.29683129332476000e+00,
    9.90236698574844500e-01, 9.68141323149857000e-01, 1.29669328718585760e+00,
    9.90023231357906400e-01, 9.67452770427422500e-01, 1.29655527027834160e+00,
    9.89809711750325600e-01, 9.66764390225154900e-01, 1.29641724260980280e+00,
    9.89596139726141200e-01, 9.66076182615040800e-01, 1.29627920418785170e+00,
    9.89382515259343100e-01, 9.65388147669002900e-01, 1.29614115502010200e+00,
    9.89168838323911400e-01, 9.64700285459025100e-01, 1.29600309511417600e+00,
    9.88955108893787300e-01, 9.64012596057058900e-01, 1.29586502447771260e+00,
    9.88741326942897000e-01, 9.63325079535101200e-01, 1.29572694311835780e+00,
    9.88527492445136200e-01, 9.62637735965144700e-01, 1.29558885104377920e+00,
    9.88313605374371700e-01, 9.61950565419182100e-01, 1.29545074826163530e+00,
    9.88099665704454100e-01, 9.61263567969246800e-01, 1.29531263477961800e+00,
    9.87885673409183700e-01, 9.60576743687306500e-01, 1.29517451060540820e+00,
    9.87671628462358900e-01, 9.59890092645415100e-01, 1.29503637574671870e+00,
    9.87457530837741700e-01, 9.59203614915603200e-01, 1.29489823021125930e+00,
    9.87243380509066200e-01, 9.58517310569907100e-01, 1.29476007400675570e+00,
    9.87029177450042400e-01, 9.57831179680378900e-01, 1.29462190714094970e+00,
    9.86814921634353200e-01, 9.57145222319079000e-01, 1.29448372962158340e+00,
    9.86600613035654000e-01, 9.56459438558072300e-01, 1.29434554145641820e+00,
    9.86386251627573300e-01, 9.55773828469433200e-01, 1.29420734265322400e+00,
    9.85960596783334800e-01, 9.54413436347224200e-01, 1.29393299241290350e+00,
    9.85531653839846000e-01, 9.53043901274378800e-01, 1.29365661636488400e+00,
    9.85102453109081300e-01, 9.51674914660365600e-01, 1.29338016821743480e+00,
    9.84673039912246200e-01, 9.50306622539981300e-01, 1.29310367737667130e+00,
    9.84243414742143500e-01, 9.48939027740132300e-01, 1.29282714435985560e+00,
    9.83813577397414600e-01, 9.47572130873879600e-01, 1.29255056923752560e+00,
    9.83383527665210800e-01, 9.46205932519170500e-01, 1.29227395207327160e+00,
    9.82953265332075900e-01, 9.44840433253541500e-01, 1.29199729293076060e+00,
    9.82522790184113600e-01, 9.43475633654652000e-01, 1.29172059187384080e+00,
    9.82092102006984800e-01, 9.42111534300285500e-01, 1.29144384896654320e+00,
    9.81661200585923300e-01, 9.40748135768397300e-01, 1.29116706427308550e+00,
    9.81230085705717200e-01, 9.39385438637065100e-01, 1.29089023785787500e+00,
    9.80798757150715500e-01, 9.38023443484510100e-01, 1.29061336978550130e+00,
    9.80367214704830700e-01, 9.36662150889111300e-01, 1.29033646012074650e+00,
    9.79935458151527600e-01, 9.35301561429374900e-01, 1.29005950892857700e+00,
    9.79503487273838500e-01, 9.33941675683983800e-01, 1.28978251627415540e+00,
    9.79071301854339000e-01, 9.32582494231728600e-01, 1.28950548222282450e+00,
    9.78638901675163100e-01, 9.31224017651556900e-01, 1.28922840684012200e+00,
    9.78206286518004800e-01, 9.29866246522584300e-01, 1.28895129019178060e+00,
    9.77773456164103200e-01, 9.28509181424050800e-01, 1.28867413234371830e+00,
    9.77340410394255800e-01, 9.27152822935366500e-01, 1.28839693336205020e+00,
    9.76907148988801400e-01, 9.25797171636063900e-01, 1.28811969331308470e+00,
    9.76473671727634100e-01, 9.24442228105843800e-01, 1.28784241226331720e+00,
    9.76039978390193200e-01, 9.23087992924548700e-01, 1.28756509027944400e+00,
    9.75606068755466400e-01, 9.21734466672179000e-01, 1.28728772742835540e+00,
    9.75171942601986800e-01, 9.20381649928884300e-01, 1.28701032377713730e+00,
    9.74737599707829900e-01, 9.19029543274961600e-01, 1.28673287939307080e+00,
    9.74303039850617100e-01, 9.17678147290867900e-01, 1.28645539434363390e+00,
    9.73868262807506500e-01, 9.16327462557199400e-01, 1.28617786869650530e+00,
    9.73433268355201000e-01, 9.14977489654716000e-01, 1.28590030251955430e+00,
    9.72998056269946400e-01, 9.13628229164343400e-01, 1.28562269588086050e+00,
    9.72562626327520700e-01, 9.12279681667144100e-01, 1.28534504884869600e+00,
    9.71704287013631000e-01, 9.09625427342829700e-01, 1.28479802737030080e+00,
    9.70832726368644300e-01, 9.06935800681394000e-01, 1.28424297268152900e+00,
    9.69959919145027200e-01, 9.04247886847342100e-01, 1.28368752124065780e+00,
    9.69086224588880800e-01, 9.01562806072346000e-01, 1.28313190364225880e+00,
    9.68211652104010600e-01, 8.98880597603478000e-01, 1.28257612759148350e+00,
    9.67336200227753900e-01, 8.96201267205143100e-01, 1.28202019388306580e+00,
    9.66459867142823400e-01, 8.93524819576241700e-01, 1.28146410309425300e+00,
    9.65582651012943200e-01, 8.90851259383234700e-01, 1.28090785579840530e+00,
    9.64704549993828400e-01, 8.88180591294008000e-01, 1.28035145257202320e+00,
    9.63825562233534200e-01, 8.85512819979082500e-01, 1.27979489399501630e+00,
    9.62945685872410000e-01, 8.82847950111617700e-01, 1.27923818065072430e+00,
    9.62064919043075500e-01, 8.80185986367481900e-01, 1.27868131312595300e+00,
    9.61183259870372100e-01, 8.77526933425249900e-01, 1.27812429201099000e+00,
    9.60300706471325800e-01, 8.74870795966231400e-01, 1.27756711789963300e+00,
    9.59417256955102600e-01, 8.72217578674484200e-01, 1.27700979138920930e+00,
    9.58532909422969400e-01, 8.69567286236842100e-01, 1.27645231308060800e+00,
    9.57647661968253500e-01, 8.66919923342937000e-01, 1.27589468357829650e+00,
    9.56761512676299700e-01, 8.64275494685218500e-01, 1.27533690349035120e+00,
    9.55874459624427700e-01, 8.61634004958973200e-01, 1.27477897342847820e+00,
    9.54986500881891100e-01, 8.58995458862351400e-01, 1.27422089400804370e+00,
    9.54097634509834500e-01, 8.56359861096386700e-01, 1.27366266584809630e+00,
    9.53207858561250200e-01, 8.53727216365016500e-01, 1.27310428957139130e+00,
    9.52317171080937300e-01, 8.51097529375111300e-01, 1.27254576580441950e+00,
    9.51425570105453900e-01, 8.48470804836484500e-01, 1.27198709517743280e+00,
    9.50533053663077500e-01, 8.45847047461924600e-01, 1.27142827832446880e+00,
    9.49639619773759200e-01, 8.43226261967214400e-01, 1.27086931588337840e+00,
    9.48745266449082700e-01, 8.40608453071160700e-01, 1.27031020849585330e+00,
    9.47849991692212600e-01, 8.37993625495598100e-01, 1.26975095680744880e+00,
    9.46953793497859800e-01, 8.35381783965441000e-01, 1.26919156146762100e+00,
    9.46056669852226800e-01, 8.32772933208677500e-01, 1.26863202312974000e+00,
    9.45158618732970100e-01, 8.30167077956415200e-01, 1.26807234245112980e+00,
    9.44259638109148400e-01, 8.27564222942887900e-01, 1.26751252009308810e+00,
    9.42407127241941500e-01, 8.22218519269909400e-01, 1.26636028885103200e+00,
    9.40600811776073200e-01, 8.17029339732315500e-01, 1.26523858778358300e+00,
    9.38790920098724700e-01, 8.11852823848982200e-01, 1.26411645064054580e+00,
    9.36977249555939400e-01, 8.06688474369130700e-01, 1.26299376739359430e+00,
    9.35159782512925400e-01, 8.01536327243576200e-01, 1.26187054316005680e+00,
    9.33338501924700600e-01, 7.96396420585767000e-01, 1.26074678358204870e+00,
    9.31513390602348100e-01, 7.91268792573652200e-01, 1.25962249437312930e+00,
    9.29684431208333400e-01, 7.86153481441676400e-01, 1.25849768131737850e+00,
    9.27851606254876400e-01, 7.81050525481562200e-01, 1.25737235027040020e+00,
    9.26014898102306900e-01, 7.75959963043118000e-01, 1.25624650716034370e+00,
    9.24174288957411900e-01, 7.70881832535087400e-01, 1.25512015798894620e+00,
    9.22329760871755000e-01, 7.65816172425989100e-01, 1.25399330883259050e+00,
    9.20481295739966000e-01, 7.60763021244949900e-01, 1.25286596584337670e+00,
    9.18628875298023500e-01, 7.55722417582585600e-01, 1.25173813525021950e+00,
    9.16772481121504300e-01, 7.50694400091858400e-01, 1.25060982335995140e+00,
    9.14912094623815000e-01, 7.45679007488964300e-01, 1.24948103655845630e+00,
    9.13047697054394600e-01, 7.40676278554211600e-01, 1.24835178131181100e+00,
    9.11179269496900900e-01, 7.35686252132925500e-01, 1.24722206416745200e+00,
    9.09306792867369800e-01, 7.30708967136358900e-01, 1.24609189175535920e+00,
    9.07430247912346700e-01, 7.25744462542599300e-01, 1.24496127078925720e+00,
    9.05549615207000300e-01, 7.20792777397505100e-01, 1.24383020806783740e+00,
    9.03664875153207300e-01, 7.15853950815643000e-01, 1.24269871047600430e+00,
    9.01776007977611400e-01, 7.10928021981229700e-01, 1.24156678498613450e+00,
    8.99882993729657100e-01, 7.06015030149089100e-01, 1.24043443865936040e+00,
    8.97985812279600500e-01, 7.01115014645623800e-01, 1.23930167864687650e+00,
    8.96084443316489300e-01, 6.96228014869788100e-01, 1.23816851219126510e+00,
    8.94178866346119400e-01, 6.91354070294079900e-01, 1.23703494662784430e+00,
    8.92269060688961200e-01, 6.86493220465530200e-01, 1.23590098938603700e+00,
    8.90355005478060700e-01, 6.81645505006716500e-01, 1.23476664799076530e+00,
    8.88436679656912300e-01, 6.76810963616779200e-01, 1.23363193006386520e+00,
    8.86514061977304000e-01, 6.71989636072452400e-01, 1.23249684332552770e+00,
    8.84587130997128400e-01, 6.67181562229093900e-01, 1.23136139559576050e+00,
    8.80750573515689700e-01, 6.57680074435819300e-01, 1.22910723516129570e+00,
    8.76868793939248500e-01, 6.48162988649492800e-01, 1.22683540372859110e+00,
    8.72968981926658600e-01, 6.38698773385583300e-01, 1.22456208130162910e+00,
    8.69051385864433500e-01, 6.29288805865015700e-01, 1.22228758754191800e+00,
    8.65115830167085200e-01, 6.19933427371933200e-01, 1.22001199717336520e+00,
    8.61162128875544900e-01, 6.10632963535324700e-01, 1.21773538270066920e+00,
    8.57190092451050900e-01, 6.01387741141420700e-01, 1.21545781848106140e+00,
    8.53199527817002000e-01, 5.92198088463378100e-01, 1.21317938085360240e+00,
    8.49190238279681900e-01, 5.83064335303817700e-01, 1.21090014820118500e+00,
    8.45162023444613300e-01, 5.73986813032632800e-01, 1.20862020101329800e+00,
    8.41114679130674500e-01, 5.64965854625300400e-01, 1.20633962195091170e+00,
    8.37047997281959100e-01, 5.56001794701777000e-01, 1.20405849591356940e+00,
    8.32961765877328100e-01, 5.47094969565981800e-01, 1.20177691010877850e+00,
    8.28855768837592700e-01, 5.38245717245846400e-01, 1.19949495412378000e+00,
    8.24729785930262900e-01, 5.29454377533907600e-01, 1.19721271999977930e+00,
    8.20583592671812900e-01, 5.20721292028445600e-01, 1.19493030230873600e+00,
    8.16416960227391900e-01, 5.12046804175122800e-01, 1.19264779823279500e+00,
    8.12229655307922200e-01, 5.03431259309110000e-01, 1.19036530764646380e+00,
    8.08021440064528600e-01, 4.94875004697677260e-01, 1.18808293320162870e+00,
    8.03792071980224200e-01, 4.86378389583198400e-01, 1.18580078041551370e+00,
    7.99541303758805600e-01, 4.77941765226555400e-01, 1.18351895776169020e+00,
    7.95268883210886500e-01, 4.69565484950888100e-01, 1.18123757676424220e+00,
    7.90974553137007700e-01, 4.61249904185644540e-01, 1.17895675209520360e+00,
    7.86658051207769800e-01, 4.52995380510888200e-01, 1.17667660167538000e+00,
    7.82319109840922200e-01, 4.44802273701799640e-01, 1.17439724677868050e+00,
    7.77957456075353900e-01, 4.36670945773313600e-01, 1.17211881214007250e+00,
    7.73572811441926900e-01, 4.28601761024818730e-01, 1.16984142606730020e+00,
    7.68960910244236900e-01, 4.20227112035507670e-01, 1.16746020980084690e+00,
    7.64721305870373200e-01, 4.12629741412390670e-01, 1.16528413772940900e+00,
    7.60057534476442000e-01, 4.04383424920820860e-01, 1.16290472772599340e+00,
    7.55566138111390700e-01, 3.96551141268285200e-01, 1.16062757513580170e+00,
    7.51053030406577800e-01, 3.88788122901708100e-01, 1.15835363911740600e+00,
    7.42037090334163300e-01, 3.73597985303104840e-01, 1.15385394853928960e+00,
    7.32844748938667800e-01, 3.58541585908809000e-01, 1.14932576344033890e+00,
    7.23546489277346500e-01, 3.43747267082826400e-01, 1.14480722748983780e+00,
    7.14140888999055600e-01, 3.29220499482615600e-01, 1.14030047017357860e+00,
    7.04625071475298400e-01, 3.14964569968772150e-01, 1.13580704480949040e+00,
    6.94996010622403000e-01, 3.00982765694906230e-01, 1.13132858499849440e+00,
    6.85250552385155000e-01, 2.87278418436151500e-01, 1.12686682409145590e+00,
    6.75385409740859600e-01, 2.73854905976346040e-01, 1.12242360215005620e+00,
    6.65121300836667700e-01, 2.60359138967450170e-01, 1.11787981757928880e+00,
    6.54976964652252900e-01, 2.47483293308394280e-01, 1.11346915795467140e+00,
    6.44703659649746000e-01, 2.34902937153920600e-01, 1.10908439982484160e+00,
    6.34294016738137600e-01, 2.22617360234246860e-01, 1.10472641972061200e+00,
    6.23743796279071500e-01, 2.10630175351321900e-01, 1.10039769407455350e+00,
    6.13048629686564900e-01, 1.98945104061696460e-01, 1.09610088025696500e+00,
    6.02203963321274200e-01, 1.87565911166105740e-01, 1.09183880526850440e+00,
    5.91205051667039100e-01, 1.76496399326404700e-01, 1.08761447598355900e+00,
    5.80046951900402300e-01, 1.65740403397493690e-01, 1.08343109013488250e+00,
    5.68724519264623300e-01, 1.55301783239811340e-01, 1.07929204761755300e+00,
    5.57232403520293700e-01, 1.45184414708316280e-01, 1.07520096200021800e+00,
    5.45565046829372200e-01, 1.35392178483544570e-01, 1.07116167209883130e+00,
    5.33716683514642600e-01, 1.25928946355422460e-01, 1.06717825341918910e+00,
    5.21324814678510000e-01, 1.16535357231737070e-01, 1.06314070877202020e+00,
    5.09050996703396000e-01, 1.07723948593127760e-01, 1.05927197654759700e+00,
    4.96582380224845470e-01, 9.92593231030434700e-02, 1.05547533652989830e+00,
    4.83904271658060500e-01, 9.11394082184619900e-02, 1.05175343687524170e+00,
    4.71009418961310200e-01, 8.33676680266044500e-02, 1.04811166878031760e+00,
    4.57890628272123600e-01, 7.59475539328696000e-02, 1.04455578351861830e+00,
    4.44540585803906200e-01, 6.88823059856497700e-02, 1.04109182499401130e+00,
    4.30587039970141700e-01, 6.20013503725896600e-02, 1.03763797789637360e+00,
    4.16698490300631400e-01, 5.56432334576156260e-02, 1.03436919523456900e+00,
    4.02562175701567600e-01, 4.96533768923020650e-02, 1.03121472180603010e+00,
    3.88160602348470330e-01, 4.40288959403860200e-02, 1.02817931740512920e+00,
    3.58279623965142400e-01, 3.38012656383037200e-02, 1.02244979228853600e+00,
    3.27109572878986900e-01, 2.50315497299075370e-02, 1.01727593357979140e+00,
    2.95019097024267930e-01, 1.78035015648331280e-02, 1.01277863302231520e+00,
    2.61270929816000140e-01, 1.19234133462343650e-02, 1.00891446288036550e+00,
    2.25845757672159920e-01, 7.37189927149933700e-03, 1.00574937087803200e+00,
    1.89427589900413800e-01, 4.12631778110466800e-03, 1.00335853774267280e+00,
    1.51483531022226970e-01, 1.97335802134401300e-03, 1.00167773755716600e+00,
    1.11974842115410810e-01, 7.27943234255545100e-04, 1.00064690281405740e+00,
    7.15006628844198500e-02, 1.65662123888571480e-04, 1.00015384265811710e+00,
    2.99493944215643500e-02, 9.37735914999092000e-06, 1.00000909659822420e+00,
    2.75493635553743340e-02, 7.11816714744094000e-06, 1.00000692211544570e+00,
    2.53416620591069800e-02, 5.40325935356294400e-06, 1.00000526636023660e+00,
    2.33108773865921700e-02, 4.10150689596567600e-06, 1.00000400591360200e+00,
    2.14428320946476700e-02, 3.11337245112272470e-06, 1.00000304662241370e+00,
    1.97244848666105700e-02, 2.36329921301466160e-06, 1.00000231668982840e+00,
    1.81438394674676200e-02, 1.79393351033913240e-06, 1.00000176138782850e+00,
    1.66898609949758730e-02, 1.36173930994226210e-06, 1.00000133901389350e+00,
    1.53523988421010430e-02, 1.03366927344563670e-06, 1.00000101780102260e+00,
    1.41221158329537140e-02, 7.84637822426477200e-07, 1.00000077355768320e+00,
    1.29904230375028730e-02, 5.95602992367114000e-07, 1.00000058786620770e+00,
    1.19494198099910600e-02, 4.52110406071970300e-07, 1.00000044670815110e+00,
    1.09918386324433970e-02, 3.43188032797144030e-07, 1.00000033941588180e+00,
    1.01109943782086800e-02, 2.60507221849754070e-07, 1.00000025787330200e+00,
    9.30073764132781900e-03, 1.97745859850512820e-07, 1.00000019590671620e+00,
    8.55541180590960400e-03, 1.50104956055963920e-07, 1.00000014882076860e+00,
    7.86981355580391700e-03, 1.13941691874589570e-07, 1.00000011304500500e+00,
    7.23915654887992000e-03, 8.64908760401191600e-08, 1.00000008586476240e+00,
    6.65903800230979200e-03, 6.56535067639761000e-08, 1.00000006521632200e+00,
    6.12540795558108150e-03, 4.98362734632039660e-08, 1.00000004953100840e+00,
    5.63454099665468450e-03, 3.78297257087571450e-08, 1.00000003761657390e+00,
    5.18301025388121900e-03, 2.87157936930502830e-08, 1.00000002856696040e+00,
    4.76766347210664000e-03, 2.17975888530151720e-08, 1.00000002169366580e+00,
    4.03415557660428450e-03, 1.25598288357429140e-08, 1.00000001250916060e+00,
    3.41349831242523160e-03, 7.23700687479189860e-09, 1.00000000721230340e+00,
    2.88832954199992960e-03, 4.16998266383521400e-09, 1.00000000415793840e+00,
    2.44395830307657300e-03, 2.40275513309999260e-09, 1.00000000239688300e+00,
    2.06795384679033700e-03, 1.38447391633245530e-09, 1.00000000138161080e+00,
    1.74979790247303810e-03, 7.97737563266357000e-10, 1.00000000079634170e+00,
    1.48059044173120100e-03, 4.59658511683602650e-10, 1.00000000045897800e+00,
    1.25280071089785330e-03, 2.64856460435521760e-10, 1.00000000026452460e+00,
    1.06005656729148920e-03, 1.52610998929393400e-10, 1.00000000015244940e+00,
    8.96966226218429900e-04, 8.79348646279184000e-11, 1.00000000008785620e+00,
    7.58967432306187500e-04, 5.06683035389074850e-11, 1.00000000005063000e+00,
    6.42199835917982200e-04, 2.91952116418654400e-11, 1.00000000002917640e+00,
    5.43397004532734000e-04, 1.68223588176539900e-11, 1.00000000001681320e+00,
    4.59795048239252150e-04, 9.69308801940992200e-12, 1.00000000000968870e+00,
    3.89055303253149400e-04, 5.58518317023571840e-12, 1.00000000000558300e+00,
    3.29198910621235640e-04, 3.21819743951768200e-12, 1.00000000000321720e+00,
    2.78551460031617060e-04, 1.85433394824202800e-12, 1.00000000000185390e+00,
    2.35696150206945430e-04, 1.06847216686563030e-12, 1.00000000000106830e+00,
    1.99434155599362000e-04, 6.15656512382164100e-13, 1.00000000000061550e+00,
    1.68751090693285370e-04, 3.54742924516660540e-13, 1.00000000000035460e+00,
    1.42788633795406780e-04, 2.04403819278561900e-13, 1.00000000000020450e+00,
    1.20820516521675130e-04, 1.17778025855173600e-13, 1.00000000000011800e+00,
    1.02232207316167510e-04, 6.78640126358770000e-14, 1.00000000000006800e+00,
    8.65037206728121500e-05, 3.91034250880186500e-14, 1.00000000000003900e+00,
    7.31950711686974200e-05, 2.25314978325628640e-14, 1.00000000000002260e+00,
    6.19339654031150800e-05, 1.29827091472439110e-14, 1.00000000000001300e+00,
    5.24053875391910360e-05, 7.48067163818726000e-15, 1.00000000000000750e+00,
    4.43427871162059840e-05, 4.31038295040745450e-15, 1.00000000000000440e+00,
    3.75206226222960050e-05, 2.48365415269924800e-15, 1.00000000000000240e+00,
    3.17480522429823700e-05, 1.43108814719980140e-15, 1.00000000000000160e+00,
    2.68635952918378580e-05, 8.24596807421826400e-16, 1.00000000000000090e+00,
];
#[rustfmt::skip]
const DFM1_KN: [i32; 128] = [
    1991057938, 0, 1611602771, 1826899878, 1918584482, 1969227037, 2001281515, 2023368125,
    2039498179, 2051788381, 2061460127, 2069267110, 2075699398, 2081089314, 2085670119, 2089610331,
    2093034710, 2096037586, 2098691595, 2101053571, 2103168620, 2105072996, 2106796166, 2108362327,
    2109791536, 2111100552, 2112303493, 2113412330, 2114437283, 2115387130, 2116269447, 2117090813,
    2117856962, 2118572919, 2119243101, 2119871411, 2120461303, 2121015852, 2121537798, 2122029592,
    2122493434, 2122931299, 2123344971, 2123736059, 2124106020, 2124456175, 2124787725, 2125101763,
    2125399283, 2125681194, 2125948325, 2126201433, 2126441213, 2126668298, 2126883268, 2127086657,
    2127278949, 2127460589, 2127631985, 2127793506, 2127945490, 2128088244, 2128222044, 2128347141,
    2128463758, 2128572095, 2128672327, 2128764606, 2128849065, 2128925811, 2128994934, 2129056501,
    2129110560, 2129157136, 2129196237, 2129227847, 2129251929, 2129268426, 2129277255, 2129278312,
    2129271467, 2129256561, 2129233410, 2129201800, 2129161480, 2129112170, 2129053545, 2128985244,
    2128906855, 2128817916, 2128717911, 2128606255, 2128482298, 2128345305, 2128194452, 2128028813,
    2127847342, 2127648860, 2127432031, 2127195339, 2126937058, 2126655214, 2126347546, 2126011445,
    2125643893, 2125241376, 2124799783, 2124314271, 2123779094, 2123187386, 2122530867, 2121799464,
    2120980787, 2120059418, 2119015917, 2117825402, 2116455471, 2114863093, 2112989789, 2110753906,
    2108037662, 2104664315, 2100355223, 2094642347, 2086670106, 2074676188, 2054300022, 2010539237,
];
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const DFM1_WN: [f64; 128] = [
    1.72904052154279800e-09, 1.26809284470029200e-10, 1.68975177731846410e-10,
    1.98626884424791160e-10, 2.22324317925000040e-10, 2.42449361254489800e-10,
    2.60161319006320950e-10, 2.76119887117039770e-10, 2.90739628177159950e-10,
    3.04299704143766170e-10, 3.16997952139542940e-10, 3.28980205271130800e-10,
    3.40357381218340840e-10, 3.51216022136647230e-10, 3.61625099505651850e-10,
    3.71640576349598000e-10, 3.81308564311059900e-10, 3.90667568099488270e-10,
    3.99750118699769100e-10, 4.08583986159844030e-10, 4.17193096401606540e-10,
    4.25598235345926260e-10, 4.33817597392551050e-10, 4.41867218125288600e-10,
    4.49761319626658200e-10, 4.57512588945882870e-10, 4.65132404814001000e-10,
    4.72631023848117600e-10, 4.80017734723256700e-10, 4.87300986779874800e-10,
    4.94488498053897300e-10, 5.01587346611961600e-10, 5.08604048242456000e-10,
    5.15544622919539000e-10, 5.22414651970631600e-10, 5.29219327500630500e-10,
    5.35963495331289000e-10, 5.42651692482061900e-10, 5.49288180034602100e-10,
    5.55876972076077300e-10, 5.62421861298358800e-10, 5.68926441734655000e-10,
    5.75394129037560300e-10, 5.81828178639089800e-10, 5.88231702081217000e-10,
    5.94607681762499600e-10, 6.00958984310830300e-10, 6.07288372762788500e-10,
    6.13598517705413500e-10, 6.19892007515592100e-10, 6.26171357814942800e-10,
    6.32439020243540100e-10, 6.38697390643573500e-10, 6.44948816733738200e-10,
    6.51195605346469700e-10, 6.57440029292859800e-10, 6.63684333913987400e-10,
    6.69930743372330100e-10, 6.76181466732744400e-10, 6.82438703879113700e-10,
    6.88704651310073300e-10, 6.94981507855166700e-10, 7.01271480351315500e-10,
    7.07576789318556000e-10, 7.13899674673584900e-10, 7.20242401519748600e-10,
    7.26607266052704700e-10, 7.32996601622086400e-10, 7.39412784991122800e-10,
    7.45858242838353900e-10, 7.52335458548348800e-10, 7.58846979341765200e-10,
    7.65395423799226300e-10, 7.71983489838440000e-10, 7.78613963209838100e-10,
    7.85289726582899700e-10, 7.92013769303409800e-10, 7.98789197911353600e-10,
    8.05619247520217000e-10, 8.12507294171396800e-10, 8.19456868292574500e-10,
    8.26471669406662500e-10, 8.33555582258784500e-10, 8.40712694553299100e-10,
    8.47947316521837200e-10, 8.55264002577609400e-10, 8.62667575351936300e-10,
    8.70163152457442400e-10, 8.77756176380328400e-10, 8.85452447973727800e-10,
    8.93258164108036900e-10, 9.01179960135660500e-10, 9.09224957951138100e-10,
    9.17400820578600500e-10, 9.25715814404012600e-10, 9.34178880398847200e-10,
    9.42799715966631400e-10, 9.51588869399888300e-10, 9.60557849383125300e-10,
    9.69719252545394400e-10, 9.79086912790890000e-10, 9.88676077068772400e-10,
    9.98503613453542500e-10, 1.00858825899144730e-09, 1.01895091686213820e-09,
    1.02961501520066680e-09, 1.04060694369998740e-09, 1.05195658927280390e-09,
    1.06369799919308710e-09, 1.07587021016458190e-09, 1.08851829606072830e-09,
    1.10169470781350440e-09, 1.11546100955971630e-09, 1.12989016134932160e-09,
    1.14506957000672370e-09, 1.16110524260223480e-09, 1.17812756094561300e-09,
    1.19629950538507560e-09, 1.21582869832955640e-09, 1.23698562908049660e-09,
    1.26013233006085250e-09, 1.28576968442051530e-09, 1.31462018496771830e-09,
    1.34778395622108550e-09, 1.38706353150670430e-09, 1.43574031918163800e-09,
    1.50086590302229930e-09, 1.60309479380911230e-09,
];
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const DFM1_FN: [f64; 128] = [
    1.00000000000000000e+00, 9.63599693127085300e-01, 9.36282681685058900e-01,
    9.13043647971739600e-01, 8.92281650784025700e-01, 8.73243048910069100e-01,
    8.55500607869450300e-01, 8.38783605295989400e-01, 8.22907211381408800e-01,
    8.07738294682960300e-01, 7.93177011771304800e-01, 7.79146085929687500e-01,
    7.65584173897704300e-01, 7.52441559174611200e-01, 7.39677243672647100e-01,
    7.27256918344184600e-01, 7.15151507410498400e-01, 7.03336099016158000e-01,
    6.91789143436675000e-01, 6.80491840997334100e-01, 6.69427667348890400e-01,
    6.58582000050088000e-01, 6.47941821110222500e-01, 6.37495477335042300e-01,
    6.27232485249927300e-01, 6.17143370818880900e-01, 6.07219536625120300e-01,
    5.97453150944516700e-01, 5.87837054434706600e-01, 5.78364681119763100e-01,
    5.69029991067950900e-01, 5.59827412704086900e-01, 5.50751793114604500e-01,
    5.41798355025425500e-01, 5.32962659383836100e-01, 5.24240572672984100e-01,
    5.15628238244001800e-01, 5.07122051075569000e-01, 4.98718635470979500e-01,
    4.90414825283844100e-01, 4.82207646329485200e-01, 4.74094300693016950e-01,
    4.66072152689456100e-01, 4.58138716267872060e-01, 4.50291643682039200e-01,
    4.42528715275468440e-01, 4.34847830249990850e-01, 4.27246998304996000e-01,
    4.19724332049574400e-01, 4.12278040102661100e-01, 4.04906420807223100e-01,
    3.97607856493873400e-01, 3.90380808237314700e-01, 3.83223811055901300e-01,
    3.76135469510562700e-01, 3.69114453664472300e-01, 3.62159495369317700e-01,
    3.55269384847917200e-01, 3.48442967546326640e-01, 3.41679141231550400e-01,
    3.34976853313589200e-01, 3.28335098372850240e-01, 3.21752915875984900e-01,
    3.15229388065010900e-01, 3.08763638006181100e-01, 3.02354827786483540e-01,
    2.96002156846933000e-01, 2.89704860442959840e-01, 2.83462208223233000e-01,
    2.77273502919188100e-01, 2.71138079138384600e-01, 2.65055302255589200e-01,
    2.59024567396204830e-01, 2.53045298507325770e-01, 2.47116947512321400e-01,
    2.41238993545439820e-01, 2.35410942263479080e-01, 2.29632325232116130e-01,
    2.23902699385008420e-01, 2.18221646554305430e-01, 2.12588773071730300e-01,
    2.07003709439926520e-01, 2.01466110074313670e-01, 1.95975653116277740e-01,
    1.90532040319137170e-01, 1.85134997008992200e-01, 1.79784272123295450e-01,
    1.74479638330789500e-01, 1.69220892237365000e-01, 1.64007854683420380e-01,
    1.58840371139479300e-01, 1.53718312208181660e-01, 1.48641574242342260e-01,
    1.43610080090627760e-01, 1.38623779984594600e-01, 1.33682652583439370e-01,
    1.28786706195943200e-01, 1.23935980202867820e-01, 1.19130546707650830e-01,
    1.14370512448866010e-01, 1.09656021014840270e-01, 1.04987255409421320e-01,
    1.00364441028655870e-01, 9.57878491217314400e-02, 9.12578008268302400e-02,
    8.67746718947801900e-02, 8.23388982422356600e-02, 7.79509825139734000e-02,
    7.36115018841134000e-02, 6.93211173935779100e-02, 6.50805852130680700e-02,
    6.08907703480404060e-02, 5.67526634810498500e-02, 5.26674019030510100e-02,
    4.86362958598678050e-02, 4.46608622004914250e-02, 4.07428680744441750e-02,
    3.68843887866562000e-02, 3.30878861462257500e-02, 2.93563174400068500e-02,
    2.56932919359342700e-02, 2.21033046159270980e-02, 1.85921027370112880e-02,
    1.51672980105465680e-02, 1.18394786578848620e-02, 8.62448441285988500e-03,
    5.54899522077134500e-03, 2.66962908388092300e-03,
];

/// `Decimator.ar(in, rate, bits)`: block-controlled sample-and-hold with PCM-style quantization.
///
/// The constructor advances the held-sample phase once with a zero input. Subsequent callbacks
/// retain both phase and held output, while `rate` and `bits` are sampled once per callback.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Decimator {
    count: f32,
    last: f32,
    rate: f32,
    bits: f32,
}

impl Decimator {
    const IN: usize = 0;
    const RATE: usize = 1;
    const BITS: usize = 2;
}

impl Unit for Decimator {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let sample_rate = ctx.own.sample_rate as f32;
        let rate = remembered_finite_control(
            ctx.ins.control(Self::RATE),
            &mut self.rate,
            0.0,
            sample_rate,
        );
        let bits = remembered_unbounded_control(ctx.ins.control(Self::BITS), &mut self.bits);
        decimator_sample(
            0.0,
            rate / sample_rate,
            bits,
            &mut self.count,
            &mut self.last,
        );
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sample_rate = ctx.own.sample_rate as f32;
        let rate = remembered_finite_control(
            ctx.ins.control(Self::RATE),
            &mut self.rate,
            0.0,
            sample_rate,
        );
        let bits = remembered_unbounded_control(ctx.ins.control(Self::BITS), &mut self.bits);
        let ratio = rate / sample_rate;

        for sample in 0..ctx.outs.audio(0).len() {
            let input = signal(&ctx.ins, Self::IN, sample);
            decimator_sample(input, ratio, bits, &mut self.count, &mut self.last);
            ctx.outs.audio(0)[sample] = self.last;
        }
        DoneAction::Nothing
    }
}

/// Retain a finite control after clamping it to the supplied safety range.
fn remembered_finite_control(value: f32, remembered: &mut f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        *remembered = value.clamp(min, max);
    }
    *remembered
}

/// Retain a finite control without changing its valid finite range.
fn remembered_unbounded_control(value: f32, remembered: &mut f32) -> f32 {
    if value.is_finite() {
        *remembered = value;
    }
    *remembered
}

/// Advance one decimator sample in the normative `f32` arithmetic order.
fn decimator_sample(input: f32, ratio: f32, bits: f32, count: &mut f32, last: &mut f32) {
    let step = if !(1.0..31.0).contains(&bits) {
        0.0
    } else {
        math::powf(0.5_f32, bits - 0.999)
    };
    let inv_step = if step == 0.0 { 1.0 } else { 1.0 / step };

    *count += ratio;
    if *count >= 1.0 {
        *count -= 1.0;
        let signed_half_step = if input < 0.0 { -step * 0.5 } else { step * 0.5 };
        let value = (input + signed_half_step) * inv_step;
        let fraction = value - math::trunc(value);
        *last = input - fraction * step;
        if !last.is_finite() {
            *last = 0.0;
        }
    }
}

/// Constructor for [`Decimator`].
pub struct DecimatorCtor;

impl UnitDef for DecimatorCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_calc(ctx, 3, 1, &[Rate::Audio])?;
        Ok(unit_spec(Decimator {
            count: 0.0,
            last: 0.0,
            rate: 44_100.0,
            bits: 24.0,
        }))
    }
}

/// `BMoog.ar(in, freq, q, mode)`: four-pole resonant filter with fixed gain compensation.
///
/// Controls are sampled once per callback. Mode one selects high-pass, mode two selects band-pass,
/// and every other finite mode selects low-pass without resetting retained poles.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BMoog {
    poles: [f32; 4],
    feedback: f32,
    retained_gain: f32,
    frequency: f32,
    resonance: f32,
    mode: f32,
}

impl BMoog {
    const IN: usize = 0;
    const FREQ: usize = 1;
    const RES: usize = 2;
    const MODE: usize = 3;
}

impl Unit for BMoog {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let nyquist = ctx.own.sample_rate as f32 * 0.5;
        let frequency = remembered_finite_control(
            ctx.ins.control(Self::FREQ),
            &mut self.frequency,
            20.0,
            nyquist,
        );
        let resonance =
            remembered_finite_control(ctx.ins.control(Self::RES), &mut self.resonance, 0.0, 1.0);
        let mode = remembered_unbounded_control(ctx.ins.control(Self::MODE), &mut self.mode);

        let fc = 2.0 * frequency / ctx.own.sample_rate as f32;
        let fc2 = fc * fc;
        let p = -0.69346 * fc2 * fc - 0.59515 * fc2 + 3.2937 * fc - 1.0072;
        let ix = (p * 99.0).clamp(-99.0, 98.0);
        let ixint = math::floor(ix) as i32;
        let ixfrac = (ix - ixint as f32).clamp(0.0, 1.0);
        let table_index = (ixint + 99) as usize;
        let gain = BMOOG_GAIN[table_index] * (1.0 - ixfrac) + BMOOG_GAIN[table_index + 1] * ixfrac;
        let retained_gain = self.retained_gain;

        for sample in 0..ctx.outs.audio(0).len() {
            let input = signal(&ctx.ins, Self::IN, sample);
            let mut y = 0.25 * (input - self.feedback);
            for pole in &mut self.poles {
                let old = *pole;
                y = (y + p * (y - old)).clamp(-0.95, 0.95);
                *pole = y;
                y = (y + old).clamp(-0.95, 0.95);
            }

            let output = if (1.0..2.0).contains(&mode) {
                input - y
            } else if (2.0..3.0).contains(&mode) {
                3.0 * self.poles[2] - y
            } else {
                y
            };
            self.feedback = y * retained_gain;
            ctx.outs.audio(0)[sample] = if output.is_finite() { output } else { 0.0 };
        }
        self.retained_gain = resonance * gain;
        DoneAction::Nothing
    }
}

/// Constructor for [`BMoog`].
pub struct BMoogCtor;

impl UnitDef for BMoogCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        validate_calc(ctx, 4, 1, &[Rate::Audio])?;
        Ok(unit_spec(BMoog {
            poles: [0.0; 4],
            feedback: 0.0,
            retained_gain: 0.0,
            frequency: 440.0,
            resonance: 0.2,
            mode: 0.0,
        }))
    }
}

/// Public gain-compensation data from DFL's Stilson Moog filter reference, rounded to `f32`.
#[rustfmt::skip]
const BMOOG_GAIN: [f32; 199] = [
    0.999969, 0.990082, 0.980347, 0.970764, 0.961304, 0.951996, 0.94281, 0.933777,
    0.924866, 0.916077, 0.90741, 0.898865, 0.890442, 0.882141, 0.873962, 0.865906,
    0.857941, 0.850067, 0.842346, 0.834686, 0.827148, 0.819733, 0.812378, 0.805145,
    0.798004, 0.790955, 0.783997, 0.77713, 0.770355, 0.763672, 0.75708, 0.75058,
    0.744141, 0.737793, 0.731537, 0.725342, 0.719238, 0.713196, 0.707245, 0.701355,
    0.695557, 0.689819, 0.684174, 0.678558, 0.673035, 0.667572, 0.66217, 0.65686,
    0.651581, 0.646393, 0.641235, 0.636169, 0.631134, 0.62619, 0.621277, 0.616425,
    0.611633, 0.606903, 0.602234, 0.597626, 0.593048, 0.588531, 0.584045, 0.579651,
    0.575287, 0.570953, 0.566681, 0.562469, 0.558289, 0.554169, 0.550079, 0.546051,
    0.542053, 0.538116, 0.53421, 0.530334, 0.52652, 0.522736, 0.518982, 0.515289,
    0.511627, 0.507996, 0.504425, 0.500885, 0.497375, 0.493896, 0.490448, 0.487061,
    0.483704, 0.480377, 0.477081, 0.473816, 0.470581, 0.467377, 0.464203, 0.46109,
    0.457977, 0.454926, 0.451874, 0.448883, 0.445892, 0.442932, 0.440033, 0.437134,
    0.434265, 0.431427, 0.428619, 0.425842, 0.423096, 0.42038, 0.417664, 0.415009,
    0.412354, 0.409729, 0.407135, 0.404572, 0.402008, 0.399506, 0.397003, 0.394501,
    0.392059, 0.389618, 0.387207, 0.384827, 0.382477, 0.380127, 0.377808, 0.375488,
    0.37323, 0.370972, 0.368713, 0.366516, 0.364319, 0.362122, 0.359985, 0.357849,
    0.355713, 0.353607, 0.351532, 0.349457, 0.347412, 0.345398, 0.343384, 0.34137,
    0.339417, 0.337463, 0.33551, 0.333588, 0.331665, 0.329773, 0.327911, 0.32605,
    0.324188, 0.322357, 0.320557, 0.318756, 0.316986, 0.315216, 0.313446, 0.311707,
    0.309998, 0.308289, 0.30658, 0.304901, 0.303223, 0.301575, 0.299927, 0.298309,
    0.296692, 0.295074, 0.293488, 0.291931, 0.290375, 0.288818, 0.287262, 0.285736,
    0.284241, 0.282715, 0.28125, 0.279755, 0.27829, 0.276825, 0.275391, 0.273956,
    0.272552, 0.271118, 0.269745, 0.268341, 0.266968, 0.265594, 0.264252, 0.262909,
    0.261566, 0.260223, 0.258911, 0.257599, 0.256317, 0.255035, 0.25375,
];
