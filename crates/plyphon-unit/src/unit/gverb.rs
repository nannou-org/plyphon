//! `GVerb` - plyphon's port of scsynth's large Griesinger-style FDN reverb (`ReverbUGens.cpp`).
//!
//! A feedback-delay-network reverb: the input is band-limited by an input damper and diffused, fed into
//! four parallel delay lines whose outputs are damped, gained (for the `revtime` decay), mixed by a
//! Hadamard matrix and recirculated; four early-reflection taps read a long tap delay, and the tail is
//! diffused into a decorrelated stereo pair. `roomsize` sets the delay lengths (and, with `spread`, the
//! diffuser lengths), `revtime` the decay, `damping` the high-frequency loss.
//!
//! All delay/diffuser buffers live in [aux memory](crate::unit::Aux), allocated when the synth starts
//! and laid out from the first values of `roomsize`, `spread` and `maxroomsize`, as scsynth's
//! `GVerb_Ctor` does. scsynth later rescales the FDN lengths when `roomsize` is modulated; plyphon
//! keeps the layout from the start, leaving `revtime`/`damping`/the levels freely modulatable. The
//! lines are zeroed on the first block.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::math;

const FDN: usize = 4;
/// The fixed early-reflection tap-delay length (scsynth's `make_fixeddelay(unit, 44000, 44000)`).
const TAP_LEN: usize = 44000;
/// Buffer indices in aux: `[0]` tap delay, `[1..5]` the four FDN lines, `[5..9]` the left diffusers,
/// `[9..13]` the right diffusers.
const NBUF: usize = 13;

/// Flush denormals/non-finite to zero (scsynth's `zapgremlins`/`flush_to_zero`) for `f32` state.
#[inline]
fn zapf(x: f32) -> f32 {
    let a = x.abs();
    if a > 1e-15 && a < 1e15 { x } else { 0.0 }
}

/// scsynth's `f_round`: round to nearest integer.
fn fround(x: f32) -> i32 {
    math::round(x as f64) as i32
}

fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// The nearest prime to `n` within `n * rerror` (scsynth's `nearestprime`); falls back to `n` if none
/// is found (rather than scsynth's `-1`, so the length stays valid).
fn nearest_prime(n: i64, rerror: f32) -> i64 {
    if is_prime(n) {
        return n;
    }
    let bound = (n as f32 * rerror) as i64;
    for k in 1..=bound {
        if is_prime(n + k) {
            return n + k;
        }
        if is_prime(n - k) {
            return n - k;
        }
    }
    n.max(1)
}

/// A one-pole lowpass "damper": `y = x*(1-damping) + delay*damping`, `delay = y`.
#[inline]
fn damper_do(delay: &mut f32, damping: f32, x: f32) -> f32 {
    let y = x * (1.0 - damping) + *delay * damping;
    *delay = zapf(y);
    y
}

/// A Schroeder allpass "diffuser" over `buf[off..off+size]` at circular `idx`.
#[inline]
fn diffuser_do(buf: &mut [f32], off: usize, size: usize, idx: &mut u32, coef: f32, x: f32) -> f32 {
    let i = off + *idx as usize;
    let bi = buf[i];
    let w = zapf(x - bi * coef);
    let y = bi + w * coef;
    buf[i] = zapf(w);
    *idx = (*idx + 1) % size as u32;
    y
}

/// Read `n` samples back in the fixed delay `buf[off..off+size]` at write index `idx`.
#[inline]
fn fixed_read(buf: &[f32], off: usize, size: usize, idx: u32, n: usize) -> f32 {
    let i = ((idx as i64 - n as i64).rem_euclid(size as i64)) as usize;
    buf[off + i]
}

/// Write `x` at the head of the fixed delay `buf[off..off+size]` and advance `idx`.
#[inline]
fn fixed_write(buf: &mut [f32], off: usize, size: usize, idx: &mut u32, x: f32) {
    buf[off + *idx as usize] = zapf(x);
    *idx = (*idx + 1) % size as u32;
}

/// The FDN's Hadamard-style mixing matrix (scsynth's `gverb_fdnmatrix`).
#[inline]
fn fdn_matrix(a: &[f32; 4]) -> [f32; 4] {
    [
        0.5 * (a[0] + a[1] - a[2] - a[3]),
        0.5 * (a[0] - a[1] - a[2] + a[3]),
        0.5 * (-a[0] + a[1] - a[2] + a[3]),
        0.5 * (a[0] + a[1] + a[2] + a[3]),
    ]
}

/// `GVerb.ar(in, roomsize, revtime, damping, inputbw, spread, drylevel, earlyreflevel, taillevel,
/// maxroomsize)` -> `[left, right]`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GVerb {
    /// The decay base `0.001^(1/(sr*revtime))`.
    alpha: f64,
    /// Aux offset of each of the 13 buffers.
    off: [u32; NBUF],
    /// Effective length of each buffer (tap = 44000, FDN = its fdnlen, diffusers = their size).
    size: [u32; NBUF],
    /// Circular index of each buffer.
    idx: [u32; NBUF],
    /// Early-reflection tap positions into the tap delay.
    taps: [u32; FDN],
    /// Diffuser coefficients (`[0..4]` left, `[4..8]` right).
    coef: [f32; 8],
    /// Per-FDN-line decay gains and their per-sample slopes (`revtime` interpolation).
    fdngains: [f32; FDN],
    fdngainslopes: [f32; FDN],
    /// Early-reflection tap gains and slopes.
    tapgains: [f32; FDN],
    tapgainslopes: [f32; FDN],
    /// Input damper (`damping`, one-pole `delay`).
    input_damping: f32,
    input_delay: f32,
    /// FDN dampers: shared `damping`, per-line `delay`.
    fdn_damping: f32,
    fdn_delay: [f32; FDN],
    /// Output levels (current) and their per-sample slopes.
    drylevel: f32,
    earlylevel: f32,
    taillevel: f32,
    drylevelslope: f32,
    earlylevelslope: f32,
    taillevelslope: f32,
    /// Last-seen modulatable inputs, for change detection.
    revtime: f32,
    damping: f32,
    inputbw: f32,
    /// `0` until [`Unit::init`] has seeded the gains/dampers.
    inited: u32,
    /// `1` until the first block has zeroed the (dirty) delay buffers.
    first: u32,
    _pad: u32,
}

impl GVerb {
    const IN: usize = 0;
    const ROOMSIZE: usize = 1;
    const REVTIME: usize = 2;
    const DAMPING: usize = 3;
    const INPUTBW: usize = 4;
    const DRYLEVEL: usize = 6;
    const EARLYLEVEL: usize = 7;
    const TAILLEVEL: usize = 8;
    const SPREAD: usize = 5;
    const MAXROOMSIZE: usize = 9;

    /// Recompute the decay base and the FDN/tap gains from `revtime`.
    fn set_revtime(&mut self, sr: f64, revtime: f32, n: usize) {
        self.alpha = math::powf(0.001f64, 1.0 / (sr * revtime as f64));
        for j in 0..FDN {
            let old = self.fdngains[j];
            let fresh = -(math::powf(self.alpha, self.size[1 + j] as f64) as f32);
            self.fdngains[j] = old;
            self.fdngainslopes[j] = (fresh - old) / n as f32;
        }
        for j in 0..FDN {
            let old = self.tapgains[j];
            let fresh = math::powf(self.alpha, self.taps[j] as f64) as f32;
            self.tapgains[j] = old;
            self.tapgainslopes[j] = (fresh - old) / n as f32;
        }
        self.revtime = revtime;
    }
}

/// scsynth's `gverb_set_roomsize`: a `roomsize` at or below 1 becomes 1, and one at or above
/// `maxroomsize` becomes `maxroomsize - 1`, so the FDN lines never outgrow their `maxroomsize`-sized
/// buffers. Applied at construction too, which scsynth's constructor comment specifies.
fn clamp_roomsize(roomsize: f32, maxroomsize: f32) -> f32 {
    if roomsize <= 1.0 {
        1.0
    } else if roomsize >= maxroomsize {
        maxroomsize - 1.0
    } else {
        roomsize
    }
}

impl Unit for GVerb {
    // `0.707100` is scsynth's literal FDN scale (not exactly 1/sqrt(2)); keep it verbatim. The layout
    // loops index parallel `[_; NBUF]`/`[_; FDN]` arrays, which reads clearest as indexed loops.
    #[allow(clippy::approx_constant, clippy::needless_range_loop)]
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        let sr = ctx.audio.sample_rate;
        let maxroomsize = ctx.ins.control(Self::MAXROOMSIZE).max(1.0001);
        let roomsize = clamp_roomsize(ctx.ins.control(Self::ROOMSIZE), maxroomsize);
        let spread = ctx.ins.control(Self::SPREAD);

        // Saturating float-to-int casts and sums throughout, so an out-of-range input asks for more
        // than any pool holds instead of wrapping into a small layout.
        let maxdelay = (sr * maxroomsize as f64 / 340.0) as u64;
        let largestdelay = sr * roomsize as f64 / 340.0;

        // FDN line lengths (scsynth's `gbmul`); line 0 is snapped to a prime once the memory exists.
        let gbmul = [1.0, 0.816_49, 0.707_1, 0.632_45];
        let mut fdnlens = [0u64; FDN];
        for j in 1..FDN {
            fdnlens[j] = fround((gbmul[j] * largestdelay) as f32).max(1) as u64;
        }

        // Diffuser lengths from the FDN and `spread` (scsynth's diffuser section).
        let diffscale = fdnlens[3] as f32 / (210.0 + 159.0 + 562.0 + 410.0);
        let dif_sizes = |sp1: f32, r1: f32, r2: f32| -> [u64; 4] {
            let b = 210i64;
            let a1 = (sp1 * r1) as i32 as i64;
            let c = 210 + 159 + a1;
            let cc = c - b;
            let a2 = (3.0 * sp1 * r2) as i32 as i64;
            let d = 210 + 159 + 562 + a2;
            let dd = d - c;
            let e = 1341 - d;
            [b, cc, dd, e].map(|x| fround(diffscale * x as f32).max(1) as u64)
        };
        let ldif = dif_sizes(spread, 0.125_541, 0.854_046);
        let rdif = dif_sizes(spread, -0.568_366, -0.126_815);

        // Lay out the 13 buffers: tap delay, four FDN lines (capacity maxdelay+1000), eight diffusers.
        let fdn_cap = maxdelay.saturating_add(1000);
        let caps: [u64; NBUF] = [
            TAP_LEN as u64,
            fdn_cap,
            fdn_cap,
            fdn_cap,
            fdn_cap,
            ldif[0],
            ldif[1],
            ldif[2],
            ldif[3],
            rdif[0],
            rdif[1],
            rdif[2],
            rdif[3],
        ];
        let mut off = [0u64; NBUF];
        let mut total = 0u64;
        for k in 0..NBUF {
            off[k] = total;
            total = total.saturating_add(caps[k]);
        }
        let bytes = total
            .checked_mul(core::mem::size_of::<f32>() as u64)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .unwrap_or(usize::MAX);
        if !aux.alloc(bytes) {
            return;
        }

        // Every length and offset now lies within the allocation, so it fits in `u32`, and the prime
        // search is bounded by memory that exists.
        let gb0 = (gbmul[0] * largestdelay) as f32;
        fdnlens[0] = nearest_prime(gb0 as i64, 0.5).max(1) as u64;
        let sizes: [u64; NBUF] = [
            TAP_LEN as u64,
            fdnlens[0],
            fdnlens[1],
            fdnlens[2],
            fdnlens[3],
            ldif[0],
            ldif[1],
            ldif[2],
            ldif[3],
            rdif[0],
            rdif[1],
            rdif[2],
            rdif[3],
        ];
        for k in 0..NBUF {
            self.off[k] = off[k] as u32;
            self.size[k] = sizes[k] as u32;
        }
        // Early-reflection taps.
        self.taps = [
            5 + (0.410 * largestdelay) as u32,
            5 + (0.300 * largestdelay) as u32,
            5 + (0.155 * largestdelay) as u32,
            5,
        ];
    }

    fn init(&mut self, ctx: &InitCtx<'_>) {
        let sr = ctx.own.sample_rate;
        let revtime = ctx.ins.control(Self::REVTIME);
        self.alpha = math::powf(0.001f64, 1.0 / (sr * revtime as f64));
        for j in 0..FDN {
            self.fdngains[j] = -(math::powf(self.alpha, self.size[1 + j] as f64) as f32);
            self.tapgains[j] = math::powf(self.alpha, self.taps[j] as f64) as f32;
        }
        self.fdn_damping = ctx.ins.control(Self::DAMPING);
        self.inputbw = ctx.ins.control(Self::INPUTBW);
        self.input_damping = 1.0 - self.inputbw;
        self.damping = self.fdn_damping;
        self.revtime = revtime;
        self.drylevel = ctx.ins.control(Self::DRYLEVEL);
        self.earlylevel = ctx.ins.control(Self::EARLYLEVEL);
        self.taillevel = ctx.ins.control(Self::TAILLEVEL);
        self.inited = 1;
    }

    #[allow(clippy::needless_range_loop)]
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.own.sample_rate;
        let ins = ctx.ins;
        let in_audio = ins.audio(Self::IN);
        let revtime = ins.control(Self::REVTIME);
        let damping = ins.control(Self::DAMPING);
        let inputbw = ins.control(Self::INPUTBW);
        let drylevel = ins.control(Self::DRYLEVEL);
        let earlylevel = ins.control(Self::EARLYLEVEL);
        let taillevel = ins.control(Self::TAILLEVEL);

        let n = ctx.outs.audio(0).len();
        let buf = ctx.aux.f32_mut();
        let total: usize = self.off[NBUF - 1] as usize + self.size[NBUF - 1] as usize;
        if buf.len() < total {
            ctx.outs.audio(0).fill(0.0);
            ctx.outs.audio(1).fill(0.0);
            return DoneAction::Nothing;
        }
        if self.first != 0 {
            buf[..total].fill(0.0);
            self.first = 0;
        }

        // Recompute the interpolation slopes when any modulatable input changes.
        if revtime != self.revtime
            || damping != self.damping
            || inputbw != self.inputbw
            || drylevel != self.drylevel
            || earlylevel != self.earlylevel
            || taillevel != self.taillevel
        {
            self.set_revtime(sr, revtime, n);
            self.fdn_damping = damping;
            self.damping = damping;
            self.input_damping = 1.0 - inputbw;
            self.inputbw = inputbw;
            self.drylevelslope = (drylevel - self.drylevel) / n as f32;
            self.earlylevelslope = (earlylevel - self.earlylevel) / n as f32;
            self.taillevelslope = (taillevel - self.taillevel) / n as f32;
        } else {
            self.fdngainslopes = [0.0; FDN];
            self.tapgainslopes = [0.0; FDN];
            self.drylevelslope = 0.0;
            self.earlylevelslope = 0.0;
            self.taillevelslope = 0.0;
        }

        let off = self.off;
        let size = self.size;
        let coef = self.coef;
        let taps = self.taps;
        let mut idx = self.idx;
        let mut fdngains = self.fdngains;
        let mut tapgains = self.tapgains;
        let mut input_delay = self.input_delay;
        let mut fdn_delay = self.fdn_delay;
        let mut drylevel_c = self.drylevel;
        let mut earlylevel_c = self.earlylevel;
        let mut taillevel_c = self.taillevel;
        let input_damping = self.input_damping;
        let fdn_damping = self.fdn_damping;

        for i in 0..n {
            let x = {
                let v = in_audio[i];
                if v.is_nan() { 0.0 } else { v }
            };
            // Band-limit and diffuse the input.
            let mut z = damper_do(&mut input_delay, input_damping, x);
            z = diffuser_do(
                buf,
                off[5] as usize,
                size[5] as usize,
                &mut idx[5],
                coef[0],
                z,
            );

            // Early-reflection taps read the long tap delay, then write the diffused input into it.
            let mut u = [0.0f32; FDN];
            for j in 0..FDN {
                u[j] = tapgains[j]
                    * fixed_read(buf, off[0] as usize, TAP_LEN, idx[0], taps[j] as usize);
            }
            fixed_write(buf, off[0] as usize, TAP_LEN, &mut idx[0], z);

            // FDN reads, damped and decay-gained.
            let mut d = [0.0f32; FDN];
            for j in 0..FDN {
                let raw = fdngains[j]
                    * fixed_read(
                        buf,
                        off[1 + j] as usize,
                        size[1 + j] as usize,
                        idx[1 + j],
                        size[1 + j] as usize,
                    );
                d[j] = damper_do(&mut fdn_delay[j], fdn_damping, raw);
            }

            // Sum with alternating sign, plus the early input.
            let mut sum = 0.0f32;
            let mut sign = 1.0f32;
            for j in 0..FDN {
                sum += sign * (taillevel_c * d[j] + earlylevel_c * u[j]);
                sign = -sign;
            }
            sum += x * earlylevel_c;
            let mut lsum = sum;
            let mut rsum = sum;

            // Recirculate through the mixing matrix.
            let f = fdn_matrix(&d);
            for j in 0..FDN {
                fixed_write(
                    buf,
                    off[1 + j] as usize,
                    size[1 + j] as usize,
                    &mut idx[1 + j],
                    u[j] + f[j],
                );
            }

            // Diffuse the tail into a decorrelated stereo pair.
            for k in 1..FDN {
                let li = 5 + k;
                lsum = diffuser_do(
                    buf,
                    off[li] as usize,
                    size[li] as usize,
                    &mut idx[li],
                    coef[k],
                    lsum,
                );
                let ri = 9 + k;
                rsum = diffuser_do(
                    buf,
                    off[ri] as usize,
                    size[ri] as usize,
                    &mut idx[ri],
                    coef[4 + k],
                    rsum,
                );
            }

            let dry = x * drylevel_c;
            ctx.outs.audio(0)[i] = lsum + dry;
            ctx.outs.audio(1)[i] = rsum + dry;

            drylevel_c += self.drylevelslope;
            earlylevel_c += self.earlylevelslope;
            taillevel_c += self.taillevelslope;
            for j in 0..FDN {
                fdngains[j] += self.fdngainslopes[j];
                tapgains[j] += self.tapgainslopes[j];
            }
        }

        self.idx = idx;
        self.fdngains = fdngains;
        self.tapgains = tapgains;
        self.input_delay = input_delay;
        self.fdn_delay = fdn_delay;
        self.drylevel = drylevel_c;
        self.earlylevel = earlylevel_c;
        self.taillevel = taillevel_c;
        DoneAction::Nothing
    }
}

/// Constructor for [`GVerb`]. The 13 delay/diffuser buffers are allocated when the synth starts, laid
/// out from the first values of `roomsize`, `spread` and `maxroomsize`.
pub struct GVerbCtor;

impl UnitDef for GVerbCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 10 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(GVerb {
            alpha: 0.0,
            off: [0; NBUF],
            size: [0; NBUF],
            idx: [0; NBUF],
            taps: [0; FDN],
            coef: [0.75, 0.75, 0.625, 0.625, 0.75, 0.75, 0.625, 0.625],
            fdngains: [0.0; FDN],
            fdngainslopes: [0.0; FDN],
            tapgains: [0.0; FDN],
            tapgainslopes: [0.0; FDN],
            input_damping: 0.0,
            input_delay: 0.0,
            fdn_damping: 0.0,
            fdn_delay: [0.0; FDN],
            drylevel: 0.0,
            earlylevel: 0.0,
            taillevel: 0.0,
            drylevelslope: 0.0,
            earlylevelslope: 0.0,
            taillevelslope: 0.0,
            revtime: 0.0,
            damping: 0.0,
            inputbw: 0.0,
            inited: 0,
            first: 1,
            _pad: 0,
        }))
    }
}
