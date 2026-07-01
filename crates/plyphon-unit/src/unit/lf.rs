//! Non-band-limited low-frequency oscillators - plyphon's ports of scsynth's `LFSaw`, `LFPulse`,
//! `Impulse`, `LFTri`, `LFPar`, `LFCub`, `VarSaw` and `SyncSaw`.
//!
//! These are control/LFO sources (and raw audio waves); for clean band-limited audio use [`Saw`] and
//! [`Pulse`]. Frequency is read at control rate (one value per block), so frequency modulation is at
//! block resolution. Each writes a full block; a `.kr` instance just publishes the first sample.
//! `LFTri`/`LFPar`/`LFCub`/`VarSaw`/`SyncSaw` use scsynth's own phase accumulators (kept in `f64`)
//! and per-cycle wrap points, seeding their initial phase in [`Unit::init`].
//!
//! [`Saw`]: crate::unit::Saw
//! [`Pulse`]: crate::unit::Pulse

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::math;

/// `LFSaw.ar/kr(freq)`: a non-band-limited sawtooth ramping from -1 to 1 each cycle.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFSaw {
    phase: f32,
}

impl Unit for LFSaw {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(0) * ctx.audio.sample_dur as f32;
        for o in ctx.outs.audio(0).iter_mut() {
            *o = 2.0 * self.phase - 1.0;
            self.phase = wrap(self.phase + inc);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`LFSaw`].
pub struct LFSawCtor;

impl UnitDef for LFSawCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(LFSaw { phase: 0.0 }))
    }
}

/// `LFPulse.ar/kr(freq, iphase, width)`: a non-band-limited pulse, output 0 or 1 with duty `width`
/// (default 0.5). `iphase` is currently ignored.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFPulse {
    phase: f32,
}

impl LFPulse {
    const FREQ: usize = 0;
    const WIDTH: usize = 2;
}

impl Unit for LFPulse {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(Self::FREQ) * ctx.audio.sample_dur as f32;
        let width = if ctx.ins.len() > Self::WIDTH {
            ctx.ins.control(Self::WIDTH)
        } else {
            0.5
        };
        for o in ctx.outs.audio(0).iter_mut() {
            *o = if self.phase < width { 1.0 } else { 0.0 };
            self.phase = wrap(self.phase + inc);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`LFPulse`].
pub struct LFPulseCtor;

impl UnitDef for LFPulseCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(LFPulse { phase: 0.0 }))
    }
}

/// `Impulse.ar/kr(freq, phase)`: a single-sample impulse of 1.0 at the start of each period, 0
/// otherwise. `phase` is currently ignored (it starts firing immediately).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Impulse {
    phase: f32,
}

impl Unit for Impulse {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(0) * ctx.audio.sample_dur as f32;
        for o in ctx.outs.audio(0).iter_mut() {
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                *o = 1.0;
            } else {
                *o = 0.0;
            }
            self.phase += inc;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Impulse`].
pub struct ImpulseCtor;

impl UnitDef for ImpulseCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // Start at the cycle boundary so the first sample is an impulse.
        Ok(unit_spec(Impulse { phase: 1.0 }))
    }
}

/// `LFTri.ar/kr(freq, iphase)`: a non-band-limited triangle from -1 to 1.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFTri {
    phase: f64,
}

impl Unit for LFTri {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.phase = math::rem_euclid(ctx.ins.control(1) as f64, 4.0);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(0) as f64 * 4.0 * ctx.audio.sample_dur;
        let mut phase = self.phase;
        for o in ctx.outs.audio(0).iter_mut() {
            let z = if phase > 1.0 { 2.0 - phase } else { phase };
            phase += freq;
            if phase >= 3.0 {
                phase -= 4.0;
            }
            *o = z as f32;
        }
        self.phase = phase;
        DoneAction::Nothing
    }
}

/// `LFPar.ar/kr(freq, iphase)`: a non-band-limited parabolic wave, a sine-like curve from parabola
/// segments.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFPar {
    phase: f64,
}

impl Unit for LFPar {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.phase = ctx.ins.control(1) as f64;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(0) as f64 * 4.0 * ctx.audio.sample_dur;
        let mut phase = self.phase;
        for o in ctx.outs.audio(0).iter_mut() {
            if phase < 1.0 {
                *o = (1.0 - phase * phase) as f32;
            } else if phase < 3.0 {
                let z = phase - 2.0;
                *o = (z * z - 1.0) as f32;
            } else {
                phase -= 4.0;
                *o = (1.0 - phase * phase) as f32;
            }
            phase += freq;
        }
        self.phase = phase;
        DoneAction::Nothing
    }
}

/// `LFCub.ar/kr(freq, iphase)`: a non-band-limited sine-like wave from a cubic curve.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFCub {
    phase: f64,
}

impl Unit for LFCub {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.phase = ctx.ins.control(1) as f64 + 0.5;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(0) as f64 * 2.0 * ctx.audio.sample_dur;
        let mut phase = self.phase;
        for o in ctx.outs.audio(0).iter_mut() {
            let z = if phase < 1.0 {
                phase
            } else if phase < 2.0 {
                2.0 - phase
            } else {
                phase -= 2.0;
                phase
            };
            *o = (z * z * (6.0 - 4.0 * z) - 1.0) as f32;
            phase += freq;
        }
        self.phase = phase;
        DoneAction::Nothing
    }
}

/// `VarSaw.ar/kr(freq, iphase, width)`: a variable-duty triangle/saw - `width` (0..1) moves the peak,
/// morphing from a rising ramp through a triangle to a falling ramp.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct VarSaw {
    phase: f64,
    duty: f32,
    invduty: f32,
    inv1duty: f32,
    _pad: u32,
}

impl VarSaw {
    fn set_duty(&mut self, duty: f32) {
        self.duty = duty.clamp(0.001, 0.999);
        self.invduty = 2.0 / self.duty;
        self.inv1duty = 2.0 / (1.0 - self.duty);
    }
}

impl Unit for VarSaw {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.phase = math::rem_euclid(ctx.ins.control(1) as f64, 1.0);
        self.set_duty(ctx.ins.control(2));
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let freq = ctx.ins.control(0) as f64 * ctx.audio.sample_dur;
        let next_duty = ctx.ins.control(2);
        let mut phase = self.phase;
        for o in ctx.outs.audio(0).iter_mut() {
            if phase >= 1.0 {
                phase -= 1.0;
                self.set_duty(next_duty);
            }
            let (duty, invduty, inv1duty) =
                (self.duty as f64, self.invduty as f64, self.inv1duty as f64);
            let z = if phase < duty {
                phase * invduty
            } else {
                (1.0 - phase) * inv1duty
            };
            phase += freq;
            *o = (z - 1.0) as f32;
        }
        self.phase = phase;
        DoneAction::Nothing
    }
}

/// `SyncSaw.ar/kr(syncFreq, sawFreq)`: a sawtooth (`sawFreq`) hard-synced to `syncFreq` - the saw
/// resets whenever the sync oscillator wraps, giving a bright, harmonically rich tone.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SyncSaw {
    phase1: f64,
    phase2: f64,
}

impl Unit for SyncSaw {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let fmul = 2.0 * ctx.audio.sample_dur;
        let freq1x = ctx.ins.control(0) as f64 * fmul;
        let freq2x = ctx.ins.control(1) as f64 * fmul;
        let (mut phase1, mut phase2) = (self.phase1, self.phase2);
        for o in ctx.outs.audio(0).iter_mut() {
            let z = phase2;
            phase2 += freq2x;
            if phase2 >= 1.0 {
                phase2 -= 2.0;
            }
            phase1 += freq1x;
            if phase1 >= 1.0 {
                phase1 -= 2.0;
                phase2 = (phase1 + 1.0) * freq2x / freq1x - 1.0;
            }
            *o = z as f32;
        }
        self.phase1 = phase1;
        self.phase2 = phase2;
        DoneAction::Nothing
    }
}

/// Constructor for [`LFTri`].
pub struct LFTriCtor;

impl UnitDef for LFTriCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(LFTri { phase: 0.0 }))
    }
}

/// Constructor for [`LFPar`].
pub struct LFParCtor;

impl UnitDef for LFParCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(LFPar { phase: 0.0 }))
    }
}

/// Constructor for [`LFCub`].
pub struct LFCubCtor;

impl UnitDef for LFCubCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(LFCub { phase: 0.0 }))
    }
}

/// Constructor for [`VarSaw`].
pub struct VarSawCtor;

impl UnitDef for VarSawCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(VarSaw {
            phase: 0.0,
            duty: 0.5,
            invduty: 4.0,
            inv1duty: 4.0,
            _pad: 0,
        }))
    }
}

/// Constructor for [`SyncSaw`].
pub struct SyncSawCtor;

impl UnitDef for SyncSawCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(SyncSaw {
            phase1: 0.0,
            phase2: 0.0,
        }))
    }
}

/// Wrap a phase into `[0, 1)` (assuming a single cycle's worth of drift at most).
#[inline]
fn wrap(phase: f32) -> f32 {
    if phase >= 1.0 {
        phase - 1.0
    } else if phase < 0.0 {
        phase + 1.0
    } else {
        phase
    }
}
