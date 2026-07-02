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

/// `LFSaw.ar/kr(freq, iphase)`: a non-band-limited sawtooth. The output *is* the phase, ramping
/// through `[-1, 1)` from `iphase` (scsynth's convention: `iphase` 0 starts at 0, rises to 1, wraps
/// to -1 - not a ramp starting at -1).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFSaw {
    phase: f64,
}

impl LFSaw {
    const FREQ: usize = 0;
    const IPHASE: usize = 1;
}

impl Unit for LFSaw {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // `iphase` is in cycles over `[0, 2)` (sclang's convention); map into the `[-1, 1)` ramp.
        let iphase = if ctx.ins.len() > Self::IPHASE {
            ctx.ins.control(Self::IPHASE) as f64
        } else {
            0.0
        };
        self.phase = math::rem_euclid(iphase + 1.0, 2.0) - 1.0;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // scsynth's `mFreqMul`: 2 units of phase per cycle.
        let inc = ctx.ins.control(Self::FREQ) as f64 * 2.0 * ctx.own.sample_dur;
        let mut phase = self.phase;
        for o in ctx.outs.audio(0).iter_mut() {
            *o = phase as f32;
            phase += inc;
            if phase >= 1.0 {
                phase -= 2.0;
            } else if phase <= -1.0 {
                phase += 2.0;
            }
        }
        self.phase = phase;
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
/// (default 0.5), starting `iphase` cycles into the waveform.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFPulse {
    phase: f64,
}

impl LFPulse {
    const FREQ: usize = 0;
    const IPHASE: usize = 1;
    const WIDTH: usize = 2;
}

impl Unit for LFPulse {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let iphase = if ctx.ins.len() > Self::IPHASE {
            ctx.ins.control(Self::IPHASE) as f64
        } else {
            0.0
        };
        self.phase = math::rem_euclid(iphase, 1.0);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(Self::FREQ) as f64 * ctx.own.sample_dur;
        let width = if ctx.ins.len() > Self::WIDTH {
            ctx.ins.control(Self::WIDTH)
        } else {
            0.5
        } as f64;
        let mut phase = self.phase;
        for o in ctx.outs.audio(0).iter_mut() {
            *o = if phase < width { 1.0 } else { 0.0 };
            phase += inc;
            if phase >= 1.0 {
                phase -= 1.0;
            }
        }
        self.phase = phase;
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
/// otherwise. `phase` is the starting phase in cycles: 0 (the default) fires immediately
/// (scsynth special-cases it to 1), and e.g. 0.9 starts nine tenths through the cycle, firing
/// after the remaining tenth.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Impulse {
    phase: f64,
}

impl Impulse {
    const FREQ: usize = 0;
    const PHASE: usize = 1;
}

impl Unit for Impulse {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        let iphase = if ctx.ins.len() > Self::PHASE {
            ctx.ins.control(Self::PHASE) as f64
        } else {
            0.0
        };
        let p = math::rem_euclid(iphase, 1.0);
        // Start at the cycle boundary when unphased, so the first sample is an impulse.
        self.phase = if p == 0.0 { 1.0 } else { p };
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let inc = ctx.ins.control(Self::FREQ) as f64 * ctx.own.sample_dur;
        let mut phase = self.phase;
        for o in ctx.outs.audio(0).iter_mut() {
            if phase >= 1.0 {
                phase -= 1.0;
                *o = 1.0;
            } else {
                *o = 0.0;
            }
            phase += inc;
        }
        self.phase = phase;
        DoneAction::Nothing
    }
}

/// Constructor for [`Impulse`].
pub struct ImpulseCtor;

impl UnitDef for ImpulseCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // `init` seeds the phase; start at the cycle boundary for the un-phased default.
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
        let freq = ctx.ins.control(0) as f64 * 4.0 * ctx.own.sample_dur;
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
        let freq = ctx.ins.control(0) as f64 * 4.0 * ctx.own.sample_dur;
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
        let freq = ctx.ins.control(0) as f64 * 2.0 * ctx.own.sample_dur;
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
        let freq = ctx.ins.control(0) as f64 * ctx.own.sample_dur;
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
        let fmul = 2.0 * ctx.own.sample_dur;
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

/// `LFGauss.ar/kr(duration, width, iphase, loop, doneAction)`: a Gaussian-shaped grain/LFO. Over
/// `duration` seconds an internal ramp sweeps `[-1, 1]`; the output is the Gaussian bump `exp(-0.5 *
/// ((phase - iphase) / width)^2)`. When the ramp completes it either loops (`loop != 0`, the default)
/// or marks itself done and fires `doneAction` (so a grain can free its own synth). All inputs are read
/// at control rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LFGauss {
    /// The ramp position, in `[-1, 1]`, advanced by `2 / (duration * sampleRate)` per sample.
    phase: f64,
}

impl LFGauss {
    const DURATION: usize = 0;
    const WIDTH: usize = 1;
    const IPHASE: usize = 2;
    const LOOP: usize = 3;
    const DONE: usize = 4;
}

impl Unit for LFGauss {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let dur = ctx.ins.control(Self::DURATION) as f64;
        // A zero/negative width or duration would poison the block with NaN/inf; floor them.
        let width = (ctx.ins.control(Self::WIDTH) as f64).abs().max(1e-6);
        let iphase = read_input(&ctx.ins, Self::IPHASE, 0.0) as f64;
        let loop_on = read_input(&ctx.ins, Self::LOOP, 1.0) != 0.0;
        let done_action = if ctx.ins.len() > Self::DONE {
            DoneAction::from_code(ctx.ins.control(Self::DONE))
        } else {
            DoneAction::Nothing
        };

        let step = 2.0 * ctx.own.sample_dur / dur.abs().max(1e-9);
        let factor = -1.0 / (2.0 * width * width);
        let mut x = self.phase - iphase;
        let mut completed = false;
        for o in ctx.outs.audio(0).iter_mut() {
            if x > 1.0 {
                if loop_on {
                    x -= 2.0;
                } else {
                    completed = true;
                }
            }
            *o = math::exp(x * x * factor) as f32;
            x += step;
        }
        self.phase = x + iphase;
        if completed {
            // Reaching the end marks the unit done (scsynth's `mDone`), so a watcher sees it even when
            // `doneAction` is 0.
            ctx.done.mark_done();
        }
        if completed {
            done_action
        } else {
            DoneAction::Nothing
        }
    }
}

/// Constructor for [`LFGauss`].
pub struct LFGaussCtor;

impl UnitDef for LFGaussCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        // Start the ramp at the beginning of the sweep (scsynth's `mPhase = -1`).
        Ok(unit_spec(LFGauss { phase: -1.0 }))
    }
}

/// Read input `i` as a single value, or `default` if the unit was built with fewer inputs.
fn read_input(ins: &crate::unit::Inputs<'_>, i: usize, default: f32) -> f32 {
    if ins.len() > i {
        ins.control(i)
    } else {
        default
    }
}
