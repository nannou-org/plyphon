//! Non-band-limited low-frequency oscillators - plyphon's ports of scsynth's `LFSaw`, `LFPulse`, and
//! `Impulse`.
//!
//! These are control/LFO sources (and raw audio waves); for clean band-limited audio use [`Saw`] and
//! [`Pulse`]. Frequency is read at control rate (one value per block), so frequency modulation is at
//! block resolution. Each writes a full block; a `.kr` instance just publishes the first sample.
//!
//! [`Saw`]: crate::unit::Saw
//! [`Pulse`]: crate::unit::Pulse

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};

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
