//! `DelayN` - a non-interpolating delay line, plyphon's port of scsynth's `DelayN`.
//!
//! This is the first unit to use per-instance [auxiliary memory](crate::unit::Aux): its delay line
//! is sized at build time from the scalar `maxdelaytime` and lives in the synth's pool block (the
//! safe stand-in for scsynth's `RTAlloc`'d `float* m_dlybuf`). The whole delay/comb/allpass family
//! follows the same shape; only the per-sample read kernel (no interp here; linear/cubic, feedback)
//! differs.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// scsynth's `DelayN`/`DelayL` `minDelaySamples` (a cubic `DelayC` would use 2). `CalcDelay` clamps
/// the requested delay in samples to `[MIN_DELAY, fdelaylen]`, so a `delaytime` of 0 still delays by
/// one sample (plyphon omits scsynth's separate scalar-zero passthrough optimization).
const MIN_DELAY: f32 = 1.0;

/// Calc-variant tags, chosen from the `delaytime` input's rate at build time (scsynth selects a
/// `_next` vs `_next_a` calc func by `INRATE(2)`). Stored as a `u32` so the state stays [`Pod`].
mod calc {
    /// `delaytime` is constant or control-rate: one value per block, slope-interpolated when it
    /// changes (scsynth's `CALCSLOPE`).
    pub const DELAY_CONTROL: u32 = 0;
    /// `delaytime` is audio-rate: recomputed every sample, no slope.
    pub const DELAY_AUDIO: u32 = 1;
}

/// `DelayN.ar(in, maxdelaytime, delaytime)`: a simple delay with no interpolation.
///
/// The delay line itself is not a field - it is the [`ProcessCtx::aux`] slice, sized to `len` `f32`s
/// at build time. Field names mirror scsynth's `DelayUnit` (minus the `float* m_dlybuf` pointer).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DelayN {
    /// Current delay in samples (`m_dsamp`), possibly mid-slope; truncated to an integer tap.
    dsamp: f32,
    /// The `delaytime` seen last block (`m_delaytime`), to detect a change and slope toward the new
    /// value. Only used for control-rate `delaytime`.
    delaytime: f32,
    /// Delay-line length in samples (`m_idelaylen`), a power of two.
    len: u32,
    /// `len - 1`, the wrap mask for the power-of-two circular buffer (`m_mask`).
    mask: u32,
    /// Monotonic write phase (`m_iwrphase`); only masked at use, so it may wrap freely once warm.
    iwrphase: u32,
    /// Samples written so far, saturating at `len` (`m_numoutput`). While `< len` the line is not yet
    /// filled, so reads behind the write head return 0 (scsynth's `_z` cold-start variants).
    numoutput: u32,
    /// Which calc variant (see [`calc`]), chosen from the `delaytime` rate at build time.
    calc: u32,
}

impl DelayN {
    const IN: usize = 0;
    const MAXDELAY: usize = 1;
    const DELAY: usize = 2;
}

/// Clamp a delay in samples to `[MIN_DELAY, max]` (scsynth's `CalcDelay`/`sc_clip`). NaN-safe: the
/// `max`/`min` order maps a NaN to `MIN_DELAY` rather than propagating it onto the read index.
#[inline]
fn clamp_delay(samples: f32, max: f32) -> f32 {
    samples.max(MIN_DELAY).min(max)
}

/// One delay sample: write `x` at the write head, read the tap `idsamp` samples behind it, advance.
/// `warm` drops the cold-start guard once the line has filled (scsynth's `_z` -> steady calc swap);
/// while cold, a tap before the start of writing reads 0. Mirrors scsynth's `DelayN_helper::perform`.
#[inline]
fn delay_tick(
    buf: &mut [f32],
    iwrphase: &mut u32,
    mask: u32,
    idsamp: i64,
    x: f32,
    warm: bool,
) -> f32 {
    buf[(*iwrphase & mask) as usize] = x;
    let y = if warm {
        buf[(iwrphase.wrapping_sub(idsamp as u32) & mask) as usize]
    } else {
        // Cold start: `iwrphase < len`, so this subtraction cannot wrap and the sign is meaningful.
        let irdphase = *iwrphase as i64 - idsamp;
        if irdphase < 0 {
            0.0
        } else {
            buf[(irdphase as u32 & mask) as usize]
        }
    };
    *iwrphase = iwrphase.wrapping_add(1);
    y
}

impl Unit for DelayN {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Seed `dsamp`/`delaytime` from the initial `delaytime` so the first block uses the steady
        // path (no ramp-from-zero), mirroring scsynth's `DelayUnit_Reset` (`m_dsamp = CalcDelay`).
        let dt = ctx.ins.control(Self::DELAY);
        self.delaytime = dt;
        self.dsamp = clamp_delay(dt * ctx.audio.sample_rate as f32, self.len as f32);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let sr = ctx.audio.sample_rate as f32;
        let max_delay = self.len as f32;
        let mask = self.mask;
        let mut iwrphase = self.iwrphase;
        let warm = self.numoutput >= self.len;

        // The signal input, per sample (audio-rate) or broadcast (control/scalar). Wires are `'a`, so
        // these borrows do not tie up `ctx` while `outs`/`aux` are borrowed mutably below.
        let in_audio = (ctx.ins.rate(Self::IN) == Rate::Audio).then(|| ctx.ins.audio(Self::IN));
        let in_ctrl = ctx.ins.control(Self::IN);
        let dt_audio = (self.calc == calc::DELAY_AUDIO).then(|| ctx.ins.audio(Self::DELAY));
        let dt_ctrl = ctx.ins.control(Self::DELAY);

        let out = ctx.outs.audio(0);
        let buf = ctx.aux.f32_mut();
        if buf.is_empty() {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        let n = out.len();

        match dt_audio {
            Some(dt) => {
                // Audio-rate delaytime: recompute the tap every sample, no slope or state carry.
                for i in 0..n {
                    let x = in_audio.map_or(in_ctrl, |s| s[i]);
                    let idsamp = clamp_delay(dt[i] * sr, max_delay) as i64;
                    out[i] = delay_tick(buf, &mut iwrphase, mask, idsamp, x, warm);
                }
            }
            None if dt_ctrl == self.delaytime => {
                // Unchanged control-rate delaytime: a fixed integer tap for the whole block.
                let idsamp = self.dsamp as i64;
                for i in 0..n {
                    let x = in_audio.map_or(in_ctrl, |s| s[i]);
                    out[i] = delay_tick(buf, &mut iwrphase, mask, idsamp, x, warm);
                }
            }
            None => {
                // Changed control-rate delaytime: slope `dsamp` from the old value to the new across
                // the block (scsynth's `CALCSLOPE`, i.e. `(next - cur) / blockSize` per sample).
                let next = clamp_delay(dt_ctrl * sr, max_delay);
                let mut dsamp = self.dsamp;
                let slope = (next - dsamp) / n as f32;
                for i in 0..n {
                    dsamp += slope;
                    let idsamp = dsamp as i64;
                    let x = in_audio.map_or(in_ctrl, |s| s[i]);
                    out[i] = delay_tick(buf, &mut iwrphase, mask, idsamp, x, warm);
                }
                self.dsamp = dsamp;
                self.delaytime = dt_ctrl;
            }
        }

        self.iwrphase = iwrphase;
        if !warm {
            self.numoutput = self.numoutput.saturating_add(n as u32).min(self.len);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DelayN`].
pub struct DelayNCtor;

impl UnitDef for DelayNCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // Needs `in`, `maxdelaytime`, `delaytime`.
        if ctx.input_rates.len() < 3 {
            return Err(BuildError::WrongInputCount);
        }
        // `maxdelaytime` sizes the line, so it must be a compile-time constant (scsynth reads it once
        // at ctor and never again).
        let max_delay =
            ctx.const_input(DelayN::MAXDELAY)
                .ok_or(BuildError::AuxRequiresConstant {
                    input: DelayN::MAXDELAY,
                })?;
        let sr = ctx.audio.sample_rate;
        let block = ctx.audio.block_size as i64;
        // scsynth's `DelayUnit_AllocDelayLine`: `NEXTPOWEROFTWO(ceil(maxdelay*SR + 1) + BUFLENGTH)`.
        // The `+1` lets a read sit one sample behind a write at the same phase; the `+block` headroom
        // keeps the write head and any delayed read from colliding within a block; the power-of-two
        // length makes circular addressing a single mask.
        let base = math::ceil(max_delay.max(0.0) as f64 * sr + 1.0) as i64;
        let len = (base + block).max(1) as u64;
        let len = len.next_power_of_two();
        let len = len as u32;
        let calc = match ctx.input_rates[DelayN::DELAY] {
            Rate::Audio => calc::DELAY_AUDIO,
            _ => calc::DELAY_CONTROL,
        };
        let aux_bytes = len as usize * core::mem::size_of::<f32>();
        Ok(unit_spec_aux(
            DelayN {
                dsamp: 0.0,
                delaytime: 0.0,
                len,
                mask: len - 1,
                iwrphase: 0,
                numoutput: 0,
                calc,
            },
            aux_bytes,
            core::mem::align_of::<f32>(),
        ))
    }
}
