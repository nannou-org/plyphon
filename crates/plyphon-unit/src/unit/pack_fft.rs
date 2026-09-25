//! `PackFFT` - plyphon's port of scsynth's `PackFFT` (`UnpackFFTUGens.cpp`). Compiled only with the
//! `fft` feature.
//!
//! It is the write half of the `Unpack1FFT` -> per-bin arithmetic -> `PackFFT` pattern sclang's
//! `pvcalc`/`pvcollect` helpers emit: a flat list of magnitude/phase values, usually demand-rate,
//! is pulled once per FFT frame and packed back into the chain buffer.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{DemandWorld, demand_next};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec};
use plyphon_dsp::math;

/// `PackFFT.kr(chain, bufsize, frombin, tobin, zeroothers, numinvals, magsphases...)`: write a
/// magnitude/phase payload into the chain buffer's spectrum.
///
/// The payload follows input `6` as `[mag, phase]` pairs, one per packed slot from `frombin` to
/// `tobin`, where slot `0` is the DC term, slot `numbins + 1` the Nyquist term, and slot `k` in
/// between is packed bin `k - 1`. The DC and Nyquist terms are purely real, so only their magnitudes
/// are pulled. Bins outside the range keep their values unless `zeroothers` is set, in which case
/// they - and the DC/Nyquist terms the range does not reach - are zeroed.
///
/// Each frame forces the buffer to Cartesian form, then pulls and writes one slot at a time in the
/// reference's order: DC, Nyquist (whose input index the reference derives from `numinvals`), then
/// each bin's magnitude and phase. Between frames it outputs `-1` and pulls nothing, so a demand
/// payload advances exactly once per frame. `frombin`, `tobin`, `zeroothers` and `numinvals` are
/// read when the synth starts, as the reference's constructor reads them. The reference trusts them
/// and reads past its inputs or its spectrum when they lie; here such a read or write is skipped.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PackFft {
    /// First packed slot the payload covers.
    frombin: i32,
    /// Last packed slot the payload covers.
    tobin: i32,
    /// Payload length, used to locate the Nyquist magnitude.
    numinvals: i32,
    /// Non-zero when the frame zeroes everything the payload does not cover.
    zeroothers: u32,
}

impl PackFft {
    const FROMBIN: usize = 2;
    const TOBIN: usize = 3;
    const ZEROOTHERS: usize = 4;
    const NUMINVALS: usize = 5;
    /// The first payload input (scsynth's `PACKFFT_INPUTSOFFSET`).
    const PAYLOAD: i64 = 6;
}

/// Pull input `input` (scsynth's `DEMANDINPUT`), or `None` when the index is not one of the unit's
/// inputs.
fn pull(ctx: &mut ProcessCtx<'_>, input: i64) -> Option<f32> {
    let input = usize::try_from(input).ok().filter(|&i| i < ctx.ins.len())?;
    let mut world = DemandWorld {
        buffers: &mut *ctx.buffers,
        local_bufs: &mut ctx.local_bufs,
        node_id: ctx.node_id,
        node_msgs: &mut ctx.node_msgs,
        buf_counter: ctx.buf_counter,
        rgen: &mut *ctx.rgen,
    };
    Some(demand_next(&ctx.ins, &mut ctx.demand, &mut world, input))
}

/// Run `write` on the chain buffer's packed spectrum, looked up afresh: a pull between two writes
/// may itself read the buffer (an `Unpack1FFT`), so the spectrum is not held across pulls.
fn with_spectrum(ctx: &mut ProcessCtx<'_>, bufnum: usize, write: impl FnOnce(pv::Spectrum<'_>)) {
    if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
        && let Some(spectrum) = pv::spectrum(&mut buffer)
    {
        write(spectrum);
    }
}

impl Unit for PackFft {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.frombin = ctx.ins.control(Self::FROMBIN) as i32;
        self.tobin = ctx.ins.control(Self::TOBIN) as i32;
        self.zeroothers = u32::from(ctx.ins.control(Self::ZEROOTHERS) > 0.0);
        self.numinvals = ctx.ins.control(Self::NUMINVALS) as i32;
        // The constructor passes the chain (input 0) straight through.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let Some(bufnum) = pv::pv_frame(ctx) else {
            return DoneAction::Nothing;
        };
        let Some(numbins) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            .and_then(|mut buffer| pv::to_complex(&mut buffer).map(|s| s.bins.len()))
        else {
            return DoneAction::Nothing;
        };
        let numbins = numbins as i64;
        let frombin = i64::from(self.frombin);
        let tobin = i64::from(self.tobin);
        let zeroothers = self.zeroothers != 0;

        if frombin == 0 {
            if let Some(dc) = pull(ctx, Self::PAYLOAD) {
                with_spectrum(ctx, bufnum, |s| *s.dc = dc);
            }
        } else if zeroothers {
            with_spectrum(ctx, bufnum, |s| *s.dc = 0.0);
        }

        if tobin == numbins + 1 {
            let slot = Self::PAYLOAD + i64::from(self.numinvals) - 2 - 2 * frombin;
            if let Some(nyq) = pull(ctx, slot) {
                with_spectrum(ctx, bufnum, |s| *s.nyq = nyq);
            }
        } else if zeroothers {
            with_spectrum(ctx, bufnum, |s| *s.nyq = 0.0);
        }

        let startat = if frombin == 0 { 0 } else { frombin - 1 };
        let endbefore = numbins.min(tobin);
        for i in startat..endbefore {
            let slot = 2 * i + Self::PAYLOAD + 2 - 2 * frombin;
            let (Some(mag), Some(phase)) = (pull(ctx, slot), pull(ctx, slot + 1)) else {
                continue;
            };
            with_spectrum(ctx, bufnum, |s| {
                if let Some(bin) = usize::try_from(i).ok().and_then(|i| s.bins.get_mut(i)) {
                    bin.x = mag * math::cos(phase);
                    bin.y = mag * math::sin(phase);
                }
            });
        }

        if zeroothers {
            with_spectrum(ctx, bufnum, |s| {
                let zero = pv::Bin { x: 0.0, y: 0.0 };
                let below = startat.clamp(0, numbins) as usize;
                let above = endbefore.clamp(0, numbins) as usize;
                s.bins[..below].fill(zero);
                s.bins[above..].fill(zero);
            });
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PackFft`]. The bin range and payload length are read when the synth starts.
pub struct PackFftCtor;

impl UnitDef for PackFftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < PackFft::PAYLOAD as usize {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PackFft::zeroed()))
    }
}
