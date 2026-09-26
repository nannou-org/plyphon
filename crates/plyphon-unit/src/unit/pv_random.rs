//! Randomised spectral (`PV_*`) operators - plyphon's ports of scsynth's `PV_MagNoise`,
//! `PV_RandComb`, `PV_RandWipe` and `PV_BinScramble` (`PV_UGens.cpp`).
//!
//! Every draw comes from the synth's random stream (scsynth's `RGET`/`RPUT` over `mParent->mRGen`),
//! in the reference's order. `PV_MagNoise` draws on every frame; the other three draw a random bin
//! ordering on the first frame and again on the first frame after each rising `trig`, holding it in
//! memory allocated from the engine's pool on that first frame, when the chain buffer's size gives
//! the bin count (as `PV_Diffuser` does). None of them converts the frame, as scsynth does not:
//! `PV_MagNoise` branches on its stored form (`PV_UGens.cpp`:462), and the other three read the
//! packed data as stored and move or zero whole bins (965, 1031-1032, 1199-1200). Compiled only with
//! the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, pv, unit_spec, unit_spec_pool};
use plyphon_dsp::buffer::SpectrumCoord;
use plyphon_dsp::rng::Rng;

/// `PV_MagNoise(buffer)`: multiply every magnitude by a fresh bipolar random value in `[-1, 1)`
/// each frame.
///
/// A Cartesian bin scales both parts by its draw and a polar bin scales its magnitude, so either
/// form keeps its phase (up to a sign flip); the frame is not converted (`PV_UGens.cpp`:462). One
/// draw per bin in order, then one for the DC term and one for the Nyquist term (scsynth's
/// `PV_MagNoise_next`).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvMagNoise {
    /// The unit is stateless; the state block must still be a non-zero-sized `Pod`.
    _pad: u32,
}

impl Unit for PvMagNoise {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if let Some(bufnum) = pv::pv_frame(ctx)
            && let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
        {
            let complex = buffer.coord() == SpectrumCoord::Complex;
            if let Some(spectrum) = pv::spectrum(&mut buffer) {
                let mut rgen = *ctx.rgen;
                for bin in spectrum.bins.iter_mut() {
                    let r = rgen.next_bipolar();
                    bin.x *= r;
                    if complex {
                        bin.y *= r;
                    }
                }
                *spectrum.dc *= rgen.next_bipolar();
                *spectrum.nyq *= rgen.next_bipolar();
                *ctx.rgen = rgen;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvMagNoise`].
pub struct PvMagNoiseCtor;

impl UnitDef for PvMagNoiseCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.is_empty() {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(PvMagNoise { _pad: 0 }))
    }
}

/// The trigger and random-ordering state shared by [`PvRandComb`], [`PvRandWipe`] and
/// [`PvBinScramble`] - scsynth's `m_prevtrig`, `m_triggered`, `m_numbins` and the non-null test on
/// the unit's table.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Reorder {
    /// Number of bins the ordering covers, from the first frame's chain buffer.
    numbins: u32,
    /// Previous block's `trig` value, for rising-edge detection across blocks.
    prev_trig: f32,
    /// `1` once a rising `trig` has been seen and not yet acted on by a frame.
    triggered: u32,
    /// `1` once the unit's table is allocated.
    allocated: u32,
}

impl Reorder {
    /// Sample `trig` - every block, frame or not, before the frame preamble - latching a rising edge
    /// for the next frame to act on.
    fn latch(&mut self, trig: f32) {
        if trig > 0.0 && self.prev_trig <= 0.0 {
            self.triggered = 1;
        }
        self.prev_trig = trig;
    }

    /// The first-frame allocation and re-choice logic. On the first frame, allocate `bytes` and
    /// fix the bin count; then (or when a latched trigger is pending) the caller draws a fresh
    /// ordering. After the first frame, a frame with a different bin count is skipped. Returns
    /// `None` to skip the frame, else whether to choose a new ordering.
    ///
    /// The first frame chooses without clearing the latch, so a trigger that rose on or before the
    /// first frame chooses again on the next one - the reference's behaviour.
    fn frame(&mut self, ctx: &mut ProcessCtx<'_>, numbins: usize, bytes: usize) -> Option<bool> {
        if self.allocated == 0 {
            if !ctx.aux.alloc(bytes) {
                return None;
            }
            self.allocated = 1;
            self.numbins = numbins as u32;
            Some(true)
        } else if numbins != self.numbins as usize {
            None
        } else {
            Some(core::mem::take(&mut self.triggered) != 0)
        }
    }
}

/// Fill `ordering` with `0..n` and scramble it with one draw per slot: slot `i` swaps with slot
/// `(int)(frand() * (n - i))`. The partner is drawn from the whole front of the table rather than
/// from `i..n`, so this is not a uniform shuffle; it is the reference's own permutation.
fn scramble(ordering: &mut [i32], rgen: &mut Rng) {
    let n = ordering.len();
    for (i, slot) in ordering.iter_mut().enumerate() {
        *slot = i as i32;
    }
    for i in 0..n {
        let j = (rgen.next_unipolar() * (n - i) as i32 as f32) as i32 as usize;
        // `frand() < 1`, so the partner is always below `n - i`; the clamp never binds.
        ordering.swap(i, j.min(n - 1));
    }
}

/// How many bins a `wipe` fraction of `numbins` selects: `(int)(wipe * numbins)`, clipped to
/// `[0, numbins]`.
fn wipe_count(wipe: f32, numbins: usize) -> usize {
    ((wipe * numbins as i32 as f32) as i32).clamp(0, numbins as i32) as usize
}

/// A zeroed bin - scsynth's `SCComplex = 0.f`, which zeroes both halves.
fn zero() -> pv::Bin {
    pv::Bin { x: 0.0, y: 0.0 }
}

/// `PV_RandComb(buffer, wipe, trig)`: zero a random `wipe` fraction of the bins.
///
/// The bins are zeroed in a random order drawn on the first frame and again on the next frame after
/// each rising `trig`, so sweeping `wipe` from `0` to `1` removes bins one by one in that order. At
/// `wipe >= 1` the DC and Nyquist terms are zeroed too, silencing the frame. Whole bins are zeroed,
/// so the frame is not converted and its coordinate form is untouched (scsynth's `PV_RandComb_next`,
/// `PV_UGens.cpp`:965).
///
/// The ordering (one `i32` per bin) lives in `aux`, allocated on the first frame.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvRandComb {
    reorder: Reorder,
}

impl PvRandComb {
    const WIPE: usize = 1;
    const TRIG: usize = 2;
}

impl Unit for PvRandComb {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.reorder.latch(ctx.ins.control(Self::TRIG));
        let Some(bufnum) = pv::pv_frame(ctx) else {
            return DoneAction::Nothing;
        };
        let Some(numbins) = frame_bins(ctx, bufnum) else {
            return DoneAction::Nothing;
        };
        let bytes = numbins * core::mem::size_of::<i32>();
        let Some(choose) = self.reorder.frame(ctx, numbins, bytes) else {
            return DoneAction::Nothing;
        };
        let ordering = &mut ctx.aux.cast_mut::<i32>()[..numbins];
        if choose {
            scramble(ordering, ctx.rgen);
        }
        let n = wipe_count(ctx.ins.control(Self::WIPE), numbins);
        if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::spectrum(&mut buffer)
        {
            for &k in &ordering[..n] {
                spectrum.bins[k as usize] = zero();
            }
            if n == numbins {
                // The DC and Nyquist terms are outside the ordering, so the full wipe clears them
                // explicitly to reach silence.
                *spectrum.dc = 0.0;
                *spectrum.nyq = 0.0;
            }
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvRandComb`]: the unit allocates its ordering on the first frame.
pub struct PvRandCombCtor;

impl UnitDef for PvRandCombCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= PvRandComb::TRIG {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvRandComb::zeroed())
        })
    }
}

/// The bin count of chain buffer `bufnum`, `(samples - 2) / 2`; `None` if there is no such buffer
/// or it cannot be viewed as a packed spectrum.
fn frame_bins(ctx: &mut ProcessCtx<'_>, bufnum: usize) -> Option<usize> {
    let mut buffer = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)?;
    pv::spectrum(&mut buffer).map(|s| s.bins.len())
}

/// `PV_RandWipe(bufferA, bufferB, wipe, trig)`: crossfade from spectrum `A` to spectrum `B` by
/// replacing a random `wipe` fraction of `A`'s bins with `B`'s.
///
/// The bins are replaced in a random order drawn on the first frame and again on the next frame
/// after each rising `trig`. The DC and Nyquist terms stay `A`'s. Each bin is copied as stored,
/// without converting either buffer, so the two chains should be in the same coordinate form - as
/// in the reference (scsynth's `PV_RandWipe_next`, `PV_UGens.cpp`:1031-1032). The frame preamble
/// is the shared `PV_GET_BUF2` port, [`pv::pv_pair_frame`], converting neither buffer.
///
/// A frame is ready only when both chains carry one; the output is then `A`'s buffer number, and
/// `-1` otherwise. Buffers of different sizes leave `A` untouched. The ordering (one `i32` per bin)
/// lives in `aux`, allocated on the first frame.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvRandWipe {
    reorder: Reorder,
}

impl PvRandWipe {
    const WIPE: usize = 2;
    const TRIG: usize = 3;
}

impl Unit for PvRandWipe {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.reorder.latch(ctx.ins.control(Self::TRIG));
        // scsynth's `PV_GET_BUF2` (`FFT_UGens.h`:116-154); the unit converts neither buffer.
        let Some((a, b)) = pv::pv_pair_frame(ctx, pv::unconverted) else {
            return DoneAction::Nothing;
        };
        let Some(numbins) = frame_bins(ctx, a) else {
            return DoneAction::Nothing;
        };
        let bytes = numbins * core::mem::size_of::<i32>();
        let Some(choose) = self.reorder.frame(ctx, numbins, bytes) else {
            return DoneAction::Nothing;
        };
        let ordering = &mut ctx.aux.cast_mut::<i32>()[..numbins];
        if choose {
            scramble(ordering, ctx.rgen);
        }
        let n = wipe_count(ctx.ins.control(Self::WIPE), numbins);
        // Each chosen bin of `A` takes `B`'s as stored (one chain wiping into itself copies each
        // bin onto itself).
        pv::pv_pair_op(ctx.buffers, &mut ctx.local_bufs, a, b, |p, q| {
            for &k in &ordering[..n] {
                let k = k as usize;
                p.bins[k] = q.bin(k, p.bins[k]);
            }
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`PvRandWipe`]: the unit allocates its ordering on the first frame.
pub struct PvRandWipeCtor;

impl UnitDef for PvRandWipeCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= PvRandWipe::TRIG {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvRandWipe::zeroed())
        })
    }
}

/// `PV_BinScramble(buffer, wipe, width, trig)`: move a `wipe` fraction of the bins to random
/// positions, each taking its content from a random bin at most `width * numbins` bins away.
///
/// On the first frame, and again on the next frame after each rising `trig`, the unit draws a
/// random ordering `to` of the bins and, for each, a source bin `from` within the `width` window
/// around it (`width` is read then, not every frame). Each frame the first `wipe * numbins` entries
/// of `to` take their content from `from`, and every other bin keeps its own. Sources may repeat,
/// so some bins are duplicated and others lost. The DC and Nyquist terms are untouched, and whole
/// bins are moved as stored, so the frame is not converted and its coordinate form is untouched
/// (scsynth's `PV_BinScramble_next`, `PV_UGens.cpp`:1199-1200).
///
/// `aux` holds the two orderings (`2 * numbins` `i32`s) and a scratch frame as large as the chain
/// buffer, allocated on the first frame.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PvBinScramble {
    reorder: Reorder,
}

impl PvBinScramble {
    const WIPE: usize = 1;
    const WIDTH: usize = 2;
    const TRIG: usize = 3;
}

/// Draw [`PvBinScramble`]'s orderings: scramble `to`, then pick each `from[i]` uniformly-ish from
/// the bins within `width` of `to[i]`, clipped to the spectrum - scsynth's `PV_BinScramble_choose`.
fn scramble_bins(to: &mut [i32], from: &mut [i32], width: f32, rgen: &mut Rng) {
    scramble(to, rgen);
    let numbins = to.len() as i32;
    let width = i64::from((width * numbins as f32) as i32);
    for (&k, source) in to.iter().zip(from.iter_mut()) {
        let k = i64::from(k);
        let minr = (k - width).max(0) as i32;
        let maxr = (k + width).min(i64::from(numbins) - 1) as i32;
        *source = (rgen.next_unipolar() * (maxr - minr) as f32 + minr as f32) as i32;
    }
}

impl Unit for PvBinScramble {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor passes input 0 through without running the calc.
        *ctx.outs.control(0) = ctx.ins.control(0);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.reorder.latch(ctx.ins.control(Self::TRIG));
        let Some(bufnum) = pv::pv_frame(ctx) else {
            return DoneAction::Nothing;
        };
        let Some(samples) =
            unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum).map(|buf| buf.data().len())
        else {
            return DoneAction::Nothing;
        };
        let Some(numbins) = frame_bins(ctx, bufnum) else {
            return DoneAction::Nothing;
        };
        // The two orderings, then a scratch as large as the chain buffer (scsynth's two
        // allocations, taken as one region).
        let bytes =
            2 * numbins * core::mem::size_of::<i32>() + samples * core::mem::size_of::<f32>();
        let Some(choose) = self.reorder.frame(ctx, numbins, bytes) else {
            return DoneAction::Nothing;
        };
        let width = ctx.ins.control(Self::WIDTH);
        let wipe = ctx.ins.control(Self::WIPE);
        let (tables, scratch) = ctx.aux.cast_mut::<i32>().split_at_mut(2 * numbins);
        let (to, from) = tables.split_at_mut(numbins);
        if choose {
            scramble_bins(to, from, width, ctx.rgen);
        }
        // The scratch holds the new bins; it is viewed as `Bin`s through its `i32` storage.
        let Ok(scratch) = bytemuck::try_cast_slice_mut::<i32, pv::Bin>(&mut scratch[..2 * numbins])
        else {
            return DoneAction::Nothing;
        };
        // `sc_clip(wipe, 0, 1)` keeps a NaN, which the conversion then reads as no bins.
        let clipped = if 1.0 < wipe { 1.0 } else { wipe };
        let clipped = if clipped < 0.0 { 0.0 } else { clipped };
        let scramble_bins = (numbins as i32 as f32 * clipped) as i32 as usize;
        if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
            && let Some(spectrum) = pv::spectrum(&mut buffer)
        {
            for (&t, &f) in to.iter().zip(from.iter()).take(scramble_bins) {
                // A source outside the spectrum only arises from a negative `width`, where the
                // reference reads outside the frame; the bin is zeroed instead.
                scratch[t as usize] = usize::try_from(f)
                    .ok()
                    .and_then(|f| spectrum.bins.get(f).copied())
                    .unwrap_or_else(zero);
            }
            for &t in &to[scramble_bins..] {
                scratch[t as usize] = spectrum.bins[t as usize];
            }
            spectrum.bins.copy_from_slice(scratch);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`PvBinScramble`]: the unit allocates its orderings and scratch on the first
/// frame.
pub struct PvBinScrambleCtor;

impl UnitDef for PvBinScrambleCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= PvBinScramble::TRIG {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(PvBinScramble::zeroed())
        })
    }
}
