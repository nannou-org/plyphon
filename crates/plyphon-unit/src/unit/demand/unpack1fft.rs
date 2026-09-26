//! `Unpack1FFT` - plyphon's port of scsynth's `Unpack1FFT` (`UnpackFFTUGens.cpp`). Compiled only
//! with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::pv;
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::math;

/// `Unpack1FFT.dr(chain, bufsize, binindex, whichmeasure)`: one bin of the current FFT frame, as a
/// demand source.
///
/// It is the read half of the `Unpack1FFT` -> per-bin arithmetic -> `PackFFT` pattern sclang's
/// `pvcalc`/`pvcollect` helpers emit: a whole spectrum is exposed as a flat list of demand sources,
/// one per magnitude and phase, which `PackFFT` pulls back into the chain buffer.
///
/// `binindex` is **one-based over the middle bins**: `0` is the packed DC term, `bufsize / 2` is the
/// Nyquist term, and every value between reads packed bin `binindex - 1`. `whichmeasure` selects
/// magnitude (`0`) or phase (anything else); the phase of the DC and Nyquist terms is the constant
/// `0`, which the reference reaches by wiring those cases to its clear-outputs function - so those
/// two cases touch neither the chain buffer nor its coordinate form. Every other case forces the
/// buffer to Cartesian form (the lookup-table `ToComplexApx`, as in scsynth) and computes
/// `hypot`/`atan2` from the bin, as the reference does.
///
/// `bufsize`, `binindex` and `whichmeasure` are read once, as the reference's constructor reads them
/// (`ZIN0`) to pick one of its five calc functions.
///
/// A pull is idempotent within a World block: the first pull of a block computes and records the
/// block, and every later pull in that block re-emits the recorded value, so a spectrum fanned out
/// across many consumers is read once per frame. Between frames (`chain < 0`) the held value is
/// re-emitted and nothing is recorded, so the next frame recomputes. There is no exhaustion - the
/// unit never yields `NaN` by itself - and a reset takes the same path as a produce (the reference's
/// calc functions ignore the reset flag entirely).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Unpack1Fft {
    /// The World block whose value [`outval`](Self::outval) holds; `u64::MAX` until the first read.
    stamp: u64,
    /// The last value read, re-emitted within a block and between frames.
    outval: f32,
    /// Which packed value this instance reads - one of the `KIND_*` constants.
    kind: u32,
    /// The packed bin index (`binindex - 1`) for the magnitude and phase kinds. `u32::MAX` when
    /// `binindex` addresses no bin, which reads as a bin the frame does not have.
    bin: u32,
    _pad: u32,
}

impl Unpack1Fft {
    const CHAIN: usize = 0;
    const BUFSIZE: usize = 1;
    const BININDEX: usize = 2;
    const WHICHMEASURE: usize = 3;

    /// [`kind`](Self::kind): the DC term's magnitude.
    const KIND_DC: u32 = 0;
    /// [`kind`](Self::kind): the Nyquist term's magnitude.
    const KIND_NYQ: u32 = 1;
    /// [`kind`](Self::kind): a middle bin's magnitude.
    const KIND_MAG: u32 = 2;
    /// [`kind`](Self::kind): a middle bin's phase.
    const KIND_PHASE: u32 = 3;
    /// [`kind`](Self::kind): the constant `0` the reference emits for the phase of the DC and
    /// Nyquist terms.
    const KIND_ZERO: u32 = 4;

    /// Select which packed value this instance reads from `bufsize`, `binindex` and
    /// `whichmeasure`, as the reference's constructor selects its calc function.
    fn latch(&mut self, ctx: &mut DemandCtx<'_>) {
        let bufsize = ctx.demand(Self::BUFSIZE) as i32;
        let binindex = ctx.demand(Self::BININDEX) as i32;
        let wantmag = ctx.demand(Self::WHICHMEASURE) == 0.0;
        let nyq_index = bufsize >> 1;
        self.kind = match (wantmag, binindex) {
            (true, 0) => Self::KIND_DC,
            (true, i) if i == nyq_index => Self::KIND_NYQ,
            (true, _) => Self::KIND_MAG,
            (false, 0) => Self::KIND_ZERO,
            (false, i) if i == nyq_index => Self::KIND_ZERO,
            (false, _) => Self::KIND_PHASE,
        };
        self.bin = u32::try_from(binindex.saturating_sub(1)).unwrap_or(u32::MAX);
    }

    /// Bring [`outval`](Self::outval) up to date for the current block, reading the chain buffer at
    /// most once per block. Shared by produce and reset, which the reference does not distinguish.
    fn read(&mut self, ctx: &mut DemandCtx<'_>) {
        if self.kind == Self::KIND_ZERO {
            return;
        }
        let block = ctx.buf_counter();
        if self.stamp == block {
            return;
        }
        // The chain index is an ordinary signal input, carrying the frame's buffer number or a
        // negative value between frames.
        let fbufnum = ctx.demand(Self::CHAIN);
        if fbufnum < 0.0 {
            return;
        }
        let fft = ctx.fft();
        let Some(mut buffer) = ctx.buffer_mut(fbufnum as usize) else {
            return;
        };
        // Magnitudes and phases are read from the Cartesian form, converting the frame if an
        // upstream unit left it polar.
        let Some(spectrum) = pv::to_complex(&mut buffer, fft.complex()) else {
            return;
        };
        let bin = self.bin as usize;
        let value = match self.kind {
            Self::KIND_DC => Some(*spectrum.dc),
            Self::KIND_NYQ => Some(*spectrum.nyq),
            Self::KIND_MAG => spectrum.bins.get(bin).map(|b| math::hypot(b.y, b.x)),
            _ => spectrum.bins.get(bin).map(|b| math::atan2(b.y, b.x)),
        };
        // A bin the frame does not have holds the previous value and records nothing, so a frame of
        // the expected size resumes reading.
        let Some(value) = value else {
            return;
        };
        self.outval = value;
        self.stamp = block;
    }
}

impl DemandUnit for Unpack1Fft {
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        self.latch(ctx);
    }

    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.read(ctx);
    }

    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        self.read(ctx);
        self.outval
    }
}

/// Constructor for [`Unpack1Fft`]. The instance selects what it reads when the synth is constructed.
pub struct Unpack1FftCtor;

impl DemandUnitDef for Unpack1FftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() <= Unpack1Fft::WHICHMEASURE {
            return Err(BuildError::WrongInputCount);
        }
        Ok(demand_unit_spec(Unpack1Fft {
            stamp: u64::MAX,
            outval: 0.0,
            kind: 0,
            bin: 0,
            _pad: 0,
        }))
    }
}
