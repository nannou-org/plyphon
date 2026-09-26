//! `MFCC` - plyphon's port of scsynth's mel-frequency cepstral coefficients (`MFCC.cpp`, Dan Stowell
//! and Nick Collins). Compiled only with the `fft` feature.
//!
//! Each frame's power spectrum is warped onto 42 mel bands through triangular filters, the bands
//! taken to dB, and a DCT-II of the 42 band levels gives the coefficients. The filterbank is
//! tabulated for a 1024-sample FFT at 44.1 kHz and at 48 kHz (the `tables` submodule).

mod tables;

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::spec_stats::check_rate;
use crate::unit::{
    self, Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, pv, unit_spec_pool,
};
use plyphon_dsp::math;
use tables::{
    CUMULINDEX_44100, CUMULINDEX_48000, DCT, ENDBIN_44100, ENDBIN_48000, MELBANDWEIGHTS_44100,
    MELBANDWEIGHTS_48000, STARTBIN_44100, STARTBIN_48000,
};

/// Number of mel bands (scsynth's `m_numbands`), and the stride of the DCT table.
const NUM_BANDS: usize = 42;
/// The most coefficients the DCT table holds.
const MAX_COEFFICIENTS: usize = 42;

/// One sample rate's filterbank: each band's first bin and one past its last, where its weights
/// start, and the weights.
struct Filterbank {
    startbin: &'static [usize; NUM_BANDS],
    endbin: &'static [usize; NUM_BANDS],
    cumulindex: &'static [usize; NUM_BANDS + 1],
    weights: &'static [f32; 761],
}

const FILTERBANK_44100: Filterbank = Filterbank {
    startbin: &STARTBIN_44100,
    endbin: &ENDBIN_44100,
    cumulindex: &CUMULINDEX_44100,
    weights: &MELBANDWEIGHTS_44100,
};

const FILTERBANK_48000: Filterbank = Filterbank {
    startbin: &STARTBIN_48000,
    endbin: &ENDBIN_48000,
    cumulindex: &CUMULINDEX_48000,
    weights: &MELBANDWEIGHTS_48000,
};

/// `MFCC.kr(chain, numcoeff = 13)`: `numcoeff` mel-frequency cepstral coefficients of each frame,
/// one per output, each roughly in `[0, 1]`, held between frames.
///
/// Band `k` sums its bins' power `re^2 + im^2` (the DC term's `re^2` for bin 0) weighted by its
/// triangular filter, and becomes `10 * (log10(max(1e-5, sum)) + 5)`. Coefficient `k` is
/// `0.25 * (0.01 * sum_j dct[k][j] * band[j] + 1)`.
///
/// The filterbank is chosen from the server's sample rate when the synth starts: rates above
/// 66150 Hz are halved first (a double-size FFT is assumed there), then 44.1 kHz selects its own
/// table and anything else the 48 kHz one. `numcoeff` is read then too, clamped to `[1, 42]`. The
/// band levels and coefficients live in memory allocated from the engine's pool when the synth
/// starts, cleared.
///
/// The chain buffer is converted to Cartesian form in place with `ToComplexApx` and must hold the
/// filterbank's highest bin (834 samples at 44.1 kHz, 768 at 48 kHz); a smaller frame is skipped.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Mfcc {
    /// The sample rate the filterbank was chosen for (scsynth's `m_srate`).
    srate: f32,
    /// Coefficients computed, `numcoeff` clamped to `[1, 42]` (scsynth's `m_numcoefficients`).
    numcoefficients: u32,
    /// `1` when the 48 kHz filterbank was chosen, `0` for 44.1 kHz.
    use_48000: u32,
    /// The outputs the SynthDef gives the unit.
    num_outputs: u32,
}

impl Mfcc {
    const NUMCOEFF: usize = 1;

    fn filterbank(&self) -> &'static Filterbank {
        if self.use_48000 != 0 {
            &FILTERBANK_48000
        } else {
            &FILTERBANK_44100
        }
    }

    /// Analyse the frame in buffer `bufnum` - scsynth's `MFCC_dofft` (`MFCC.cpp:3377`) and
    /// `MFCC_prepareMel` (`MFCC.cpp:3429`).
    fn dofft(&self, ctx: &mut ProcessCtx<'_>, bufnum: usize) {
        let fb = self.filterbank();
        let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum) else {
            return;
        };
        if buffer.data().len() < 2 * fb.endbin[NUM_BANDS - 1] {
            return;
        }
        // `ToComplexApx(buf)` (`MFCC.cpp:3397`).
        pv::to_complex(&mut buffer, ctx.fft.complex());
        let data = buffer.data();
        let numcoefficients = self.numcoefficients as usize;
        let (bands, mfcc) = ctx.aux.f32_mut().split_at_mut(NUM_BANDS);

        for (k, band) in bands.iter_mut().enumerate() {
            let bandstart = fb.startbin[k];
            let index2 = fb.cumulindex[k] - bandstart;
            let mut bsum = 0.0f32;
            for j in bandstart..fb.endbin[k] {
                let real = data[2 * j];
                let imag = data[2 * j + 1];
                // Bin 0 is the packed DC term, whose partner slot is the Nyquist term.
                let power = if j == 0 {
                    real * real
                } else {
                    real * real + imag * imag
                };
                bsum += power * fb.weights[index2 + j];
            }
            // `sc_max(1e-5f, bsum)`, then the single-precision `std::log10`.
            let floored = if 1e-5 > bsum { 1e-5 } else { bsum };
            *band = 10.0 * (math::log10(floored) + 5.0);
        }

        for (k, coefficient) in mfcc[..numcoefficients].iter_mut().enumerate() {
            let basis = &DCT[k * NUM_BANDS..(k + 1) * NUM_BANDS];
            let mut sum = 0.0f32;
            for (&d, &band) in basis.iter().zip(bands.iter()) {
                sum += d * band;
            }
            // `MFCC_prepareMel` returns the multiplier `0.01f`.
            *coefficient = 0.25 * ((sum * 0.01) + 1.0);
        }
    }
}

impl Unit for Mfcc {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // `MFCC_Ctor` (`MFCC.cpp:3323-3335`): the coefficient count, clamped, sizes the memory.
        let n = (ctx.ins.control(Self::NUMCOEFF) as i32).clamp(1, MAX_COEFFICIENTS as i32);
        self.numcoefficients = n as u32;
        aux.alloc((NUM_BANDS + n as usize) * core::mem::size_of::<f32>());
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `MFCC_Ctor` (`MFCC.cpp:3298`): choose the filterbank, clear the memory and the outputs,
        // and run no calc.
        let mut srate = ctx.audio.sample_rate as f32;
        if f64::from(srate) > 44100.0 * 1.5 {
            srate = (f64::from(srate) * 0.5) as f32;
        }
        self.srate = srate;
        self.use_48000 = u32::from((f64::from(srate) + 0.01) as i32 != 44100);
        ctx.aux.f32_mut().fill(0.0);
        for k in 0..self.numcoefficients.min(self.num_outputs) as usize {
            *ctx.outs.control(k) = 0.0;
        }
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `MFCC_next` (`MFCC.cpp:3354`): a chain above `-0.01` is a frame, the buffer number
        // truncated as `(uint32)`; the held coefficients are output every block.
        let fbufnum = ctx.ins.control(0);
        if fbufnum > -0.01 {
            self.dofft(ctx, fbufnum as u32 as usize);
        }
        let n = self.numcoefficients.min(self.num_outputs) as usize;
        let mfcc = &ctx.aux.f32_mut()[NUM_BANDS..];
        for (k, &coefficient) in mfcc.iter().take(n).enumerate() {
            *ctx.outs.control(k) = coefficient;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Mfcc`]: the unit allocates its band and coefficient memory when the synth
/// starts.
pub struct MfccCtor;

impl UnitDef for MfccCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        check_rate(ctx, Mfcc::NUMCOEFF + 1)?;
        Ok(unit_spec_pool(Mfcc {
            num_outputs: ctx.num_outputs as u32,
            ..Mfcc::zeroed()
        }))
    }
}
