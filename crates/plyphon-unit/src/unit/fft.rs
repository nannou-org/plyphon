//! `FFT` (analysis) and `IFFT` (resynthesis) - plyphon's port of scsynth's short-time Fourier
//! transform chain. Compiled only with the `fft` feature.
//!
//! `FFT` accumulates its audio input into a per-unit ring (its [`aux`](crate::unit::Aux) memory); every
//! `hop` samples it windows the last `fftsize` samples, forward-transforms them via the engine's shared
//! [`FftTables`](plyphon_dsp::fft::FftTables) into a user-allocated *chain buffer* (the packed
//! spectrum), and emits that buffer's number on its control-rate output (`-1` on the blocks between
//! frames). Any number of `PV_*` units may rewrite the buffer in place; `IFFT` then reads it, inverse-
//! transforms, windows, and overlap-adds into its own output ring to resynthesize audio.
//!
//! Unlike scsynth - which `RTAlloc`s per-unit memory at the first call, when the chain buffer (hence
//! the FFT size) is known - plyphon sizes a unit's `aux` at SynthDef-compile time. So the FFT size is
//! taken from the **`winsize`** input, which must be a constant power of two in `[64, 16384]` (scsynth
//! treats `winsize = 0` as "use the buffer size"); the chain buffer must be allocated to match. For the
//! overlap-add to line up, `hop * fftsize` should be a whole number of control blocks.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, Inputs, ProcessCtx, Unit, pv, unit_spec_aux};
use plyphon_dsp::buffer::SpectrumCoord;
use plyphon_dsp::fft::{WindowType, is_supported_size};
use plyphon_dsp::math;
use plyphon_dsp::rate::Rate;

/// `FFT(buffer, in, hop = 0.5, wintype = 0, active = 1, winsize)`: short-time Fourier analysis. The
/// audio `in` is accumulated and, every `hop * fftsize` samples, transformed into `buffer` (the packed
/// spectrum); the control-rate output is `buffer` on a frame and `-1` otherwise.
///
/// `aux` holds two `fftsize`-sample regions: the circular input history, then a windowing scratch.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Fft {
    /// FFT size (a supported power of two), baked from the constant `winsize` input.
    fftsize: u32,
    /// Samples between frames, `round(hop * fftsize)`, baked from the constant `hop` input.
    hop_size: u32,
    /// The window type code, baked from the constant `wintype` input.
    wintype: i32,
    /// Circular write head into the input ring.
    pos: u32,
    /// Samples accumulated since the last frame; a frame fires when it reaches `hop_size`.
    counter: u32,
    /// `0` until the first block zeros the ring (the `aux` is not zeroed at instantiation).
    warmed: u32,
}

impl Fft {
    const BUFFER: usize = 0;
    const IN: usize = 1;
    const HOP: usize = 2;
    const WINTYPE: usize = 3;
    const ACTIVE: usize = 4;
    const WINSIZE: usize = 5;
}

impl Unit for Fft {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let n = self.fftsize as usize;
        let bs = ctx.audio.block_size;
        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`.
        let bufnum = ins.control(Self::BUFFER).max(0.0) as usize;
        let active = ins.control(Self::ACTIVE) > 0.0;

        let win = ctx
            .fft
            .window(n, WindowType::from_code(self.wintype as f32));
        let aux = ctx.aux.f32_mut();
        if self.warmed == 0 {
            aux.fill(0.0);
            self.warmed = 1;
        }
        let (ring, windowed) = aux.split_at_mut(n);

        let mut out_val = -1.0f32;
        if active {
            for i in 0..bs {
                ring[self.pos as usize] = sample_in(&ins, Self::IN, i);
                self.pos = (self.pos + 1) % n as u32;
                self.counter += 1;
                if self.counter < self.hop_size {
                    continue;
                }
                self.counter = 0;
                // Window the `n` most-recent samples in chronological order (oldest is at `pos`).
                for (j, w) in windowed.iter_mut().enumerate() {
                    let s = ring[(self.pos as usize + j) % n];
                    *w = s * win.get(j).copied().unwrap_or(1.0);
                }
                if let Some(buffer) =
                    unit::buffer_at_mut(ctx.buffers, bufnum).filter(|b| b.num_frames() == n)
                    && ctx.fft.forward(n, windowed, buffer.data_mut())
                {
                    // The forward transform writes the Cartesian packed spectrum (scsynth sets
                    // `coord_Complex`); a downstream polar `PV_*` will flip it as needed.
                    buffer.set_coord(SpectrumCoord::Complex);
                    out_val = bufnum as f32;
                }
            }
        }
        *ctx.outs.control(0) = out_val;
        DoneAction::Nothing
    }
}

/// Constructor for [`Fft`]: bakes the FFT size (from the constant `winsize`), hop, and window type.
pub struct FftCtor;

impl UnitDef for FftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Fft::WINSIZE {
            return Err(BuildError::WrongInputCount);
        }
        let fftsize = const_fftsize(ctx, Fft::WINSIZE)?;
        let hop_frac = ctx.const_input(Fft::HOP).unwrap_or(0.5);
        let hop_size = (math::floor((hop_frac * fftsize as f32) + 0.5) as u32).max(1);
        let wintype = ctx.const_input(Fft::WINTYPE).unwrap_or(0.0) as i32;
        // aux = input ring + windowing scratch, both `fftsize` f32.
        Ok(unit_spec_aux(
            Fft {
                fftsize: fftsize as u32,
                hop_size,
                wintype,
                pos: 0,
                counter: 0,
                warmed: 0,
            },
            2 * fftsize * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

/// `IFFT(buffer, wintype = 0, winsize)`: short-time Fourier resynthesis. Input `buffer` is the frame-
/// ready signal from `FFT`/`PV_*` (the packed-spectrum buffer number, or `< 0` when no frame is ready);
/// on each ready frame it inverse-transforms, windows, and overlap-adds into its output ring.
///
/// `aux` holds two `fftsize`-sample regions: the overlap-add ring, then the inverse-transform scratch.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Ifft {
    /// FFT size, baked from the constant `winsize` input.
    fftsize: u32,
    /// The window type code, baked from the constant `wintype` input.
    wintype: i32,
    /// Read/write head into the overlap-add ring.
    pos: u32,
    /// `0` until the first block zeros the ring.
    warmed: u32,
}

impl Ifft {
    const BUFFER: usize = 0;
    const WINTYPE: usize = 1;
    const WINSIZE: usize = 2;
}

impl Unit for Ifft {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let n = self.fftsize as usize;
        let bs = ctx.audio.block_size;
        let fbufnum = ctx.ins.control(Self::BUFFER);

        let win = ctx
            .fft
            .window(n, WindowType::from_code(self.wintype as f32));
        let aux = ctx.aux.f32_mut();
        if self.warmed == 0 {
            aux.fill(0.0);
            self.warmed = 1;
        }
        let (ola, temp) = aux.split_at_mut(n);

        // A ready frame (fbufnum >= 0): inverse-transform it and overlap-add into the ring at `pos`.
        if fbufnum >= 0.0 {
            let bufnum = fbufnum as usize;
            if let Some(buffer) =
                unit::buffer_at_mut(ctx.buffers, bufnum).filter(|b| b.num_frames() == n)
            {
                // A polar `PV_*` unit may have left the frame in polar form; restore Cartesian
                // before the inverse transform (scsynth's `ToComplexApx` in `IFFT_next`).
                pv::to_complex(buffer);
                if ctx.fft.inverse(n, &buffer.data()[..n], temp) {
                    for (j, w) in win.iter().enumerate().take(n) {
                        ola[(self.pos as usize + j) % n] += temp[j] * w;
                    }
                }
            }
        }

        // Emit `bs` samples from the ring, clearing each consumed slot for the next overlap-add.
        let out = ctx.outs.audio(0);
        for o in out.iter_mut().take(bs) {
            *o = ola[self.pos as usize];
            ola[self.pos as usize] = 0.0;
            self.pos = (self.pos + 1) % n as u32;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Ifft`]: bakes the FFT size (from the constant `winsize`) and window type.
pub struct IfftCtor;

impl UnitDef for IfftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Ifft::WINSIZE {
            return Err(BuildError::WrongInputCount);
        }
        let fftsize = const_fftsize(ctx, Ifft::WINSIZE)?;
        let wintype = ctx.const_input(Ifft::WINTYPE).unwrap_or(0.0) as i32;
        Ok(unit_spec_aux(
            Ifft {
                fftsize: fftsize as u32,
                wintype,
                pos: 0,
                warmed: 0,
            },
            2 * fftsize * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

/// Read input `i` at within-block index `k`: the audio sample if it is audio-rate, else the broadcast
/// control value.
fn sample_in(ins: &Inputs<'_>, i: usize, k: usize) -> f32 {
    if ins.rate(i) == Rate::Audio {
        ins.audio(i)[k]
    } else {
        ins.control(i)
    }
}

/// The constant FFT size at input `winsize`, validated as a supported power of two.
fn const_fftsize(ctx: &BuildContext<'_>, winsize: usize) -> Result<usize, BuildError> {
    let size = ctx
        .const_input(winsize)
        .ok_or(BuildError::AuxRequiresConstant { input: winsize })? as usize;
    if !is_supported_size(size) {
        return Err(BuildError::UnsupportedFftSize { size });
    }
    Ok(size)
}
