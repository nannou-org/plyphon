//! `Convolution` - plyphon's port of scsynth's signal-by-signal convolver (`Convolution.cpp`).
//!
//! Both the input and the kernel are live audio signals. The unit collects `framesize` samples of
//! each, zero-pads both to `2 * framesize`, forward-transforms them through the engine's shared
//! [`FftTables`](plyphon_dsp::fft::FftTables), multiplies the two packed spectra bin by bin, and
//! inverse-transforms the product - the frequency-domain form of a convolution. Padding to twice the
//! frame length makes that product a *linear* convolution rather than a circular one, so each frame's
//! result is `2 * framesize` samples: the first half is emitted over the next frame and the second
//! half is held back and summed into the frame after it (overlap-add), which joins consecutive frames
//! seamlessly.
//!
//! The kernel is re-transformed on every frame, so a kernel that changes between frames convolves each
//! input frame with its own contemporaneous kernel - a single one-sample impulse in the kernel signal
//! passes the input through for one frame only.
//!
//! Each block is copied into the frame *before* the frame-boundary check, so the block that completes
//! a frame already emits that frame's first samples: the latency is `framesize - block_size` samples,
//! not a whole frame. It therefore depends on the graph's block size (a `Reblock`ed graph is
//! correspondingly closer to `framesize`).
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::fft::is_supported_size;
use plyphon_dsp::rate::Rate;

/// Frame-length `f32` spans the unit allocates, as the reference's constructor does: the two
/// collected frames (one each), the two spectra and the output frame (two each), and the held
/// overlap (one).
const AUX_FRAMES: usize = 9;

/// `Convolution.ar(in, kernel, framesize)`: convolve two live audio signals frame by frame, by
/// multiplying their spectra and overlap-adding the results.
///
/// `framesize` is read when the synth starts (`Convolution_Ctor`), which allocates the whole working
/// set and zeroes the output and overlap spans. The reference fails its FFT setup, and so silences
/// the unit and marks it done (`ClearUnitIfMemFailed`), when `2 * framesize` is outside its FFT size
/// range; here the same happens for any `2 * framesize` the engine has no plan for (a power of two in
/// `[64, 16384]`) and for a `framesize` the unit's block size does not divide, which the reference
/// would overrun.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Convolution {
    /// Samples per frame; `0` when the synth started with an unusable `framesize`.
    framesize: u32,
    /// Samples of the current frame collected so far, in `0..framesize`. It doubles as the read
    /// offset into the output and overlap spans, which keeps collection and emission in step.
    pos: u32,
}

impl Convolution {
    const IN: usize = 0;
    const KERNEL: usize = 1;
    const FRAMESIZE: usize = 2;
}

impl Unit for Convolution {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        let framesize = ctx.ins.control(Self::FRAMESIZE) as i32;
        let Ok(framesize) = usize::try_from(framesize) else {
            return;
        };
        // The unit's own calc length: a block at audio rate, one sample at control rate.
        let calc_len = ctx.own.block_size.max(1);
        if !is_supported_size(framesize.saturating_mul(2)) || !framesize.is_multiple_of(calc_len) {
            return;
        }
        if !aux.alloc(AUX_FRAMES * framesize * core::mem::size_of::<f32>()) {
            return;
        }
        // The reference zeroes the output and the overlap; the other spans are written before
        // they are read.
        aux.f32_mut()[4 * framesize..AUX_FRAMES * framesize].fill(0.0);
        self.framesize = framesize as u32;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let framesize = self.framesize as usize;
        if framesize == 0 {
            ctx.outs.audio(0).fill(0.0);
            ctx.done.mark_done();
            return DoneAction::Nothing;
        }
        let fftsize = 2 * framesize;
        // The unit's own calc length, exactly the sample count the reference drives its
        // collection and emission from - so a control-rate instance stays self-consistent
        // (one sample per block) instead of desynchronizing the two loops.
        let bs = ctx.outs.audio(0).len();
        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`.
        let in_audio = (ins.rate(Self::IN) == Rate::Audio).then(|| ins.audio(Self::IN));
        let in_ctrl = ins.control(Self::IN);
        let kernel_audio = (ins.rate(Self::KERNEL) == Rate::Audio).then(|| ins.audio(Self::KERNEL));
        let kernel_ctrl = ins.control(Self::KERNEL);

        let aux = &mut ctx.aux.f32_mut()[..AUX_FRAMES * framesize];
        let (inbuf, rest) = aux.split_at_mut(framesize);
        let (kernbuf, rest) = rest.split_at_mut(framesize);
        let (spec_in, rest) = rest.split_at_mut(fftsize);
        let (spec_kernel, rest) = rest.split_at_mut(fftsize);
        let (outbuf, overlap) = rest.split_at_mut(fftsize);

        // Collect this block into the current frame. This happens before the boundary check below,
        // so the block that completes a frame goes on to emit that frame's first samples - which is
        // what makes the latency `framesize - bs` rather than a full frame.
        let pos = self.pos as usize;
        for (i, (x, k)) in inbuf[pos..]
            .iter_mut()
            .zip(&mut kernbuf[pos..])
            .take(bs)
            .enumerate()
        {
            *x = in_audio.map_or(in_ctrl, |b| b[i]);
            *k = kernel_audio.map_or(kernel_ctrl, |b| b[i]);
        }
        self.pos += bs as u32;

        // The calc length divides `framesize`, so the collection counter steps exactly onto
        // `framesize` - the only value it can reach with that bit set (the reference's
        // `m_pos & framesize`).
        if self.pos & self.framesize != 0 {
            self.pos = 0;

            // The previous frame's second half overlaps the frame about to be emitted; hold it
            // before the output span is reused as transform scratch.
            overlap.copy_from_slice(&outbuf[framesize..]);

            // Zero-pad each collected frame to `fftsize` in the output span and transform it out.
            outbuf[..framesize].copy_from_slice(inbuf);
            outbuf[framesize..].fill(0.0);
            let mut ready = ctx.fft.forward(fftsize, outbuf, spec_in);
            outbuf[..framesize].copy_from_slice(kernbuf);
            outbuf[framesize..].fill(0.0);
            ready &= ctx.fft.forward(fftsize, outbuf, spec_kernel);
            if ready {
                multiply_packed(spec_in, spec_kernel);
                ready = ctx.fft.inverse(fftsize, spec_in, outbuf);
            }
            if !ready {
                // Unreachable: the size was checked against the engine's plans at synth start.
                // Clearing keeps a hypothetical failure silent rather than emitting the transform
                // scratch left behind in the output span.
                outbuf.fill(0.0);
            }
        }

        // Emit the current frame's samples with the previous frame's held tail summed in.
        let pos = self.pos as usize;
        for ((o, &head), &tail) in ctx
            .outs
            .audio(0)
            .iter_mut()
            .zip(&outbuf[pos..])
            .zip(&overlap[pos..])
        {
            *o = head + tail;
        }
        DoneAction::Nothing
    }
}

/// Multiply the packed spectrum `a` by `b`, bin by bin, in place. The packed layout is
/// `[DC, Nyquist, re1, im1, ...]`, so the two purely real terms multiply as plain scalars and every
/// remaining bin as a complex product (`Convolution_next`'s `p1[0] *= p2[0]; p1[1] *= p2[1];` and
/// its complex-multiply loop).
fn multiply_packed(a: &mut [f32], b: &[f32]) {
    let (a_real, a_bins) = a.split_at_mut(2);
    let (b_real, b_bins) = b.split_at(2);
    a_real[0] *= b_real[0];
    a_real[1] *= b_real[1];
    for (x, y) in a_bins.chunks_exact_mut(2).zip(b_bins.chunks_exact(2)) {
        let (re, im) = (x[0], x[1]);
        x[0] = re * y[0] - im * y[1];
        x[1] = re * y[1] + im * y[0];
    }
}

/// Constructor for [`Convolution`]: the unit reads `framesize` and allocates its working set when
/// the synth starts.
pub struct ConvolutionCtor;

impl UnitDef for ConvolutionCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 3 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(Convolution::zeroed()))
    }
}
