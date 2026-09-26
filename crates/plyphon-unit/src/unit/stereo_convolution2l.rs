//! `StereoConvolution2L` - plyphon's port of scsynth's stereo crossfading convolver
//! (`Convolution.cpp`).
//!
//! It is `Convolution2L` run twice over one input, once per kernel buffer, with two outputs. The port
//! follows the reference transform for transform, including how its constructor wires its `scfft`
//! objects, which differs from the two-channel mirror of `Convolution2L` the unit describes:
//!
//! | `scfft`       | transforms, in place | created as |
//! |---------------|----------------------|------------|
//! | `m_scfft2[0]` | left kernel A        | forward    |
//! | `m_scfft2[1]` | left output          | backward   |
//! | `m_scfft3[0]` | right kernel A       | forward    |
//! | `m_scfft3[1]` | left product         | backward   |
//! | `m_scfftR[0]` | left kernel B        | forward    |
//! | `m_scfftR[1]` | right output         | backward   |
//! | `m_scfftR2[0]`| right kernel B       | forward    |
//! | `m_scfftR2[1]`| right product        | backward   |
//!
//! So, as in the reference:
//!
//! - loading set A (at the start, or on a trigger while set B is current) transforms its left kernel
//!   but copies its right kernel in as samples, and forward-transforms the left output frame, with
//!   the backward gain, in its place;
//! - loading set B (a trigger while set A is current) copies both its kernels in as samples, and
//!   instead forward-transforms set A's right kernel and the left product span in place;
//! - each frame inverse-transforms set B's left kernel in place, unnormalized, rather than the left
//!   output frame, which is emitted as the product spectrum it holds; while crossfading it does the
//!   same to set B's right kernel;
//! - the products never set their DC and Nyquist terms, which keep whatever the product spans last
//!   held.
//!
//! Only the right output's inverse transform is the one the unit describes, and it multiplies by
//! whatever the current set's right-kernel span holds.
//!
//! The gains follow scsynth's FFTW build (forward `1`, backward `1 / N`), as every plyphon transform
//! does; scsynth's macOS build scales its `scfft`s differently, which changes this unit's crossed
//! transforms there.
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::convolution2::{
    Direction, collect, crossfade, dofft, doifft, load_kernel, usable_framesize,
};
use crate::unit::convolution3::conv_get_buffer;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::fft::FftTables;
use plyphon_dsp::rate::Rate;

/// `StereoConvolution2L.ar(in, kernelL, kernelR, trigger, framesize, crossfade)`: two-kernel,
/// two-output [`Convolution2L`](crate::unit::convolution2::Convolution2L), with the reference's
/// transform wiring (see the module docs).
///
/// `framesize` is read when the synth starts (`StereoConvolution2L_Ctor`); each kernel buffer must
/// hold `framesize` samples. `crossfade` is read then and at every trigger.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct StereoConvolution2L {
    /// Samples per frame; `0` once the unit is silenced.
    framesize: u32,
    /// Samples of the current frame collected so far; also the read offset into the output spans.
    pos: u32,
    /// Frames of the current crossfade done (`m_cfpos`).
    cfpos: i32,
    /// Frames the crossfade lasts (`m_cflength`).
    cflength: i32,
    /// Which kernel set is current: `0` set A, `1` set B (`m_curbuf`).
    curbuf: u32,
    /// The previous block's trigger value, for rising-edge detection.
    prevtrig: f32,
}

/// The spans of a [`StereoConvolution2L`]'s memory, in `framesize` units: the input frame (1), the
/// input spectrum (2), kernel sets A and B (2 per channel each), the two product spans (2 each), the
/// two output frames (2 each) and held overlaps (1 each), and the transform buffer of the reference's
/// `scfft`s (2).
const STEREO_FRAMES: usize = 23;

/// [`StereoConvolution2L`]'s memory, split into its spans (named as the reference's buffers).
struct Spans<'a> {
    inbuf1: &'a mut [f32],
    fftbuf1: &'a mut [f32],
    fftbuf2: [&'a mut [f32]; 2],
    fftbuf3: [&'a mut [f32]; 2],
    tempbuf: [&'a mut [f32]; 2],
    outbuf: [&'a mut [f32]; 2],
    overlap: [&'a mut [f32]; 2],
    trbuf: &'a mut [f32],
}

impl<'a> Spans<'a> {
    fn new(aux: &'a mut Aux<'_>, fs: usize) -> Self {
        let aux = &mut aux.f32_mut()[..STEREO_FRAMES * fs];
        let (inbuf1, rest) = aux.split_at_mut(fs);
        let (fftbuf1, rest) = rest.split_at_mut(2 * fs);
        let (fftbuf2_l, rest) = rest.split_at_mut(2 * fs);
        let (fftbuf2_r, rest) = rest.split_at_mut(2 * fs);
        let (fftbuf3_l, rest) = rest.split_at_mut(2 * fs);
        let (fftbuf3_r, rest) = rest.split_at_mut(2 * fs);
        let (tempbuf_l, rest) = rest.split_at_mut(2 * fs);
        let (tempbuf_r, rest) = rest.split_at_mut(2 * fs);
        let (outbuf_l, rest) = rest.split_at_mut(2 * fs);
        let (overlap_l, rest) = rest.split_at_mut(fs);
        let (outbuf_r, rest) = rest.split_at_mut(2 * fs);
        let (overlap_r, trbuf) = rest.split_at_mut(fs);
        Spans {
            inbuf1,
            fftbuf1,
            fftbuf2: [fftbuf2_l, fftbuf2_r],
            fftbuf3: [fftbuf3_l, fftbuf3_r],
            tempbuf: [tempbuf_l, tempbuf_r],
            outbuf: [outbuf_l, outbuf_r],
            overlap: [overlap_l, overlap_r],
            trbuf,
        }
    }
}

/// `scfft_dofft` in place over `data` through an `scfft` created as `dir`.
fn dofft_in_place(fft: &FftTables, dir: Direction, data: &mut [f32], trbuf: &mut [f32]) {
    trbuf.copy_from_slice(data);
    dofft(fft, dir, trbuf, data);
}

/// `scfft_doifft` in place over `data` through an `scfft` created as `dir`.
fn doifft_in_place(fft: &FftTables, dir: Direction, data: &mut [f32], trbuf: &mut [f32]) {
    trbuf.copy_from_slice(data);
    doifft(fft, dir, trbuf, data);
}

/// `tempbuf[c] = input * kernels[c]` for both channels, over every bin but DC and Nyquist (the
/// reference's stereo complex-multiply loops start at bin 1 and never write the first two floats).
fn multiply_bins(input: &[f32], kernels: [&[f32]; 2], tempbuf: &mut [&mut [f32]; 2]) {
    for (p2, p3) in kernels.into_iter().zip(tempbuf.iter_mut()) {
        for ((o, x), y) in p3[2..]
            .chunks_exact_mut(2)
            .zip(input[2..].chunks_exact(2))
            .zip(p2[2..].chunks_exact(2))
        {
            o[0] = x[0] * y[0] - x[1] * y[1];
            o[1] = x[0] * y[1] + x[1] * y[0];
        }
    }
}

impl StereoConvolution2L {
    const IN: usize = 0;
    const KERNEL_L: usize = 1;
    const KERNEL_R: usize = 2;
    const TRIGGER: usize = 3;
    const FRAMESIZE: usize = 4;
    const CROSSFADE: usize = 5;

    /// Silence the unit for the rest of its life (`SETCALC(ClearUnitOutputs)`, `mDone = true`),
    /// clearing `len` samples of each output.
    fn clear(&mut self, ctx: &mut ProcessCtx<'_>, len: usize) {
        self.framesize = 0;
        ctx.outs.audio(0)[..len].fill(0.0);
        ctx.outs.audio(1)[..len].fill(0.0);
        ctx.done.mark_done();
    }
}

impl Unit for StereoConvolution2L {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        let framesize = ctx.ins.control(Self::FRAMESIZE) as i32;
        // The reference's calc always runs a whole block (`FULLBUFLENGTH`).
        let Some(fs) = usable_framesize(framesize, ctx.audio.block_size) else {
            return;
        };
        if aux.alloc(STEREO_FRAMES * fs * core::mem::size_of::<f32>()) {
            self.framesize = fs as u32;
        }
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.cflength = ctx.ins.control(Self::CROSSFADE) as i32;
        self.curbuf = 0;
        self.cfpos = self.cflength;
        if self.framesize == 0 {
            self.clear(ctx, 1);
            return DoneAction::Nothing;
        }
        let fs = self.framesize as usize;
        let s = Spans::new(&mut ctx.aux, fs);
        for c in 0..2 {
            s.outbuf[c].fill(0.0);
            s.overlap[c].fill(0.0);
            // The reference leaves these uninitialized; kernel set B and the products' DC and
            // Nyquist terms are read before they are first written (see the module docs).
            s.fftbuf3[c].fill(0.0);
            s.tempbuf[c].fill(0.0);
        }

        let Some(buf) = conv_get_buffer(
            ctx.buffers,
            &ctx.local_bufs,
            ctx.ins.control(Self::KERNEL_L),
        ) else {
            self.clear(ctx, 1);
            return DoneAction::Nothing;
        };
        let s = Spans::new(&mut ctx.aux, fs);
        let [left, _] = s.fftbuf2;
        load_kernel(ctx.fft, buf, fs, s.trbuf, left);

        let Some(buf) = conv_get_buffer(
            ctx.buffers,
            &ctx.local_bufs,
            ctx.ins.control(Self::KERNEL_R),
        ) else {
            self.clear(ctx, 1);
            return DoneAction::Nothing;
        };
        let s = Spans::new(&mut ctx.aux, fs);
        load_right_a(ctx.fft, buf, fs, s);

        self.pos = 0;
        self.prevtrig = 0.0;
        let in0 = ctx.ins.control(Self::IN);
        *ctx.outs.control(0) = in0;
        *ctx.outs.control(1) = in0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let n = ctx.outs.audio(0).len();
        if self.framesize == 0 {
            self.clear(ctx, n);
            return DoneAction::Nothing;
        }
        let fs = self.framesize as usize;
        let curtrig = ctx.ins.control(Self::TRIGGER);

        let pos = self.pos as usize;
        collect(
            ctx.ins,
            Self::IN,
            &mut Spans::new(&mut ctx.aux, fs).inbuf1[pos..pos + n],
        );
        self.pos += n as u32;

        if self.prevtrig <= 0.0 && curtrig > 0.0 {
            self.cflength = ctx.ins.control(Self::CROSSFADE) as i32;
            let buf_l = conv_get_buffer(
                ctx.buffers,
                &ctx.local_bufs,
                ctx.ins.control(Self::KERNEL_L),
            );
            let buf_r = conv_get_buffer(
                ctx.buffers,
                &ctx.local_bufs,
                ctx.ins.control(Self::KERNEL_R),
            );
            let (Some(buf_l), Some(buf_r)) = (buf_l, buf_r) else {
                self.clear(ctx, n);
                return DoneAction::Nothing;
            };
            self.cfpos = 0;
            let s = Spans::new(&mut ctx.aux, fs);
            if self.curbuf == 1 {
                let [left, _] = s.fftbuf2;
                load_kernel(ctx.fft, buf_l, fs, s.trbuf, left);
                let s = Spans::new(&mut ctx.aux, fs);
                load_right_a(ctx.fft, buf_r, fs, s);
            } else {
                // Set B's kernels are copied in untransformed; `m_scfft3[0]` transforms the right
                // kernel of set A in place and `m_scfft3[1]` the left product.
                let Spans {
                    fftbuf2: [_, right_a],
                    fftbuf3: [left_b, right_b],
                    tempbuf: [temp_l, _],
                    trbuf,
                    ..
                } = s;
                copy_padded(left_b, buf_l, fs);
                dofft_in_place(ctx.fft, Direction::Forward, right_a, trbuf);
                copy_padded(right_b, buf_r, fs);
                dofft_in_place(ctx.fft, Direction::Backward, temp_l, trbuf);
            }
        }

        let s = Spans::new(&mut ctx.aux, fs);
        let Spans {
            inbuf1,
            fftbuf1,
            fftbuf2,
            fftbuf3,
            mut tempbuf,
            outbuf: [out_l, out_r],
            overlap: [overlap_l, overlap_r],
            trbuf,
        } = s;
        let [a_l, a_r] = fftbuf2;
        let [b_l, b_r] = fftbuf3;
        if self.pos & self.framesize != 0 {
            self.pos = 0;
            let (head, tail) = trbuf.split_at_mut(fs);
            head.copy_from_slice(inbuf1);
            tail.fill(0.0);
            dofft(ctx.fft, Direction::Forward, trbuf, fftbuf1);

            let current = if self.curbuf == 0 {
                [&*a_l, &*a_r]
            } else {
                [&*b_l, &*b_r]
            };
            multiply_bins(fftbuf1, current, &mut tempbuf);

            // Left: the product is copied to the output frame, but `m_scfftR[0]` inverse-transforms
            // the left kernel of set B instead.
            overlap_l.copy_from_slice(&out_l[fs..]);
            out_l.copy_from_slice(&tempbuf[0][..]);
            doifft_in_place(ctx.fft, Direction::Forward, b_l, trbuf);
            // Right: `m_scfftR[1]` inverse-transforms the output frame.
            overlap_r.copy_from_slice(&out_r[fs..]);
            out_r.copy_from_slice(&tempbuf[1][..]);
            doifft_in_place(ctx.fft, Direction::Backward, out_r, trbuf);

            if self.cfpos < self.cflength {
                let next = if self.curbuf == 0 {
                    [&*b_l, &*b_r]
                } else {
                    [&*a_l, &*a_r]
                };
                multiply_bins(fftbuf1, next, &mut tempbuf);
                // `m_scfftR2[0]` inverse-transforms the right kernel of set B; `m_scfftR2[1]` the
                // right product.
                doifft_in_place(ctx.fft, Direction::Forward, b_r, trbuf);
                let [temp_l, temp_r] = tempbuf;
                doifft_in_place(ctx.fft, Direction::Backward, temp_r, trbuf);
                crossfade(
                    self.cfpos,
                    self.cflength,
                    fs,
                    &mut [(&mut *out_l, &*temp_l), (&mut *out_r, &*temp_r)],
                );
                self.cfpos += 1;
                if self.cfpos == self.cflength {
                    self.curbuf = if self.curbuf == 0 { 1 } else { 0 };
                }
            }
        }

        let pos = self.pos as usize;
        for (channel, (outbuf, overlap)) in [(out_l, overlap_l), (out_r, overlap_r)]
            .into_iter()
            .enumerate()
        {
            for ((o, &head), &tail) in ctx
                .outs
                .audio(channel)
                .iter_mut()
                .zip(&outbuf[pos..])
                .zip(&overlap[pos..])
            {
                *o = head + tail;
            }
        }
        self.prevtrig = curtrig;
        DoneAction::Nothing
    }
}

/// Copy a kernel of `fs` samples from `buf` into the first half of `dst` and zero the second (the
/// reference's `memcpy`/`memset` before each kernel transform).
fn copy_padded(dst: &mut [f32], buf: plyphon_dsp::buffer::BufView<'_>, fs: usize) {
    let (head, tail) = dst.split_at_mut(fs);
    crate::unit::convolution3::copy_kernel(head, buf);
    tail.fill(0.0);
}

/// Load the right kernel into set A as the reference does: the kernel is copied in untransformed,
/// and `m_scfft2[1]` forward-transforms the left output frame in place with its backward gain.
fn load_right_a(fft: &FftTables, buf: plyphon_dsp::buffer::BufView<'_>, fs: usize, s: Spans<'_>) {
    let Spans {
        fftbuf2: [_, right_a],
        outbuf: [out_l, _],
        trbuf,
        ..
    } = s;
    copy_padded(right_a, buf, fs);
    dofft_in_place(fft, Direction::Backward, out_l, trbuf);
}

/// Constructor for [`StereoConvolution2L`].
///
/// Audio rate only: the reference's calc always processes a whole audio block (`FULLBUFLENGTH`).
/// `2 * framesize` must be a power of two in `[8, 262144]` and `framesize` a multiple of the block;
/// the reference fails its FFT setup outside that size range and overruns its buffers otherwise, and
/// any such `framesize` silences the unit and marks it done here.
pub struct StereoConvolution2LCtor;

impl UnitDef for StereoConvolution2LCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 6 {
            return Err(BuildError::WrongInputCount);
        }
        if ctx.rate != Rate::Audio {
            return Err(BuildError::UnsupportedRate(ctx.rate));
        }
        Ok(unit_spec_pool(StereoConvolution2L::zeroed()))
    }
}
