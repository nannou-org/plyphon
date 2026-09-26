//! `Convolution2` and `Convolution2L` - plyphon's ports of scsynth's fixed-kernel convolvers
//! (`Convolution.cpp`), plus the `scfft` transform helpers the buffer-kernel convolvers share.
//!
//! Both read a kernel from a buffer when the synth starts, zero-pad it to `2 * framesize` and keep its
//! spectrum. The input is collected `framesize` samples at a time; each full frame is zero-padded,
//! transformed, multiplied by the kernel spectrum and transformed back, and the `2 * framesize`
//! result is emitted with the previous frame's second half summed in (overlap-add), as `Convolution`
//! does. A rising `trigger` re-reads the kernel buffer (whose number may have changed) immediately,
//! so the frame completed in that block already uses the new kernel.
//!
//! `Convolution2` swaps kernels outright. `Convolution2L` keeps two kernel spectra and, on a trigger,
//! loads the new kernel into the idle one and crossfades linearly from the old kernel's result to the
//! new one's over `crossfade` frames.
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::convolution3::{conv_get_buffer, copy_kernel};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, Inputs, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::buffer::BufView;
use plyphon_dsp::fft::{FftTables, is_supported_size};
use plyphon_dsp::rate::Rate;

/// The direction an `scfft` was created with (`scfft_create`'s `direction`), which fixes its gain
/// (`scalefac`, with FFTW's conventions): `1` for [`Forward`](Direction::Forward), `1 / N` for
/// [`Backward`](Direction::Backward). `scfft_dofft` and `scfft_doifft` apply that gain whichever way
/// they transform, so a transform run through an `scfft` created the other way is scaled
/// accordingly (`StereoConvolution2L` does this).
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum Direction {
    /// Unity gain.
    Forward,
    /// Gain `1 / N`.
    Backward,
}

/// scsynth's `scfft_dofft` for a rectangular-window `scfft` of size `trbuf.len()`: `trbuf` holds the
/// frame already copied in (the reference's first `memcpy` into its transform buffer); it is scaled by
/// the `scfft`'s gain and forward-transformed into `out` in the packed layout
/// `[DC, Nyquist, re1, im1, ...]`.
pub(crate) fn dofft(fft: &FftTables, dir: Direction, trbuf: &mut [f32], out: &mut [f32]) {
    let n = trbuf.len();
    if dir == Direction::Backward {
        let scalefac = 1.0 / n as f32;
        for x in trbuf.iter_mut() {
            *x *= scalefac;
        }
    }
    // The size was checked against the engine's plans when the synth started.
    let _ = fft.forward(n, trbuf, out);
}

/// scsynth's `scfft_doifft` for a rectangular-window `scfft` of size `input.len()`: inverse-transform
/// the packed spectrum `input` into `out` and apply the `scfft`'s gain. The engine's inverse is
/// normalized by `1 / N`, which is the backward gain; the forward gain of `1` undoes it (a
/// power-of-two rescale, exact unless the normalized value underflowed).
pub(crate) fn doifft(fft: &FftTables, dir: Direction, input: &[f32], out: &mut [f32]) {
    let n = input.len();
    // The size was checked against the engine's plans when the synth started.
    let _ = fft.inverse(n, input, out);
    if dir == Direction::Forward {
        let unscale = n as f32;
        for x in out.iter_mut() {
            *x *= unscale;
        }
    }
}

/// Load a kernel into `spectrum` as the reference does: copy `len` samples of the buffer's raw
/// storage, zero-pad to the transform size, and forward-transform in place through a
/// forward-created `scfft` (`trbuf` is its transform buffer).
pub(crate) fn load_kernel(
    fft: &FftTables,
    buf: BufView<'_>,
    len: usize,
    trbuf: &mut [f32],
    spectrum: &mut [f32],
) {
    copy_kernel(&mut trbuf[..len], buf);
    trbuf[len..].fill(0.0);
    dofft(fft, Direction::Forward, trbuf, spectrum);
}

/// Zero-pad the collected frame `inbuf` into `trbuf` and forward-transform it into `spectrum` (the
/// `memcpy`/`memset`/`scfft_dofft` each frame begins with).
fn transform_frame(fft: &FftTables, inbuf: &[f32], trbuf: &mut [f32], spectrum: &mut [f32]) {
    let (head, tail) = trbuf.split_at_mut(inbuf.len());
    head.copy_from_slice(inbuf);
    tail.fill(0.0);
    dofft(fft, Direction::Forward, trbuf, spectrum);
}

/// `out = a * b`, bin by bin, over packed spectra: the DC and Nyquist terms are real and multiply as
/// scalars, every other bin as a complex product (the reference's `p3[0] = p1[0] * p2[0]; ...` and
/// complex-multiply loop).
fn multiply_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    out[0] = a[0] * b[0];
    out[1] = a[1] * b[1];
    for ((o, x), y) in out[2..]
        .chunks_exact_mut(2)
        .zip(a[2..].chunks_exact(2))
        .zip(b[2..].chunks_exact(2))
    {
        o[0] = x[0] * y[0] - x[1] * y[1];
        o[1] = x[0] * y[1] + x[1] * y[0];
    }
}

/// `a *= b` over packed spectra, as [`multiply_into`].
fn multiply_in_place(a: &mut [f32], b: &[f32]) {
    a[0] *= b[0];
    a[1] *= b[1];
    for (x, y) in a[2..].chunks_exact_mut(2).zip(b[2..].chunks_exact(2)) {
        let (re, im) = (x[0], x[1]);
        x[0] = re * y[0] - im * y[1];
        x[1] = re * y[1] + im * y[0];
    }
}

/// Collect a block of input into `inbuf`, the frame's next `inbuf.len()` samples. The reference
/// copies `IN(0)` directly, which is only a whole block for an audio-rate input; any other input is
/// held for the block, as `Convolution` does.
pub(crate) fn collect(ins: Inputs<'_>, input: usize, inbuf: &mut [f32]) {
    if ins.rate(input) == Rate::Audio {
        inbuf.copy_from_slice(&ins.audio(input)[..inbuf.len()]);
    } else {
        inbuf.fill(ins.control(input));
    }
}

/// Whether `framesize` is one the FFT convolvers can run: its transform size `2 * framesize` has an
/// engine plan (a power of two in `[8, 262144]`; the reference's `scfft_create` fails outside that
/// range and overruns its buffers for a size that is not a power of two), and the calc length
/// `calc_len` steps the collection counter exactly onto it (the reference overruns its input frame
/// otherwise).
pub(crate) fn usable_framesize(framesize: i32, calc_len: usize) -> Option<usize> {
    let framesize = usize::try_from(framesize).ok()?;
    let calc_len = calc_len.max(1);
    (is_supported_size(framesize.saturating_mul(2)) && framesize.is_multiple_of(calc_len))
        .then_some(framesize)
}

/// `Convolution2.ar(in, kernel, trigger, framesize)`: convolve `in` with a kernel read from buffer
/// `kernel`, re-read (from the buffer `kernel` then names) on every rising `trigger`.
///
/// `framesize` (`<= 0` means the kernel buffer's frame count) is read when the synth starts
/// (`Convolution2_Ctor`), which allocates the working set and transforms the kernel, truncated to
/// `framesize` samples. The unit is silenced and marked done if the kernel buffer cannot be resolved
/// (then or on a later trigger), if `framesize` is smaller than the block, or if it is otherwise
/// unusable (see [`Convolution2Ctor`]).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Convolution2 {
    /// Samples per frame; `0` once the unit is silenced.
    framesize: u32,
    /// Samples of the current frame collected so far; also the read offset into the output spans.
    pos: u32,
    /// The previous block's trigger value, for rising-edge detection.
    prevtrig: f32,
    _pad: u32,
}

/// The spans of a [`Convolution2`]'s memory, in `framesize` units: the input frame (1), the input
/// spectrum (2), the kernel spectrum (2), the output frame (2), the held overlap (1), and the
/// transform buffer of the reference's `scfft`s (2).
const CONV2_FRAMES: usize = 10;

impl Convolution2 {
    const IN: usize = 0;
    const KERNEL: usize = 1;
    const TRIGGER: usize = 2;
    const FRAMESIZE: usize = 3;

    /// Silence the unit for the rest of its life (`SETCALC(ClearUnitOutputs)`, `mDone = true`).
    fn clear(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.framesize = 0;
        ctx.outs.audio(0).fill(0.0);
        ctx.done.mark_done();
    }

    /// Read the kernel from `buf` (at most `framesize` of its frames) and transform it.
    fn load(&self, fft: &FftTables, buf: BufView<'_>, aux: &mut Aux<'_>) {
        let fs = self.framesize as usize;
        let len = buf.num_frames().min(fs);
        let s = Conv2Spans::new(aux, fs);
        load_kernel(fft, buf, len, s.trbuf, s.kernel);
    }
}

/// [`Convolution2`]'s memory, split into its spans.
struct Conv2Spans<'a> {
    inbuf: &'a mut [f32],
    spectrum: &'a mut [f32],
    kernel: &'a mut [f32],
    outbuf: &'a mut [f32],
    overlap: &'a mut [f32],
    trbuf: &'a mut [f32],
}

impl<'a> Conv2Spans<'a> {
    fn new(aux: &'a mut Aux<'_>, fs: usize) -> Self {
        let aux = &mut aux.f32_mut()[..CONV2_FRAMES * fs];
        let (inbuf, rest) = aux.split_at_mut(fs);
        let (spectrum, rest) = rest.split_at_mut(2 * fs);
        let (kernel, rest) = rest.split_at_mut(2 * fs);
        let (outbuf, rest) = rest.split_at_mut(2 * fs);
        let (overlap, trbuf) = rest.split_at_mut(fs);
        Conv2Spans {
            inbuf,
            spectrum,
            kernel,
            outbuf,
            overlap,
            trbuf,
        }
    }
}

impl Unit for Convolution2 {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        let bufnum = ctx.ins.control(Self::KERNEL);
        let Some(buf) = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum) else {
            return;
        };
        let mut framesize = ctx.ins.control(Self::FRAMESIZE) as i32;
        if framesize <= 0 {
            framesize = buf.num_frames() as i32;
        }
        // The reference's calc always runs a whole block (`FULLBUFLENGTH`).
        let Some(fs) = usable_framesize(framesize, ctx.audio.block_size) else {
            return;
        };
        if aux.alloc(CONV2_FRAMES * fs * core::mem::size_of::<f32>()) {
            self.framesize = fs as u32;
        }
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bufnum = ctx.ins.control(Self::KERNEL);
        let buf = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum);
        let Some(buf) = buf.filter(|_| self.framesize != 0) else {
            self.clear(ctx);
            return DoneAction::Nothing;
        };
        self.load(ctx.fft, buf, &mut ctx.aux);
        let s = Conv2Spans::new(&mut ctx.aux, self.framesize as usize);
        s.outbuf.fill(0.0);
        s.overlap.fill(0.0);
        self.pos = 0;
        self.prevtrig = 0.0;
        *ctx.outs.control(0) = ctx.ins.control(Self::IN);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.framesize == 0 {
            self.clear(ctx);
            return DoneAction::Nothing;
        }
        let fs = self.framesize as usize;
        let n = ctx.outs.audio(0).len();
        let curtrig = ctx.ins.control(Self::TRIGGER);

        let pos = self.pos as usize;
        collect(
            ctx.ins,
            Self::IN,
            &mut Conv2Spans::new(&mut ctx.aux, fs).inbuf[pos..pos + n],
        );
        self.pos += n as u32;

        if self.prevtrig <= 0.0 && curtrig > 0.0 {
            let bufnum = ctx.ins.control(Self::KERNEL);
            let Some(buf) = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum) else {
                self.clear(ctx);
                return DoneAction::Nothing;
            };
            self.load(ctx.fft, buf, &mut ctx.aux);
        }

        let s = Conv2Spans::new(&mut ctx.aux, fs);
        if self.pos as usize >= fs {
            self.pos = 0;
            transform_frame(ctx.fft, s.inbuf, s.trbuf, s.spectrum);
            multiply_in_place(s.spectrum, s.kernel);
            s.overlap.copy_from_slice(&s.outbuf[fs..]);
            doifft(ctx.fft, Direction::Backward, s.spectrum, s.outbuf);
        }

        let pos = self.pos as usize;
        for ((o, &head), &tail) in ctx
            .outs
            .audio(0)
            .iter_mut()
            .zip(&s.outbuf[pos..])
            .zip(&s.overlap[pos..])
        {
            *o = head + tail;
        }
        self.prevtrig = curtrig;
        DoneAction::Nothing
    }
}

/// Constructor for [`Convolution2`].
///
/// Audio rate only: the reference's calc always processes a whole audio block (`FULLBUFLENGTH`),
/// which a control-rate unit's single output sample cannot hold. `2 * framesize` must be a power of
/// two in `[8, 262144]` and `framesize` a multiple of the block (the reference silences itself for a
/// `framesize` smaller than the block, fails its FFT setup outside that size range, and overruns its
/// buffers otherwise); any other `framesize` silences the unit and marks it done.
pub struct Convolution2Ctor;

impl UnitDef for Convolution2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 4 {
            return Err(BuildError::WrongInputCount);
        }
        if ctx.rate != Rate::Audio {
            return Err(BuildError::UnsupportedRate(ctx.rate));
        }
        Ok(unit_spec_pool(Convolution2::zeroed()))
    }
}

/// `Convolution2L.ar(in, kernel, trigger, framesize, crossfade)`: as [`Convolution2`], but a rising
/// `trigger` loads the new kernel beside the current one and crossfades linearly to it over
/// `crossfade` frames.
///
/// `framesize` is read when the synth starts (`Convolution2L_Ctor`) and taken as given (no
/// buffer-size fallback); the kernel buffer must hold `framesize` samples. `crossfade` is read then
/// and at every trigger. A `crossfade` below `1` never crossfades - nor, as in the reference, ever
/// switches to a newly loaded kernel.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Convolution2L {
    /// Samples per frame; `0` once the unit is silenced.
    framesize: u32,
    /// Samples of the current frame collected so far; also the read offset into the output spans.
    pos: u32,
    /// Frames of the current crossfade done (`m_cfpos`).
    cfpos: i32,
    /// Frames the crossfade lasts (`m_cflength`).
    cflength: i32,
    /// Which kernel spectrum is current: `0` the first, `1` the second (`m_curbuf`).
    curbuf: u32,
    /// The previous block's trigger value, for rising-edge detection.
    prevtrig: f32,
}

/// The spans of a [`Convolution2L`]'s memory, in `framesize` units: the input frame (1), the input
/// spectrum (2), the two kernel spectra (2 each), the product spectrum and crossfade target (2), the
/// output frame (2), the held overlap (1), and the transform buffer of the reference's `scfft`s (2).
const CONV2L_FRAMES: usize = 14;

/// [`Convolution2L`]'s memory, split into its spans (named as the reference's buffers).
struct Conv2LSpans<'a> {
    inbuf1: &'a mut [f32],
    fftbuf1: &'a mut [f32],
    fftbuf2: &'a mut [f32],
    fftbuf3: &'a mut [f32],
    tempbuf: &'a mut [f32],
    outbuf: &'a mut [f32],
    overlap: &'a mut [f32],
    trbuf: &'a mut [f32],
}

impl<'a> Conv2LSpans<'a> {
    fn new(aux: &'a mut Aux<'_>, fs: usize) -> Self {
        let aux = &mut aux.f32_mut()[..CONV2L_FRAMES * fs];
        let (inbuf1, rest) = aux.split_at_mut(fs);
        let (fftbuf1, rest) = rest.split_at_mut(2 * fs);
        let (fftbuf2, rest) = rest.split_at_mut(2 * fs);
        let (fftbuf3, rest) = rest.split_at_mut(2 * fs);
        let (tempbuf, rest) = rest.split_at_mut(2 * fs);
        let (outbuf, rest) = rest.split_at_mut(2 * fs);
        let (overlap, trbuf) = rest.split_at_mut(fs);
        Conv2LSpans {
            inbuf1,
            fftbuf1,
            fftbuf2,
            fftbuf3,
            tempbuf,
            outbuf,
            overlap,
            trbuf,
        }
    }
}

impl Convolution2L {
    const IN: usize = 0;
    const KERNEL: usize = 1;
    const TRIGGER: usize = 2;
    const FRAMESIZE: usize = 3;
    const CROSSFADE: usize = 4;

    /// Silence the unit for the rest of its life (`SETCALC(ClearUnitOutputs)`, `mDone = true`).
    fn clear(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.framesize = 0;
        ctx.outs.audio(0).fill(0.0);
        ctx.done.mark_done();
    }
}

impl Unit for Convolution2L {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        let framesize = ctx.ins.control(Self::FRAMESIZE) as i32;
        let Some(fs) = usable_framesize(framesize, ctx.own.block_size) else {
            return;
        };
        if aux.alloc(CONV2L_FRAMES * fs * core::mem::size_of::<f32>()) {
            self.framesize = fs as u32;
        }
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.cflength = ctx.ins.control(Self::CROSSFADE) as i32;
        self.curbuf = 0;
        self.cfpos = self.cflength;
        let bufnum = ctx.ins.control(Self::KERNEL);
        let buf = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum);
        let Some(buf) = buf.filter(|_| self.framesize != 0) else {
            self.clear(ctx);
            return DoneAction::Nothing;
        };
        let fs = self.framesize as usize;
        let s = Conv2LSpans::new(&mut ctx.aux, fs);
        s.outbuf.fill(0.0);
        s.overlap.fill(0.0);
        load_kernel(ctx.fft, buf, fs, s.trbuf, s.fftbuf2);
        self.pos = 0;
        self.prevtrig = 0.0;
        *ctx.outs.control(0) = ctx.ins.control(Self::IN);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.framesize == 0 {
            self.clear(ctx);
            return DoneAction::Nothing;
        }
        let fs = self.framesize as usize;
        let n = ctx.outs.audio(0).len();
        let curtrig = ctx.ins.control(Self::TRIGGER);

        let pos = self.pos as usize;
        collect(
            ctx.ins,
            Self::IN,
            &mut Conv2LSpans::new(&mut ctx.aux, fs).inbuf1[pos..pos + n],
        );
        self.pos += n as u32;

        if self.prevtrig <= 0.0 && curtrig > 0.0 {
            let bufnum = ctx.ins.control(Self::KERNEL);
            let Some(buf) = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum) else {
                self.clear(ctx);
                return DoneAction::Nothing;
            };
            self.cflength = ctx.ins.control(Self::CROSSFADE) as i32;
            self.cfpos = 0;
            // The new kernel goes into the spectrum not in use.
            let s = Conv2LSpans::new(&mut ctx.aux, fs);
            let idle = if self.curbuf == 1 {
                s.fftbuf2
            } else {
                s.fftbuf3
            };
            load_kernel(ctx.fft, buf, fs, s.trbuf, idle);
        }

        let s = Conv2LSpans::new(&mut ctx.aux, fs);
        if self.pos & self.framesize != 0 {
            self.pos = 0;
            transform_frame(ctx.fft, s.inbuf1, s.trbuf, s.fftbuf1);
            let (current, next) = if self.curbuf == 0 {
                (&*s.fftbuf2, &*s.fftbuf3)
            } else {
                (&*s.fftbuf3, &*s.fftbuf2)
            };
            multiply_into(s.fftbuf1, current, s.tempbuf);
            s.overlap.copy_from_slice(&s.outbuf[fs..]);
            doifft(ctx.fft, Direction::Backward, s.tempbuf, s.outbuf);

            if self.cfpos < self.cflength {
                // The same frame through the incoming kernel, then a linear crossfade to it.
                multiply_in_place(s.fftbuf1, next);
                doifft(ctx.fft, Direction::Backward, s.fftbuf1, s.tempbuf);
                crossfade(
                    self.cfpos,
                    self.cflength,
                    fs,
                    &mut [(&mut *s.outbuf, &*s.tempbuf)],
                );
                self.cfpos += 1;
                if self.cfpos == self.cflength {
                    self.curbuf = if self.curbuf == 0 { 1 } else { 0 };
                }
            }
        }

        let pos = self.pos as usize;
        for ((o, &head), &tail) in ctx
            .outs
            .audio(0)
            .iter_mut()
            .zip(&s.outbuf[pos..])
            .zip(&s.overlap[pos..])
        {
            *o = head + tail;
        }
        self.prevtrig = curtrig;
        DoneAction::Nothing
    }
}

/// Crossfade each `(outbuf, tempbuf)` pair in place - `outbuf` from its own result towards
/// `tempbuf`'s - over the frame, as the reference's crossfade loops do.
///
/// The mix starts at `cfpos / cflength` and rises by `1 / (cflength * framesize)` per sample, shared
/// by every pair. Over the second half the reference's loop starts one sample late (at
/// `framesize + 1`), leaving sample `framesize` unfaded, except for a one-frame crossfade, which
/// takes the whole second half from `tempbuf`.
pub(crate) fn crossfade(cfpos: i32, cflength: i32, fs: usize, pairs: &mut [(&mut [f32], &[f32])]) {
    let mut fact1 = cfpos as f32 / cflength as f32;
    let rc = 1.0 / cflength.wrapping_mul(fs as i32) as f32;
    let mut fade = |i: usize, fact1: f32| {
        for (outbuf, tempbuf) in pairs.iter_mut() {
            outbuf[i] = (1.0 - fact1) * outbuf[i] + fact1 * tempbuf[i];
        }
    };
    for i in 0..fs {
        fade(i, fact1);
        fact1 += rc;
    }
    if cflength == 1 {
        for (outbuf, tempbuf) in pairs.iter_mut() {
            outbuf[fs..].copy_from_slice(&tempbuf[fs..]);
        }
    } else {
        for i in fs + 1..2 * fs {
            fade(i, fact1);
            fact1 += rc;
        }
    }
}

/// Constructor for [`Convolution2L`].
///
/// Audio or control rate (the reference's calc runs the unit's own calc length). `2 * framesize`
/// must be a power of two in `[8, 262144]` and `framesize` a multiple of the calc length; the
/// reference fails its FFT setup outside that size range and overruns its buffers otherwise, and any
/// such `framesize` silences the unit and marks it done here.
pub struct Convolution2LCtor;

impl UnitDef for Convolution2LCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 5 {
            return Err(BuildError::WrongInputCount);
        }
        if !matches!(ctx.rate, Rate::Audio | Rate::Control) {
            return Err(BuildError::UnsupportedRate(ctx.rate));
        }
        Ok(unit_spec_pool(Convolution2L::zeroed()))
    }
}
