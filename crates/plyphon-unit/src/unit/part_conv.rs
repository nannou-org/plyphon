//! `PartConv` - plyphon's port of scsynth's partitioned convolver (`PartitionedConvolution.cpp`).
//!
//! The impulse response is not read as samples: its buffer holds the spectra of the response's
//! consecutive `fftsize / 2`-sample partitions, each zero-padded to `fftsize` and forward-transformed,
//! one `fftsize`-float packed spectrum after another - the layout scsynth's `PreparePartConv` buffer
//! generator writes (`/b_gen ... PreparePartConv srcbuf fftsize`, sized by
//! `PartConv.calcBufSize(fftsize, irbuffer)`). plyphon's `/b_gen` does not provide that generator,
//! so the spectra must be written into the buffer directly.
//!
//! The input is collected `fftsize / 2` samples at a time; each full half-frame is zero-padded and
//! transformed, and its spectrum is multiplied by every partition's spectrum and accumulated into a
//! ring of per-frame spectra, partition `i` landing `i` frames ahead. The block that completes a
//! half-frame multiplies in the first partition, inverse-transforms the ring's current frame, shifts
//! the output frame down by half and sums the result in, then clears that ring slot and advances.
//! The other partitions are spread across the remaining blocks of the half-frame (amortisation), so
//! the calc stays near-constant per block.
//!
//! Compiled only with the `fft` feature.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::convolution2::{Direction, dofft, doifft};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{Aux, BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec_pool};
use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::fft::is_supported_size;
use plyphon_dsp::rate::Rate;

/// `PartConv.ar(in, fftsize, irbufnum)`: convolve `in` with the impulse response whose partition
/// spectra buffer `irbufnum` holds (see the module docs).
///
/// `fftsize` and `irbufnum` are read when the synth starts (`PartConv_Ctor`), which allocates the
/// working set, including a spectral accumulator as large as the spectra buffer. The unit is silenced
/// and marked done if `fftsize` is unusable, if the buffer is missing or empty or its size is not a
/// multiple of `fftsize`, if the block does not divide `fftsize / 2`, or if a half-frame is a single
/// block (nothing to amortise over). The buffer is read live each block, and the unit silences itself
/// if it is freed.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PartConv {
    /// The transform size; `0` once the unit is silenced.
    fftsize: u32,
    /// The spectra buffer's number (`m_specbufnumcheck`).
    bufnum: u32,
    /// The spectra buffer's size in floats, and so the accumulator's (`m_fullsize`).
    fullsize: u32,
    /// Partitions in the impulse response (`m_partitions`).
    partitions: i32,
    /// Samples of the current half-frame collected so far (`m_pos`).
    pos: u32,
    /// Read offset into the output frame (`m_outputpos`).
    outputpos: u32,
    /// The accumulator frame the next half-frame completes (`m_fd_accum_pos`).
    accum_pos: u32,
    /// Blocks per half-frame after the first (`m_spareblocks`).
    spareblocks: i32,
    /// Partitions multiplied per amortisation block (`m_numamort`).
    numamort: i32,
    /// Partitions multiplied in the last amortisation block (`m_lastamort`).
    lastamort: i32,
    /// Amortisation blocks done this half-frame; `-1` until the first half-frame completes
    /// (`m_amortcount`).
    amortcount: i32,
    /// Partitions multiplied in so far this half-frame (`m_partitionsdone`).
    partitionsdone: i32,
}

impl PartConv {
    const IN: usize = 0;
    const FFTSIZE: usize = 1;
    const IRBUFNUM: usize = 2;

    /// Silence the unit for the rest of its life (`SETCALC(ClearUnitOutputs)`, `mDone = true`).
    fn clear(&mut self, ctx: &mut ProcessCtx<'_>) {
        self.fftsize = 0;
        ctx.outs.audio(0).fill(0.0);
        ctx.done.mark_done();
    }

    /// The spectra buffer, read live: scsynth keeps the buffer's data pointer and checks each block
    /// only that the buffer still has data. `None` once it has none - or holds fewer than the
    /// `fullsize` floats the unit was built for, which the reference would read past.
    fn spectra<'a>(&self, buffers: &'a BufferTable) -> Option<&'a [f32]> {
        buffers
            .get(self.bufnum as usize)
            .and_then(|buf| buf.data().get(..self.fullsize as usize))
    }
}

/// The spans of a [`PartConv`]'s memory, in `fftsize` units, before the accumulator: the input frame,
/// the input spectrum, the inverse transform's input and output, the output frame, and the transform
/// buffer of the reference's `scfft`s.
const PARTCONV_FRAMES: usize = 6;

/// [`PartConv`]'s memory, split into its spans (named as the reference's buffers).
struct Spans<'a> {
    inputbuf: &'a mut [f32],
    spectrum: &'a mut [f32],
    inputbuf2: &'a mut [f32],
    spectrum2: &'a mut [f32],
    output: &'a mut [f32],
    trbuf: &'a mut [f32],
    accumulate: &'a mut [f32],
}

impl<'a> Spans<'a> {
    fn new(aux: &'a mut Aux<'_>, fftsize: usize, fullsize: usize) -> Self {
        let aux = &mut aux.f32_mut()[..PARTCONV_FRAMES * fftsize + fullsize];
        let (inputbuf, rest) = aux.split_at_mut(fftsize);
        let (spectrum, rest) = rest.split_at_mut(fftsize);
        let (inputbuf2, rest) = rest.split_at_mut(fftsize);
        let (spectrum2, rest) = rest.split_at_mut(fftsize);
        let (output, rest) = rest.split_at_mut(fftsize);
        let (trbuf, accumulate) = rest.split_at_mut(fftsize);
        Spans {
            inputbuf,
            spectrum,
            inputbuf2,
            spectrum2,
            output,
            trbuf,
            accumulate,
        }
    }
}

/// `target += ir * spectrum` over packed spectra: the DC and Nyquist terms as real products, every
/// other bin as a complex product (the reference's multiply-accumulate loops).
fn multiply_accumulate(target: &mut [f32], ir: &[f32], spectrum: &[f32]) {
    target[0] += ir[0] * spectrum[0];
    target[1] += ir[1] * spectrum[1];
    for ((t, i), s) in target[2..]
        .chunks_exact_mut(2)
        .zip(ir[2..].chunks_exact(2))
        .zip(spectrum[2..].chunks_exact(2))
    {
        t[0] += (i[0] * s[0]) - (i[1] * s[1]);
        t[1] += (i[1] * s[0]) + (i[0] * s[1]);
    }
}

impl Unit for PartConv {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // `scfft_create` fails outside `[8, 262144]` and the reference's transforms overrun their
        // buffers for a size that is not a power of two.
        let fftsize = ctx.ins.control(Self::FFTSIZE) as i32;
        let Some(fftsize) = usize::try_from(fftsize)
            .ok()
            .filter(|&n| is_supported_size(n))
        else {
            return;
        };
        // `(uint32)ZIN0(2)`. A number past the world's buffers selects a graph-local buffer in the
        // reference's first check, but the constructor then reads it as a world buffer regardless,
        // past the end of the table; only world buffers are resolved here.
        let bufnum = ctx.ins.control(Self::IRBUFNUM) as i32;
        let Ok(bufnum) = usize::try_from(bufnum) else {
            return;
        };
        let Some(buf) = ctx.buffers.get(bufnum) else {
            return;
        };
        let fullsize = buf.data().len();
        if fullsize == 0 || !fullsize.is_multiple_of(fftsize) {
            return;
        }
        let nover2 = fftsize / 2;
        let blocksize = ctx.audio.block_size;
        if !nover2.is_multiple_of(blocksize) {
            return;
        }
        let spareblocks = (nover2 / blocksize) as i32 - 1;
        if spareblocks < 1 {
            return;
        }
        let Some(bytes) = fullsize
            .checked_add(PARTCONV_FRAMES * fftsize)
            .and_then(|n| n.checked_mul(core::mem::size_of::<f32>()))
        else {
            return;
        };
        if !aux.alloc(bytes) {
            return;
        }
        let partitions = (fullsize / fftsize) as i32;
        self.fftsize = fftsize as u32;
        self.bufnum = bufnum as u32;
        self.fullsize = fullsize as u32;
        self.partitions = partitions;
        self.spareblocks = spareblocks;
        self.numamort = (partitions - 1) / spareblocks;
        self.lastamort = (partitions - 1) - ((spareblocks - 1) * self.numamort);
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // The constructor writes `0`; every failure also leaves the output at `0`.
        if self.fftsize == 0 {
            ctx.done.mark_done();
            return DoneAction::Nothing;
        }
        let s = Spans::new(&mut ctx.aux, self.fftsize as usize, self.fullsize as usize);
        s.output.fill(0.0);
        s.inputbuf.fill(0.0);
        s.accumulate.fill(0.0);
        self.pos = 0;
        self.outputpos = 0;
        self.accum_pos = 0;
        self.amortcount = -1;
        self.partitionsdone = 1;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.fftsize == 0 {
            self.clear(ctx);
            return DoneAction::Nothing;
        }
        let Some(irspectra) = self.spectra(ctx.buffers) else {
            self.clear(ctx);
            return DoneAction::Nothing;
        };
        let fftsize = self.fftsize as usize;
        let nover2 = fftsize / 2;
        let fullsize = self.fullsize as usize;
        let n = ctx.outs.audio(0).len();
        let s = Spans::new(&mut ctx.aux, fftsize, fullsize);

        let pos = self.pos as usize;
        let input = &mut s.inputbuf[pos..pos + n];
        if ctx.ins.rate(Self::IN) == Rate::Audio {
            input.copy_from_slice(&ctx.ins.audio(Self::IN)[..n]);
        } else {
            // The reference copies `IN(0)` as a whole block, which only an audio-rate input is;
            // any other input is held for the block.
            input.fill(ctx.ins.control(Self::IN));
        }
        self.pos += n as u32;

        if self.pos as usize == nover2 {
            // The second half of the input frame is never written, so it stays zero.
            s.trbuf.copy_from_slice(s.inputbuf);
            dofft(ctx.fft, Direction::Forward, s.trbuf, s.spectrum);
            self.pos = 0;
            self.outputpos = 0;

            // The first partition now; the rest are amortised over the following blocks.
            let accum_pos = self.accum_pos as usize;
            multiply_accumulate(
                &mut s.accumulate[accum_pos..accum_pos + fftsize],
                &irspectra[..fftsize],
                s.spectrum,
            );

            s.inputbuf2
                .copy_from_slice(&s.accumulate[accum_pos..accum_pos + fftsize]);
            doifft(ctx.fft, Direction::Backward, s.inputbuf2, s.spectrum2);

            s.output.copy_within(nover2.., 0);
            s.output[nover2..].fill(0.0);
            for (o, &x) in s.output.iter_mut().zip(s.spectrum2.iter()) {
                *o += x;
            }

            s.accumulate[accum_pos..accum_pos + fftsize].fill(0.0);
            self.accum_pos = ((accum_pos + fftsize) % fullsize) as u32;
            self.amortcount = 0;
            self.partitionsdone = 1;
        } else if self.amortcount >= 0 {
            let number = if self.amortcount == self.spareblocks - 1 {
                self.lastamort
            } else {
                self.numamort
            };
            let starti = self.partitionsdone;
            let stopi = starti + number - 1;
            self.partitionsdone += number;
            self.amortcount += 1;
            let accum_pos = self.accum_pos as usize;
            for i in starti..=stopi {
                // The ring has already advanced past this half-frame's slot, hence `i - 1`.
                let posnow = (accum_pos + (i as usize - 1) * fftsize) % fullsize;
                let irpos = i as usize * fftsize;
                multiply_accumulate(
                    &mut s.accumulate[posnow..posnow + fftsize],
                    &irspectra[irpos..irpos + fftsize],
                    s.spectrum,
                );
            }
        }

        let outputpos = self.outputpos as usize;
        ctx.outs
            .audio(0)
            .copy_from_slice(&s.output[outputpos..outputpos + n]);
        self.outputpos += n as u32;
        DoneAction::Nothing
    }
}

/// Constructor for [`PartConv`].
///
/// Audio rate only: the amortisation schedule counts audio blocks, and a control-rate calc would
/// step past the last partition. `fftsize` must be a power of two in `[8, 262144]` (the reference
/// fails its FFT setup outside that range and overruns its buffers for other sizes); an unusable
/// `fftsize` silences the unit and marks it done.
pub struct PartConvCtor;

impl UnitDef for PartConvCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 3 {
            return Err(BuildError::WrongInputCount);
        }
        if ctx.rate != Rate::Audio {
            return Err(BuildError::UnsupportedRate(ctx.rate));
        }
        Ok(unit_spec_pool(PartConv::zeroed()))
    }
}
