//! `Convolution3` - plyphon's port of scsynth's time-domain convolver (`Convolution.cpp`), plus the
//! kernel-buffer lookup (`ConvGetBuffer`) that file's buffer-kernel convolvers share.
//!
//! The kernel is read from a buffer when the synth starts and again on every rising trigger. Each
//! input sample is multiplied by the whole kernel and summed into a circular accumulator of
//! `framesize` samples, starting at the current position; the output reads the accumulator at that
//! position. As in scsynth the accumulator is never cleared after a sample is read, so every
//! contribution keeps recirculating: the output is the running sum of the input convolved with the
//! kernel, folded modulo `framesize`.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    Aux, BuiltUnit, DoneAction, InitCtx, LocalBufs, ProcessCtx, Unit, buffer_at, unit_spec_pool,
};
use plyphon_dsp::buffer::{BufView, BufferTable};
use plyphon_dsp::rate::Rate;

/// scsynth's `ConvGetBuffer`: resolve the kernel buffer number `bufnum` (the raw input value, cast
/// as `(int32)ZIN0(i)`), world or graph-local. `None` for a negative number, a number that names no
/// buffer, or a buffer with no storage; the caller then silences itself for good and marks itself
/// done, as the reference does (`SETCALC(ClearUnitOutputs)`, `mDone = true`).
pub(crate) fn conv_get_buffer<'a>(
    buffers: &'a BufferTable,
    local: &'a LocalBufs<'_>,
    bufnum: f32,
) -> Option<BufView<'a>> {
    let bufnum = bufnum as i32;
    let index = usize::try_from(bufnum).ok()?;
    buffer_at(buffers, local, index)
}

/// Copy a kernel of `dst.len()` samples from `buf`'s raw interleaved storage, as the reference's
/// `memcpy(dst, buf->data, framesize * sizeof(float))` does. That `memcpy` reads past the end of a
/// buffer holding fewer samples; here the samples the buffer lacks are zero.
pub(crate) fn copy_kernel(dst: &mut [f32], buf: BufView<'_>) {
    let data = buf.data();
    let n = dst.len().min(data.len());
    dst[..n].copy_from_slice(&data[..n]);
    dst[n..].fill(0.0);
}

/// `Convolution3.ar/kr(in, kernel, trigger, framesize)`: time-domain convolution of `in` with a
/// kernel read from buffer `kernel`, re-read on every rising `trigger`.
///
/// `framesize` (`<= 0` means the kernel buffer's frame count) is read when the synth starts
/// (`Convolution3_Ctor`), which allocates the input, kernel and accumulator spans. An audio-rate `in`
/// runs `Convolution3_next_a` over the block; any other `in` runs `Convolution3_next_k`, one sample
/// per block.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Convolution3 {
    /// Kernel and accumulator length; `0` once the unit is silenced (no kernel buffer, an unusable
    /// `framesize`, or a failed re-read).
    framesize: u32,
    /// The accumulator position of the next output sample, in `0..=framesize`.
    pos: u32,
    /// The previous block's trigger value, for rising-edge detection.
    prevtrig: f32,
    /// `1` when `in` is audio rate (`Convolution3_next_a`), `0` otherwise (`Convolution3_next_k`).
    audio_in: u32,
}

impl Convolution3 {
    const IN: usize = 0;
    const KERNEL: usize = 1;
    const TRIGGER: usize = 2;
    const FRAMESIZE: usize = 3;

    /// Split the memory into the input frame, the kernel, and the accumulator. The accumulator has
    /// one sample past `framesize`: `Convolution3_next_k` reads `pout[framesize]` once per cycle,
    /// one past the end of the reference's allocation. Here that slot is zeroed and never written,
    /// so the read yields `0`.
    fn spans<'a>(&self, aux: &'a mut Aux<'_>) -> (&'a mut [f32], &'a mut [f32], &'a mut [f32]) {
        let size = self.framesize as usize;
        let aux = &mut aux.f32_mut()[..3 * size + 1];
        let (inbuf, rest) = aux.split_at_mut(size);
        let (kernel, outbuf) = rest.split_at_mut(size);
        (inbuf, kernel, outbuf)
    }

    /// Silence the unit for the rest of its life (`SETCALC(ClearUnitOutputs)`, `mDone = true`).
    fn clear(&mut self, ctx: &mut ProcessCtx<'_>, len: usize) {
        self.framesize = 0;
        ctx.outs.audio(0)[..len].fill(0.0);
        ctx.done.mark_done();
    }

    /// The rising-trigger kernel re-read shared by both calcs. Returns `false` (having silenced the
    /// unit, clearing `len` output samples) when the buffer cannot be resolved.
    fn retrigger(&mut self, ctx: &mut ProcessCtx<'_>, len: usize) -> bool {
        let curtrig = ctx.ins.control(Self::TRIGGER);
        if self.prevtrig <= 0.0 && curtrig > 0.0 {
            let bufnum = ctx.ins.control(Self::KERNEL);
            let Some(buf) = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum) else {
                self.clear(ctx, len);
                return false;
            };
            let (_, kernel, _) = self.spans(&mut ctx.aux);
            copy_kernel(kernel, buf);
        }
        true
    }

    /// `Convolution3_next_a`: accumulate the whole block, then read it out.
    fn next_a(&mut self, ctx: &mut ProcessCtx<'_>) {
        let n = ctx.outs.audio(0).len();
        let input = ctx.ins.audio(Self::IN);
        let (inbuf, _, _) = self.spans(&mut ctx.aux);
        inbuf[..n].copy_from_slice(&input[..n]);
        if !self.retrigger(ctx, n) {
            return;
        }
        let size = self.framesize as usize;
        let pos = self.pos as usize;
        let (inbuf, kernel, outbuf) = self.spans(&mut ctx.aux);
        for (j, &x) in inbuf[..n].iter().enumerate() {
            for (i, &k) in kernel.iter().enumerate() {
                let ind = (pos + i + j) % size;
                outbuf[ind] += k * x;
            }
        }
        for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
            *o = outbuf[(pos + i) % size];
        }
        let pos = pos + n;
        self.pos = if pos > size { pos - size } else { pos } as u32;
        self.prevtrig = ctx.ins.control(Self::TRIGGER);
    }

    /// `Convolution3_next_k`: accumulate one sample and output one. At audio rate only the first
    /// sample of the block is written, as in the reference (see [`Convolution3Ctor`]).
    fn next_k(&mut self, ctx: &mut ProcessCtx<'_>) {
        let input = ctx.ins.control(Self::IN);
        if !self.retrigger(ctx, 1) {
            return;
        }
        let size = self.framesize as usize;
        let pos = self.pos as usize;
        let (_, kernel, outbuf) = self.spans(&mut ctx.aux);
        for (i, &k) in kernel.iter().enumerate() {
            let ind = (pos + i) % size;
            outbuf[ind] += k * input;
        }
        *ctx.outs.control(0) = outbuf[pos];
        // `if (++pos > size) m_pos = 0; else m_pos++;` - the position reaches `size` itself before
        // wrapping.
        self.pos = if pos + 1 > size { 0 } else { pos + 1 } as u32;
        self.prevtrig = ctx.ins.control(Self::TRIGGER);
    }
}

impl Unit for Convolution3 {
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        let bufnum = ctx.ins.control(Self::KERNEL);
        let Some(buf) = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum) else {
            return;
        };
        let mut framesize = ctx.ins.control(Self::FRAMESIZE) as i32;
        if framesize <= 0 {
            framesize = buf.num_frames() as i32;
        }
        // A zero `framesize` would divide by zero in the reference's position arithmetic, and
        // `Convolution3_next_a` copies a whole block into the `framesize`-sample input frame; both
        // are refused here rather than faulting.
        let Ok(size) = usize::try_from(framesize) else {
            return;
        };
        if size == 0 || (self.audio_in != 0 && size < ctx.audio.block_size) {
            return;
        }
        let Some(bytes) = size
            .checked_mul(3)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(core::mem::size_of::<f32>()))
        else {
            return;
        };
        if aux.alloc(bytes) {
            self.framesize = size as u32;
        }
    }

    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bufnum = ctx.ins.control(Self::KERNEL);
        let buf = conv_get_buffer(ctx.buffers, &ctx.local_bufs, bufnum);
        let Some(buf) = buf.filter(|_| self.framesize != 0) else {
            // No kernel buffer (`ConvGetBuffer` clears the output and marks the unit done), or a
            // `framesize` the unit cannot run with.
            self.clear(ctx, 1);
            return DoneAction::Nothing;
        };
        let (_, kernel, outbuf) = self.spans(&mut ctx.aux);
        copy_kernel(kernel, buf);
        outbuf.fill(0.0);
        self.pos = 0;
        self.prevtrig = 0.0;
        *ctx.outs.control(0) = ctx.ins.control(Self::IN);
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        if self.framesize == 0 {
            let len = ctx.outs.audio(0).len();
            self.clear(ctx, len);
        } else if self.audio_in != 0 {
            self.next_a(ctx);
        } else {
            self.next_k(ctx);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Convolution3`]: picks the calc from `in`'s rate, as `Convolution3_Ctor` does.
///
/// The reference runs `Convolution3_next_a` for an audio-rate `in` whatever the unit's own rate, and
/// that calc writes a full audio block of output; a control-rate unit has one output sample, so
/// `Convolution3.kr` over an audio-rate `in` overruns its output in the reference and is refused
/// here. An audio-rate unit over a non-audio `in` runs `Convolution3_next_k`, which writes only the
/// block's first output sample; the rest of the block is left as the reference leaves it, unwritten.
pub struct Convolution3Ctor;

impl UnitDef for Convolution3Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() != 4 {
            return Err(BuildError::WrongInputCount);
        }
        let audio_in = ctx.input_rates[Convolution3::IN] == Rate::Audio;
        match ctx.rate {
            Rate::Audio => {}
            Rate::Control if !audio_in => {}
            rate => return Err(BuildError::UnsupportedRate(rate)),
        }
        Ok(unit_spec_pool(Convolution3 {
            audio_in: audio_in as u32,
            ..Convolution3::zeroed()
        }))
    }
}
