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
//! As in scsynth, the FFT size is the chain buffer's frame count, and each unit allocates its
//! memory from the engine's pool once that size is known. `FFT` reads the buffer when the synth
//! starts (`FFTBase_Ctor`); `IFFT` reads it from the chain's first ready frame, the first block its
//! input carries the buffer number. A positive `winsize` may only name the buffer's own size:
//! scsynth's zero-padded analysis (a window smaller than the buffer) is not supported, and such a
//! unit stays silent. The size must be a power of two in `[8, 262144]`, scsynth's range. For the
//! overlap-add to line up, `hop * fftsize` should be a whole number of control blocks.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{
    self, Aux, BuiltUnit, DoneAction, InitCtx, Inputs, LocalBufs, ProcessCtx, Unit, pv,
    unit_spec_pool,
};
use plyphon_dsp::buffer::{BufferTable, SpectrumCoord};
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
    /// FFT size, the chain buffer's frame count when the synth started; `0` if that buffer was
    /// missing or unusable, which leaves the unit outputting `-1` for the rest of its life (scsynth's
    /// `FFT_ClearUnitOutputs`).
    fftsize: u32,
    /// Samples between frames, `round(hop * fftsize)`, from the `hop` input when the synth started.
    hop_size: u32,
    /// The window type code, from the `wintype` input when the synth started.
    wintype: i32,
    /// Circular write head into the input ring.
    pos: u32,
    /// Samples accumulated since the last frame; a frame fires when it reaches `hop_size`.
    counter: u32,
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
    fn alloc(&mut self, ctx: &InitCtx<'_>, aux: &mut Aux<'_>) {
        // scsynth's `FFT_Ctor`: size the unit from the chain buffer, then allocate the input ring
        // and zero it. Without a usable buffer the unit never allocates and outputs `-1`.
        let ins = ctx.ins;
        let Some(n) = chain_fftsize(
            ctx.buffers,
            &ctx.local_bufs,
            ins.control(Self::BUFFER),
            ins.control(Self::WINSIZE),
        ) else {
            return;
        };
        if !aux.alloc(2 * n * core::mem::size_of::<f32>()) {
            return;
        }
        aux.f32_mut()[..n].fill(0.0);
        self.fftsize = n as u32;
        self.hop_size = (math::floor(ins.control(Self::HOP) * n as f32 + 0.5) as u32).max(1);
        self.wintype = ins.control(Self::WINTYPE) as i32;
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let n = self.fftsize as usize;
        if n == 0 {
            *ctx.outs.control(0) = -1.0;
            return DoneAction::Nothing;
        }
        let bs = ctx.audio.block_size;
        let ins = ctx.ins; // `Copy`; borrows the wires, not `ctx`.
        let bufnum = ins.control(Self::BUFFER).max(0.0) as usize;
        let active = ins.control(Self::ACTIVE) > 0.0;

        let win = ctx
            .fft
            .window(n, WindowType::from_code(self.wintype as f32));
        let (ring, rest) = ctx.aux.f32_mut().split_at_mut(n);
        let windowed = &mut rest[..n];

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
                if let Some(mut buffer) =
                    unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
                        .filter(|b| b.num_frames() == n)
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

/// Constructor for [`Fft`]: the unit sizes and allocates its memory when the synth starts.
pub struct FftCtor;

impl UnitDef for FftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Fft::WINSIZE {
            return Err(BuildError::WrongInputCount);
        }
        Ok(BuiltUnit {
            cleared_output: -1.0,
            ..unit_spec_pool(Fft::zeroed())
        })
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
    /// FFT size, the chain buffer's frame count at the first ready frame; `0` until then.
    fftsize: u32,
    /// The window type code, from the `wintype` input at the first ready frame.
    wintype: i32,
    /// Read/write head into the overlap-add ring.
    pos: u32,
    /// `1` once the first ready frame's buffer proved unusable: the unit then outputs silence for
    /// the rest of its life, as scsynth's `IFFT_Ctor` falls back to `ClearUnitOutputs`.
    dead: u32,
}

impl Ifft {
    const BUFFER: usize = 0;
    const WINTYPE: usize = 1;
    const WINSIZE: usize = 2;
}

impl Unit for Ifft {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let bs = ctx.audio.block_size;
        let fbufnum = ctx.ins.control(Self::BUFFER);

        // scsynth's `IFFT_Ctor` sizes the unit from the chain buffer and allocates its overlap-add
        // ring. The buffer number only reaches an `IFFT` with the chain's first ready frame, so the
        // unit does this then, and is silent until.
        if self.fftsize == 0 {
            if self.dead != 0 || fbufnum < 0.0 {
                ctx.outs.audio(0).fill(0.0);
                return DoneAction::Nothing;
            }
            let ins = ctx.ins;
            let n = match chain_fftsize(
                ctx.buffers,
                &ctx.local_bufs,
                fbufnum,
                ins.control(Self::WINSIZE),
            ) {
                Some(n) => n,
                None => {
                    self.dead = 1;
                    ctx.outs.audio(0).fill(0.0);
                    return DoneAction::Nothing;
                }
            };
            if !ctx.aux.alloc(2 * n * core::mem::size_of::<f32>()) {
                return DoneAction::Nothing;
            }
            ctx.aux.f32_mut()[..n].fill(0.0);
            self.fftsize = n as u32;
            self.wintype = ins.control(Self::WINTYPE) as i32;
        }
        let n = self.fftsize as usize;

        let win = ctx
            .fft
            .window(n, WindowType::from_code(self.wintype as f32));
        let (ola, rest) = ctx.aux.f32_mut().split_at_mut(n);
        let temp = &mut rest[..n];

        // A ready frame (fbufnum >= 0): inverse-transform it and overlap-add into the ring at `pos`.
        if fbufnum >= 0.0 {
            let bufnum = fbufnum as usize;
            if let Some(mut buffer) = unit::buffer_at_mut(ctx.buffers, &mut ctx.local_bufs, bufnum)
                .filter(|b| b.num_frames() == n)
            {
                // A polar `PV_*` unit may have left the frame in polar form; restore Cartesian
                // before the inverse transform (scsynth's `ToComplexApx` in `IFFT_next`).
                pv::to_complex(&mut buffer);
                if ctx.fft.inverse(n, &buffer.data()[..n], temp) {
                    // An empty window is the rectangular one: all ones.
                    for (j, &t) in temp.iter().enumerate() {
                        let w = win.get(j).copied().unwrap_or(1.0);
                        ola[(self.pos as usize + j) % n] += t * w;
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

/// Constructor for [`Ifft`]: the unit sizes and allocates its memory at the chain's first frame.
pub struct IfftCtor;

impl UnitDef for IfftCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= Ifft::WINSIZE {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_pool(Ifft::zeroed()))
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

/// The FFT size for chain buffer `bufnum` - scsynth's `FFTBase_Ctor`: the buffer's frame count,
/// which a positive `winsize` caps (`m_audiosize = min(buf->samples, winsize)`). `None` when the
/// buffer does not exist, its size is not a power of two in `[8, 262144]`, or `winsize` asks for a
/// window smaller than the buffer (scsynth's zero-padded analysis, which plyphon does not support).
fn chain_fftsize(
    buffers: &BufferTable,
    local: &LocalBufs<'_>,
    bufnum: f32,
    winsize: f32,
) -> Option<usize> {
    let frames = unit::buffer_at(buffers, local, bufnum.max(0.0) as usize)?.num_frames();
    let winsize = winsize as i32;
    let audiosize = if winsize < 1 {
        frames
    } else {
        frames.min(winsize as usize)
    };
    (audiosize == frames && is_supported_size(frames)).then_some(frames)
}
