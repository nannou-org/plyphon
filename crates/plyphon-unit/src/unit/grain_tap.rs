//! `GrainTap` - plyphon's port of scsynth's granulating delay tap (`DelayUGens.cpp`).
//!
//! `GrainTap` reads a buffer that something else keeps writing as a delay line, assuming the writer
//! advances one block per block from position 0, and plays up to 32 overlapping grains from it.
//! Each grain has a parabolic envelope, reads at `pchRatio` (with random dispersion) relative to
//! the write head, and starts at a randomly dispersed delay. A new grain starts every
//! `grainDur / overlap`. The random draws come from the synth's stream.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{self, BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::rate::Rate;

const BUFNUM: usize = 0;
const GRAIN_DUR: usize = 1;
const PCH_RATIO: usize = 2;
const PCH_DISP: usize = 3;
const TIME_DISP: usize = 4;
const OVERLAP: usize = 5;

/// Grains a `GrainTap` can play at once (scsynth's `MAXDGRAINS`).
const MAX_GRAINS: usize = 32;

/// The end of a grain list (scsynth's `nullptr`).
const NONE: i32 = -1;

/// One grain (scsynth's `GrainTap1`). Grains live in two singly linked lists threaded through
/// `next`, the active and the free list, as scsynth keeps them: the active list's order is the
/// order grains are summed into the output.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Grain {
    /// Samples left to play.
    counter: i64,
    /// Read distance behind the write head, in samples.
    pos: f32,
    /// Change in read distance per sample (`1 - pitch`).
    rate: f32,
    /// Envelope level.
    level: f32,
    /// Envelope slope.
    slope: f32,
    /// Envelope curvature (the slope's change per sample).
    curve: f32,
    /// The next grain in this grain's list, or [`NONE`].
    next: i32,
}

impl Grain {
    /// Play `out.len()` samples of the grain, adding them to `out`, with the write head at
    /// `iwrphase` before the first. The loop body shared by scsynth's two grain loops.
    fn play(&mut self, out: &mut [f32], dly: &[f32], mask: i64, mut iwrphase: i64) {
        let mut dsamp = self.pos;
        let dsamp_slope = self.rate;
        let mut level = self.level;
        let mut slope = self.slope;
        let curve = self.curve;
        for o in out.iter_mut() {
            dsamp += dsamp_slope;
            let idsamp = dsamp as i64;
            let frac = dsamp - idsamp as f32;
            iwrphase = (iwrphase + 1) & mask;
            let irdphase = (iwrphase - idsamp) & mask;
            let irdphaseb = (irdphase - 1) & mask;
            let d1 = dly[irdphase as usize];
            let d2 = dly[irdphaseb as usize];
            *o += (d1 + frac * (d2 - d1)) * level;
            level += slope;
            slope += curve;
        }
        self.pos = dsamp;
        self.level = level;
        self.slope = slope;
        self.counter -= out.len() as i64;
    }
}

/// `GrainTap.ar(bufnum, grainDur, pchRatio, pchDispersion, timeDispersion, overlap)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GrainTap {
    grains: [Grain; MAX_GRAINS],
    /// The write head's position at the block's start (`iwrphase`).
    iwrphase: i64,
    /// Samples until the next grain starts (`nextTime`).
    next_time: i64,
    /// The longest usable delay, `bufSamples - 2 * blockSize - 3` (`fdelaylen`).
    fdelay_len: f32,
    /// The buffer's sample count at construction (`bufsize`); a buffer of another size silences the
    /// block.
    bufsize: u32,
    /// Head of the active grain list (`firstActive`).
    first_active: i32,
    /// Head of the free grain list (`firstFree`).
    first_free: i32,
    /// Whether the constructor accepted the buffer (its sample count a power of two); otherwise the
    /// unit outputs silence for its life.
    live: u32,
    _pad: u32,
}

/// `(bufnum, bufSamples)` of the buffer `GET_BUF` resolves from input 0, with 0 samples for a
/// missing buffer.
fn buffer_samples(ctx: &ProcessCtx<'_>) -> (usize, u32) {
    let bufnum = ctx.ins.control(BUFNUM).max(0.0) as usize;
    let samples = unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum)
        .map_or(0, |b| (b.num_frames() * b.num_channels()) as u32);
    (bufnum, samples)
}

impl Unit for GrainTap {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `GrainTap_Ctor`: a buffer whose sample count is not a power of two silences the unit for
        // good; otherwise the delay length is fixed from it. The output sample is 0 either way.
        let (_, samples) = buffer_samples(ctx);
        let samples_i = samples as i32;
        if samples_i <= 0 || samples_i & (samples_i - 1) != 0 {
            return DoneAction::Nothing;
        }
        let block = ctx.audio.block_size as u32;
        self.fdelay_len = samples.wrapping_sub(2 * block).wrapping_sub(3) as f32;
        self.bufsize = samples;
        self.iwrphase = 0;
        self.next_time = 0;
        self.live = 1;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let (bufnum, samples) = buffer_samples(ctx);
        let out = ctx.outs.audio(0);
        out.fill(0.0);
        if self.live == 0 || samples != self.bufsize {
            return DoneAction::Nothing;
        }
        let Some(buf) = unit::buffer_at(ctx.buffers, &ctx.local_bufs, bufnum) else {
            return DoneAction::Nothing;
        };
        let dly = buf.data();
        let mask = (samples - 1) as i32 as i64;
        let sample_rate = ctx.audio.sample_rate;
        let block = out.len();
        let ins = ctx.ins;
        let mut rgen = *ctx.rgen;

        // `sc_max(0.0001, density)`, a double compared with the float.
        let density = ins.control(OVERLAP);
        let density = if 0.0001 > density as f64 {
            0.0001f64 as f32
        } else {
            density
        };
        let fdelay_len = self.fdelay_len;
        let iwrphase0 = self.iwrphase;

        // Play every active grain, in list order, unlinking the ones that finish.
        let mut prev = NONE;
        let mut g = self.first_active;
        while g != NONE {
            let gi = g as usize;
            let nsmps = self.grains[gi].counter.min(block as i64).max(0) as usize;
            self.grains[gi].play(&mut out[..nsmps], dly, mask, iwrphase0);
            let next = self.grains[gi].next;
            if self.grains[gi].counter <= 0 {
                if prev != NONE {
                    self.grains[prev as usize].next = next;
                } else {
                    self.first_active = next;
                }
                self.grains[gi].next = self.first_free;
                self.first_free = g;
            } else {
                prev = g;
            }
            g = next;
        }

        // Start new grains.
        let mut remain = block as i64;
        while self.next_time <= remain {
            remain -= self.next_time;
            let sdur = (ins.control(GRAIN_DUR) as f64 * sample_rate) as f32;
            let sdur = if sdur > 4.0 { sdur } else { 4.0 };

            let g = self.first_free;
            if g != NONE {
                let gi = g as usize;
                self.first_free = self.grains[gi].next;
                self.grains[gi].next = self.first_active;
                self.first_active = g;

                let koffset = block as i64 - remain;
                let iwrphase = (iwrphase0 + koffset) & mask;
                self.grains[gi].counter = sdur as i64;

                let timedisp = ins.control(TIME_DISP);
                let timedisp = if timedisp > 0.0 { timedisp } else { 0.0 };
                let timedisp = ((rgen.next_unipolar() * timedisp) as f64 * sample_rate) as f32;

                let pitch = ins.control(PCH_RATIO) + rgen.next_bipolar() * ins.control(PCH_DISP);
                // The block offset, as scsynth's `BUFLENGTH + koffset` (a long), then as a float.
                let offset = (block as i64 + koffset) as f32;
                let (dsamp_slope, dsamp) = if pitch >= 1.0 {
                    let maxpitch = 1.0 + fdelay_len / sdur;
                    let pitch = if pitch < maxpitch { pitch } else { maxpitch };
                    let dsamp_slope = 1.0 - pitch;
                    let maxtimedisp = fdelay_len + sdur * dsamp_slope;
                    let timedisp = if timedisp < maxtimedisp {
                        timedisp
                    } else {
                        maxtimedisp
                    };
                    let dsamp = offset + 2.0 + timedisp - sdur * dsamp_slope;
                    (dsamp_slope, dsamp)
                } else {
                    let maxpitch = -(1.0 + fdelay_len / sdur);
                    let pitch = if pitch > maxpitch { pitch } else { maxpitch };
                    let dsamp_slope = 1.0 - pitch;
                    let maxtimedisp = fdelay_len - sdur * dsamp_slope;
                    let timedisp = if timedisp < maxtimedisp {
                        timedisp
                    } else {
                        maxtimedisp
                    };
                    (dsamp_slope, offset + 2.0 + timedisp)
                };
                let dsamp = if dsamp < fdelay_len {
                    dsamp
                } else {
                    fdelay_len
                };

                // A parabolic envelope over `sdur` samples, rising from and returning to 0.
                let rdur = 1.0 / sdur;
                let rdur2 = rdur * rdur;
                let grain = &mut self.grains[gi];
                grain.rate = dsamp_slope;
                grain.pos = dsamp;
                grain.level = 0.0;
                grain.slope = (4.0 * (rdur - rdur2) as f64) as f32;
                grain.curve = (-8.0 * rdur2 as f64) as f32;
                grain.play(&mut out[koffset as usize..], dly, mask, iwrphase);
                if grain.counter <= 0 {
                    // It is at the head of the active list.
                    self.first_active = grain.next;
                    grain.next = self.first_free;
                    self.first_free = g;
                }
            }
            self.next_time = ((sdur / density) as i64).max(1);
        }
        self.next_time = (self.next_time - remain).max(0);
        self.iwrphase = (iwrphase0 + block as i64) & mask;
        *ctx.rgen = rgen;
        DoneAction::Nothing
    }
}

/// Constructor for [`GrainTap`]: audio rate only, as scsynth's calc writes a block.
pub struct GrainTapCtor;

impl UnitDef for GrainTapCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() <= OVERLAP {
            return Err(BuildError::WrongInputCount);
        }
        if ctx.rate != Rate::Audio {
            return Err(BuildError::UnsupportedRate(ctx.rate));
        }
        let mut state = GrainTap::zeroed();
        // Every grain starts on the free list, in index order.
        for (i, grain) in state.grains.iter_mut().enumerate() {
            grain.next = if i + 1 < MAX_GRAINS {
                i as i32 + 1
            } else {
                NONE
            };
        }
        state.first_free = 0;
        state.first_active = NONE;
        Ok(unit_spec(state))
    }
}
