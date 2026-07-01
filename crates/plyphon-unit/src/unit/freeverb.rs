//! `FreeVerb`/`FreeVerb2` - plyphon's ports of scsynth's Schroeder/Moorer "freeverb" reverb
//! (`ReverbUGens.cpp`).
//!
//! The classic freeverb: the input feeds eight parallel comb filters (each a delay line with a one-pole
//! lowpass in its feedback - the `damp` control), whose sum runs through four series Schroeder allpasses
//! (feedback `0.5`), scaled and mixed back with the dry signal by `mix`. `room` sets the comb feedback
//! (ring time) and `damp` the high-frequency decay. [`FreeVerb`] is mono; [`FreeVerb2`] is the true-
//! stereo version - two input channels driving two banks of combs/allpasses whose lengths are offset by
//! a fixed stereo spread (23 samples), cross-mixed to two outputs. Both share one `process_bank`
//! kernel.
//!
//! The delay lines are fixed-size (the classic 44.1 kHz freeverb tunings) and live in
//! [aux memory](crate::unit::Aux), zeroed on the first block so the dirty arena never leaks.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec_aux};
use plyphon_dsp::rate::Rate;

/// One bank's twelve line lengths: four series allpasses then eight parallel combs (the 44.1 kHz
/// freeverb tunings). The stereo right bank ([`SIZES_R`]) adds a 23-sample spread to each.
const SIZES: [usize; 12] = [
    225, 341, 441, 556, 1617, 1557, 1491, 1422, 1277, 1116, 1188, 1356,
];
/// The cumulative offset of each line within one bank (prefix sum of [`SIZES`]).
const OFFSETS: [usize; 12] = [
    0, 225, 566, 1007, 1563, 3180, 4737, 6228, 7650, 8927, 10043, 11231,
];
/// Total samples in one bank.
const BANK: usize = 12587;

/// Clamp a control input to `[0, 1]` (scsynth clips `mix`/`room`/`damp`).
fn unit_clamp(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Run one freeverb bank for a single sample: advance its twelve lines in `buf` (at `offsets`, with
/// wrap `sizes`), run the eight damped combs and four series allpasses, and return the bank's output
/// (`R0_1`). `scaled` is the shared `0.015 * input` excitation; `feedback` the comb feedback
/// (`0.7 + 0.28*room`); `damp`/`damp1` the one-pole lowpass coefficients. `iota` (12), `r0` (20) and
/// `r1` (4) are the bank's persistent state.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn process_bank(
    buf: &mut [f32],
    offsets: &[usize; 12],
    sizes: &[usize; 12],
    iota: &mut [u32],
    r0: &mut [f32],
    r1: &mut [f32],
    scaled: f32,
    feedback: f32,
    damp: f32,
    damp1: f32,
) -> f32 {
    // Read the four allpass taps (their lines are rewritten below, in the allpass chain).
    let mut t = [0.0f32; 4];
    for a in 0..4 {
        iota[a] += 1;
        if iota[a] == sizes[a] as u32 {
            iota[a] = 0;
        }
        t[a] = buf[offsets[a] + iota[a] as usize];
    }

    // Eight parallel damped combs; each delayed read R*_0 sums into the allpass input.
    for cmb in 0..8 {
        let k = 4 + cmb;
        iota[k] += 1;
        if iota[k] == sizes[k] as u32 {
            iota[k] = 0;
        }
        let idx = offsets[k] + iota[k] as usize;
        let tc = buf[idx];
        let (re, ro) = (4 + 2 * cmb, 5 + 2 * cmb);
        r0[ro] = damp1 * r0[re] + damp * r0[ro];
        buf[idx] = scaled + feedback * r0[ro];
        r0[re] = tc;
    }
    let combsum = r0[4] + r0[6] + r0[8] + r0[10] + r0[12] + r0[14] + r0[16] + r0[18];

    // Four series Schroeder allpasses (feedback 0.5); the last is fed by the comb sum.
    buf[offsets[3] + iota[3] as usize] = 0.5 * r0[3] + combsum;
    r0[3] = t[3];
    r1[3] = r0[3] - combsum;
    for a in (0..3).rev() {
        buf[offsets[a] + iota[a] as usize] = 0.5 * r0[a] + r1[a + 1];
        r0[a] = t[a];
        r1[a] = r0[a] - r1[a + 1];
    }
    r1[0]
}

/// The right-bank line lengths (left + a 23-sample stereo spread) and their offsets within the second
/// bank.
const SIZES_R: [usize; 12] = [
    248, 364, 464, 579, 1640, 1580, 1514, 1445, 1300, 1139, 1211, 1379,
];
const OFFSETS_R: [usize; 12] = [
    BANK,
    BANK + 248,
    BANK + 612,
    BANK + 1076,
    BANK + 1655,
    BANK + 3295,
    BANK + 4875,
    BANK + 6389,
    BANK + 7834,
    BANK + 9134,
    BANK + 10273,
    BANK + 11484,
];
/// Total samples across both banks of [`FreeVerb2`].
const BANK2: usize = BANK + 12863;

/// `FreeVerb.ar(in, mix, room, damp)`: the mono freeverb reverb.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FreeVerb {
    /// Circular index of each of the twelve lines.
    iota: [u32; 12],
    /// `R*_0` states: `[0..4]` the allpass delayed reads, `[4..20]` the combs' (delayed, damper) pairs.
    r0: [f32; 20],
    /// `R*_1` allpass intermediate values.
    r1: [f32; 4],
    /// `1` until the first block has zeroed the (dirty) lines.
    first: u32,
}

impl Unit for FreeVerb {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let wet = unit_clamp(ins.control(1));
        let dry = 1.0 - wet;
        let feedback = 0.7 + 0.28 * unit_clamp(ins.control(2));
        let damp = 0.4 * unit_clamp(ins.control(3));
        let damp1 = 1.0 - damp;
        let in_audio = (ins.rate(0) == Rate::Audio).then(|| ins.audio(0));
        let in_ctrl = ins.control(0);

        let out = ctx.outs.audio(0);
        let buf = ctx.aux.f32_mut();
        if buf.len() < BANK {
            out.fill(0.0);
            return DoneAction::Nothing;
        }
        if self.first != 0 {
            buf[..BANK].fill(0.0);
            self.first = 0;
        }

        let mut iota = self.iota;
        let mut r0 = self.r0;
        let mut r1 = self.r1;
        for (i, o) in out.iter_mut().enumerate() {
            let input = in_audio.map_or(in_ctrl, |s| s[i]);
            let wet_sig = process_bank(
                buf,
                &OFFSETS,
                &SIZES,
                &mut iota,
                &mut r0,
                &mut r1,
                0.015 * input,
                feedback,
                damp,
                damp1,
            );
            *o = dry * input + wet * wet_sig;
        }
        self.iota = iota;
        self.r0 = r0;
        self.r1 = r1;
        DoneAction::Nothing
    }
}

/// Constructor for [`FreeVerb`].
pub struct FreeVerbCtor;

impl UnitDef for FreeVerbCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 4 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_aux(
            FreeVerb {
                iota: [0; 12],
                r0: [0.0; 20],
                r1: [0.0; 4],
                first: 1,
            },
            BANK * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}

/// `FreeVerb2.ar(in, in2, mix, room, damp)`: the true-stereo freeverb. Both banks are excited by the
/// same `0.015 * (in + in2)` sum, but their line lengths differ by the stereo spread, so the two
/// outputs decorrelate; each output mixes its dry input channel with its bank's wet signal.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FreeVerb2 {
    /// Circular index of each of the 24 lines (`[0..12]` left bank, `[12..24]` right).
    iota: [u32; 24],
    /// The left bank's `R*_0` states.
    r0_l: [f32; 20],
    /// The right bank's `R*_0` states.
    r0_r: [f32; 20],
    /// The two banks' `R*_1` states (`[0..4]` left, `[4..8]` right).
    r1: [f32; 8],
    /// `1` until the first block has zeroed the (dirty) lines.
    first: u32,
}

impl Unit for FreeVerb2 {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let wet = unit_clamp(ins.control(2));
        let dry = 1.0 - wet;
        let feedback = 0.7 + 0.28 * unit_clamp(ins.control(3));
        let damp = 0.4 * unit_clamp(ins.control(4));
        let damp1 = 1.0 - damp;
        let in0_audio = (ins.rate(0) == Rate::Audio).then(|| ins.audio(0));
        let in0_ctrl = ins.control(0);
        let in1_audio = (ins.rate(1) == Rate::Audio).then(|| ins.audio(1));
        let in1_ctrl = ins.control(1);

        let n = ctx.outs.audio(0).len();
        let buf = ctx.aux.f32_mut();
        if buf.len() < BANK2 {
            ctx.outs.audio(0).fill(0.0);
            ctx.outs.audio(1).fill(0.0);
            return DoneAction::Nothing;
        }
        if self.first != 0 {
            buf[..BANK2].fill(0.0);
            self.first = 0;
        }

        let mut iota = self.iota;
        let mut r0_l = self.r0_l;
        let mut r0_r = self.r0_r;
        let mut r1 = self.r1;
        for i in 0..n {
            let in0 = in0_audio.map_or(in0_ctrl, |s| s[i]);
            let in1 = in1_audio.map_or(in1_ctrl, |s| s[i]);
            let scaled = 0.015 * (in0 + in1);
            let left = process_bank(
                buf,
                &OFFSETS,
                &SIZES,
                &mut iota[0..12],
                &mut r0_l,
                &mut r1[0..4],
                scaled,
                feedback,
                damp,
                damp1,
            );
            let right = process_bank(
                buf,
                &OFFSETS_R,
                &SIZES_R,
                &mut iota[12..24],
                &mut r0_r,
                &mut r1[4..8],
                scaled,
                feedback,
                damp,
                damp1,
            );
            ctx.outs.audio(0)[i] = dry * in0 + wet * left;
            ctx.outs.audio(1)[i] = dry * in1 + wet * right;
        }
        self.iota = iota;
        self.r0_l = r0_l;
        self.r0_r = r0_r;
        self.r1 = r1;
        DoneAction::Nothing
    }
}

/// Constructor for [`FreeVerb2`].
pub struct FreeVerb2Ctor;

impl UnitDef for FreeVerb2Ctor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 5 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec_aux(
            FreeVerb2 {
                iota: [0; 24],
                r0_l: [0.0; 20],
                r0_r: [0.0; 20],
                r1: [0.0; 8],
                first: 1,
            },
            BANK2 * core::mem::size_of::<f32>(),
            core::mem::align_of::<f32>(),
        ))
    }
}
