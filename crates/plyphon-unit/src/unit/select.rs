//! Selection and buffer-lookup units - plyphon's ports of scsynth's `Select`, `TWindex`, the
//! `Index`/`IndexL`/`WrapIndex`/`FoldIndex` family, `IndexInBetween`, `DetectIndex`, `Shaper` and
//! `DegreeToKey` (`OscUGens.cpp`).
//!
//! `Select` passes through one of its trailing signal inputs, chosen by an index. `TWindex` chooses an
//! index at random on each trigger, weighted by its trailing inputs. The rest read a value
//! out of a `/b_alloc`'d buffer: the `Index` family treats the buffer as a lookup table indexed by
//! `in` (differing only in how an out-of-range or fractional index is treated - `Index` clips, `IndexL`
//! interpolates, `WrapIndex`/`FoldIndex` wrap/fold); `IndexInBetween` and `DetectIndex` search the
//! table for `in` and output where it falls or where it is; `Shaper` treats it as a `(a, b)`-format
//! transfer function and waveshapes `in`; and `DegreeToKey` treats it as a scale and maps a degree to
//! a key with octave wrapping. All read their index/signal at whatever rate the SynthDef assigns and
//! output at the unit's own rate.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::{buffer_at, sample_channel};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, Inputs, LocalBufs, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::interp::lininterp;
use plyphon_dsp::math;
use plyphon_dsp::ops;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::rng::Rng;
use plyphon_dsp::wavetable::shape_wavetable;

/// The input index a `Select` reads for selector `which`: truncate toward zero, then clamp into
/// `1..=num_inputs - 1`. The increment wraps like scsynth's `(int32)in + 1`, so an out-of-range
/// selector picks the first input.
fn select_index(which: f32, num_inputs: usize) -> usize {
    let maxindex = (num_inputs as i32 - 1).max(1);
    (which as i32).wrapping_add(1).clamp(1, maxindex) as usize
}

/// `Select.ar/kr(which, array)`: outputs the `array` input selected by `which` (rounded and clamped
/// into range). Input `0` is `which`; inputs `1..` are the selectable signals.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Select {
    audio: u32,
}

impl Unit for Select {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let ins = ctx.ins; // `Copy`; its slices are `'a`, so it coexists with the `&mut` output.
        drive(ctx, audio, |i| {
            let index = select_index(sample_channel(&ins, 0, i), ins.len());
            sample_channel(&ins, index, i)
        });
        DoneAction::Nothing
    }
}

/// Constructor for [`Select`].
pub struct SelectCtor;

impl UnitDef for SelectCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // Needs `which` plus at least one selectable input.
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Select {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `TWindex.ar/kr(in, array, normalize)`: on each rising edge of the trigger `in`, a random index
/// into `array` chosen with probability proportional to its weights, held between triggers. Input
/// `0` is `in`, `1` is `normalize`, `2..` are the weights.
///
/// A draw scales one uniform value by the weights' total (their sum when `normalize` is exactly
/// `1`, else `1`) and picks the first weight at which the running sum reaches it. When no running
/// sum reaches it - weights summing below `1` without `normalize`, or a `NaN` weight - the index is
/// the unit's input count, as in scsynth. Weights and `normalize` are read once per block, as is
/// the total, however many triggers the block holds.
///
/// The constructor draws the first index and outputs it, then treats the trigger as high, so a
/// trigger already high on the first sample does not draw again and the first block starts with
/// the constructor's index. The trigger is edge-detected per sample when it is audio-rate.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TWindex {
    /// The index chosen on the last trigger (scsynth's `m_prevIndex`).
    prev_index: i32,
    /// The previous trigger value (scsynth's `m_trig`).
    prev_trig: f32,
    /// The weights' total for this block, negative until the first draw computes it (scsynth's
    /// `m_maxSum`).
    max_sum: f32,
}

impl TWindex {
    const TRIG: usize = 0;
    const NORMALIZE: usize = 1;
    const WEIGHTS: usize = 2;

    /// Draw a new index (scsynth's `TWindex_chooseNewIndex`), computing this block's weight total
    /// first if it has not been yet.
    fn choose(&mut self, ins: &Inputs<'_>, rgen: &mut Rng) -> i32 {
        let max_index = ins.len() as i32;
        let mut index = max_index;
        let normalize = ins.control(Self::NORMALIZE);
        let mut max_sum = self.max_sum;
        if max_sum < 0.0 {
            max_sum = 0.0;
            if normalize == 1.0 {
                for k in Self::WEIGHTS..ins.len() {
                    max_sum += ins.control(k);
                }
            } else {
                max_sum = 1.0;
            }
            self.max_sum = max_sum;
        }
        let max = max_sum * rgen.next_unipolar();
        let mut sum = 0.0f32;
        for k in Self::WEIGHTS..ins.len() {
            sum += ins.control(k);
            if sum >= max {
                index = (k - Self::WEIGHTS) as i32;
                break;
            }
        }
        index
    }
}

impl Unit for TWindex {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        self.max_sum = -1.0;
        let index = self.choose(&ctx.ins, ctx.rgen);
        *ctx.outs.control(0) = index as f32;
        self.prev_index = index;
        self.prev_trig = 1.0;
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ProcessCtx {
            ins, outs, rgen, ..
        } = ctx;
        self.max_sum = -1.0;
        let trig = sig(ins, Self::TRIG);
        // The calc length: one sample at control rate, a block at audio rate. A control-rate trigger
        // is the same at every sample, so only the first can be an edge (`TWindex_next_k`); an
        // audio-rate one is checked per sample (`TWindex_next_a`).
        for (i, o) in outs.audio(0).iter_mut().enumerate() {
            let cur = trig.at(i);
            if cur > 0.0 && self.prev_trig <= 0.0 {
                self.prev_index = self.choose(ins, rgen);
            }
            *o = self.prev_index as f32;
            self.prev_trig = cur;
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`TWindex`].
pub struct TWindexCtor;

impl UnitDef for TWindexCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        // Needs `trig` and `normalize`; the weights may be empty.
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(TWindex {
            prev_index: 0,
            prev_trig: 0.0,
            max_sum: -1.0,
        }))
    }
}

/// How an [`Index`] unit treats a raw index against the table bounds.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum IndexMode {
    /// Clip to the nearest in-range whole slot (`Index`).
    Clip,
    /// Linearly interpolate between adjacent slots (`IndexL`).
    Lin,
    /// Wrap the whole index back into range (`WrapIndex`).
    Wrap,
    /// Fold the whole index back into range (`FoldIndex`).
    Fold,
}

impl IndexMode {
    fn to_tag(self) -> u32 {
        match self {
            IndexMode::Clip => 0,
            IndexMode::Lin => 1,
            IndexMode::Wrap => 2,
            IndexMode::Fold => 3,
        }
    }

    fn from_tag(tag: u32) -> IndexMode {
        match tag {
            1 => IndexMode::Lin,
            2 => IndexMode::Wrap,
            3 => IndexMode::Fold,
            _ => IndexMode::Clip,
        }
    }
}

/// Look up `findex` in `table` per [`IndexMode`]; `0.0` for an empty/missing table.
fn index_table(table: &[f32], mode: IndexMode, findex: f32) -> f32 {
    if table.is_empty() {
        return 0.0;
    }
    let max = (table.len() - 1) as i32;
    match mode {
        IndexMode::Clip => table[(findex as i32).clamp(0, max) as usize],
        IndexMode::Wrap => table[ops::iwrap(findex as i32, 0, max) as usize],
        IndexMode::Fold => table[ops::ifold(findex as i32, 0, max) as usize],
        IndexMode::Lin => {
            let i1 = (findex as i32).clamp(0, max);
            let i2 = (i1 + 1).clamp(0, max);
            let frac = findex - math::floor(findex);
            lininterp(frac, table[i1 as usize], table[i2 as usize])
        }
    }
}

/// `Index/IndexL/WrapIndex/FoldIndex.ar/kr(bufnum, in)`: reads the buffer `bufnum` as a lookup table,
/// indexed by `in`. Input `0` is `bufnum`; input `1` is the index signal.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Index {
    /// The [`IndexMode`] tag.
    mode: u32,
    audio: u32,
}

impl Index {
    const BUF: usize = 0;
    const INDEX: usize = 1;
}

impl Unit for Index {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let mode = IndexMode::from_tag(self.mode);
        let ins = ctx.ins;
        let bufnum = ins.control(Self::BUF).max(0.0) as usize;
        let idx = sig(&ins, Self::INDEX);
        // The table (`ctx.buffers`) and the output (`ctx.outs`) are disjoint `ctx` fields, so both
        // borrows coexist; a missing buffer yields an empty table (silent output).
        let table = buffer_at(ctx.buffers, &ctx.local_bufs, bufnum)
            .map(|b| b.data())
            .unwrap_or(&[]);
        if audio {
            for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
                *o = index_table(table, mode, idx.at(i));
            }
        } else {
            *ctx.outs.control(0) = index_table(table, mode, idx.at(0));
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Index`] and its variants, parameterized by [`IndexMode`].
pub struct IndexCtor(pub IndexMode);

impl UnitDef for IndexCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Index {
            mode: self.0.to_tag(),
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `Shaper.ar/kr(bufnum, in)`: waveshapes `in` (nominally in `[-1, 1]`) through the transfer function
/// stored in buffer `bufnum`, read in scsynth's `(a, b)` wavetable format - fill it with
/// `/b_gen cheby … wavetable` for a Chebyshev waveshaper. Input `0` is `bufnum`; input `1` is the
/// signal to shape.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Shaper {
    audio: u32,
}

impl Shaper {
    const BUF: usize = 0;
    const IN: usize = 1;
}

impl Unit for Shaper {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let ins = ctx.ins;
        let bufnum = ins.control(Self::BUF).max(0.0) as usize;
        let input = sig(&ins, Self::IN);
        let table = buffer_at(ctx.buffers, &ctx.local_bufs, bufnum)
            .map(|b| b.data())
            .unwrap_or(&[]);
        if audio {
            for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
                *o = shape_wavetable(table, input.at(i));
            }
        } else {
            *ctx.outs.control(0) = shape_wavetable(table, input.at(0));
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Shaper`].
pub struct ShaperCtor;

impl UnitDef for ShaperCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(Shaper {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// Map scale-degree `findex` (floored to an integer) through the scale `table` to a key, transposing by
/// `octave` per whole octave wrapped. Uses Euclidean modulo/division so negative degrees wrap correctly
/// (a small correctness fix over scsynth's C `%`, which mishandles exact octave multiples). `0.0` for an
/// empty table.
fn degree_to_key(table: &[f32], findex: f32, octave: f32) -> f32 {
    if table.is_empty() {
        return 0.0;
    }
    let n = table.len() as i32;
    let degree = math::floor(findex) as i32;
    let key = degree.rem_euclid(n) as usize;
    let oct = degree.div_euclid(n);
    table[key] + octave * oct as f32
}

/// `DegreeToKey.ar/kr(bufnum, in, octave)`: reads buffer `bufnum` as a scale table and maps the
/// scale-degree `in` (floored) to a key, transposing whole octaves by `octave` (default 12). Input `0`
/// is `bufnum`, `1` the degree signal, `2` the octave interval.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DegreeToKey {
    audio: u32,
}

impl DegreeToKey {
    const BUF: usize = 0;
    const IN: usize = 1;
    const OCTAVE: usize = 2;
}

impl Unit for DegreeToKey {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let audio = self.audio != 0;
        let ins = ctx.ins;
        let bufnum = ins.control(Self::BUF).max(0.0) as usize;
        let octave = if ins.len() > Self::OCTAVE {
            ins.control(Self::OCTAVE)
        } else {
            12.0
        };
        let degree = sig(&ins, Self::IN);
        let table = buffer_at(ctx.buffers, &ctx.local_bufs, bufnum)
            .map(|b| b.data())
            .unwrap_or(&[]);
        if audio {
            for (i, o) in ctx.outs.audio(0).iter_mut().enumerate() {
                *o = degree_to_key(table, degree.at(i), octave);
            }
        } else {
            *ctx.outs.control(0) = degree_to_key(table, degree.at(0), octave);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DegreeToKey`].
pub struct DegreeToKeyCtor;

impl UnitDef for DegreeToKeyCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DegreeToKey {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// The table buffer `bufnum` reads, as scsynth's `UnitGetTable` resolves it (`(uint32)bufnum`), or
/// `None` when there is none - the caller then clears its output for the calc and leaves its state
/// alone, as `GET_TABLE` returns early after `ClearUnitOutputs`.
fn get_table<'a>(
    ins: &Inputs<'_>,
    buffers: &'a BufferTable,
    local_bufs: &'a LocalBufs<'_>,
) -> Option<&'a [f32]> {
    let bufnum = ins.control(0) as u32 as usize;
    buffer_at(buffers, local_bufs, bufnum).map(|buf| buf.data())
}

/// `IndexInBetween_FindIndex`: the fractional index at which `x` falls in the ascending `table` -
/// the first entry greater than `x` and the one before it, interpolated linearly - clamped to 0
/// below the first entry and to the last index when no entry exceeds `x`.
fn index_in_between(table: &[f32], x: f32) -> f32 {
    let maxindex = table.len() as i32 - 1;
    for i in 0..=maxindex {
        let hi = table[i as usize];
        if hi > x {
            if i == 0 {
                return 0.0;
            }
            let lo = table[i as usize - 1];
            // `(in - lo) / (hi - lo) + i - 1`, evaluated left to right in single precision.
            return (x - lo) / (hi - lo) + i as f32 - 1.0;
        }
    }
    maxindex as f32
}

/// `IndexInBetween.ar/kr(bufnum, in)`: the inverse of `IndexL` - finds where `in` falls in the
/// ascending table in buffer `bufnum` and outputs that fractional index. Input `0` is `bufnum`;
/// input `1` the value to look up.
///
/// A direct port of scsynth's `IndexInBetween`: at control rate it looks up the first sample of
/// `in` (`next_1`); at audio rate every sample of an audio-rate `in` (`next_a`), or the one value of
/// any other `in` for the whole block (`next_k`). A missing buffer outputs zero.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct IndexInBetween {
    audio: u32,
}

impl IndexInBetween {
    const IN: usize = 1;
}

impl Unit for IndexInBetween {
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let Some(table) = get_table(&ins, ctx.buffers, &ctx.local_bufs) else {
            ctx.outs.audio(0).fill(0.0);
            return DoneAction::Nothing;
        };
        if self.audio != 0 && ins.rate(Self::IN) == Rate::Audio {
            for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ins.audio(Self::IN)) {
                *o = index_in_between(table, x);
            }
        } else {
            let value = index_in_between(table, ins.control(Self::IN));
            ctx.outs.audio(0).fill(value);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`IndexInBetween`].
pub struct IndexInBetweenCtor;

impl UnitDef for IndexInBetweenCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(IndexInBetween {
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}

/// `DetectIndex_FindIndex`: the index of the first entry of `table` equal to `x`, or `-1`.
fn detect_index(table: &[f32], x: f32) -> i32 {
    table
        .iter()
        .position(|&v| v == x)
        .map_or(-1, |index| index as i32)
}

/// `DetectIndex.ar/kr(bufnum, in)`: the index of the first entry of the table in buffer `bufnum`
/// equal to `in`, or `-1` if there is none. Input `0` is `bufnum`; input `1` the value to find.
///
/// A direct port of scsynth's `DetectIndex`, which searches again only when `in` changes: the last
/// input and its index persist across blocks (the index as a float, as scsynth keeps it). At
/// control rate it looks up the first sample of `in` (`next_1`); at audio rate every sample of an
/// audio-rate `in` (`next_a`), or the one value of any other `in` for the whole block (`next_k`). A
/// missing buffer outputs zero and leaves the remembered input alone.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DetectIndex {
    /// The last index found (scsynth's `mPrev`).
    prev: f32,
    /// The input it was found for (scsynth's `mPrevIn`), `NaN` before the first search.
    prev_in: f32,
    audio: u32,
}

impl DetectIndex {
    const IN: usize = 1;

    /// `next_1`/`next_k`: the index for the one value `x`, searching only if it changed.
    fn lookup(&mut self, table: &[f32], x: f32) -> f32 {
        let index = if x == self.prev_in {
            self.prev as i32
        } else {
            let index = detect_index(table, x);
            self.prev = index as f32;
            self.prev_in = x;
            index
        };
        index as f32
    }
}

impl Unit for DetectIndex {
    fn init(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        // `DetectIndex_next_1(unit, 1)`.
        let ins = ctx.ins;
        *ctx.outs.control(0) = match get_table(&ins, ctx.buffers, &ctx.local_bufs) {
            Some(table) => self.lookup(table, ins.control(Self::IN)),
            None => 0.0,
        };
        DoneAction::Nothing
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        let ins = ctx.ins;
        let Some(table) = get_table(&ins, ctx.buffers, &ctx.local_bufs) else {
            ctx.outs.audio(0).fill(0.0);
            return DoneAction::Nothing;
        };
        if self.audio != 0 && ins.rate(Self::IN) == Rate::Audio {
            // `DetectIndex_next_a`.
            let mut prev = self.prev_in;
            let mut prev_index = self.prev as i32;
            for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ins.audio(Self::IN)) {
                if x != prev {
                    prev_index = detect_index(table, x);
                }
                prev = x;
                *o = prev_index as f32;
            }
            self.prev = prev_index as f32;
            self.prev_in = prev;
        } else {
            let value = self.lookup(table, ins.control(Self::IN));
            ctx.outs.audio(0).fill(value);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`DetectIndex`].
pub struct DetectIndexCtor;

impl UnitDef for DetectIndexCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        Ok(unit_spec(DetectIndex {
            prev: -1.0,
            // Ensures the first input differs from it.
            prev_in: f32::NAN,
            audio: (ctx.rate == Rate::Audio) as u32,
        }))
    }
}
