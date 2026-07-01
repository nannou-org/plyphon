//! Selection and buffer-lookup units - plyphon's ports of scsynth's `Select`, the `Index`/`IndexL`/
//! `WrapIndex`/`FoldIndex` family, `Shaper` and `DegreeToKey` (`OscUGens.cpp`).
//!
//! `Select` passes through one of its trailing signal inputs, chosen by an index. The rest read a value
//! out of a `/b_alloc`'d buffer: the `Index` family treats the buffer as a lookup table indexed by
//! `in` (differing only in how an out-of-range or fractional index is treated - `Index` clips, `IndexL`
//! interpolates, `WrapIndex`/`FoldIndex` wrap/fold); `Shaper` treats it as a `(a, b)`-format transfer
//! function and waveshapes `in`; and `DegreeToKey` treats it as a scale and maps a degree to a key with
//! octave wrapping. All read their index/signal at whatever rate the SynthDef assigns and output at the
//! unit's own rate.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::io::{buffer_at, sample_channel};
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::trigger::{drive, sig};
use crate::unit::{BuiltUnit, DoneAction, ProcessCtx, Unit, unit_spec};
use plyphon_dsp::interp::lininterp;
use plyphon_dsp::math;
use plyphon_dsp::ops;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::wavetable::shape_wavetable;

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
        // scsynth's `maxindex = mNumInputs - 1`; items live at inputs `1..=maxindex`.
        let maxindex = (ins.len() as i32 - 1).max(1);
        drive(ctx, audio, |i| {
            let which = sample_channel(&ins, 0, i) as i32;
            let index = (which + 1).clamp(1, maxindex) as usize;
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
        let table = buffer_at(ctx.buffers, bufnum)
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
        let table = buffer_at(ctx.buffers, bufnum)
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
        let table = buffer_at(ctx.buffers, bufnum)
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
