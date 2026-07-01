//! Selection and buffer-indexing units - plyphon's ports of scsynth's `Select`, `Index`, `IndexL`,
//! `WrapIndex` and `FoldIndex` (`OscUGens.cpp`).
//!
//! `Select` passes through one of its trailing signal inputs, chosen by an index. The `Index` family
//! instead reads a single value out of a `/b_alloc`'d buffer used as a lookup table, differing only in
//! how an out-of-range or fractional index is treated: `Index` clips to the nearest whole slot,
//! `IndexL` linearly interpolates between slots, and `WrapIndex`/`FoldIndex` wrap/fold the whole index
//! back into range. All read their index at whatever rate the SynthDef assigns and output at the
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
