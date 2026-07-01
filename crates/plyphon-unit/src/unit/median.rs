//! `Median` - plyphon's port of scsynth's running-median filter.
//!
//! Keeps the last `length` input samples in a parallel value/age array pair, always sorted by value,
//! and outputs the middle element. `length` is a compile-time constant (odd, at most 32), so the state
//! is a fixed in-struct array - no allocation. Good for removing impulsive spikes while preserving
//! edges better than an averaging low-pass.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::registry::{BuildContext, UnitDef};
use crate::unit::{BuiltUnit, DoneAction, InitCtx, ProcessCtx, Unit, unit_spec};

/// scsynth's `kMAXMEDIANSIZE`.
const MAX_MEDIAN: usize = 32;

/// `Median.ar/kr(length, in)`: the running median of the last `length` samples (`length` a constant
/// odd number, capped at 32). Input `0` is `length`; input `1` is the signal.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Median {
    /// The window's values, kept sorted ascending.
    values: [f32; MAX_MEDIAN],
    /// Each value's age in samples; the oldest (age `size - 1`) is the one replaced next.
    ages: [i32; MAX_MEDIAN],
    size: u32,
}

impl Median {
    const LENGTH: usize = 0;
    const IN: usize = 1;

    /// Insert `value` (evicting the oldest) and return the new median (scsynth's `Median_InsertMedian`).
    fn insert(&mut self, value: f32) -> f32 {
        let size = self.size as usize;
        let last = size as i32 - 1;
        // Age every bin, and find the oldest (its slot is the one we reuse).
        let mut pos: i32 = -1;
        for i in 0..size {
            if self.ages[i] == last {
                pos = i as i32;
            } else {
                self.ages[i] += 1;
            }
        }
        if pos < 0 {
            // The invariant guarantees an oldest bin; bail safely rather than index out of range.
            return self.values[size >> 1];
        }
        // Shift the open slot down while the new value is smaller than its lower neighbour, ...
        while pos != 0 && value < self.values[(pos - 1) as usize] {
            self.values[pos as usize] = self.values[(pos - 1) as usize];
            self.ages[pos as usize] = self.ages[(pos - 1) as usize];
            pos -= 1;
        }
        // ... or up while it is larger than its upper neighbour, keeping the array sorted.
        while pos != last && value > self.values[(pos + 1) as usize] {
            self.values[pos as usize] = self.values[(pos + 1) as usize];
            self.ages[pos as usize] = self.ages[(pos + 1) as usize];
            pos += 1;
        }
        self.values[pos as usize] = value;
        self.ages[pos as usize] = 0;
        self.values[size >> 1]
    }
}

impl Unit for Median {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        // Seed the window with the first input, ages ascending (scsynth's `Median_InitMedian`).
        let v = ctx.ins.control(Self::IN);
        for i in 0..self.size as usize {
            self.values[i] = v;
            self.ages[i] = i as i32;
        }
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        for (o, &x) in ctx.outs.audio(0).iter_mut().zip(ctx.ins.audio(Self::IN)) {
            *o = self.insert(x);
        }
        DoneAction::Nothing
    }
}

/// Constructor for [`Median`]. The window `length` must be a compile-time constant (as in scsynth,
/// where it is read once at ctor).
pub struct MedianCtor;

impl UnitDef for MedianCtor {
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        if ctx.input_rates.len() < 2 {
            return Err(BuildError::WrongInputCount);
        }
        let length = ctx
            .const_input(Median::LENGTH)
            .ok_or(BuildError::AuxRequiresConstant {
                input: Median::LENGTH,
            })?;
        let size = (length as i32).clamp(1, MAX_MEDIAN as i32) as u32;
        Ok(unit_spec(Median {
            values: [0.0; MAX_MEDIAN],
            ages: [0; MAX_MEDIAN],
            size,
        }))
    }
}
