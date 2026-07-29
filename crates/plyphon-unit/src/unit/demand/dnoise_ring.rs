//! `DNoiseRing` - a demand-rate rotating bit-ring noise source.
//!
//! The state transition is based on NoiseRing 1.0 by Julian Parker and Till Bovermann,
//! Copyright 2013, licensed under GPL-2.0-or-later. Defined source behavior is preserved; Rust's
//! saturating float conversion and a no-op rotation safely replace out-of-range C++ conversions
//! and shifts.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
use plyphon_dsp::rate::Rate;

/// `DNoiseRing(change, chance, shift, numBits, resetval)`.
///
/// Every operation reads all five inputs in order. Constructor and reset calls read the retained
/// value of nested demand inputs after resetting them, then reset inputs zero through three again,
/// matching the source's `inNumSamples == 0` path.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DNoiseRing {
    state: u32,
}

impl DNoiseRing {
    const CHANGE: usize = 0;
    const CHANCE: usize = 1;
    const SHIFT: usize = 2;
    const NUM_BITS: usize = 3;
    const RESET: usize = 4;

    /// Convert a source float-to-unsigned cast without invoking C++ undefined behavior.
    ///
    /// Rust's cast matches truncation for every finite value in the source-defined range and
    /// saturates negative, non-finite, and out-of-range values.
    fn source_uint(value: f32) -> u32 {
        value as u32
    }

    /// Apply the source rotate expression when each C++ shift is defined.
    ///
    /// The original expression uses a signed `1 << shift`, so `shift == 31`, a shift count of 32,
    /// or `numBits - shift` underflow are outside its defined domain. Those cases leave the state
    /// unchanged.
    fn rotate(state: u32, shift: u32, num_bits: u32) -> u32 {
        let Some(left_shift) = num_bits.checked_sub(shift) else {
            return state;
        };
        if shift >= 31 || left_shift >= 32 {
            return state;
        }
        let low_bits = state & ((1u32 << shift) - 1);
        (state >> shift) | (low_bits << left_shift)
    }

    /// Run the source's constructor/reset branch.
    fn reset_from_inputs(&mut self, ctx: &mut DemandCtx<'_>) {
        let _change = ctx.reset_value(Self::CHANGE);
        let _chance = ctx.reset_value(Self::CHANCE);
        let _shift = Self::source_uint(ctx.reset_value(Self::SHIFT));
        let _num_bits = Self::source_uint(ctx.reset_value(Self::NUM_BITS));
        self.state = Self::source_uint(ctx.reset_value(Self::RESET));

        for input in Self::CHANGE..=Self::NUM_BITS {
            ctx.reset(input);
        }
    }
}

impl DemandUnit for DNoiseRing {
    /// Loads the constructor-time initial state and reproduces nested-input reset cadence.
    fn init(&mut self, ctx: &mut DemandCtx<'_>) {
        self.reset_from_inputs(ctx);
    }

    /// Reloads the ring through the source's `inNumSamples == 0` branch.
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        self.reset_from_inputs(ctx);
    }

    /// Pulls every inlet, advances the source rotate expression, and performs its one-or-two draws.
    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let change = ctx.demand(Self::CHANGE);
        let chance = ctx.demand(Self::CHANCE);
        let shift = Self::source_uint(ctx.demand(Self::SHIFT));
        let num_bits = Self::source_uint(ctx.demand(Self::NUM_BITS));
        let _initial_state = Self::source_uint(ctx.demand(Self::RESET));

        let mut state = Self::rotate(self.state, shift, num_bits);
        if ctx.random_unipolar() < change {
            if ctx.random_unipolar() < chance {
                state |= 1;
            } else {
                state &= !1;
            }
        }
        self.state = state;
        state as f32
    }
}

/// Constructor for [`DNoiseRing`].
pub(in crate::unit) struct DNoiseRingCtor;

impl DemandUnitDef for DNoiseRingCtor {
    /// Validates the demand-rate ABI and constructs the ring state.
    fn build(&self, ctx: &BuildContext<'_>) -> Result<BuiltDemandUnit, BuildError> {
        if ctx.input_rates.len() != 5 {
            return Err(BuildError::WrongInputCount);
        }
        if ctx.num_outputs != 1 {
            return Err(BuildError::WrongOutputCount {
                expected: 1,
                actual: ctx.num_outputs,
            });
        }
        if ctx.special_index != 0 {
            return Err(BuildError::UnsupportedOp(ctx.special_index));
        }
        if ctx.rate != Rate::Demand {
            return Err(BuildError::UnsupportedUnitRate);
        }
        Ok(demand_unit_spec(DNoiseRing { state: 0 }))
    }
}
