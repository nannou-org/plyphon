//! `DNoiseRing` - a demand-rate rotating bit-ring noise source.
//!
//! The state transition is based on NoiseRing 1.0 by Julian Parker and Till Bovermann,
//! Copyright 2013, licensed under GPL-2.0-or-later. Plyphon defines finite conversion, reset
//! aliasing, and lifecycle behavior that the original host leaves undefined.

use bytemuck::{Pod, Zeroable};

use crate::error::BuildError;
use crate::unit::demand::{BuiltDemandUnit, DemandCtx, DemandUnit, demand_unit_spec};
use crate::unit::registry::{BuildContext, DemandUnitDef};
#[cfg(not(feature = "std"))]
use plyphon_dsp::math::Real;
use plyphon_dsp::rate::Rate;

/// The default value remembered for each runtime control before its first finite pull.
const DEFAULT_CHANGE: f32 = 0.5;
const DEFAULT_CHANCE: f32 = 0.5;
const DEFAULT_SHIFT: f32 = 1.0;
const DEFAULT_NUM_BITS: f32 = 8.0;
const DEFAULT_RESET: f32 = 0.0;

/// A complete, sanitized set of controls for one successful production step.
struct Controls {
    change: f32,
    chance: f32,
    shift: f32,
    num_bits: f32,
    reset: f32,
}

/// `DNoiseRing(change, chance, shift, numBits, resetval)`.
///
/// The five inputs are pulled in order. The ring lazily initializes from `resetval` on the first
/// successful pull, then rotates right and optionally replaces bit zero using the synth-shared
/// random stream. All state is fixed-size and every pull/reset is allocation-free.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct DNoiseRing {
    state: u32,
    initialized: u32,
    last_change: f32,
    last_chance: f32,
    last_shift: f32,
    last_num_bits: f32,
    last_reset: f32,
}

impl DNoiseRing {
    const CHANGE: usize = 0;
    const CHANCE: usize = 1;
    const SHIFT: usize = 2;
    const NUM_BITS: usize = 3;
    const RESET: usize = 4;

    /// Return a finite input value, falling back to the remembered value for a non-demand
    /// non-finite input. A `NaN` from a nested demand source is exhaustion.
    fn pull(ctx: &mut DemandCtx<'_>, input: usize, remembered: f32) -> Option<f32> {
        let value = ctx.demand(input);
        if value.is_nan() && ctx.is_demand(input) {
            None
        } else if value.is_finite() {
            Some(value)
        } else {
            Some(remembered)
        }
    }

    /// Pull and sanitize all controls without mutating this unit. This makes exhaustion at a later
    /// inlet commit child effects but leave the outer ring and its remembered controls unchanged.
    fn controls(&self, ctx: &mut DemandCtx<'_>) -> Option<Controls> {
        Some(Controls {
            change: Self::pull(ctx, Self::CHANGE, self.last_change)?.clamp(0.0, 1.0),
            chance: Self::pull(ctx, Self::CHANCE, self.last_chance)?.clamp(0.0, 1.0),
            shift: Self::pull(ctx, Self::SHIFT, self.last_shift)?,
            num_bits: Self::pull(ctx, Self::NUM_BITS, self.last_num_bits)?,
            reset: Self::pull(ctx, Self::RESET, self.last_reset)?,
        })
    }

    /// Truncate a finite float toward zero using Rust's saturating float-to-integer conversion.
    fn trunc_i64(value: f32) -> i64 {
        value.trunc() as i64
    }

    /// Sanitize a finite `numBits` value to the inclusive range `1..=32`.
    fn num_bits(value: f32) -> u32 {
        Self::trunc_i64(value).clamp(1, 32) as u32
    }

    /// Sanitize a finite shift using signed Euclidean remainder at the selected ring width.
    fn shift(value: f32, num_bits: u32) -> u32 {
        Self::trunc_i64(value).rem_euclid(num_bits as i64) as u32
    }

    /// The mask for a ring of `num_bits`.
    fn mask(num_bits: u32) -> u32 {
        if num_bits == 32 {
            u32::MAX
        } else {
            (1u32 << num_bits) - 1
        }
    }

    /// Clamp, truncate, and mask a finite reset value.
    fn reset_value(value: f32, mask: u32) -> u32 {
        value.clamp(0.0, u32::MAX as f32) as u32 & mask
    }

    /// Rotate the masked ring right within its selected width.
    fn rotate(state: u32, shift: u32, num_bits: u32, mask: u32) -> u32 {
        if shift == 0 {
            state & mask
        } else {
            ((state >> shift) | (state << (num_bits - shift))) & mask
        }
    }

    /// Commit the successfully pulled finite-control memories.
    fn remember(&mut self, controls: &Controls) {
        self.last_change = controls.change;
        self.last_chance = controls.chance;
        self.last_shift = controls.shift;
        self.last_num_bits = controls.num_bits;
        self.last_reset = controls.reset;
    }
}

impl DemandUnit for DNoiseRing {
    /// Resets nested demand inputs and reloads the finite ring seed.
    fn reset(&mut self, ctx: &mut DemandCtx<'_>) {
        for input in Self::CHANGE..=Self::NUM_BITS {
            ctx.reset(input);
        }

        let Some(reset) = Self::pull(ctx, Self::RESET, self.last_reset) else {
            return;
        };
        let num_bits = Self::num_bits(self.last_num_bits);
        let mask = Self::mask(num_bits);
        self.state = Self::reset_value(reset, mask);
        self.initialized = 1;
        if reset.is_finite() {
            self.last_reset = reset;
        }
    }

    /// Pulls one complete control set and advances the rotating ring once.
    fn produce(&mut self, ctx: &mut DemandCtx<'_>) -> f32 {
        let Some(controls) = self.controls(ctx) else {
            return f32::NAN;
        };

        let num_bits = Self::num_bits(controls.num_bits);
        let shift = Self::shift(controls.shift, num_bits);
        let mask = Self::mask(num_bits);

        if self.initialized == 0 {
            self.state = Self::reset_value(controls.reset, mask);
            self.initialized = 1;
        }
        self.remember(&controls);

        let mut state = Self::rotate(self.state & mask, shift, num_bits, mask);
        if ctx.random_unipolar() < controls.change {
            if ctx.random_unipolar() < controls.chance {
                state |= 1;
            } else {
                state &= !1;
            }
        }
        self.state = state & mask;
        self.state as f32
    }
}

/// Constructor for [`DNoiseRing`].
pub struct DNoiseRingCtor;

impl DemandUnitDef for DNoiseRingCtor {
    /// Validates the demand-rate ABI and constructs the fixed-size ring state.
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
        Ok(demand_unit_spec(DNoiseRing {
            state: 0,
            initialized: 0,
            last_change: DEFAULT_CHANGE,
            last_chance: DEFAULT_CHANCE,
            last_shift: DEFAULT_SHIFT,
            last_num_bits: DEFAULT_NUM_BITS,
            last_reset: DEFAULT_RESET,
        }))
    }
}
