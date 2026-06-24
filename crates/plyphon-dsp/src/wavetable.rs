//! Shared wavetables owned by the engine.
//!
//! SuperCollider keeps its sine table in process-global statics reached through the plugin
//! `InterfaceTable`. plyphon instead owns the tables in a [`Wavetables`] value held by the engine
//! and lends them to units by argument while they process, so there is no global
//! mutable state and multiple engines can coexist.

use core::f64::consts::TAU;

/// Number of samples in one cycle of the sine table (matching scsynth's default).
pub const SINE_SIZE: usize = 16384;

/// The wavetables shared by oscillator units.
///
/// Tables carry one guard sample (a copy of index 0) past the end so linear interpolation can read
/// `table[i + 1]` without bounds juggling.
#[derive(Clone, Debug)]
pub struct Wavetables {
    /// One cycle of a sine, `SINE_SIZE + 1` samples (`sine[i] == sin(TAU * i / SINE_SIZE)`).
    sine: Vec<f32>,
}

impl Wavetables {
    /// Build the default wavetable bank.
    pub fn new() -> Self {
        let mut sine = Vec::with_capacity(SINE_SIZE + 1);
        for i in 0..=SINE_SIZE {
            let phase = (i as f64) / (SINE_SIZE as f64) * TAU;
            sine.push(phase.sin() as f32);
        }
        Wavetables { sine }
    }

    /// One cycle of a sine with a trailing guard sample (`SINE_SIZE + 1` samples).
    pub fn sine(&self) -> &[f32] {
        &self.sine
    }
}

impl Default for Wavetables {
    fn default() -> Self {
        Self::new()
    }
}

/// Linearly interpolate `table` (a one-cycle table with a trailing guard sample) at normalised
/// `phase` in cycles. Only the fractional part of `phase` is used.
#[inline]
pub fn lookup_cycle(table: &[f32], phase: f32) -> f32 {
    let n = table.len() - 1; // last entry is the guard sample
    let frac_phase = phase - phase.floor(); // wrap into [0, 1)
    let pos = frac_phase * n as f32;
    let i = pos as usize; // 0..=n-1 (frac_phase < 1)
    let frac = pos - i as f32;
    let a = table[i];
    let b = table[i + 1];
    a + frac * (b - a)
}
