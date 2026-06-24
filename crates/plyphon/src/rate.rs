//! Calculation rates and the derived per-block constants units compute with.
//!
//! SuperCollider overloads the word "Rate" for two distinct things; plyphon splits them:
//! [`Rate`] is the per-wire calculation rate (SC's `calc_*Rate` enum), while [`RateInfo`] is the
//! struct of derived constants for a given sample rate and block size (SC's `struct Rate`).

use core::f64::consts::TAU;

/// The calculation rate of a unit output or input wire (SC's `calc_ScalarRate` etc.).
///
/// Demand rate is intentionally omitted until demand-rate units are ported.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Rate {
    /// Computed once at construction; constant for the synth's lifetime.
    Scalar,
    /// One value per control block.
    Control,
    /// One value per sample (`block_size` values per control block).
    Audio,
}

/// Derived per-block constants for a given sample rate and block size.
///
/// This is the struct SC also calls `Rate`. It is owned by the engine and lent to units through
/// [`crate::unit::ProcessCtx`] - plyphon keeps no global rate/wavetable state.
#[derive(Copy, Clone, Debug)]
pub struct RateInfo {
    /// Samples per second.
    pub sample_rate: f64,
    /// Seconds per sample (`1 / sample_rate`).
    pub sample_dur: f64,
    /// Control blocks per second (`sample_rate / block_size`).
    pub buf_rate: f64,
    /// Seconds per control block (`block_size / sample_rate`).
    pub buf_dur: f64,
    /// Control-to-audio interpolation slope factor (`1 / block_size`).
    pub slope_factor: f64,
    /// Radians advanced per sample at 1 Hz (`TAU / sample_rate`).
    pub radians_per_sample: f64,
    /// Samples per control block.
    pub block_size: usize,
}

impl RateInfo {
    /// Derive the constants for `sample_rate` (Hz) and `block_size` (samples per control block).
    pub fn new(sample_rate: f64, block_size: usize) -> Self {
        let bs = block_size as f64;
        RateInfo {
            sample_rate,
            sample_dur: 1.0 / sample_rate,
            buf_rate: sample_rate / bs,
            buf_dur: bs / sample_rate,
            slope_factor: 1.0 / bs,
            radians_per_sample: TAU / sample_rate,
            block_size,
        }
    }
}
