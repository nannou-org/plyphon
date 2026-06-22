//! plyphon: a pure-Rust rewrite of SuperCollider's `scsynth` audio engine core.
//!
//! This crate currently provides only a minimal placeholder: a [`SinOsc`] oscillator and a
//! [`Source`] abstraction for filling interleaved `f32` output blocks. It compiles for native and
//! `wasm32-unknown-unknown` alike with no platform-specific dependencies - the property the whole
//! engine core must preserve as it grows into the real real-time synthesis engine.
//!
//! See the project plan for the staged roadmap: the next milestone replaces this placeholder with
//! the lock-free `World`/`Controller` engine, a `Ugen` trait, a node tree, and SynthDef
//! instantiation - all without `unsafe` or global mutable state, hence the crate-wide
//! `forbid(unsafe_code)` below.

#![forbid(unsafe_code)]

use core::f32::consts::TAU;

/// Anything that can fill an interleaved, `channels`-wide block of `f32` output samples.
///
/// This is the seed of the engine's host-facing interface: a host (e.g. a `cpal` callback) hands
/// us an interleaved output buffer to fill, one block at a time.
pub trait Source {
    /// Fill `output` (interleaved, `channels` samples per frame) with the next block of audio.
    fn fill(&mut self, output: &mut [f32], channels: usize);
}

/// A sine oscillator driven by a normalised phase accumulator.
///
/// `phase` advances in `[0, 1)` cycles per sample; the emitted sample is
/// `sin(phase * TAU) * amplitude`. This is a deliberately simple placeholder - the real engine's
/// `SinOsc` will port scsynth's wavetable + fixed-point phase implementation.
#[derive(Clone, Debug)]
pub struct SinOsc {
    /// Current phase in cycles, kept within `[0, 1)`.
    phase: f32,
    /// Phase increment per sample: `freq / sample_rate`.
    phase_inc: f32,
    /// Peak amplitude of the emitted sine.
    amplitude: f32,
}

impl SinOsc {
    /// Create a sine oscillator at `freq` Hz for the given `sample_rate`, with peak `amplitude`.
    pub fn new(freq: f32, sample_rate: f32, amplitude: f32) -> Self {
        SinOsc {
            phase: 0.0,
            phase_inc: freq / sample_rate,
            amplitude,
        }
    }

    /// Advance one sample and return its value.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let value = (self.phase * TAU).sin() * self.amplitude;
        self.phase = (self.phase + self.phase_inc).fract();
        value
    }
}

impl Source for SinOsc {
    fn fill(&mut self, output: &mut [f32], channels: usize) {
        for frame in output.chunks_mut(channels.max(1)) {
            let value = self.next_sample();
            for sample in frame.iter_mut() {
                *sample = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_is_bounded_and_nonzero() {
        let mut osc = SinOsc::new(440.0, 48_000.0, 0.2);
        let mut buf = [0.0f32; 256];
        osc.fill(&mut buf, 2);
        assert!(buf.iter().all(|s| s.abs() <= 0.2 + 1e-6));
        assert!(buf.iter().any(|s| s.abs() > 0.0));
    }

    #[test]
    fn stereo_frames_are_identical_across_channels() {
        let mut osc = SinOsc::new(440.0, 48_000.0, 0.5);
        let mut buf = [0.0f32; 64];
        osc.fill(&mut buf, 2);
        for frame in buf.chunks(2) {
            assert_eq!(frame[0], frame[1]);
        }
    }
}
