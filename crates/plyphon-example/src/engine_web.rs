//! Web audio source: a [`plyphon::SinOsc`].
//!
//! Identical to the native source today. Kept separate so the web path can grow an
//! AudioWorklet/SharedArrayBuffer backend independently of the native build.

pub use plyphon::Source;

const FREQ: f32 = 440.0;
const AMPLITUDE: f32 = 0.2;

/// Create the web sine source.
pub fn new(sample_rate: f32, _channels: usize) -> plyphon::SinOsc {
    plyphon::SinOsc::new(FREQ, sample_rate, AMPLITUDE)
}
