//! Native audio source: a [`plyphon::SinOsc`].
//!
//! Identical to the web source today. Kept separate so the native path can later drive the full
//! `plyphon` engine (`World`/`Controller`) without disturbing the web build.

pub use plyphon::Source;

const FREQ: f32 = 440.0;
const AMPLITUDE: f32 = 0.2;

/// Create the native sine source.
pub fn new(sample_rate: f32, _channels: usize) -> plyphon::SinOsc {
    plyphon::SinOsc::new(FREQ, sample_rate, AMPLITUDE)
}
