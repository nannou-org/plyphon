//! Web audio source: a [`plyphon::World`] playing the demo sine.
//!
//! Identical to the native source today. Kept separate so the web path can grow an
//! AudioWorklet/SharedArrayBuffer backend independently of the native build.

/// Create the web source: a `World` already playing the demo sine.
pub fn new(sample_rate: f32, channels: usize) -> plyphon::World {
    crate::sine::build_world(sample_rate, channels)
}
