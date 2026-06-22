//! Native audio source: a [`plyphon::World`] playing the demo sine.
//!
//! Identical to the web source today. Kept separate so the native path can diverge later (e.g.
//! richer device/driver handling) without disturbing the web build.

/// Create the native source: a `World` already playing the demo sine.
pub fn new(sample_rate: f32, channels: usize) -> plyphon::World {
    crate::sine::build_world(sample_rate, channels)
}
