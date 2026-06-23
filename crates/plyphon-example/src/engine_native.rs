//! Native engine wiring: the demo's `Controls` plus the audio `World`.
//!
//! Identical to the web wiring today. Kept separate so the native path can diverge later (e.g.
//! richer device/driver handling) without disturbing the web build.

/// Create the native engine: the control plane plus the audio `World`.
pub fn new(sample_rate: f32, channels: usize) -> (crate::demo::Controls, plyphon::World) {
    crate::demo::build(sample_rate, channels)
}
