//! Web engine wiring: the demo's `Controls` plus the audio `World`.
//!
//! Identical to the native wiring today. Kept separate so the web path can grow an
//! AudioWorklet/SharedArrayBuffer backend independently of the native build.

/// Create the web engine: the control plane plus the audio `World`.
pub fn new(sample_rate: f32, channels: usize) -> (crate::demo::Controls, plyphon::World) {
    crate::demo::build(sample_rate, channels)
}
