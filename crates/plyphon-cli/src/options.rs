//! Mapping CLI argument groups onto [`plyphon::Options`].

use plyphon::Options;

use crate::cli::EngineArgs;

/// Build engine [`Options`] from the shared engine flags, a resolved `sample_rate`, and the channel
/// counts (which come from the audio device for `server`/`play`, or from `--channels` for `render`).
///
/// The engine flags default to scsynth's values, so an unflagged invocation reproduces
/// [`Options::default`] (bar the channel/rate fields the caller resolves).
pub fn engine_options(
    engine: &EngineArgs,
    sample_rate: f64,
    output_channels: usize,
    input_channels: usize,
) -> Options {
    Options {
        sample_rate,
        block_size: engine.block_size,
        output_channels,
        input_channels,
        audio_bus_channels: engine.audio_buses,
        control_bus_channels: engine.control_buses,
        max_nodes: engine.max_nodes,
        max_buffers: engine.max_buffers,
        max_synthdefs: engine.max_synthdefs,
        pool_bytes: engine.rt_memory_kib * 1024,
        ..Options::default()
    }
}
