//! Engine configuration ([`Options`]) and the always-present root group id.

/// The client id of the always-present root group.
pub const ROOT_GROUP_ID: i32 = 0;

/// Engine configuration.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Audio sample rate in Hz.
    pub sample_rate: f64,
    /// Samples per control block (scsynth's `mBufLength`, typically 64).
    pub block_size: usize,
    /// Number of hardware output bus channels (the first audio bus channels).
    pub output_channels: usize,
    /// Number of hardware input bus channels (the audio bus channels following the outputs).
    pub input_channels: usize,
    /// Number of private audio bus channels for routing between synths (after output and input).
    pub audio_bus_channels: usize,
    /// Number of control bus channels (for `In.kr`/`Out.kr`, `/c_set`, and `/n_map`).
    pub control_bus_channels: usize,
    /// Maximum number of live nodes (sizes the node arena and id map; never exceeded at runtime).
    pub max_nodes: usize,
    /// Number of buffer table slots (sizes the buffer table; indices `0..max_buffers` are valid).
    pub max_buffers: usize,
    /// Number of compiled-def table slots (sizes the resident def table; `def_id`s index it).
    pub max_synthdefs: usize,
    /// Bytes of real-time pool backing per-synth state blocks (scsynth's `mRealTimeMemorySize`).
    pub pool_bytes: usize,
    /// Max audio wires any one synth may use; sizes the World-shared wire scratch
    /// (`max_wire_bufs * block_size` f32). A def needing more fails to compile.
    pub max_wire_bufs: usize,
    /// Max outputs any one unit may have; sizes the World-shared output scratch
    /// (`max_unit_outputs * block_size` f32). A def with a wider unit fails to compile.
    pub max_unit_outputs: usize,
    /// Capacity of the control -> RT command ring.
    pub command_capacity: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            sample_rate: 48_000.0,
            block_size: 64,
            output_channels: 2,
            input_channels: 2,
            audio_bus_channels: 128,
            control_bus_channels: 4096,
            max_nodes: 1024,
            max_buffers: 1024,
            max_synthdefs: 1024,
            // 8 MiB, matching scsynth's default real-time memory size.
            pool_bytes: 8 * 1024 * 1024,
            max_wire_bufs: 1024,
            max_unit_outputs: 128,
            command_capacity: 1024,
        }
    }
}
