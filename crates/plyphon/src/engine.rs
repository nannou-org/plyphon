//! Engine construction: the [`Options`] and the [`engine`] function that wires a paired
//! [`Controller`] (control side) and [`World`] (RT side) together over the command/trash rings.

use rtrb::RingBuffer;

use crate::command::{Command, Trash};
use crate::controller::Controller;
use crate::rate::RateInfo;
use crate::world::World;

/// The client id of the always-present root group.
pub const ROOT_GROUP_ID: i32 = 0;

/// Engine configuration.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Audio sample rate in Hz.
    pub sample_rate: f64,
    /// Samples per control block (scsynth's `mBufLength`, typically 64).
    pub block_size: usize,
    /// Number of output bus channels.
    pub output_channels: usize,
    /// Maximum number of live nodes (sizes the node arena and id map; never exceeded at runtime).
    pub max_nodes: usize,
    /// Capacity of the control -> RT command ring.
    pub command_capacity: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            sample_rate: 48_000.0,
            block_size: 64,
            output_channels: 2,
            max_nodes: 1024,
            command_capacity: 1024,
        }
    }
}

/// Build a paired [`Controller`] and [`World`] from `options`.
///
/// The `Controller` stays on the control side and the `World` moves to the audio thread; they share
/// only the two lock-free rings created here.
pub fn engine(options: Options) -> (Controller, World) {
    let (cmd_tx, cmd_rx) = RingBuffer::<Command>::new(options.command_capacity.max(1));
    let (trash_tx, trash_rx) = RingBuffer::<Trash>::new(options.max_nodes.max(1));

    let audio = RateInfo::new(options.sample_rate, options.block_size);
    // Control rate: one value per control block.
    let control = RateInfo::new(options.sample_rate / options.block_size as f64, 1);

    let world = World::new(&options, audio, control, cmd_rx, trash_tx);
    let controller = Controller::new(audio, control, cmd_tx, trash_rx);
    (controller, world)
}
