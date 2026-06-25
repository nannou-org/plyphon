//! Engine construction: the [`engine`] function that wires a [`Controller`] (control side), [`Nrt`]
//! (NRT side), and [`World`] (RT side) together over lock-free rings.

use rtrb::RingBuffer;

use plyphon_dsp::rate::RateInfo;
use plyphon_rt::{Event, Nrt, Options, TimedCommand, Trash, World};

use crate::controller::Controller;

/// Build a [`Controller`], [`Nrt`], and [`World`] from `options`.
///
/// Move the `World` to the audio thread, run the `Nrt` on an NRT thread (or timer), and keep the
/// `Controller` wherever commands originate. They share only the lock-free rings created here. See
/// the [`nrt`](plyphon_rt::nrt) module for the intended threading lifecycle.
pub fn engine(options: Options) -> (Controller, Nrt, World) {
    let (cmd_tx, cmd_rx) = RingBuffer::<TimedCommand>::new(options.command_capacity.max(1));
    // The trash ring carries freed/replaced buffers and streams (freed synths return to the pool
    // directly, on the audio thread, so they never trash).
    let (trash_tx, trash_rx) = RingBuffer::<Trash>::new(options.max_buffers.max(1));
    let (events_tx, events_rx) = RingBuffer::<Event>::new(options.max_nodes.max(1));

    let audio = RateInfo::new(options.sample_rate, options.block_size);
    // Control rate: one value per control block.
    let control = RateInfo::new(options.sample_rate / options.block_size as f64, 1);

    let world = World::new(&options, audio, control, cmd_rx, trash_tx, events_tx);
    let nrt = Nrt::new(trash_rx, events_rx);
    let controller = Controller::new(&options, audio, control, cmd_tx);
    (controller, nrt, world)
}
