//! Engine construction: the [`engine`] function that wires a [`Controller`] (control side), [`Nrt`]
//! (NRT side), and [`World`] (RT side) together over lock-free rings.

use rtrb::RingBuffer;

use plyphon_dsp::rate::RateInfo;
use plyphon_rt::{Event, NodeMsg, Nrt, Options, Reply, TimedCommand, Trash, Trigger, World};

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
    // The reply ring carries query answers (the getters) back for the dispatcher to reassemble. Size
    // it to hold a whole `/g_queryTree` dump in one block (the World caps a dump at this many records,
    // ~4 per node); a backlog beyond that queues in the World's `pending_replies`.
    let (replies_tx, replies_rx) =
        RingBuffer::<Reply>::new(options.max_nodes.saturating_mul(4).max(1));
    // The trigger ring carries `SendTrig` `/tr`s back for the dispatcher to broadcast. It is separate
    // from the events ring so a burst of audio-rate triggers can never starve or delay node
    // lifecycle notifications; triggers beyond its capacity are dropped (best-effort, like scsynth).
    let (triggers_tx, triggers_rx) = RingBuffer::<Trigger>::new(options.max_triggers.max(1));
    // The node-message ring carries `SendReply` messages back for the dispatcher to emit as OSC. Like
    // the trigger ring it is separate and best-effort; excess is dropped when the host lags.
    let (node_msgs_tx, node_msgs_rx) = RingBuffer::<NodeMsg>::new(options.max_node_msgs.max(1));

    let audio = RateInfo::new(options.sample_rate, options.block_size);
    // Control rate: one value per control block.
    let control = RateInfo::new(options.sample_rate / options.block_size as f64, 1);

    let world = World::new(
        &options,
        audio,
        cmd_rx,
        trash_tx,
        events_tx,
        replies_tx,
        triggers_tx,
        node_msgs_tx,
    );
    let nrt = Nrt::new(trash_rx, events_rx, replies_rx, triggers_rx, node_msgs_rx);
    let controller = Controller::new(&options, audio, control, cmd_tx);
    (controller, nrt, world)
}
