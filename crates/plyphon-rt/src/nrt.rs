//! The non-real-time (NRT) side of the engine - the state machine the consumer runs on an
//! associated NRT thread.
//!
//! # Why the NRT side exists
//!
//! [`World::fill`](crate::world::World::fill) must never allocate, block, take a lock, or run a
//! `Drop` on the audio thread. The [`Nrt`] is what *enables* those guarantees: it absorbs the work
//! the audio thread is forbidden from doing. Concretely it:
//!
//! - **Frees memory off the audio thread.** When a node is freed - explicitly via
//!   `Controller::free` or by a unit's
//!   [`DoneAction`](plyphon_unit::unit::DoneAction) - or a buffer is replaced or freed, the
//!   [`World`](crate::world::World) only unlinks/swaps it (O(1)) and ships its `Box` over the trash
//!   ring; the actual `Drop`/`free` happens here, in [`Nrt::process`].
//! - **Surfaces notifications.** Ownership-critical start/fail/end events and best-effort
//!   pause/resume/move events flow over isolated rings. Consumers can merge them with [`Nrt::poll`]
//!   or drain them by class.
//!
//! Buffers follow the same off-RT→RT→off-RT pattern as synths: a buffer is built off the audio
//! thread, the `World` swaps it into its table on the audio thread (O(1)), and the replaced buffer
//! is dropped here. *Loading* sample data from storage (sound files, key-value stores, the network)
//! is deliberately left to the application - see [`Buffer`](plyphon_dsp::buffer::Buffer) and
//! `Controller::buffer_set` - so it can use whatever
//! native or web I/O it likes; plyphon only ever installs a finished buffer.
//!
//! # Lifecycle
//!
//! The engine builder returns `(Controller, Nrt, World)`. Move the `World` to the
//! audio thread, run the `Nrt` on a thread (or timer, or even the same thread as the `Controller`)
//! of your choosing, and keep the `Controller` wherever commands originate:
//!
//! ```text
//!   control thread           NRT thread              audio thread
//!   ┌────────────┐  commands                         ┌──────────┐
//!   │ Controller │ ───────────────── ring ─────────▶ │  World   │ ◀─ audio callback: World::fill
//!   └────────────┘                                   └────┬─────┘
//!                           ┌──────────┐  trash + events       │
//!                           │   Nrt    │ ◀──── rings ──────────┘
//!                           └──────────┘
//!                       Nrt::process()  (drop trash)
//!                       Nrt::poll()     (drain events)
//! ```
//!
//! Each NRT tick: call [`Nrt::process`] and drain either [`Nrt::poll`] or both isolated event polls
//! in loops.
//!
//! **Shutdown ordering matters.** Stop the audio thread first so the `World` is no longer driven.
//! Then keep ticking the `Nrt` until [`Nrt::process`] returns `0`, so every freed `Box` still in
//! flight is dropped here rather than on the audio thread. Whoever finally drops the `World` drops
//! the live node tree (and its synths), so do that off the audio thread once it has stopped.

use rtrb::Consumer;

use crate::command::{Event, Reply, StampedEvent, Trash};
use plyphon_unit::unit::{NodeMsg, Trigger};

/// The NRT-side state machine: drops freed synths, surfaces node notifications, surfaces query
/// answers, and surfaces `SendTrig` triggers and `SendReply` messages.
pub struct Nrt {
    trash_rx: Consumer<Trash>,
    /// Lossless ownership-critical start/fail/end stream.
    critical_rx: Consumer<StampedEvent>,
    /// Best-effort pause/resume/move stream.
    advisory_rx: Consumer<StampedEvent>,
    /// One critical head retained while the merged poll compares ring sequences.
    critical_head: Option<StampedEvent>,
    /// One advisory head retained while the merged poll compares ring sequences.
    advisory_head: Option<StampedEvent>,
    replies_rx: Consumer<Reply>,
    triggers_rx: Consumer<Trigger>,
    node_msgs_rx: Consumer<NodeMsg>,
}

impl Nrt {
    pub fn new(
        trash_rx: Consumer<Trash>,
        critical_rx: Consumer<StampedEvent>,
        advisory_rx: Consumer<StampedEvent>,
        replies_rx: Consumer<Reply>,
        triggers_rx: Consumer<Trigger>,
        node_msgs_rx: Consumer<NodeMsg>,
    ) -> Self {
        Nrt {
            trash_rx,
            critical_rx,
            advisory_rx,
            critical_head: None,
            advisory_head: None,
            replies_rx,
            triggers_rx,
            node_msgs_rx,
        }
    }

    /// Drain the trash ring, dropping every freed synth and buffer here (off the audio thread).
    /// Returns the number dropped, so a shutdown loop can tell when the audio thread's trash is clear.
    pub fn process(&mut self) -> usize {
        let mut dropped = 0;
        while let Ok(trash) = self.trash_rx.pop() {
            drop(trash); // the heavy `Drop` runs here, never on the audio thread
            dropped += 1;
        }
        dropped
    }

    /// Pop the next retained node notification in its original RT emission order.
    ///
    /// This backward-compatible merged view compares sequence stamps from the critical and
    /// advisory rings. A host should use this method exclusively, or use both isolated polling
    /// methods exclusively, for an engine lifetime; mixing drain styles makes a single global
    /// ordering meaningless.
    pub fn poll(&mut self) -> Option<Event> {
        if self.critical_head.is_none() {
            self.critical_head = self.critical_rx.pop().ok();
        }
        if self.advisory_head.is_none() {
            self.advisory_head = self.advisory_rx.pop().ok();
        }
        match (self.critical_head, self.advisory_head) {
            (Some(critical), Some(advisory)) if critical.sequence <= advisory.sequence => {
                self.critical_head = None;
                Some(critical.event)
            }
            (Some(_), Some(advisory)) => {
                self.advisory_head = None;
                Some(advisory.event)
            }
            (Some(critical), None) => {
                self.critical_head = None;
                Some(critical.event)
            }
            (None, Some(advisory)) => {
                self.advisory_head = None;
                Some(advisory.event)
            }
            (None, None) => None,
        }
    }

    /// Pop the next ownership-critical node notification, if any.
    ///
    /// This isolated lossless stream contains only [`Event::NodeStarted`],
    /// [`Event::SynthFailed`], and [`Event::NodeEnded`]. Drain it together with
    /// [`poll_advisory`](Self::poll_advisory), and do not mix either isolated drain with
    /// [`poll`](Self::poll) during one engine lifetime.
    pub fn poll_critical(&mut self) -> Option<Event> {
        self.critical_head
            .take()
            .or_else(|| self.critical_rx.pop().ok())
            .map(|stamped| stamped.event)
    }

    /// Pop the next best-effort advisory node notification, if any.
    ///
    /// This isolated stream contains only pause, resume, and move events. Advisory events may be
    /// dropped under backpressure and never consume ownership-critical transport capacity.
    pub fn poll_advisory(&mut self) -> Option<Event> {
        self.advisory_head
            .take()
            .or_else(|| self.advisory_rx.pop().ok())
            .map(|stamped| stamped.event)
    }

    /// Pop the next query answer, if any (the getters). Drain in a loop alongside [`poll`](Self::poll),
    /// feeding each to the dispatcher's reply reassembler.
    pub fn poll_reply(&mut self) -> Option<Reply> {
        self.replies_rx.pop().ok()
    }

    /// Pop the next `SendTrig` trigger, if any. Drain in a loop alongside [`poll`](Self::poll),
    /// feeding each to the dispatcher's `notify_trigger` to emit a `/tr`.
    pub fn poll_trigger(&mut self) -> Option<Trigger> {
        self.triggers_rx.pop().ok()
    }

    /// Pop the next `SendReply` message, if any. Drain in a loop alongside [`poll`](Self::poll),
    /// feeding each to the dispatcher's `notify_node_msg` to emit its OSC reply.
    pub fn poll_node_msg(&mut self) -> Option<NodeMsg> {
        self.node_msgs_rx.pop().ok()
    }
}
