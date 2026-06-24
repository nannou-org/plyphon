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
//! - **Surfaces notifications.** Node started/ended/paused/resumed [`Event`]s flow over the events
//!   ring; the consumer drains them with [`Nrt::poll`].
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
//! Each NRT tick: call [`Nrt::process`] (drops freed synths) and drain [`Nrt::poll`] in a loop.
//!
//! **Shutdown ordering matters.** Stop the audio thread first so the `World` is no longer driven.
//! Then keep ticking the `Nrt` until [`Nrt::process`] returns `0`, so every freed `Box` still in
//! flight is dropped here rather than on the audio thread. Whoever finally drops the `World` drops
//! the live node tree (and its synths), so do that off the audio thread once it has stopped.

use rtrb::Consumer;

use crate::command::{Event, Trash};

/// The NRT-side state machine: drops freed synths and surfaces node notifications.
pub struct Nrt {
    trash_rx: Consumer<Trash>,
    events_rx: Consumer<Event>,
}

impl Nrt {
    pub fn new(trash_rx: Consumer<Trash>, events_rx: Consumer<Event>) -> Self {
        Nrt {
            trash_rx,
            events_rx,
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

    /// Pop the next node notification, if any. Drain in a loop: `while let Some(e) = nrt.poll() {}`.
    pub fn poll(&mut self) -> Option<Event> {
        self.events_rx.pop().ok()
    }
}
