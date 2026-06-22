//! The messages crossing the control/RT boundary.
//!
//! [`Command`]s flow control-side -> RT-side over a lock-free ring. Anything needing allocation
//! (instantiating a synth) is pre-built control-side, so applying a command on the audio thread is
//! pure link manipulation. Two streams flow back RT-side -> NRT-side, both drained by the
//! [`Nrt`](crate::nrt::Nrt): [`Trash`] carries freed `Box`es to be dropped off the audio thread, and
//! [`Event`] carries notifications (node started/ended/paused/resumed) for the consumer.

use crate::synth::Synth;
use crate::tree::AddAction;

/// A command from the [`Controller`](crate::controller::Controller) to the
/// [`World`](crate::world::World).
pub enum Command {
    /// Link an already-built synth into the tree under group `target`.
    AddSynth {
        /// Client id for the new synth.
        id: i32,
        /// The pre-built synth (all allocation already done control-side).
        synth: Box<Synth>,
        /// Target group's client id.
        target: i32,
        /// Placement within the target group.
        action: AddAction,
    },
    /// Create an empty group under group `target`.
    AddGroup {
        /// Client id for the new group.
        id: i32,
        /// Target group's client id.
        target: i32,
        /// Placement within the target group.
        action: AddAction,
    },
    /// Set control parameter `param` of node `node` to `value`.
    SetControl {
        /// Target node's client id.
        node: i32,
        /// Parameter index.
        param: usize,
        /// New value.
        value: f32,
    },
    /// Free node `node`, returning any owned synth to the trash ring.
    FreeNode {
        /// Target node's client id.
        node: i32,
    },
    /// Pause or resume node `node` (scsynth's `/n_run`).
    NodeRun {
        /// Target node's client id.
        node: i32,
        /// Run the node (`true`) or pause it (`false`).
        run: bool,
    },
}

/// Heap-owning values handed back to the NRT side to be dropped off the audio thread.
pub enum Trash {
    /// A freed synth.
    Synth(Box<Synth>),
}

/// A notification flowing RT-side -> NRT-side, surfaced to the consumer by the
/// [`Nrt`](crate::nrt::Nrt). Each mirrors a scsynth node-notification message.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Event {
    /// A node was added to the tree (`/n_go`).
    NodeStarted {
        /// The node's client id.
        id: i32,
    },
    /// A node was freed (`/n_end`), whether explicitly or by a done action.
    NodeEnded {
        /// The node's client id.
        id: i32,
    },
    /// A node was paused (`/n_off`).
    NodePaused {
        /// The node's client id.
        id: i32,
    },
    /// A node was resumed (`/n_on`).
    NodeResumed {
        /// The node's client id.
        id: i32,
    },
}
