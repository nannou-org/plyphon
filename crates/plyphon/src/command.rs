//! The messages crossing the control/RT boundary.
//!
//! [`Command`]s flow control-side -> RT-side over a lock-free ring. Anything needing allocation
//! (instantiating a synth) is pre-built control-side, so applying a command on the audio thread is
//! pure link manipulation. [`Trash`] flows back RT-side -> control-side so freed `Box`es are dropped
//! off the audio thread.

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
}

/// Heap-owning values handed back to the control side to be dropped off the audio thread.
pub enum Trash {
    /// A freed synth.
    Synth(Box<Synth>),
}
