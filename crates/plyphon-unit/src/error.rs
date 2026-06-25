//! Control-side error types. These never surface on the audio thread.

use alloc::string::String;

use thiserror::Error;

/// Errors from compiling a `SynthDef` into a [`GraphDef`](crate::graphdef::GraphDef).
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum BuildError {
    /// The SynthDef references a unit name not present in the registry.
    #[error("unknown unit: {0}")]
    UnknownUnit(String),
    /// An input reference (parameter or unit index) is out of range.
    #[error("input reference out of range")]
    BadInputRef,
    /// A unit used a `special_index` operator that is not implemented.
    #[error("unsupported operator index: {0}")]
    UnsupportedOp(i16),
    /// A unit was instantiated with the wrong number of inputs.
    #[error("wrong number of inputs for unit")]
    WrongInputCount,
    /// The def needs more audio wire buffers than the engine's `max_wire_bufs` allows.
    #[error("def needs {needed} audio wires but the engine allows {limit}")]
    TooManyWires {
        /// Audio wires the def requires.
        needed: usize,
        /// The engine's `max_wire_bufs` limit.
        limit: usize,
    },
    /// A unit has more outputs than the engine's `max_unit_outputs` scratch allows.
    #[error("a unit has {needed} outputs but the engine allows {limit}")]
    TooManyOutputs {
        /// Outputs the widest unit requires.
        needed: usize,
        /// The engine's `max_unit_outputs` limit.
        limit: usize,
    },
    /// A demand-rate unit's state is too large for the fixed stack buffer the audio thread uses to
    /// pull it (`MAX_DEMAND_STATE`). Rejected off-RT so the RT path never over-runs the buffer.
    #[error("a demand unit needs {needed} state bytes but the limit is {limit}")]
    DemandStateTooLarge {
        /// State bytes the demand unit requires.
        needed: usize,
        /// The `MAX_DEMAND_STATE` limit.
        limit: usize,
    },
    /// A demand-rate graph nests deeper than `MAX_DEMAND_DEPTH`. Each level recurses the audio
    /// thread's stack, so deeper graphs are rejected off-RT to keep the recursion bounded.
    #[error("a demand graph nests {depth} deep but the limit is {limit}")]
    DemandNestingTooDeep {
        /// The deepest demand-input chain in the def.
        depth: usize,
        /// The `MAX_DEMAND_DEPTH` limit.
        limit: usize,
    },
    /// A demand-rate source was given more than one output. Demand sources are single-output (they
    /// produce one value per pull); a multi-output demand input cannot be resolved.
    #[error("a demand source has {0} outputs but must have exactly one")]
    DemandMultiOutput(usize),
}
