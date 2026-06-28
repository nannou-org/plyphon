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
    /// A def has more than one `LocalIn` or `LocalOut`. The v1 feedback bus supports exactly one of
    /// each (its channel count is taken from the single `LocalIn`).
    #[error("a def may have at most one LocalIn and one LocalOut")]
    MultipleLocalBuses,
    /// `LocalOut` writes a different channel count than the `LocalIn` declares (or there is a
    /// `LocalOut` with no `LocalIn` to size the bus). The two must agree.
    #[error("LocalOut writes {local_out} channels but LocalIn declares {local_in}")]
    LocalBusMismatch {
        /// Channels the `LocalIn` declares (its output count; `0` if there is no `LocalIn`).
        local_in: usize,
        /// Channels the `LocalOut` writes (its input count).
        local_out: usize,
    },
    /// A unit that sizes per-instance auxiliary memory (a delay line) from a scalar input was given a
    /// non-constant for that input. The size must be known at compile time, so - like scsynth's
    /// instantiation-only `maxdelaytime` (`ZIN0` at ctor) - the input must be a baked constant.
    #[error("input {input} must be a compile-time constant to size auxiliary memory")]
    AuxRequiresConstant {
        /// The index of the offending input.
        input: usize,
    },
}
