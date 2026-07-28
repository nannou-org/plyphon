//! Control-side error types. These never surface on the audio thread.

use alloc::string::String;

use thiserror::Error;

/// Why an allocation-sizing input could not reduce to a compile-time constant.
///
/// Carried by [`BuildError::AuxRequiresConstant`] so hosts can classify the failure without
/// re-deriving the expression: a genuinely signal-driven size, an init-time random size (which
/// scsynth folds at ctor but plyphon deliberately does not), a size gated on buffer metadata the
/// host has not supplied, or an expression outside the initialization evaluator's proven set.
/// Raise sites outside the evaluator (unit builders reached by a bare `compile`) always construct
/// [`AuxDynamicCause::Unsupported`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuxDynamicCause {
    /// The expression reads a live signal: a bus (`In`/`InFeedback`/`LocalIn`), a trigger, a demand
    /// unit, a reply, or any unit output the evaluator does not prove.
    Signal,
    /// The expression draws init-time randomness (the `Rand` family).
    Random,
    /// The expression is rooted in a `Buf*` info unit and no buffer metadata is available.
    BufferMetadataUnavailable,
    /// An unresolved operator index, an out-of-range parameter reference, or any other expression
    /// outside the evaluator's enumerated set.
    Unsupported,
}

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
    /// A unit that sizes per-instance auxiliary memory (a delay line) from a scalar input was given a
    /// non-constant for that input. The size must be known at compile time, so - like scsynth's
    /// instantiation-only `maxdelaytime` (`ZIN0` at ctor) - the input must be a baked constant.
    /// `cause` records why the input could not reduce to a constant, so hosts can classify the
    /// failure (signal-driven vs init-time-random vs buffer-metadata-gated) without re-deriving it.
    #[error("input {input} must be a compile-time constant to size auxiliary memory")]
    AuxRequiresConstant {
        /// The index of the offending input.
        input: usize,
        /// Why the input is not a compile-time constant.
        cause: AuxDynamicCause,
    },
    /// The initialization evaluator proved an allocation-sizing input to a value that is NaN or
    /// infinite. Aux sizes must be finite, so the definition is rejected deterministically at
    /// specialization time rather than allocating from a nonsense size.
    #[error("unit {unit} input {input} proved to a non-finite auxiliary size")]
    AuxNonFinite {
        /// The registry name of the unit whose input proved non-finite.
        unit: String,
        /// The index of the offending input.
        input: usize,
    },
    /// An allocation site's total element count exceeds [`MAX_AUX_ELEMS`](crate::unit::MAX_AUX_ELEMS).
    /// The count is accumulated with saturating arithmetic before any narrowing cast, so an
    /// out-of-range size fails here deterministically instead of truncating or wrapping into an
    /// undersized allocation that the audio thread would then index.
    #[error("unit {unit} input {input} sizes {elements} aux elements but the limit is {limit}")]
    AuxSizeOutOfRange {
        /// The registry name of the unit whose allocation overflows the bound.
        unit: String,
        /// The index of the sizing input.
        input: usize,
        /// The saturating total element count the site would allocate.
        elements: u64,
        /// The `MAX_AUX_ELEMS` bound.
        limit: u64,
    },
    /// An emitting unit (`SendReply`) was given a non-constant label length or character. The OSC path
    /// is encoded as constant float inputs (scsynth's scheme), so it must be known at compile time.
    #[error("an emitting unit's label must be encoded as compile-time constant inputs")]
    EmitBadLabel,
    /// An emitting unit's label is longer than the inline carrier allows (`MAX_LABEL`).
    #[error("an emitting unit's label is {len} bytes but the limit is {limit}")]
    EmitLabelTooLong {
        /// The requested label length.
        len: usize,
        /// The `MAX_LABEL` limit.
        limit: usize,
    },
    /// An emitting unit (`SendReply`) carries more values than the inline carrier allows (`MAX_VALUES`).
    #[error("an emitting unit carries {count} values but the limit is {limit}")]
    EmitTooManyValues {
        /// The requested value count.
        count: usize,
        /// The `MAX_VALUES` limit.
        limit: usize,
    },
    /// An `FFT`/`IFFT` unit was given an unsupported FFT size. The size (its constant `winsize` input)
    /// must be a power of two the engine has a plan for - `[64, 16384]`.
    #[error("unsupported FFT size {size}: must be a power of two in [64, 16384]")]
    UnsupportedFftSize {
        /// The requested FFT size.
        size: usize,
    },
    /// A reblocked def (scsynth's `Reblock(n)`) requested a block size that is not a power of two, is
    /// zero, or exceeds the World block - none of which scsynth's reblocking allows.
    #[error("invalid reblock block size {block_size}: must be a power of two in [1, {world}]")]
    InvalidReblock {
        /// The requested graph block size.
        block_size: usize,
        /// The World control block size it must divide.
        world: usize,
    },
    /// A resampled def (scsynth's `Resample(n)`) requested an oversample factor that is not a power of
    /// two or is zero. scsynth allows only power-of-two oversampling (no downsampling).
    #[error("invalid resample factor {factor}: must be a power of two >= 1")]
    InvalidResample {
        /// The requested oversample factor.
        factor: usize,
    },
}
