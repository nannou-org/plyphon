//! Initialization specialization: proving allocation-only unit inputs to constants.
//!
//! Imported SynthDefs frequently compute allocation-sizing inputs - a delay's `maxdelaytime`, a
//! `LocalBuf`'s frames, `PitchShift`'s window - from sample-rate info, parameter values, and
//! scalar math rather than baking literals. scsynth accepts these because a ctor reads each input
//! chain's *first computed sample* at instantiation; plyphon's
//! [`BuildContext::const_input`](plyphon_unit::unit::registry::BuildContext::const_input)
//! accepts only syntactic constants, so such defs fail with
//! [`BuildError::AuxRequiresConstant`].
//!
//! [`SynthDef::specialize_init`] closes that gap off the audio thread: it lazily evaluates each
//! *declared* init-only input (the table in
//! [`init_only`](plyphon_unit::unit::init_only)) under scsynth's constructor-time
//! first-sample semantics and rewrites proven expressions to [`InputRef::Constant`], leaving
//! every other use of the same parameters and units untouched. The host records which parameters
//! each proof traversed ([`InitDependencies`]) and which inputs were rewritten
//! ([`SpecializedSynthDef::rewritten`]), so specialized definitions get distinct identities and
//! re-specialize when a dependency changes.
//!
//! Everything the evaluator computes goes through the *shipped* scalar kernels - the
//! unary/binary operator tables, the `Select` index conversion, `Clip`/`LinExp`/`Sanitize` - and
//! every rate/info read narrows to `f32` at the leaf exactly as the runtime units do, so the
//! evaluator can never land on the other side of a rounding boundary from an equivalent runtime
//! chain. Evaluation is deterministic: the same definition and environment always produce the
//! same result, bit for bit.

use alloc::vec::Vec;

use plyphon_dsp::ops;
use plyphon_dsp::rate::RateInfo;
use plyphon_unit::error::{AuxDynamicCause, BuildError};
use plyphon_unit::unit::binary_op::binary_op;
use plyphon_unit::unit::init_only::init_only_inputs;
use plyphon_unit::unit::select::select_index;
use plyphon_unit::unit::shape::{lin_exp_apply, lin_exp_coeffs};
use plyphon_unit::unit::test::sanitize_scalar;
use plyphon_unit::unit::unary_op::unary_op;

use super::{InputRef, SynthDef};

/// The immutable inputs one initialization specialization evaluates against.
#[derive(Clone, Debug)]
pub struct InitEnvironment {
    /// The graph audio rate the definition will compile with - derive it with
    /// [`graph_rates`](super::graph_rates) from the same host rate and reblock/resample the
    /// compile call will use, so the evaluator and the compiled units can never disagree.
    /// `SampleRate`/`SampleDur`/`RadiansPerSample` read its rate fields; `ControlRate`/
    /// `ControlDur` read its `buf_rate`/`buf_dur`, mirroring the shipped info units.
    pub audio: RateInfo,
    /// One authored base value per authored param index; `-0.0` normalized to `0.0` by the
    /// caller. Entries may be non-finite - only a non-finite *proven result* at a declared
    /// allocation site is an error, so a bad value on a non-dependency param cannot fail a
    /// definition that never reads it. Array-span lanes carry their authored defaults.
    pub params: Vec<f32>,
}

/// The parameters a specialization's proofs traversed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InitDependencies {
    /// Sorted authored param indices traversed by any proof, including `Select` selectors -
    /// every param whose value influenced a proven result, not only the leaf that supplied the
    /// number. A change to any of these (and only these) can change the specialization.
    pub params: Vec<usize>,
}

/// A failed specialization still reports what it traversed, so the host can watch those inputs
/// and re-trigger when one changes (a corrective base edit).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InitError {
    /// The deterministic build failure.
    pub error: BuildError,
    /// The dependency set attempted up to and including the failure point.
    pub dependencies: InitDependencies,
}

/// The result of [`SynthDef::specialize_init`]: the rewritten definition plus the provenance the
/// host folds into def identity and its re-specialization watch.
#[derive(Clone, Debug)]
pub struct SpecializedSynthDef {
    /// The input definition with every proven init-only input rewritten to a constant;
    /// structurally identical to the input when `rewritten` is empty (`SynthDef` has no byte
    /// encoding - equality is structural).
    pub def: SynthDef,
    /// The parameters the proofs traversed.
    pub dependencies: InitDependencies,
    /// Exactly the `(unit index, input index)` pairs the evaluator replaced with constants.
    /// This one predicate decides everything downstream: hosts fold the specialized-def digest
    /// into identity iff it is non-empty and scope def retirement on the same predicate.
    pub rewritten: Vec<(u32, u32)>,
}

/// The outcome of evaluating one expression.
#[derive(Clone, Copy)]
enum InitValue {
    /// Reduced to a finite-or-not scalar under constructor-time semantics.
    Proven(f32),
    /// Not provable, with the classification cause of the first dynamic leaf.
    Dynamic(AuxDynamicCause),
}

/// Scalar-rate random units scsynth folds at instantiation but plyphon deliberately does not:
/// proving them would bake one seed's draw into the definition identity.
fn is_random_unit(name: &str) -> bool {
    matches!(name, "Rand" | "IRand" | "ExpRand" | "LinRand" | "NRand")
}

/// The buffer-info units whose proofs are gated on host-supplied buffer metadata. No metadata
/// channel exists yet, so every `Buf*`-rooted proof classifies as
/// [`AuxDynamicCause::BufferMetadataUnavailable`] unconditionally.
fn is_buffer_info_unit(name: &str) -> bool {
    matches!(
        name,
        "BufSampleRate" | "BufFrames" | "BufDur" | "BufChannels" | "BufSamples" | "BufRateScale"
    )
}

/// Recursion bound for expression evaluation: a well-formed def's inputs reference earlier
/// units, so real chains are shallow; a malformed (cyclic or forward-referencing) def hits the
/// bound and classifies `Unsupported` instead of overflowing the stack.
const MAX_EVAL_DEPTH: usize = 256;

/// One specialization's evaluation state: walks expressions over the definition and environment,
/// accumulating traversed parameters and memoized unit outcomes as it goes.
struct Evaluator<'a> {
    def: &'a SynthDef,
    env: &'a InitEnvironment,
    /// Authored param indices traversed so far, in insertion order (sorted + deduped at the end).
    deps: Vec<usize>,
    /// Current expression recursion depth, bounded by [`MAX_EVAL_DEPTH`].
    depth: usize,
    /// Per-unit memo of output 0's outcome. Chains of shared subexpressions form diamond DAGs
    /// (sclang's `n.do { a = a + a }` idiom), which naive recursion re-evaluates `2^depth`
    /// times; the memo makes evaluation linear in the unit count. Keying by unit index alone is
    /// sound because a non-zero output short-circuits to `Signal` before any recursion, and a
    /// memo hit contributes no new dependencies (the first evaluation already recorded them).
    memo: Vec<Option<InitValue>>,
}

impl Evaluator<'_> {
    /// Note that `param` influenced the evaluation (kept unique, insertion-ordered).
    fn record_dep(&mut self, param: usize) {
        if !self.deps.contains(&param) {
            self.deps.push(param);
        }
    }

    /// Evaluate one input reference: constants prove to themselves, parameters to their
    /// environment base (recorded as a dependency), unit outputs by recursion.
    fn eval_input(&mut self, input: &InputRef) -> InitValue {
        match *input {
            InputRef::Constant(v) => InitValue::Proven(v),
            InputRef::Param(p) => {
                let p = p as usize;
                // The session transform may append host params after the authored ones; an
                // index at or past the environment is unprovable, never a panic or a
                // neighbouring read.
                match self.env.params.get(p) {
                    Some(&base) => {
                        self.record_dep(p);
                        InitValue::Proven(base)
                    }
                    None => InitValue::Dynamic(AuxDynamicCause::Unsupported),
                }
            }
            InputRef::Unit { unit, output } => self.eval_unit(unit as usize, output as usize),
        }
    }

    /// Evaluate one unit output through the depth guard and the memo; the expression semantics
    /// live in [`Self::eval_unit_inner`].
    fn eval_unit(&mut self, unit: usize, output: usize) -> InitValue {
        if self.depth >= MAX_EVAL_DEPTH {
            return InitValue::Dynamic(AuxDynamicCause::Unsupported);
        }
        // Only output 0 is memoizable (and only output 0 recurses; see below).
        if output == 0
            && let Some(Some(value)) = self.memo.get(unit)
        {
            return *value;
        }
        self.depth += 1;
        let value = self.eval_unit_inner(unit, output);
        self.depth -= 1;
        if output == 0
            && let Some(slot) = self.memo.get_mut(unit)
        {
            *slot = Some(value);
        }
        value
    }

    /// The evaluable-expression set itself: one arm per provable unit (compile-context
    /// invariants, the shipped operator tables, lazy `Select`), with every unproven output
    /// classified by its [`AuxDynamicCause`].
    fn eval_unit_inner(&mut self, unit: usize, output: usize) -> InitValue {
        let Some(spec) = self.def.units.get(unit) else {
            return InitValue::Dynamic(AuxDynamicCause::Unsupported);
        };
        // Every unit the evaluator proves is single-output; a multi-output tap is a signal.
        if output != 0 {
            return InitValue::Dynamic(AuxDynamicCause::Signal);
        }
        let ins = &spec.inputs;
        match spec.name.as_str() {
            // Compile-context invariants, each narrowed to `f32` at the leaf exactly as the
            // shipped info units read them (`ControlRate`/`ControlDur` from the *audio*
            // `RateInfo`'s block fields).
            "SampleRate" => InitValue::Proven(self.env.audio.sample_rate as f32),
            "SampleDur" => InitValue::Proven(self.env.audio.sample_dur as f32),
            "RadiansPerSample" => InitValue::Proven(self.env.audio.radians_per_sample as f32),
            "ControlRate" => InitValue::Proven(self.env.audio.buf_rate as f32),
            "ControlDur" => InitValue::Proven(self.env.audio.buf_dur as f32),
            "UnaryOpUGen" => {
                let Some(op) = unary_op(spec.special_index) else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                let [a] = ins.as_slice() else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                self.map1(a, op)
            }
            "BinaryOpUGen" => {
                let Some(op) = binary_op(spec.special_index) else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                let [a, b] = ins.as_slice() else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                self.map2(a, b, op)
            }
            "MulAdd" => {
                let [a, b, c] = ins.as_slice() else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                match (self.eval_input(a), self.eval_input(b), self.eval_input(c)) {
                    (InitValue::Proven(a), InitValue::Proven(b), InitValue::Proven(c)) => {
                        InitValue::Proven(a * b + c)
                    }
                    (a, b, c) => first_dynamic([a, b, c]),
                }
            }
            "Sum3" => self.sum(ins, 3),
            "Sum4" => self.sum(ins, 4),
            "Select" => {
                // Lazy: only the selected branch is evaluated, so an unselected signal branch
                // (the corpus's `x_ar` lane) cannot poison the proof. The selector converts
                // through the shipped runtime helper.
                if ins.len() < 2 {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                }
                let which = match self.eval_input(&ins[0]) {
                    InitValue::Proven(v) => v,
                    dynamic => return dynamic,
                };
                let index = select_index(which, ins.len());
                self.eval_input(&ins[index])
            }
            // Rate lifts are identities over their operand.
            "K2A" | "A2K" => {
                let [a] = ins.as_slice() else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                self.eval_input(a)
            }
            "Clip" => {
                let [a, lo, hi] = ins.as_slice() else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                match (self.eval_input(a), self.eval_input(lo), self.eval_input(hi)) {
                    (InitValue::Proven(a), InitValue::Proven(lo), InitValue::Proven(hi)) => {
                        InitValue::Proven(ops::clip(a, lo, hi))
                    }
                    (a, lo, hi) => first_dynamic([a, lo, hi]),
                }
            }
            "LinExp" => {
                let [x, srclo, srchi, dstlo, dsthi] = ins.as_slice() else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                match (
                    self.eval_input(x),
                    self.eval_input(srclo),
                    self.eval_input(srchi),
                    self.eval_input(dstlo),
                    self.eval_input(dsthi),
                ) {
                    (
                        InitValue::Proven(x),
                        InitValue::Proven(srclo),
                        InitValue::Proven(srchi),
                        InitValue::Proven(dstlo),
                        InitValue::Proven(dsthi),
                    ) => {
                        let (dstratio, rsrcrange, rrminuslo) =
                            lin_exp_coeffs(srclo, srchi, dstlo, dsthi);
                        InitValue::Proven(lin_exp_apply(x, dstlo, dstratio, rsrcrange, rrminuslo))
                    }
                    (a, b, c, d, e) => first_dynamic([a, b, c, d, e]),
                }
            }
            "Sanitize" => {
                let [x, replace] = ins.as_slice() else {
                    return InitValue::Dynamic(AuxDynamicCause::Unsupported);
                };
                match (self.eval_input(x), self.eval_input(replace)) {
                    (InitValue::Proven(x), InitValue::Proven(replace)) => {
                        InitValue::Proven(sanitize_scalar(x, replace))
                    }
                    (a, b) => first_dynamic([a, b]),
                }
            }
            name if is_random_unit(name) => InitValue::Dynamic(AuxDynamicCause::Random),
            name if is_buffer_info_unit(name) => {
                // Traverse the buffer input for the attempted-dependency record (so a host can
                // watch the buffer-selecting param for a corrective edit) before classifying;
                // its value is deliberately unused - no metadata channel exists here.
                if let Some(buffer_input) = ins.first() {
                    let _ = self.eval_input(buffer_input);
                }
                InitValue::Dynamic(AuxDynamicCause::BufferMetadataUnavailable)
            }
            // Everything else - bus reads, triggers, demand units, replies, ordinary signal
            // units - is a live signal.
            _ => InitValue::Dynamic(AuxDynamicCause::Signal),
        }
    }

    /// Apply a unary operator to a proven operand, or pass the dynamic outcome through.
    fn map1(&mut self, a: &InputRef, op: fn(f32) -> f32) -> InitValue {
        match self.eval_input(a) {
            InitValue::Proven(a) => InitValue::Proven(op(a)),
            dynamic => dynamic,
        }
    }

    /// Apply a binary operator to two proven operands; both are evaluated eagerly so every
    /// traversed parameter is recorded, with the first dynamic outcome (in input order) winning.
    fn map2(&mut self, a: &InputRef, b: &InputRef, op: fn(f32, f32) -> f32) -> InitValue {
        match (self.eval_input(a), self.eval_input(b)) {
            (InitValue::Proven(a), InitValue::Proven(b)) => InitValue::Proven(op(a, b)),
            (a, b) => first_dynamic([a, b]),
        }
    }

    /// `Sum3`/`Sum4`: add exactly `arity` operands, stopping at the first dynamic one (its
    /// cause is the whole sum's classification, so later operands stay untraversed).
    fn sum(&mut self, ins: &[InputRef], arity: usize) -> InitValue {
        if ins.len() != arity {
            return InitValue::Dynamic(AuxDynamicCause::Unsupported);
        }
        let mut total = 0.0f32;
        for input in ins {
            match self.eval_input(input) {
                InitValue::Proven(v) => total += v,
                dynamic => return dynamic,
            }
        }
        InitValue::Proven(total)
    }
}

/// The first `Dynamic` outcome in declared-input order - the deterministic cause when several
/// leaves are dynamic.
fn first_dynamic<const N: usize>(values: [InitValue; N]) -> InitValue {
    for value in values {
        if matches!(value, InitValue::Dynamic(_)) {
            return value;
        }
    }
    InitValue::Dynamic(AuxDynamicCause::Unsupported)
}

impl SynthDef {
    /// Prove every declared init-only input under scsynth's constructor-time first-sample
    /// semantics and rewrite the proven ones to constants, off the audio thread.
    ///
    /// Returns the rewritten definition with its dependency and rewrite provenance. A declared
    /// input that is neither a syntactic constant nor provable fails with
    /// [`BuildError::AuxRequiresConstant`] carrying its classification cause; a proven value
    /// that is NaN or infinite fails with [`BuildError::AuxNonFinite`]. Either failure still
    /// reports the dependencies traversed up to that point, so the host can watch them for a
    /// corrective edit. When nothing needed rewriting the returned definition is structurally
    /// identical to the input and `rewritten` is empty.
    pub fn specialize_init(&self, env: &InitEnvironment) -> Result<SpecializedSynthDef, InitError> {
        let mut evaluator = Evaluator {
            def: self,
            env,
            deps: Vec::new(),
            depth: 0,
            memo: vec![None; self.units.len()],
        };
        let mut rewrites: Vec<(u32, u32, f32)> = Vec::new();
        for (u, spec) in self.units.iter().enumerate() {
            for &input_index in init_only_inputs(&spec.name) {
                let Some(input) = spec.inputs.get(input_index) else {
                    // Too few inputs: leave it for `compile`'s existing arity error.
                    continue;
                };
                if matches!(input, InputRef::Constant(_)) {
                    continue;
                }
                match evaluator.eval_input(input) {
                    InitValue::Proven(value) => {
                        if !value.is_finite() {
                            return Err(InitError {
                                error: BuildError::AuxNonFinite {
                                    unit: spec.name.clone(),
                                    input: input_index,
                                },
                                dependencies: finish_deps(evaluator.deps),
                            });
                        }
                        rewrites.push((u as u32, input_index as u32, value));
                    }
                    InitValue::Dynamic(cause) => {
                        return Err(InitError {
                            error: BuildError::AuxRequiresConstant {
                                input: input_index,
                                cause,
                            },
                            dependencies: finish_deps(evaluator.deps),
                        });
                    }
                }
            }
        }

        let mut def = self.clone();
        let mut rewritten = Vec::with_capacity(rewrites.len());
        for (u, i, value) in rewrites {
            def.units[u as usize].inputs[i as usize] = InputRef::Constant(value);
            rewritten.push((u, i));
        }
        Ok(SpecializedSynthDef {
            def,
            dependencies: finish_deps(evaluator.deps),
            rewritten,
        })
    }
}

/// Normalize the traversal-ordered dependency record into its published sorted, deduplicated form.
fn finish_deps(mut deps: Vec<usize>) -> InitDependencies {
    deps.sort_unstable();
    deps.dedup();
    InitDependencies { params: deps }
}
