//! Initialization specialization: proving allocation-sizing unit inputs to constants.
//!
//! Some unit inputs only size per-instance memory - a delay's `maxdelaytime`, a `LocalBuf`'s
//! frames, `PitchShift`'s window. scsynth reads them once in the unit constructor, as the *first
//! computed sample* of whatever is wired there, so `DelayC.ar(in, SampleRate.ir * 0.1)` works.
//! plyphon sizes that memory at compile time and so requires a syntactic constant, and such
//! definitions fail with [`BuildError::AuxRequiresConstant`](plyphon_unit::BuildError).
//!
//! [`SynthDef::specialize_init`] closes the gap ahead of compile: it evaluates each
//! allocation-sizing input from the rate environment and the parameter values, and rewrites the
//! ones it proves to [`InputRef::Constant`]. Every other use of the same parameters stays live.
//! Evaluation goes through the shipped operator tables and `Select` index conversion, so a proven
//! value is the value the compiled units would compute.

use alloc::vec::Vec;

use plyphon_dsp::ops;
use plyphon_dsp::rate::RateInfo;
use plyphon_unit::unit::binary_op::binary_op;
use plyphon_unit::unit::select::select_index;
use plyphon_unit::unit::unary_op::unary_op;
use thiserror::Error;

use super::{InputRef, SynthDef};

/// Why an allocation-sizing input could not be rewritten to a constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitCause {
    /// The expression reads a live signal: a bus, a trigger, a demand unit, or any unit output the
    /// evaluator does not prove.
    Signal,
    /// The expression draws init-time randomness (the `Rand` family). scsynth draws per instance;
    /// baking one draw into the definition would give every instance the same value.
    Random,
    /// The expression reads a `Buf*` info unit, and no buffer metadata is available.
    BufferMetadataUnavailable,
    /// An unresolved operator, an out-of-range parameter reference, or any other expression the
    /// evaluator does not cover.
    Unsupported,
    /// The expression proved to a NaN or infinite value, which cannot size memory.
    NonFinite,
}

/// A [`SynthDef::specialize_init`] failure at one allocation-sizing input.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("unit {unit} input {input} does not reduce to an allocation size: {cause:?}")]
pub struct InitError {
    /// The index of the unit whose input failed.
    pub unit: usize,
    /// The index of the failing input.
    pub input: usize,
    /// Why it failed.
    pub cause: InitCause,
    /// Sorted parameter indices traversed before the failure, so a host can retry when one of them
    /// changes.
    pub dependencies: Vec<usize>,
}

/// The result of [`SynthDef::specialize_init`].
#[derive(Clone, Debug)]
pub struct SpecializedSynthDef {
    /// The definition with every proven allocation-sizing input rewritten to a constant. Equal to
    /// the input when `rewritten` is empty.
    pub def: SynthDef,
    /// Sorted parameter indices whose values influenced any proven input. The specialization can
    /// only change when one of these does.
    pub dependencies: Vec<usize>,
    /// The `(unit index, input index)` pairs that were rewritten.
    pub rewritten: Vec<(u32, u32)>,
}

/// The allocation-sizing inputs of `unit_name`, which scsynth reads only in its constructor.
///
/// These are exactly the inputs whose builders raise `AuxRequiresConstant`. A test probes the
/// registry to keep the two in step.
fn init_only_inputs(unit_name: &str) -> &'static [usize] {
    match unit_name {
        "FFT" => &[5],
        "IFFT" => &[2],
        "DelayN" | "DelayL" | "DelayC" | "CombN" | "CombL" | "CombC" | "AllpassN" | "AllpassL"
        | "AllpassC" => &[1],
        "Pluck" => &[2],
        "PitchShift" => &[1],
        "LocalBuf" => &[0, 1],
        "Gendy1" => &[8],
        "Limiter" | "Normalizer" => &[2],
        "Median" => &[0],
        "GVerb" => &[1, 5, 9],
        _ => &[],
    }
}

/// Recursion bound: a malformed (cyclic or forward-referencing) definition reports
/// [`InitCause::Unsupported`] instead of overflowing the stack.
const MAX_EVAL_DEPTH: usize = 256;

type Value = Result<f32, InitCause>;

/// One specialization's evaluation state.
struct Evaluator<'a> {
    def: &'a SynthDef,
    audio: &'a RateInfo,
    params: &'a [f32],
    /// Parameter indices traversed so far.
    deps: Vec<usize>,
    depth: usize,
    /// Output 0 of each unit, once evaluated. Shared subexpressions form diamond DAGs (sclang's
    /// `n.do { a = a + a }`), which plain recursion would evaluate `2^depth` times.
    memo: Vec<Option<Value>>,
}

impl Evaluator<'_> {
    fn eval_input(&mut self, input: &InputRef) -> Value {
        match *input {
            InputRef::Constant(v) => Ok(v),
            // A `TrigControl` outputs 0 from its constructor; its value only appears from the
            // first block on.
            InputRef::Param(p) if self.def.params.get(p as usize).is_some_and(|p| p.is_trig) => {
                Ok(0.0)
            }
            InputRef::Param(p) => {
                let value = *self.params.get(p as usize).ok_or(InitCause::Unsupported)?;
                if !self.deps.contains(&(p as usize)) {
                    self.deps.push(p as usize);
                }
                Ok(value)
            }
            InputRef::Unit { unit, output } => {
                // Every unit the evaluator proves is single-output.
                if output != 0 {
                    return Err(InitCause::Signal);
                }
                let unit = unit as usize;
                if let Some(Some(value)) = self.memo.get(unit) {
                    return *value;
                }
                if self.depth >= MAX_EVAL_DEPTH {
                    return Err(InitCause::Unsupported);
                }
                self.depth += 1;
                let value = self.eval_unit(unit);
                self.depth -= 1;
                if let Some(slot) = self.memo.get_mut(unit) {
                    *slot = Some(value);
                }
                value
            }
        }
    }

    /// Output 0 of `unit` under constructor-time first-sample semantics.
    fn eval_unit(&mut self, unit: usize) -> Value {
        let spec = self.def.units.get(unit).ok_or(InitCause::Unsupported)?;
        let ins = spec.inputs.as_slice();
        // Rate reads narrow to `f32` exactly as the info units do.
        match (spec.name.as_str(), ins) {
            ("SampleRate", _) => Ok(self.audio.sample_rate as f32),
            ("SampleDur", _) => Ok(self.audio.sample_dur as f32),
            ("RadiansPerSample", _) => Ok(self.audio.radians_per_sample as f32),
            ("ControlRate", _) => Ok(self.audio.buf_rate as f32),
            ("ControlDur", _) => Ok(self.audio.buf_dur as f32),
            ("UnaryOpUGen", [a]) => {
                let op = unary_op(spec.special_index).ok_or(InitCause::Unsupported)?;
                Ok(op(self.eval_input(a)?))
            }
            ("BinaryOpUGen", [a, b]) => {
                let op = binary_op(spec.special_index).ok_or(InitCause::Unsupported)?;
                // Both sides are evaluated so every traversed parameter is recorded.
                let (a, b) = (self.eval_input(a), self.eval_input(b));
                Ok(op(a?, b?))
            }
            ("MulAdd", [a, b, c]) => {
                let (a, b, c) = (self.eval_input(a), self.eval_input(b), self.eval_input(c));
                Ok(a? * b? + c?)
            }
            ("Sum3", [_, _, _]) | ("Sum4", [_, _, _, _]) => {
                let values = ins.iter().map(|x| self.eval_input(x));
                values
                    .collect::<Result<Vec<f32>, _>>()
                    .map(|v| v.into_iter().sum())
            }
            // Only the selected branch is evaluated, so an unselected signal branch cannot block
            // the proof - a `Select`'s first sample is its selected input's first sample.
            ("Select", [which, ..]) if ins.len() >= 2 => {
                let which = self.eval_input(which)?;
                self.eval_input(&ins[select_index(which, ins.len())])
            }
            ("K2A" | "A2K", [a]) => self.eval_input(a),
            ("Clip", [a, lo, hi]) => {
                let (a, lo, hi) = (self.eval_input(a), self.eval_input(lo), self.eval_input(hi));
                Ok(ops::clip(a?, lo?, hi?))
            }
            ("Rand" | "IRand" | "ExpRand" | "LinRand" | "NRand", _) => Err(InitCause::Random),
            (
                "BufSampleRate" | "BufFrames" | "BufDur" | "BufChannels" | "BufSamples"
                | "BufRateScale",
                _,
            ) => {
                // Record the buffer-selecting parameter so a host can retry when it changes.
                if let Some(buffer) = ins.first() {
                    let _ = self.eval_input(buffer);
                }
                Err(InitCause::BufferMetadataUnavailable)
            }
            (
                "UnaryOpUGen" | "BinaryOpUGen" | "MulAdd" | "Sum3" | "Sum4" | "Select" | "K2A"
                | "A2K" | "Clip",
                _,
            ) => Err(InitCause::Unsupported),
            _ => Err(InitCause::Signal),
        }
    }
}

impl SynthDef {
    /// Prove every allocation-sizing input under scsynth's constructor-time semantics and rewrite
    /// the proven ones to constants, so the result compiles where this definition would fail with
    /// `AuxRequiresConstant`.
    ///
    /// `audio` is the graph audio rate the definition will compile with (for a reblocked or
    /// resampled def, its own rate, not the World's). `params` holds one value per parameter
    /// index; a reference past its end is [`InitCause::Unsupported`].
    ///
    /// Inputs that are already constants are left alone, so a definition that compiles today
    /// comes back unchanged with nothing `rewritten`.
    pub fn specialize_init(
        &self,
        audio: &RateInfo,
        params: &[f32],
    ) -> Result<SpecializedSynthDef, InitError> {
        let mut eval = Evaluator {
            def: self,
            audio,
            params,
            deps: Vec::new(),
            depth: 0,
            memo: vec![None; self.units.len()],
        };
        let mut def = self.clone();
        let mut rewritten = Vec::new();
        for (unit, spec) in self.units.iter().enumerate() {
            for &input in init_only_inputs(&spec.name) {
                // A missing input is left for `compile`'s arity check.
                let Some(input_ref) = spec.inputs.get(input) else {
                    continue;
                };
                if matches!(input_ref, InputRef::Constant(_)) {
                    continue;
                }
                let cause = match eval.eval_input(input_ref) {
                    Ok(value) if value.is_finite() => {
                        def.units[unit].inputs[input] = InputRef::Constant(value);
                        rewritten.push((unit as u32, input as u32));
                        continue;
                    }
                    Ok(_) => InitCause::NonFinite,
                    Err(cause) => cause,
                };
                return Err(InitError {
                    unit,
                    input,
                    cause,
                    dependencies: sorted(eval.deps),
                });
            }
        }
        Ok(SpecializedSynthDef {
            def,
            dependencies: sorted(eval.deps),
            rewritten,
        })
    }
}

fn sorted(mut deps: Vec<usize>) -> Vec<usize> {
    deps.sort_unstable();
    deps
}

#[cfg(test)]
mod tests {
    use super::init_only_inputs;
    use plyphon_dsp::rate::{Rate, RateInfo};
    use plyphon_unit::BuildError;
    use plyphon_unit::unit::InputSource;
    use plyphon_unit::unit::registry::{BuildContext, UnitRegistry};

    /// Probe every registered unit with all-wire inputs: each input its builder demands as a
    /// constant must be declared in [`init_only_inputs`], so a new allocation-sizing input fails
    /// here until it is declared.
    #[test]
    fn every_aux_constant_input_is_declared() {
        let registry = UnitRegistry::with_builtins();
        let audio = RateInfo::new(48_000.0, 64);
        let control = RateInfo::new(48_000.0 / 64.0, 1);
        for name in registry.names() {
            let unit = registry.get(name).expect("listed unit");
            let mut sources = vec![InputSource::Control(0); 16];
            loop {
                let rates = vec![Rate::Control; sources.len()];
                let units = vec![None; sources.len()];
                let ctx = BuildContext {
                    input_rates: &rates,
                    input_units: &units,
                    input_sources: &sources,
                    rate: Rate::Control,
                    num_outputs: 1,
                    audio: &audio,
                    control: &control,
                    special_index: 0,
                    seed: 0,
                    local_bufs_so_far: 0,
                };
                let Err(BuildError::AuxRequiresConstant { input }) = unit.build(&ctx) else {
                    break;
                };
                assert!(
                    init_only_inputs(name).contains(&input),
                    "{name} requires input {input} to be constant but does not declare it"
                );
                // Satisfy this input so units with several (LocalBuf, GVerb) reveal the next.
                sources[input] = InputSource::Constant(64.0);
            }
        }
    }
}
