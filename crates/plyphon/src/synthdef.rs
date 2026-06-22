//! Synth definitions and their instantiation into live [`Synth`]s.
//!
//! A [`SynthDef`] is the template (the analogue of scsynth's `GraphDef` / the SCgf binary format),
//! built programmatically for now. [`SynthDef::instantiate`] turns it into a `Box<Synth>` off the
//! audio thread - this is where all the per-synth allocation and UGen construction happens, so the
//! audio thread only ever has to link the finished synth into the tree.

use std::collections::HashMap;

use crate::error::BuildError;
use crate::rate::{Rate, RateInfo};
use crate::synth::Synth;
use crate::ugen::InputSource;
use crate::ugen::registry::{BuildContext, UgenRegistry};

/// A named control parameter with a default value (settable later via `set_control`).
#[derive(Clone, Debug)]
pub struct Param {
    /// Parameter name (for client-side name -> index resolution).
    pub name: String,
    /// Initial value.
    pub default: f32,
}

/// Where a UGen input comes from, as specified in a [`SynthDef`].
#[derive(Clone, Copy, Debug)]
pub enum InputRef {
    /// A constant value baked into the def.
    Constant(f32),
    /// The value of control parameter `index`.
    Param(u32),
    /// Output `output` of an earlier UGen `ugen` in the def.
    Ugen { ugen: u32, output: u32 },
}

/// One UGen within a [`SynthDef`] graph. UGens are listed in topological calc order.
#[derive(Clone, Debug)]
pub struct UgenSpec {
    /// Registry name (e.g. `"SinOsc"`).
    pub name: String,
    /// The UGen's calc rate.
    pub rate: Rate,
    /// Inputs, in order.
    pub inputs: Vec<InputRef>,
    /// Number of (audio-rate) outputs.
    pub num_outputs: usize,
    /// scsynth's `mSpecialIndex` (e.g. binary-op selector). Default `0`.
    pub special_index: i16,
}

impl UgenSpec {
    /// A convenience constructor with `special_index = 0`.
    pub fn new(
        name: impl Into<String>,
        rate: Rate,
        inputs: Vec<InputRef>,
        num_outputs: usize,
    ) -> Self {
        UgenSpec {
            name: name.into(),
            rate,
            inputs,
            num_outputs,
            special_index: 0,
        }
    }
}

/// A synth definition: a template instantiated into a [`Synth`].
#[derive(Clone, Debug)]
pub struct SynthDef {
    /// Definition name.
    pub name: String,
    /// Control parameters.
    pub params: Vec<Param>,
    /// UGens in topological calc order.
    pub ugens: Vec<UgenSpec>,
}

impl SynthDef {
    /// Resolve the index of the parameter named `name`, if any.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    /// Instantiate this def into a live [`Synth`] using `registry` for UGen construction.
    ///
    /// Runs entirely off the audio thread. All wires, scratch, and UGen state are allocated here.
    pub fn instantiate(
        &self,
        registry: &UgenRegistry,
        audio: &RateInfo,
        control: &RateInfo,
    ) -> Result<Box<Synth>, BuildError> {
        let block_size = audio.block_size;

        // Control wires: one per parameter, initialised to its default.
        let mut control_wires = vec![0.0f32; self.params.len()];
        let mut param_wires = Vec::with_capacity(self.params.len());
        for (i, param) in self.params.iter().enumerate() {
            control_wires[i] = param.default;
            param_wires.push(i as u32);
        }

        // Assign a distinct audio wire to each (ugen, output); record each UGen's first wire.
        let mut wire_base = Vec::with_capacity(self.ugens.len());
        let mut num_audio_wires = 0u32;
        for spec in &self.ugens {
            wire_base.push(num_audio_wires);
            num_audio_wires += spec.num_outputs as u32;
        }

        let mut ugens = Vec::with_capacity(self.ugens.len());
        let mut inputs_plan = Vec::with_capacity(self.ugens.len());
        let mut outputs_plan = Vec::with_capacity(self.ugens.len());
        let mut max_outputs = 0usize;

        for (u, spec) in self.ugens.iter().enumerate() {
            // Resolve each input to a concrete source (constant / control wire / audio wire).
            let mut sources = Vec::with_capacity(spec.inputs.len());
            for input in &spec.inputs {
                let source = match *input {
                    InputRef::Constant(v) => InputSource::Constant(v),
                    InputRef::Param(p) => {
                        let wire = *param_wires.get(p as usize).ok_or(BuildError::BadInputRef)?;
                        InputSource::Control(wire)
                    }
                    InputRef::Ugen { ugen, output } => {
                        let base = *wire_base
                            .get(ugen as usize)
                            .ok_or(BuildError::BadInputRef)?;
                        InputSource::Audio(base + output)
                    }
                };
                sources.push(source);
            }

            let input_rates: Vec<Rate> = sources.iter().map(|s| s.rate()).collect();
            let build_ctx = BuildContext {
                input_rates: &input_rates,
                audio,
                control,
                special_index: spec.special_index,
            };
            let ctor = registry
                .get(&spec.name)
                .ok_or_else(|| BuildError::UnknownUgen(spec.name.clone()))?;
            ugens.push(ctor.build(&build_ctx));
            inputs_plan.push(sources.into_boxed_slice());

            let base = wire_base[u];
            let out_wires: Vec<u32> = (0..spec.num_outputs as u32).map(|o| base + o).collect();
            outputs_plan.push(out_wires.into_boxed_slice());
            max_outputs = max_outputs.max(spec.num_outputs);
        }

        let audio_wires = vec![0.0f32; num_audio_wires as usize * block_size];
        let scratch = vec![0.0f32; max_outputs * block_size];

        Ok(Box::new(Synth::from_parts(
            ugens,
            audio_wires,
            control_wires,
            scratch,
            inputs_plan,
            outputs_plan,
            param_wires,
            block_size,
        )))
    }
}

/// A library of named synth definitions, owned control-side (scsynth's `GrafDefTable`).
#[derive(Clone, Debug, Default)]
pub struct SynthDefLibrary {
    map: HashMap<String, SynthDef>,
}

impl SynthDefLibrary {
    /// An empty library.
    pub fn new() -> Self {
        SynthDefLibrary {
            map: HashMap::new(),
        }
    }

    /// Add (or replace) a definition.
    pub fn insert(&mut self, def: SynthDef) {
        self.map.insert(def.name.clone(), def);
    }

    /// Look up a definition by name.
    pub fn get(&self, name: &str) -> Option<&SynthDef> {
        self.map.get(name)
    }
}
