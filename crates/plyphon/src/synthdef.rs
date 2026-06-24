//! Synth definitions and their compilation into [`GraphDef`]s.
//!
//! A [`SynthDef`] is the authored/parsed definition - SuperCollider's client-side `SynthDef`. It can
//! be built programmatically or parsed from the binary SCgf format (see [`read`]), so
//! `sclang`-compiled definitions load directly. [`SynthDef::compile`] turns it (off the audio thread,
//! once) into the immutable, shareable [`GraphDef`] - scsynth's server-side compiled form - from
//! which the audio thread constructs live [`Graph`](crate::graph::Graph)s with a single pool
//! allocation.

pub mod read;

use std::collections::HashMap;

use crate::error::BuildError;
use crate::graphdef::{GraphDef, OutputWire, UgenVtbl, build_layout};
use crate::rate::{Rate, RateInfo};
use crate::ugen::registry::{BuildContext, UgenRegistry};
use crate::ugen::{BuiltUgen, InputSource};

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

    /// Compile this def into an immutable [`GraphDef`] using `registry` for UGen construction.
    ///
    /// Runs entirely off the audio thread (the analogue of scsynth's `GraphDef_Recv`): it resolves
    /// the wiring, builds each UGen's vtable + initial state image, and computes the per-graph block
    /// layout. `max_wire_bufs`/`max_ugen_outputs` are the engine's shared-scratch capacities; a def
    /// exceeding either fails here rather than on the audio thread.
    pub fn compile(
        &self,
        registry: &UgenRegistry,
        audio: &RateInfo,
        control: &RateInfo,
        max_wire_bufs: usize,
        max_ugen_outputs: usize,
    ) -> Result<GraphDef, BuildError> {
        let block_size = audio.block_size;

        // Parameters occupy the first control wires.
        let num_params = self.params.len();
        let param_wires: Vec<u32> = (0..num_params as u32).collect();

        // Pass 1: assign a wire to each (ugen, output) by rate. Audio outputs go to audio wires;
        // control/scalar outputs go to control wires following the parameter wires.
        let mut num_audio_wires = 0u32;
        let mut num_control_wires = num_params as u32;
        let mut outputs_plan: Vec<Box<[OutputWire]>> = Vec::with_capacity(self.ugens.len());
        for spec in &self.ugens {
            let mut wires = Vec::with_capacity(spec.num_outputs);
            for _ in 0..spec.num_outputs {
                let wire = match spec.rate {
                    Rate::Audio => {
                        let w = num_audio_wires;
                        num_audio_wires += 1;
                        OutputWire {
                            rate: Rate::Audio,
                            wire: w,
                        }
                    }
                    Rate::Control | Rate::Scalar => {
                        let w = num_control_wires;
                        num_control_wires += 1;
                        OutputWire {
                            rate: spec.rate,
                            wire: w,
                        }
                    }
                };
                wires.push(wire);
            }
            outputs_plan.push(wires.into_boxed_slice());
        }

        // Control-wire defaults: parameters at their defaults, the rest (control-rate UGen outputs)
        // zeroed. These seed the per-graph control wires when an instance is built on the RT thread.
        let mut control_defaults = vec![0.0f32; num_control_wires as usize];
        for (i, param) in self.params.iter().enumerate() {
            control_defaults[i] = param.default;
        }

        // Pass 2: build each UGen and resolve its inputs against the assigned wires.
        let mut built: Vec<BuiltUgen> = Vec::with_capacity(self.ugens.len());
        let mut inputs_plan: Vec<Box<[InputSource]>> = Vec::with_capacity(self.ugens.len());
        let mut max_outputs = 0usize;
        for (u, spec) in self.ugens.iter().enumerate() {
            let mut sources = Vec::with_capacity(spec.inputs.len());
            for input in &spec.inputs {
                let source = match *input {
                    InputRef::Constant(v) => InputSource::Constant(v),
                    InputRef::Param(p) => {
                        let wire = *param_wires.get(p as usize).ok_or(BuildError::BadInputRef)?;
                        InputSource::Control(wire)
                    }
                    InputRef::Ugen { ugen, output } => {
                        let from = outputs_plan
                            .get(ugen as usize)
                            .and_then(|outs| outs.get(output as usize))
                            .ok_or(BuildError::BadInputRef)?;
                        match from.rate {
                            Rate::Audio => InputSource::Audio(from.wire),
                            Rate::Control | Rate::Scalar => InputSource::Control(from.wire),
                        }
                    }
                };
                sources.push(source);
            }

            let input_rates: Vec<Rate> = sources.iter().map(|s| s.rate()).collect();
            // A deterministic build-time seed; the real per-instance seed is applied on the RT thread
            // via `Ugen::reseed`, so this is only a placeholder for the baked state image.
            let build_ctx = BuildContext {
                input_rates: &input_rates,
                audio,
                control,
                rate: spec.rate,
                num_outputs: spec.num_outputs,
                special_index: spec.special_index,
                seed: (u as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            };
            let def = registry
                .get(&spec.name)
                .ok_or_else(|| BuildError::UnknownUgen(spec.name.clone()))?;
            built.push(def.build(&build_ctx)?);
            inputs_plan.push(sources.into_boxed_slice());
            max_outputs = max_outputs.max(spec.num_outputs);
        }

        // The shared-scratch capacities are fixed at boot; reject a def that would overflow them.
        if num_audio_wires as usize > max_wire_bufs {
            return Err(BuildError::TooManyWires {
                needed: num_audio_wires as usize,
                limit: max_wire_bufs,
            });
        }
        if max_outputs > max_ugen_outputs {
            return Err(BuildError::TooManyOutputs {
                needed: max_outputs,
                limit: max_ugen_outputs,
            });
        }

        // Lay out the per-graph block (state arena | control wires | param maps) and pack the initial
        // state-arena image from each UGen's initial state bytes.
        let state_slots: Vec<(usize, usize)> = built.iter().map(|b| (b.size, b.align)).collect();
        let (layout, state_offsets) =
            build_layout(&state_slots, num_control_wires as usize, num_params);
        let mut state_image = vec![0u8; layout.state.len];
        for (b, &off) in built.iter().zip(&state_offsets) {
            state_image[off..off + b.size].copy_from_slice(&b.init_bytes);
        }

        let ugens: Vec<UgenVtbl> = built
            .into_iter()
            .zip(inputs_plan)
            .zip(outputs_plan)
            .zip(state_offsets)
            .map(|(((b, inputs), outputs), state_offset)| UgenVtbl {
                process: b.process,
                init: b.init,
                reseed: b.reseed,
                inputs,
                outputs,
                state_offset,
                state_size: b.size,
            })
            .collect();

        Ok(GraphDef {
            ugens: ugens.into_boxed_slice(),
            layout,
            state_image: state_image.into_boxed_slice(),
            control_defaults: control_defaults.into_boxed_slice(),
            param_wires: param_wires.into_boxed_slice(),
            num_params,
            block_size,
        })
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
