//! Synth definitions and their compilation into [`GraphDef`]s.
//!
//! A [`SynthDef`] is the authored/parsed definition - SuperCollider's client-side `SynthDef`. It can
//! be built programmatically or parsed from the binary SCgf format (see [`read`]), so
//! `sclang`-compiled definitions load directly. [`SynthDef::compile`] turns it (off the audio thread,
//! once) into the immutable, shareable [`GraphDef`] - scsynth's server-side compiled form - from
//! which the audio thread constructs live [`Graph`](plyphon_rt::graph::Graph)s with a single pool
//! allocation.

pub mod read;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;

use plyphon_dsp::rate::{Rate, RateInfo};
use plyphon_unit::error::BuildError;
use plyphon_unit::graphdef::{GraphDef, OutputWire, UnitVtbl, build_layout};
use plyphon_unit::unit::demand::{BuiltDemandUnit, DemandVtbl, MAX_DEMAND_DEPTH, MAX_DEMAND_STATE};
use plyphon_unit::unit::registry::{BuildContext, UnitRegistry};
use plyphon_unit::unit::{BuiltUnit, InputSource};

/// A named control parameter with a default value (settable later via `set_control`).
#[derive(Clone, Debug)]
pub struct Param {
    /// Parameter name (for client-side name -> index resolution).
    pub name: String,
    /// Initial value.
    pub default: f32,
}

/// Where a unit input comes from, as specified in a [`SynthDef`].
#[derive(Clone, Copy, Debug)]
pub enum InputRef {
    /// A constant value baked into the def.
    Constant(f32),
    /// The value of control parameter `index`.
    Param(u32),
    /// Output `output` of an earlier unit `unit` in the def.
    Unit { unit: u32, output: u32 },
}

/// One unit within a [`SynthDef`] graph. units are listed in topological calc order.
#[derive(Clone, Debug)]
pub struct UnitSpec {
    /// Registry name (e.g. `"SinOsc"`).
    pub name: String,
    /// The unit's calc rate.
    pub rate: Rate,
    /// Inputs, in order.
    pub inputs: Vec<InputRef>,
    /// Number of (audio-rate) outputs.
    pub num_outputs: usize,
    /// scsynth's `mSpecialIndex` (e.g. binary-op selector). Default `0`.
    pub special_index: i16,
}

impl UnitSpec {
    /// A convenience constructor with `special_index = 0`.
    pub fn new(
        name: impl Into<String>,
        rate: Rate,
        inputs: Vec<InputRef>,
        num_outputs: usize,
    ) -> Self {
        UnitSpec {
            name: name.into(),
            rate,
            inputs,
            num_outputs,
            special_index: 0,
        }
    }
}

/// A synth definition: a template instantiated into a [`plyphon_rt::graph::Graph`].
#[derive(Clone, Debug)]
pub struct SynthDef {
    /// Definition name.
    pub name: String,
    /// Control parameters.
    pub params: Vec<Param>,
    /// units in topological calc order.
    pub units: Vec<UnitSpec>,
}

impl SynthDef {
    /// Resolve the index of the parameter named `name`, if any.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    /// Compile this def into an immutable [`GraphDef`] using `registry` for unit construction.
    ///
    /// Runs entirely off the audio thread (the analogue of scsynth's `GraphDef_Recv`): it resolves
    /// the wiring, builds each unit's vtable + initial state image, and computes the per-graph block
    /// layout. `max_wire_bufs`/`max_unit_outputs` are the engine's shared-scratch capacities; a def
    /// exceeding either fails here rather than on the audio thread.
    pub fn compile(
        &self,
        registry: &UnitRegistry,
        audio: &RateInfo,
        control: &RateInfo,
        max_wire_bufs: usize,
        max_unit_outputs: usize,
    ) -> Result<GraphDef, BuildError> {
        let block_size = audio.block_size;

        // Parameters occupy the first control wires.
        let num_params = self.params.len();
        let param_wires: Vec<u32> = (0..num_params as u32).collect();

        // Pre-scan: tag each unit as demand-rate (with its index in the demand plan) or calc-rate.
        // Demand units are pulled on demand, so they get no wire and stay out of the per-block calc
        // list; the SynthDef is already topologically sorted, so the filtered lists stay in order.
        let mut demand_index: Vec<Option<u32>> = Vec::with_capacity(self.units.len());
        let mut next_demand = 0u32;
        for spec in &self.units {
            if spec.rate == Rate::Demand {
                // A demand source produces one value per pull, so it is single-output.
                if spec.num_outputs != 1 {
                    return Err(BuildError::DemandMultiOutput(spec.num_outputs));
                }
                demand_index.push(Some(next_demand));
                next_demand += 1;
            } else {
                demand_index.push(None);
            }
        }

        // Pass 1: assign a wire to each (unit, output) of the *calc* units by rate. Audio outputs go
        // to audio wires; control/scalar outputs go to control wires after the parameter wires.
        // Demand units get an empty slot so `outputs_plan` stays indexable by original unit index.
        let mut num_audio_wires = 0u32;
        let mut num_control_wires = num_params as u32;
        let mut outputs_plan: Vec<Box<[OutputWire]>> = Vec::with_capacity(self.units.len());
        for spec in &self.units {
            if spec.rate == Rate::Demand {
                outputs_plan.push(Vec::new().into_boxed_slice());
                continue;
            }
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
                    Rate::Demand => unreachable!("demand units are handled above"),
                };
                wires.push(wire);
            }
            outputs_plan.push(wires.into_boxed_slice());
        }

        // Control-wire defaults: parameters at their defaults, the rest (control-rate unit outputs)
        // zeroed. These seed the per-graph control wires when an instance is built on the RT thread.
        let mut control_defaults = vec![0.0f32; num_control_wires as usize];
        for (i, param) in self.params.iter().enumerate() {
            control_defaults[i] = param.default;
        }

        // Pass 2: build each unit, resolve its inputs, and partition into the calc list and the demand
        // plan. A unit input that references a demand unit resolves to `InputSource::Demand` (whether
        // the consumer is a calc unit or a nested demand unit); everything else resolves to a wire.
        let mut calc_built: Vec<BuiltUnit> = Vec::new();
        let mut calc_inputs: Vec<Box<[InputSource]>> = Vec::new();
        let mut calc_outputs: Vec<Box<[OutputWire]>> = Vec::new();
        let mut demand_built: Vec<BuiltDemandUnit> = Vec::new();
        let mut demand_inputs: Vec<Box<[InputSource]>> = Vec::new();
        let mut max_outputs = 0usize;
        for (u, spec) in self.units.iter().enumerate() {
            let mut sources = Vec::with_capacity(spec.inputs.len());
            for input in &spec.inputs {
                let source = match *input {
                    InputRef::Constant(v) => InputSource::Constant(v),
                    InputRef::Param(p) => {
                        let wire = *param_wires.get(p as usize).ok_or(BuildError::BadInputRef)?;
                        InputSource::Control(wire)
                    }
                    InputRef::Unit { unit, output } => {
                        let kind = *demand_index
                            .get(unit as usize)
                            .ok_or(BuildError::BadInputRef)?;
                        match kind {
                            Some(di) => {
                                // A demand source is single-output, so only output 0 is valid.
                                if output != 0 {
                                    return Err(BuildError::BadInputRef);
                                }
                                InputSource::Demand(di)
                            }
                            None => {
                                let from = outputs_plan
                                    .get(unit as usize)
                                    .and_then(|outs| outs.get(output as usize))
                                    .ok_or(BuildError::BadInputRef)?;
                                match from.rate {
                                    Rate::Audio => InputSource::Audio(from.wire),
                                    Rate::Control | Rate::Scalar => InputSource::Control(from.wire),
                                    Rate::Demand => {
                                        unreachable!("demand outputs are never wired")
                                    }
                                }
                            }
                        }
                    }
                };
                sources.push(source);
            }

            let input_rates: Vec<Rate> = sources.iter().map(|s| s.rate()).collect();
            // A deterministic build-time seed; the real per-instance seed is applied on the RT thread
            // via `reseed`, so this is only a placeholder for the baked state image.
            let build_ctx = BuildContext {
                input_rates: &input_rates,
                audio,
                control,
                rate: spec.rate,
                num_outputs: spec.num_outputs,
                special_index: spec.special_index,
                seed: (u as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            };

            if spec.rate == Rate::Demand {
                let def = registry
                    .get_demand(&spec.name)
                    .ok_or_else(|| BuildError::UnknownUnit(spec.name.clone()))?;
                let built = def.build(&build_ctx)?;
                // The audio thread pulls a demand unit into a fixed stack buffer; reject oversize state.
                if built.size > MAX_DEMAND_STATE {
                    return Err(BuildError::DemandStateTooLarge {
                        needed: built.size,
                        limit: MAX_DEMAND_STATE,
                    });
                }
                demand_built.push(built);
                demand_inputs.push(sources.into_boxed_slice());
            } else {
                let def = registry
                    .get(&spec.name)
                    .ok_or_else(|| BuildError::UnknownUnit(spec.name.clone()))?;
                calc_built.push(def.build(&build_ctx)?);
                calc_inputs.push(sources.into_boxed_slice());
                calc_outputs.push(outputs_plan[u].clone());
                max_outputs = max_outputs.max(spec.num_outputs);
            }
        }

        // The shared-scratch capacities are fixed at boot; reject a def that would overflow them.
        // Demand units have no wires or output scratch, so only the calc units count here.
        if num_audio_wires as usize > max_wire_bufs {
            return Err(BuildError::TooManyWires {
                needed: num_audio_wires as usize,
                limit: max_wire_bufs,
            });
        }
        if max_outputs > max_unit_outputs {
            return Err(BuildError::TooManyOutputs {
                needed: max_outputs,
                limit: max_unit_outputs,
            });
        }

        // Reject demand graphs that nest deeper than the audio thread's recursion bound. `depth[di]`
        // is the longest chain of nested demand units rooted at `di`; demand inputs reference earlier
        // demand units (topological order), so a single forward pass suffices.
        let mut depth = vec![0usize; demand_built.len()];
        for (di, inputs) in demand_inputs.iter().enumerate() {
            let mut d = 1usize;
            for src in inputs.iter() {
                if let InputSource::Demand(child) = *src {
                    d = d.max(1 + depth[child as usize]);
                }
            }
            depth[di] = d;
        }
        let max_depth = depth.iter().copied().max().unwrap_or(0);
        if max_depth > MAX_DEMAND_DEPTH {
            return Err(BuildError::DemandNestingTooDeep {
                depth: max_depth,
                limit: MAX_DEMAND_DEPTH,
            });
        }

        // Lay out the per-graph block (calc state | demand state | control wires | param maps) and
        // pack each arena's initial image from the units' initial state bytes.
        let state_slots: Vec<(usize, usize)> =
            calc_built.iter().map(|b| (b.size, b.align)).collect();
        let demand_state_slots: Vec<(usize, usize)> =
            demand_built.iter().map(|b| (b.size, b.align)).collect();
        let (layout, state_offsets, demand_offsets) = build_layout(
            &state_slots,
            &demand_state_slots,
            num_control_wires as usize,
            num_params,
        );
        let mut state_image = vec![0u8; layout.state.len];
        for (b, &off) in calc_built.iter().zip(&state_offsets) {
            state_image[off..off + b.size].copy_from_slice(&b.init_bytes);
        }
        let mut demand_state_image = vec![0u8; layout.demand_state.len];
        for (b, &off) in demand_built.iter().zip(&demand_offsets) {
            demand_state_image[off..off + b.size].copy_from_slice(&b.init_bytes);
        }

        let units: Vec<UnitVtbl> = calc_built
            .into_iter()
            .zip(calc_inputs)
            .zip(calc_outputs)
            .zip(state_offsets)
            .map(|(((b, inputs), outputs), state_offset)| UnitVtbl {
                process: b.process,
                init: b.init,
                reseed: b.reseed,
                inputs,
                outputs,
                state_offset,
                state_size: b.size,
            })
            .collect();

        let demand_units: Vec<DemandVtbl> = demand_built
            .into_iter()
            .zip(demand_inputs)
            .zip(demand_offsets)
            .map(|((b, inputs), state_offset)| DemandVtbl {
                produce: b.produce,
                reset: b.reset,
                reseed: b.reseed,
                inputs,
                state_offset,
                state_size: b.size,
            })
            .collect();

        Ok(GraphDef::new(
            units.into_boxed_slice(),
            demand_units.into_boxed_slice(),
            layout,
            state_image.into_boxed_slice(),
            demand_state_image.into_boxed_slice(),
            control_defaults.into_boxed_slice(),
            param_wires.into_boxed_slice(),
            num_params,
            block_size,
        ))
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

    /// Remove a definition by name, returning it if present (scsynth's `/d_free`).
    pub fn remove(&mut self, name: &str) -> Option<SynthDef> {
        self.map.remove(name)
    }

    /// The names of every definition currently in the library.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }
}
