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

use plyphon_dsp::math;
use plyphon_dsp::rate::{Rate, RateInfo};
use plyphon_unit::error::BuildError;
use plyphon_unit::graphdef::{
    AudioParam, GraphDef, LagParam, LocalBufSpec, OutputWire, UnitVtbl, build_layout,
};
use plyphon_unit::unit::demand::{BuiltDemandUnit, DemandVtbl, MAX_DEMAND_DEPTH, MAX_DEMAND_STATE};
use plyphon_unit::unit::registry::{BuildContext, UnitRegistry};
use plyphon_unit::unit::{BuiltUnit, InputSource};

/// `ln(0.001)` - scsynth's -60 dB decay target for lag coefficients.
const LOG001: f32 = -6.907_755;

/// A `LagControl` one-pole coefficient: the per-control-block multiplier that decays to 0.001 of a
/// step over `lag` seconds (`0` - immediate - for a non-positive lag). scsynth computes this from the
/// *control* rate, since the lag updates once per block.
fn lag_coef(lag: f32, control_rate: f32) -> f32 {
    if lag > 0.0 {
        math::exp(LOG001 / (lag * control_rate))
    } else {
        0.0
    }
}

/// A named control parameter with a default value (settable later via `set_control`).
#[derive(Clone, Debug)]
pub struct Param {
    /// Parameter name (for client-side name -> index resolution).
    pub name: String,
    /// Initial value.
    pub default: f32,
    /// The rate of the parameter's *output*. `Control` (the default) is an ordinary control; `Audio`
    /// is an `AudioControl` - its value is lifted to an audio wire each block, so it can feed
    /// audio-rate inputs and be mapped to an audio bus with `/n_mapa`. `Scalar`/`Demand` behave as
    /// `Control`. The `/n_set`/`/n_map`/`/c_get` target (the parameter's stored value) is unaffected.
    pub rate: Rate,
    /// Whether this is a `TrigControl`: its value is seen for exactly the block it is set, then resets
    /// to `0` (scsynth's "output then zero the control").
    pub is_trig: bool,
    /// `Some(lagTime)` for a `LagControl`: its value is smoothed by a one-pole that decays to within
    /// 0.1% of a new target over `lagTime` seconds (one step per control block). `None` otherwise.
    pub lag: Option<f32>,
}

impl Param {
    /// A control-rate parameter (the common case).
    pub fn control(name: impl Into<String>, default: f32) -> Self {
        Param {
            name: name.into(),
            default,
            rate: Rate::Control,
            is_trig: false,
            lag: None,
        }
    }

    /// An audio-rate parameter (`AudioControl`): its value feeds audio-rate inputs and can be mapped
    /// to an audio bus with `/n_mapa`.
    pub fn audio(name: impl Into<String>, default: f32) -> Self {
        Param {
            name: name.into(),
            default,
            rate: Rate::Audio,
            is_trig: false,
            lag: None,
        }
    }

    /// A trigger parameter (`TrigControl`): a `/n_set` is seen for one control block, then resets to 0.
    pub fn trig(name: impl Into<String>, default: f32) -> Self {
        Param {
            name: name.into(),
            default,
            rate: Rate::Control,
            is_trig: true,
            lag: None,
        }
    }

    /// A lagged parameter (`LagControl`): its value is de-zippered by a one-pole over `lag` seconds.
    pub fn lag(name: impl Into<String>, default: f32, lag: f32) -> Self {
        Param {
            name: name.into(),
            default,
            rate: Rate::Control,
            is_trig: false,
            lag: Some(lag),
        }
    }
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

    /// A `SendReply` unit that emits `/<path> [nodeID, replyID, values...]` on each rising edge of
    /// `trig`. Expands `path` to scsynth's constant-char input layout
    /// (`[trig, replyID, len, chars..., values...]`), so callers pass a plain `&str` instead of
    /// hand-encoding it. `rate` is the trigger rate (`Rate::Audio` or `Rate::Control`).
    pub fn send_reply(
        rate: Rate,
        trig: InputRef,
        reply_id: InputRef,
        path: &str,
        values: &[InputRef],
    ) -> Self {
        let mut inputs = vec![trig, reply_id, InputRef::Constant(path.len() as f32)];
        inputs.extend(path.bytes().map(|b| InputRef::Constant(b as f32)));
        inputs.extend_from_slice(values);
        UnitSpec::new("SendReply", rate, inputs, 0)
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
    /// exceeding either fails here rather than on the audio thread. `reblock`/`resample` are the
    /// graph's `Reblock`/`Resample` overrides (`None`/`1` for an ordinary def).
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        &self,
        registry: &UnitRegistry,
        audio: &RateInfo,
        control: &RateInfo,
        max_wire_bufs: usize,
        max_unit_outputs: usize,
        reblock: Option<usize>,
        resample: usize,
    ) -> Result<GraphDef, BuildError> {
        // The graph's control block: the World block, or a smaller power-of-two reblock (scsynth's
        // `Reblock(n)`) that divides it. The graph's units run at this finer rate.
        let block_size = match reblock {
            Some(b) if b >= 1 && b.is_power_of_two() && b <= audio.block_size => b,
            Some(b) => {
                return Err(BuildError::InvalidReblock {
                    block_size: b,
                    world: audio.block_size,
                });
            }
            None => audio.block_size,
        };
        // The oversample factor (scsynth's `Resample(n)`): the graph runs at `factor`x the World
        // sample rate, anti-aliasing nonlinear units. Power-of-two only (no downsampling).
        if resample == 0 || !resample.is_power_of_two() {
            return Err(BuildError::InvalidResample { factor: resample });
        }
        // The graph's own rate pair, baked into the units and handed to each `ProcessCtx`. An ordinary
        // def (World block, no oversampling) reuses the World's rates verbatim, so its output is
        // unchanged; a reblocked/resampled def derives a smaller-block / higher-rate pair.
        let graph_sr = audio.sample_rate * resample as f64;
        let (graph_audio, graph_control) = if block_size == audio.block_size && resample == 1 {
            (*audio, *control)
        } else {
            (
                RateInfo::new(graph_sr, block_size),
                RateInfo::new(graph_sr / block_size as f64, 1),
            )
        };

        // Parameters occupy the first control wires.
        let num_params = self.params.len();
        let param_wires: Vec<u32> = (0..num_params as u32).collect();

        // Pre-scan: tag each unit as demand-rate (with its index in the demand plan) or calc-rate.
        // Demand units are pulled on demand, so they get no wire and stay out of the per-block calc
        // list; the SynthDef is already topologically sorted, so the filtered lists stay in order.
        let mut demand_index: Vec<Option<u32>> = Vec::with_capacity(self.units.len());
        // Each non-demand unit's index into the runtime calc-unit list (and the `done_flags` span),
        // so a done-watching unit can resolve its source's calc index (scsynth's `mFromUnit`).
        let mut calc_index: Vec<Option<u32>> = Vec::with_capacity(self.units.len());
        let mut next_demand = 0u32;
        let mut next_calc = 0u32;
        for spec in &self.units {
            if spec.rate == Rate::Demand {
                // A demand source produces one value per pull, so it is single-output.
                if spec.num_outputs != 1 {
                    return Err(BuildError::DemandMultiOutput(spec.num_outputs));
                }
                demand_index.push(Some(next_demand));
                calc_index.push(None);
                next_demand += 1;
            } else {
                demand_index.push(None);
                calc_index.push(Some(next_calc));
                next_calc += 1;
            }
        }

        // Pre-scan the feedback bus (`LocalIn`/`LocalOut`): at most one of each in v1. The bus width
        // is the `LocalIn`'s output count; a `LocalOut`, if present, must write that many channels.
        let mut local_in_channels: Option<usize> = None;
        let mut local_out_channels: Option<usize> = None;
        for spec in &self.units {
            match spec.name.as_str() {
                "LocalIn" => {
                    if local_in_channels.is_some() {
                        return Err(BuildError::MultipleLocalBuses);
                    }
                    local_in_channels = Some(spec.num_outputs);
                }
                "LocalOut" => {
                    if local_out_channels.is_some() {
                        return Err(BuildError::MultipleLocalBuses);
                    }
                    local_out_channels = Some(spec.inputs.len());
                }
                _ => {}
            }
        }
        let num_local_channels = local_in_channels.unwrap_or(0);
        if let Some(local_out) = local_out_channels
            && local_out != num_local_channels
        {
            return Err(BuildError::LocalBusMismatch {
                local_in: num_local_channels,
                local_out,
            });
        }

        // Resolve each parameter. Every param's value lives in its control wire (`p`, the
        // `/n_set`/`/n_map`/`control_value` target). A *control* param's output is that wire directly;
        // an *audio* param (`AudioControl`) outputs an audio wire lifted from the value each block; a
        // *lagged* param (`LagControl`) outputs a separate control wire, one-poled from the value. A
        // *trig* param (`TrigControl`) outputs its value slot but is zeroed after the block.
        let mut num_audio_wires = 0u32;
        let mut num_control_wires = num_params as u32;
        let mut param_source: Vec<InputSource> = Vec::with_capacity(num_params);
        let mut audio_params: Vec<AudioParam> = Vec::new();
        let mut trig_params: Vec<u32> = Vec::new();
        let mut lag_params: Vec<LagParam> = Vec::new();
        let control_rate = control.sample_rate as f32;
        for (p, param) in self.params.iter().enumerate() {
            let value_slot = param_wires[p];
            if param.rate == Rate::Audio {
                let wire = num_audio_wires;
                num_audio_wires += 1;
                param_source.push(InputSource::Audio(wire));
                audio_params.push(AudioParam {
                    param: p as u32,
                    value_slot,
                    wire,
                });
            } else if let Some(lag) = param.lag {
                let wire = num_control_wires;
                num_control_wires += 1;
                param_source.push(InputSource::Control(wire));
                lag_params.push(LagParam {
                    value_slot,
                    wire,
                    b1: lag_coef(lag, control_rate),
                });
            } else {
                param_source.push(InputSource::Control(value_slot));
                if param.is_trig {
                    trig_params.push(value_slot);
                }
            }
        }

        // Pass 1: assign a wire to each (unit, output) of the *calc* units by rate. Audio outputs go
        // to audio wires (after the audio param wires); control/scalar outputs go to control wires
        // after the parameter and lag wires. Demand units get an empty slot so `outputs_plan` indexes.
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

        // The wire count is final here, so reject an oversized def before building any unit state -
        // a def thousands of wires over the cap would otherwise pay full graph construction (every
        // unit's state image) just to be refused.
        if num_audio_wires as usize > max_wire_bufs {
            return Err(BuildError::TooManyWires {
                needed: num_audio_wires as usize,
                limit: max_wire_bufs,
            });
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
        let mut calc_rates: Vec<Rate> = Vec::new();
        let mut demand_built: Vec<BuiltDemandUnit> = Vec::new();
        let mut demand_inputs: Vec<Box<[InputSource]>> = Vec::new();
        let mut max_outputs = 0usize;
        // Graph-local buffers (`LocalBuf`), collected in unit order: each built unit that declares
        // one gets the next declaration index (which the unit baked into its state from
        // `local_bufs_so_far`) and the next sample offset in the block's local-buffer span.
        let mut local_buf_specs: Vec<LocalBufSpec> = Vec::new();
        let mut local_buf_samples = 0usize;
        for (u, spec) in self.units.iter().enumerate() {
            let mut sources = Vec::with_capacity(spec.inputs.len());
            for input in &spec.inputs {
                let source = match *input {
                    InputRef::Constant(v) => InputSource::Constant(v),
                    InputRef::Param(p) => *param_source
                        .get(p as usize)
                        .ok_or(BuildError::BadInputRef)?,
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
            // Each input's source calc-unit index (scsynth's `mFromUnit`): the producing calc unit for
            // a unit input, else `None` (constants, params, and demand sources have no `mDone`).
            let input_units: Vec<Option<u32>> = spec
                .inputs
                .iter()
                .map(|input| match *input {
                    InputRef::Unit { unit, .. } => calc_index.get(unit as usize).copied().flatten(),
                    _ => None,
                })
                .collect();
            // A deterministic build-time seed; the real per-instance seed is applied on the RT thread
            // via `reseed`, so this is only a placeholder for the baked state image.
            let build_ctx = BuildContext {
                input_rates: &input_rates,
                input_units: &input_units,
                input_sources: &sources,
                audio: &graph_audio,
                control: &graph_control,
                rate: spec.rate,
                num_outputs: spec.num_outputs,
                special_index: spec.special_index,
                seed: (u as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                local_bufs_so_far: local_buf_specs.len(),
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
                let built = def.build(&build_ctx)?;
                // Collect a graph-local buffer declaration (`LocalBuf`), advancing the running
                // declaration index (`local_bufs_so_far` above) and the sample offset.
                if let Some((channels, frames)) = built.local_buf {
                    local_buf_specs.push(LocalBufSpec {
                        channels: channels as u32,
                        frames: frames as u32,
                        offset: local_buf_samples,
                    });
                    local_buf_samples += channels * frames;
                }
                calc_built.push(built);
                calc_inputs.push(sources.into_boxed_slice());
                calc_outputs.push(outputs_plan[u].clone());
                calc_rates.push(spec.rate);
                max_outputs = max_outputs.max(spec.num_outputs);
            }
        }

        // The shared-scratch capacities are fixed at boot; reject a def that would overflow them.
        // Demand units have no wires or output scratch, so only the calc units count here (the
        // audio-wire check ran before Pass 2, where the count was already final).
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
        // Per-calc-unit auxiliary memory (delay lines), in calc order - parallel to `state_slots`.
        let aux_slots: Vec<(usize, usize)> = calc_built
            .iter()
            .map(|b| (b.aux_bytes, b.aux_align))
            .collect();
        let (layout, state_offsets, demand_offsets, aux_offsets) = build_layout(
            &state_slots,
            &demand_state_slots,
            &aux_slots,
            num_control_wires as usize,
            num_params,
            num_local_channels,
            local_buf_specs.len(),
            local_buf_samples,
            lag_params.len(),
            block_size,
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
            .zip(calc_rates)
            .zip(state_offsets)
            .zip(aux_offsets)
            .map(
                |(((((b, inputs), outputs), rate), state_offset), aux_offset)| UnitVtbl {
                    rate,
                    process: b.process,
                    init: b.init,
                    reseed: b.reseed,
                    inputs,
                    outputs,
                    state_offset,
                    state_size: b.size,
                    aux_offset,
                    aux_size: b.aux_bytes,
                },
            )
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
            audio_params.into_boxed_slice(),
            trig_params.into_boxed_slice(),
            lag_params.into_boxed_slice(),
            local_buf_specs.into_boxed_slice(),
            num_params,
            graph_audio,
            graph_control,
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
