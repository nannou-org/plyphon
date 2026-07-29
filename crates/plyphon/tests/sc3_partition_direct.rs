//! Direct callback-partition checks for the SC3 processing and spectral units.
//!
//! These checks deliberately bypass [`plyphon::World`]. `World::fill` reblocks host reads to the
//! graph's configured block size, so splitting host output does not prove that a unit's own
//! callback is partition invariant.

use plyphon::{
    Buffer, BuildContext, BuiltUnit, InitCtx, Inputs, Outputs, ProcessCtx, Rate, RateInfo,
};
use plyphon_dsp::buffer::{BufferTable, SpectrumCoord};
use plyphon_dsp::bus::Buses;
use plyphon_dsp::fft::FftTables;
use plyphon_dsp::rng::Rng;
use plyphon_dsp::wavetable::Wavetables;
use plyphon_unit::UnitRegistry;
use plyphon_unit::unit::demand::{DemandAccess, DemandVtbl, DemandWorld};
use plyphon_unit::unit::{
    Aux, DoneState, InputSource, LocalBufs, LocalBus, NodeMsg, NodeMsgSink, NodeOp, NodeOpSink,
    Trigger, TriggerSink,
};

/// Direct callback lengths whose sum is one ordinary 64-sample callback.
const PARTITION: [usize; 4] = [1, 7, 13, 43];
/// Number of samples used to put each separately constructed sample unit in the same warm state.
const WARM_SAMPLES: usize = 64;
/// Fixed seed shared by the two sides of every direct comparison.
const UNIT_SEED: u64 = 0x5C31_00C0_FFEE;

/// Aligned owned bytes suitable for invoking a type-erased unit vtable.
///
/// The public vtables validate alignment when they reinterpret state. Backing the bytes with
/// `u128` words gives every current built-in state and auxiliary region at least its requested
/// alignment without adding a production-only test hook.
struct AlignedBytes {
    words: Vec<u128>,
    len: usize,
}

impl AlignedBytes {
    /// Copy an initial state image into aligned storage.
    fn from_bytes(bytes: &[u8], required_alignment: usize) -> Self {
        assert!(
            required_alignment <= align_of::<u128>(),
            "test harness needs a wider aligned backing type"
        );
        let mut storage = Self::zeroed(bytes.len(), required_alignment);
        storage.as_mut().copy_from_slice(bytes);
        storage
    }

    /// Allocate a zeroed aligned byte region.
    fn zeroed(len: usize, required_alignment: usize) -> Self {
        assert!(
            required_alignment <= align_of::<u128>(),
            "test harness needs a wider aligned backing type"
        );
        Self {
            words: vec![0; len.div_ceil(size_of::<u128>())],
            len,
        }
    }

    /// Borrow the logical byte region mutably.
    fn as_mut(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.words)[..self.len]
    }

    /// Snapshot the logical byte region.
    fn to_vec(&self) -> Vec<u8> {
        bytemuck::cast_slice(&self.words)[..self.len].to_vec()
    }
}

/// One fixed control or deterministic changing audio inlet in a sample-unit case.
#[derive(Clone, Copy)]
enum PartitionInput {
    /// A callback-retained control value.
    Control(f32),
    /// An audio signal spanning `low..=high`, generated from the absolute sample position.
    Audio {
        low: f32,
        high: f32,
        stride: usize,
        offset: usize,
    },
}

impl PartitionInput {
    /// Produce one deterministic audio sample while preserving absolute sample order across calls.
    fn sample(self, absolute_sample: usize) -> f32 {
        let Self::Audio {
            low,
            high,
            stride,
            offset,
        } = self
        else {
            unreachable!("only an audio input produces samples");
        };
        let phase = ((absolute_sample * stride + offset) % 257) as f32 / 256.0;
        low + (high - low) * phase
    }
}

/// One direct sample-processor partition case.
struct SampleCase {
    name: &'static str,
    inputs: Vec<PartitionInput>,
    outputs: usize,
    tolerance: f32,
}

impl SampleCase {
    /// Resolve the build-time source wiring for this case.
    fn sources(&self) -> Vec<InputSource> {
        let mut audio_wire = 0;
        let mut control_wire = 0;
        self.inputs
            .iter()
            .map(|input| match input {
                PartitionInput::Audio { .. } => {
                    let source = InputSource::Audio(audio_wire);
                    audio_wire += 1;
                    source
                }
                PartitionInput::Control(_) => {
                    let source = InputSource::Control(control_wire);
                    control_wire += 1;
                    source
                }
            })
            .collect()
    }

    /// Build flat audio and control wires for an absolute sample range.
    fn input_block(&self, start: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
        let mut audio = Vec::new();
        let mut controls = Vec::new();
        for input in &self.inputs {
            match *input {
                PartitionInput::Audio { .. } => {
                    audio.extend((start..start + len).map(|sample| input.sample(sample)));
                }
                PartitionInput::Control(value) => controls.push(value),
            }
        }
        (audio, controls)
    }
}

/// A type-erased calc unit with just enough world state to invoke its callback directly.
struct DirectCalc {
    built: BuiltUnit,
    state: AlignedBytes,
    aux: AlignedBytes,
    sources: Vec<InputSource>,
    sample_rate: f64,
    configured_block_size: usize,
    rate: Rate,
    outputs: usize,
    constructed: bool,
    initialized: bool,
    observed_callbacks: Vec<usize>,
    wavetables: Wavetables,
    fft: FftTables,
    buses: Buses,
    buffers: BufferTable,
    rgen: Rng,
}

impl DirectCalc {
    /// Construct one calc unit through the same public registry and build context used by SynthDefs.
    fn new(
        name: &str,
        sources: Vec<InputSource>,
        rate: Rate,
        outputs: usize,
        sample_rate: f64,
        configured_block_size: usize,
    ) -> Self {
        Self::new_with_special(
            name,
            sources,
            rate,
            outputs,
            sample_rate,
            configured_block_size,
            0,
        )
    }

    /// Construct one calc unit with an explicit SynthDef special index.
    fn new_with_special(
        name: &str,
        sources: Vec<InputSource>,
        rate: Rate,
        outputs: usize,
        sample_rate: f64,
        configured_block_size: usize,
        special_index: i16,
    ) -> Self {
        let input_rates = sources
            .iter()
            .copied()
            .map(InputSource::rate)
            .collect::<Vec<_>>();
        let input_units = vec![None; sources.len()];
        let audio = RateInfo::new(sample_rate, configured_block_size);
        let control = RateInfo::new(sample_rate / configured_block_size as f64, 1);
        let registry = UnitRegistry::with_builtins();
        let build = BuildContext {
            input_rates: &input_rates,
            input_units: &input_units,
            input_sources: &sources,
            rate,
            num_outputs: outputs,
            audio: &audio,
            control: &control,
            special_index,
            seed: UNIT_SEED,
            local_bufs_so_far: 0,
        };
        let built = registry
            .get(name)
            .unwrap_or_else(|| panic!("{name} calc registration"))
            .build(&build)
            .unwrap_or_else(|error| panic!("{name} direct build failed: {error:?}"));
        let mut state = AlignedBytes::from_bytes(&built.init_bytes, built.align);
        (built.reseed)(state.as_mut(), UNIT_SEED);
        let aux = AlignedBytes::zeroed(built.aux_bytes, built.aux_align);
        Self {
            built,
            state,
            aux,
            sources,
            sample_rate,
            configured_block_size,
            rate,
            outputs,
            constructed: false,
            initialized: false,
            observed_callbacks: Vec::new(),
            wavetables: Wavetables::new(),
            fft: FftTables::new(),
            buses: Buses::new(0, 0, 0, 0, configured_block_size),
            buffers: BufferTable::new(2),
            rgen: Rng::new(UNIT_SEED),
        }
    }

    /// Run the source constructor callback once and return its published output wires.
    fn construct_once(
        &mut self,
        audio_wires: &[f32],
        control_wires: &[f32],
        callback_len: usize,
    ) -> Vec<Vec<f32>> {
        assert!(!self.constructed);
        let calc_len = if self.rate == Rate::Audio {
            callback_len
        } else {
            1
        };
        let audio = RateInfo::new(self.sample_rate, self.configured_block_size);
        let control = RateInfo::new(self.sample_rate / self.configured_block_size as f64, 1);
        let own = if self.rate == Rate::Audio {
            &audio
        } else {
            &control
        };
        let inputs = Inputs::new(&self.sources, audio_wires, control_wires, callback_len);
        let mut scratch = vec![0.0; self.outputs * calc_len];
        let mut demand_state = [];
        let mut triggers: Vec<Trigger> = Vec::new();
        let mut node_messages: Vec<NodeMsg> = Vec::new();
        let mut node_ops: Vec<NodeOp> = Vec::new();
        let mut own_done = 0;
        let mut local_bus = [];
        let mut local_samples = [];
        let mut local_coords = [];
        let mut ctx = ProcessCtx {
            audio: &audio,
            control: &control,
            own,
            wavetables: &self.wavetables,
            fft: &self.fft,
            ins: inputs,
            outs: Outputs::new(&mut scratch, calc_len),
            buses: &mut self.buses,
            buffers: &mut self.buffers,
            buf_counter: 1,
            tick: 0,
            resample_factor: 1,
            sample_offset: 0,
            subsample_offset: 0.0,
            demand: DemandAccess::new(
                &[],
                &mut demand_state,
                audio_wires,
                control_wires,
                callback_len,
            ),
            node_id: 1000,
            triggers: TriggerSink::new(&mut triggers, 0),
            node_msgs: NodeMsgSink::new(&mut node_messages, 0),
            running_synths: 1,
            done: DoneState::new(&[], &mut own_done),
            node_ops: NodeOpSink::new(&mut node_ops, 0),
            local: LocalBus::new(&mut local_bus, callback_len),
            local_bufs: LocalBufs::new(
                &[],
                &mut local_samples,
                &mut local_coords,
                self.sample_rate,
            ),
            aux: Aux::new(self.aux.as_mut()),
            rgen: &mut self.rgen,
        };
        (self.built.construct)(self.state.as_mut(), &mut ctx);
        self.constructed = true;

        (0..self.outputs)
            .map(|output| scratch[output * calc_len..(output + 1) * calc_len].to_vec())
            .collect()
    }

    /// Invoke one real unit callback and return its outputs split by output index.
    fn process(
        &mut self,
        audio_wires: &[f32],
        control_wires: &[f32],
        callback_len: usize,
    ) -> Vec<Vec<f32>> {
        assert!(callback_len > 0);
        self.observed_callbacks.push(callback_len);
        if !self.constructed {
            let _ = self.construct_once(audio_wires, control_wires, callback_len);
        }
        let calc_len = if self.rate == Rate::Audio {
            callback_len
        } else {
            1
        };
        let audio = RateInfo::new(self.sample_rate, self.configured_block_size);
        let control = RateInfo::new(self.sample_rate / self.configured_block_size as f64, 1);
        let own = if self.rate == Rate::Audio {
            &audio
        } else {
            &control
        };
        let inputs = Inputs::new(&self.sources, audio_wires, control_wires, callback_len);

        if !self.initialized {
            let mut local_samples = [];
            let mut local_coords = [];
            let init = InitCtx {
                audio: &audio,
                control: &control,
                own,
                wavetables: &self.wavetables,
                fft: &self.fft,
                ins: inputs,
                buses: &self.buses,
                buffers: &self.buffers,
                local_bufs: LocalBufs::new(
                    &[],
                    &mut local_samples,
                    &mut local_coords,
                    self.sample_rate,
                ),
                buf_counter: 1,
            };
            (self.built.init)(self.state.as_mut(), &init);
            self.initialized = true;
        }

        let mut scratch = vec![0.0; self.outputs * calc_len];
        let mut demand_state = [];
        let mut triggers: Vec<Trigger> = Vec::new();
        let mut node_messages: Vec<NodeMsg> = Vec::new();
        let mut node_ops: Vec<NodeOp> = Vec::new();
        let mut own_done = 0;
        let mut local_bus = [];
        let mut local_samples = [];
        let mut local_coords = [];
        let mut ctx = ProcessCtx {
            audio: &audio,
            control: &control,
            own,
            wavetables: &self.wavetables,
            fft: &self.fft,
            ins: inputs,
            outs: Outputs::new(&mut scratch, calc_len),
            buses: &mut self.buses,
            buffers: &mut self.buffers,
            buf_counter: 1,
            tick: 0,
            resample_factor: 1,
            sample_offset: 0,
            subsample_offset: 0.0,
            demand: DemandAccess::new(
                &[],
                &mut demand_state,
                audio_wires,
                control_wires,
                callback_len,
            ),
            node_id: 1000,
            triggers: TriggerSink::new(&mut triggers, 0),
            node_msgs: NodeMsgSink::new(&mut node_messages, 0),
            running_synths: 1,
            done: DoneState::new(&[], &mut own_done),
            node_ops: NodeOpSink::new(&mut node_ops, 0),
            local: LocalBus::new(&mut local_bus, callback_len),
            local_bufs: LocalBufs::new(
                &[],
                &mut local_samples,
                &mut local_coords,
                self.sample_rate,
            ),
            aux: Aux::new(self.aux.as_mut()),
            rgen: &mut self.rgen,
        };
        (self.built.process)(self.state.as_mut(), &mut ctx);

        (0..self.outputs)
            .map(|output| scratch[output * calc_len..(output + 1) * calc_len].to_vec())
            .collect()
    }

    /// Forget warm-up callback observations without changing unit state.
    fn clear_observations(&mut self) {
        self.observed_callbacks.clear();
    }

    /// Snapshot the complete type-erased state and auxiliary memory.
    fn state_snapshot(&self) -> (Vec<u8>, Vec<u8>) {
        (self.state.to_vec(), self.aux.to_vec())
    }
}

/// A direct demand-rate source and the minimum world reach required to pull it.
struct DirectDemand {
    plan: Vec<DemandVtbl>,
    state: AlignedBytes,
    buffers: BufferTable,
    rgen: Rng,
    sample_rate: f64,
    block_size: usize,
    observed_groups: Vec<usize>,
    pull_count: usize,
}

impl DirectDemand {
    /// Construct one `DNoiseRing` source with fixed controls and an identical shared RNG.
    fn new(sample_rate: f64, block_size: usize) -> Self {
        let sources = vec![
            InputSource::Constant(0.625),
            InputSource::Constant(0.375),
            InputSource::Constant(3.0),
            InputSource::Constant(11.0),
            InputSource::Constant(341.0),
        ];
        let rates = sources
            .iter()
            .copied()
            .map(InputSource::rate)
            .collect::<Vec<_>>();
        let input_units = vec![None; sources.len()];
        let audio = RateInfo::new(sample_rate, block_size);
        let control = RateInfo::new(sample_rate / block_size as f64, 1);
        let registry = UnitRegistry::with_builtins();
        let build = BuildContext {
            input_rates: &rates,
            input_units: &input_units,
            input_sources: &sources,
            rate: Rate::Demand,
            num_outputs: 1,
            audio: &audio,
            control: &control,
            special_index: 0,
            seed: UNIT_SEED,
            local_bufs_so_far: 0,
        };
        let built = registry
            .get_demand("DNoiseRing")
            .expect("DNoiseRing demand registration")
            .build(&build)
            .expect("DNoiseRing direct build");
        let output_offset = (built.size + 3) & !3;
        let mut init_bytes = vec![0; output_offset + core::mem::size_of::<f32>()];
        init_bytes[..built.init_bytes.len()].copy_from_slice(&built.init_bytes);
        let mut state = AlignedBytes::from_bytes(&init_bytes, built.align.max(4));
        (built.reseed)(&mut state.as_mut()[..built.size], UNIT_SEED);
        let plan = vec![DemandVtbl {
            init: built.init,
            produce: built.produce,
            reset: built.reset,
            reseed: built.reseed,
            inputs: sources.into_boxed_slice(),
            state_offset: 0,
            state_size: built.size,
            output_offset,
        }];
        let mut demand = Self {
            plan,
            state,
            buffers: BufferTable::new(0),
            rgen: Rng::new(UNIT_SEED),
            sample_rate,
            block_size,
            observed_groups: Vec::new(),
            pull_count: 0,
        };
        demand.initialize();
        demand
    }

    /// Run the demand plan's constructor callbacks before the first direct pull.
    fn initialize(&mut self) {
        let mut local_samples = [];
        let mut local_coords = [];
        let mut local_bufs =
            LocalBufs::new(&[], &mut local_samples, &mut local_coords, self.sample_rate);
        let mut node_messages: Vec<NodeMsg> = Vec::new();
        let mut node_sink = NodeMsgSink::new(&mut node_messages, 0);
        let unit_count = self.plan.len();
        let mut access =
            DemandAccess::new(&self.plan, self.state.as_mut(), &[], &[], self.block_size);
        let mut world = DemandWorld {
            buffers: &mut self.buffers,
            local_bufs: &mut local_bufs,
            node_id: 1000,
            node_msgs: &mut node_sink,
            rgen: &mut self.rgen,
        };
        for unit in 0..unit_count {
            access.init(&mut world, unit);
        }
    }

    /// Pull exactly `count` consecutive values while recording the caller's grouping.
    fn pull_group(&mut self, count: usize) -> Vec<f32> {
        self.observed_groups.push(count);
        let mut local_samples = [];
        let mut local_coords = [];
        let mut local_bufs =
            LocalBufs::new(&[], &mut local_samples, &mut local_coords, self.sample_rate);
        let mut node_messages: Vec<NodeMsg> = Vec::new();
        let mut node_sink = NodeMsgSink::new(&mut node_messages, 0);
        let mut access =
            DemandAccess::new(&self.plan, self.state.as_mut(), &[], &[], self.block_size);
        let mut world = DemandWorld {
            buffers: &mut self.buffers,
            local_bufs: &mut local_bufs,
            node_id: 1000,
            node_msgs: &mut node_sink,
            rgen: &mut self.rgen,
        };
        let values = (0..count)
            .map(|_| access.produce(&mut world, 0, 1))
            .collect::<Vec<_>>();
        self.pull_count += count;
        values
    }

    /// Reset only the observations used by the assertion after an identical warm-up.
    fn clear_observations(&mut self) {
        self.observed_groups.clear();
        self.pull_count = 0;
    }

    /// Snapshot the complete demand arena and synth-shared random generator state.
    fn state_snapshot(&self) -> (Vec<u8>, Vec<u8>) {
        (self.state.to_vec(), bytemuck::bytes_of(&self.rgen).to_vec())
    }
}

/// A full packed-spectrum snapshot, including its coordinate interpretation.
#[derive(Debug, PartialEq)]
struct SpectrumSnapshot {
    data_bits: Vec<u32>,
    coord: SpectrumCoord,
}

/// Return all sample-processor families with changing signal inlets and retained controls.
fn sample_cases() -> Vec<SampleCase> {
    let signal = || PartitionInput::Audio {
        low: -0.75,
        high: 0.75,
        stride: 37,
        offset: 11,
    };
    let frequency = || PartitionInput::Control(1_467.0);
    vec![
        SampleCase {
            name: "EnvDetect",
            inputs: vec![
                signal(),
                PartitionInput::Control(0.002),
                PartitionInput::Control(0.01),
            ],
            outputs: 1,
            tolerance: 0.0,
        },
        SampleCase {
            name: "Decimator",
            inputs: vec![
                signal(),
                PartitionInput::Control(12_000.0),
                PartitionInput::Control(8.0),
            ],
            outputs: 1,
            tolerance: 0.0,
        },
        SampleCase {
            name: "DFM1",
            inputs: vec![
                signal(),
                PartitionInput::Control(1_400.0),
                PartitionInput::Control(0.25),
                PartitionInput::Control(0.75),
                PartitionInput::Control(0.0),
                PartitionInput::Control(0.0),
            ],
            outputs: 1,
            tolerance: 2.0e-6,
        },
        SampleCase {
            name: "BMoog",
            inputs: vec![
                signal(),
                PartitionInput::Control(1_400.0),
                PartitionInput::Control(0.65),
                PartitionInput::Control(0.0),
            ],
            outputs: 1,
            tolerance: 2.0e-6,
        },
        SampleCase {
            name: "MoogLadder",
            inputs: vec![
                signal(),
                PartitionInput::Control(1_400.0),
                PartitionInput::Control(0.35),
            ],
            outputs: 1,
            tolerance: 2.0e-6,
        },
        SampleCase {
            name: "MoogVCF",
            inputs: vec![
                signal(),
                PartitionInput::Control(1_400.0),
                PartitionInput::Control(0.35),
            ],
            outputs: 1,
            tolerance: 2.0e-6,
        },
        SampleCase {
            name: "BlitB3",
            inputs: vec![frequency()],
            outputs: 1,
            tolerance: 0.0,
        },
        SampleCase {
            name: "BlitB3Saw",
            inputs: vec![frequency(), PartitionInput::Control(0.4)],
            outputs: 1,
            tolerance: 0.0,
        },
        SampleCase {
            name: "BlitB3Square",
            inputs: vec![frequency(), PartitionInput::Control(0.4)],
            outputs: 1,
            tolerance: 0.0,
        },
        SampleCase {
            name: "BlitB3Tri",
            inputs: vec![
                frequency(),
                PartitionInput::Control(0.4),
                PartitionInput::Control(0.7),
            ],
            outputs: 1,
            tolerance: 0.0,
        },
        SampleCase {
            name: "Perlin3",
            inputs: vec![
                PartitionInput::Audio {
                    low: -1.5,
                    high: 1.5,
                    stride: 17,
                    offset: 3,
                },
                PartitionInput::Audio {
                    low: -0.75,
                    high: 2.0,
                    stride: 29,
                    offset: 5,
                },
                PartitionInput::Audio {
                    low: 0.25,
                    high: 3.0,
                    stride: 43,
                    offset: 13,
                },
            ],
            outputs: 1,
            tolerance: 0.0,
        },
        SampleCase {
            name: "RosslerL",
            inputs: vec![
                PartitionInput::Control(6_000.0),
                PartitionInput::Control(0.2),
                PartitionInput::Control(0.2),
                PartitionInput::Control(5.7),
                PartitionInput::Control(0.05),
                PartitionInput::Control(0.1),
                PartitionInput::Control(0.0),
                PartitionInput::Control(0.0),
            ],
            outputs: 3,
            tolerance: 0.0,
        },
    ]
}

/// One direct runtime-safety case with mutable wire values for every logical inlet.
struct RuntimeSafetyCase {
    label: &'static str,
    name: &'static str,
    rate: Rate,
    sources: Vec<InputSource>,
    baseline: Vec<f32>,
    recovery: Vec<f32>,
    outputs: usize,
}

impl RuntimeSafetyCase {
    /// Construct a case by assigning dense wire indices to the requested inlet rates.
    fn new(
        label: &'static str,
        name: &'static str,
        rate: Rate,
        inlet_rates: &[Rate],
        baseline: &[f32],
        recovery: &[f32],
        outputs: usize,
    ) -> Self {
        assert_eq!(inlet_rates.len(), baseline.len());
        assert_eq!(baseline.len(), recovery.len());
        let mut audio_wire = 0;
        let mut control_wire = 0;
        let sources = inlet_rates
            .iter()
            .map(|rate| match rate {
                Rate::Audio => {
                    let source = InputSource::Audio(audio_wire);
                    audio_wire += 1;
                    source
                }
                Rate::Control | Rate::Scalar => {
                    let source = InputSource::Control(control_wire);
                    control_wire += 1;
                    source
                }
                Rate::Demand => unreachable!("sample safety cases have no demand inlets"),
            })
            .collect();
        Self {
            label,
            name,
            rate,
            sources,
            baseline: baseline.to_vec(),
            recovery: recovery.to_vec(),
            outputs,
        }
    }

    /// Expand logical inlet values into the channel-major audio and scalar control wire arenas.
    fn input_block(&self, values: &[f32], len: usize) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(values.len(), self.sources.len());
        let audio_count = self
            .sources
            .iter()
            .filter(|source| matches!(source, InputSource::Audio(_)))
            .count();
        let control_count = self.sources.len() - audio_count;
        let mut audio = vec![0.0; audio_count * len];
        let mut controls = vec![0.0; control_count];
        for (&source, &value) in self.sources.iter().zip(values) {
            match source {
                InputSource::Audio(wire) => {
                    audio[wire as usize * len..(wire as usize + 1) * len].fill(value);
                }
                InputSource::Control(wire) => controls[wire as usize] = value,
                InputSource::Constant(_) | InputSource::Demand(_) => {
                    unreachable!("runtime safety values use mutable calc wires")
                }
            }
        }
        (audio, controls)
    }
}

/// Return every sample-processor configuration whose runtime safety contract needs exact state.
fn runtime_safety_cases() -> Vec<RuntimeSafetyCase> {
    use Rate::{Audio, Control};

    vec![
        RuntimeSafetyCase::new(
            "EnvDetect.ar",
            "EnvDetect",
            Audio,
            &[Audio, Control, Control],
            &[0.375, 0.002, 0.01],
            &[-0.25, 0.004, 0.02],
            1,
        ),
        RuntimeSafetyCase::new(
            "Decimator.ar",
            "Decimator",
            Audio,
            &[Audio, Control, Control],
            &[0.375, 12_000.0, 8.0],
            &[-0.25, 24_000.0, 12.0],
            1,
        ),
        RuntimeSafetyCase::new(
            "DFM1.ar",
            "DFM1",
            Audio,
            &[Audio, Control, Control, Control, Control, Control],
            &[0.375, 1_000.0, 0.2, 1.0, 0.0, 0.025],
            &[-0.25, 2_000.0, 0.4, 0.75, 1.0, 0.01],
            1,
        ),
        RuntimeSafetyCase::new(
            "BMoog.ar",
            "BMoog",
            Audio,
            &[Audio, Control, Control, Control],
            &[0.375, 1_000.0, 0.2, 0.0],
            &[-0.25, 2_000.0, 0.4, 2.0],
            1,
        ),
        RuntimeSafetyCase::new(
            "MoogLadder.ar controls",
            "MoogLadder",
            Audio,
            &[Audio, Control, Control],
            &[0.375, 1_000.0, 0.2],
            &[-0.25, 2_000.0, 0.4],
            1,
        ),
        RuntimeSafetyCase::new(
            "MoogLadder.ar audio controls",
            "MoogLadder",
            Audio,
            &[Audio, Audio, Audio],
            &[0.375, 1_000.0, 0.2],
            &[-0.25, 2_000.0, 0.4],
            1,
        ),
        RuntimeSafetyCase::new(
            "MoogLadder.kr",
            "MoogLadder",
            Control,
            &[Control, Control, Control],
            &[0.375, 1_000.0, 0.2],
            &[-0.25, 2_000.0, 0.4],
            1,
        ),
        RuntimeSafetyCase::new(
            "MoogVCF.ar controls",
            "MoogVCF",
            Audio,
            &[Audio, Control, Control],
            &[0.375, 1_000.0, 0.2],
            &[-0.25, 2_000.0, 0.4],
            1,
        ),
        RuntimeSafetyCase::new(
            "MoogVCF.ar audio controls",
            "MoogVCF",
            Audio,
            &[Audio, Audio, Audio],
            &[0.375, 1_000.0, 0.2],
            &[-0.25, 2_000.0, 0.4],
            1,
        ),
        RuntimeSafetyCase::new(
            "BlitB3.ar",
            "BlitB3",
            Audio,
            &[Control],
            &[440.0],
            &[880.0],
            1,
        ),
        RuntimeSafetyCase::new(
            "BlitB3Saw.ar",
            "BlitB3Saw",
            Audio,
            &[Control, Control],
            &[440.0, 0.5],
            &[880.0, 0.25],
            1,
        ),
        RuntimeSafetyCase::new(
            "BlitB3Square.ar",
            "BlitB3Square",
            Audio,
            &[Control, Control],
            &[440.0, 0.5],
            &[880.0, 0.25],
            1,
        ),
        RuntimeSafetyCase::new(
            "BlitB3Tri.ar",
            "BlitB3Tri",
            Audio,
            &[Control, Control, Control],
            &[440.0, 0.5, 0.5],
            &[880.0, 0.25, 0.75],
            1,
        ),
        RuntimeSafetyCase::new(
            "Perlin3.ar",
            "Perlin3",
            Audio,
            &[Audio, Audio, Audio],
            &[0.125, 0.25, 0.375],
            &[0.625, 0.75, 0.875],
            1,
        ),
        RuntimeSafetyCase::new(
            "Perlin3.kr",
            "Perlin3",
            Control,
            &[Control, Control, Control],
            &[0.125, 0.25, 0.375],
            &[0.625, 0.75, 0.875],
            1,
        ),
        RuntimeSafetyCase::new(
            "RosslerL.ar",
            "RosslerL",
            Audio,
            &[Control; 8],
            &[48_000.0, 0.2, 0.2, 5.7, 0.05, 0.1, 0.0, 0.0],
            &[48_000.0, 0.1, 0.3, 5.5, 0.025, 0.2, 0.1, -0.1],
            3,
        ),
    ]
}

/// Append output-major callback results to one continuous stream per output.
fn append_outputs(accumulator: &mut [Vec<f32>], block: Vec<Vec<f32>>) {
    assert_eq!(accumulator.len(), block.len());
    for (accumulator, block) in accumulator.iter_mut().zip(block) {
        accumulator.extend(block);
    }
}

/// Compare exact-output cases bitwise and nonlinear-filter cases with a tight absolute tolerance.
fn assert_outputs_equal(expected: &[Vec<f32>], actual: &[Vec<f32>], tolerance: f32, label: &str) {
    assert_eq!(expected.len(), actual.len(), "{label}: output count");
    for (output, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        assert_eq!(
            expected.len(),
            actual.len(),
            "{label}: output {output} length"
        );
        for (sample, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            if tolerance == 0.0 {
                assert_eq!(
                    expected.to_bits(),
                    actual.to_bits(),
                    "{label}: output {output}, sample {sample}"
                );
            } else {
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{label}: output {output}, sample {sample}: expected {expected}, got {actual}"
                );
            }
        }
    }
}

/// Exercises each sample processor over its ordinary finite runtime domain.
#[test]
fn sc3_processing_runtime_defined_domain_is_finite() {
    for case in runtime_safety_cases() {
        let mut unit = DirectCalc::new(
            case.name,
            case.sources.clone(),
            case.rate,
            case.outputs,
            48_000.0,
            64,
        );
        let len = if case.rate == Rate::Audio { 16 } else { 1 };
        for controls in [&case.baseline, &case.recovery] {
            let (audio, control) = case.input_block(controls, len);
            let output = unit.process(&audio, &control, len);
            assert!(
                output.iter().flatten().all(|sample| sample.is_finite()),
                "{} ordinary finite controls emitted a non-finite sample",
                case.label
            );
        }
    }
}

/// Pins Decimator's raw lower-rate and non-finite control arithmetic.
#[test]
fn decimator_non_finite_controls_follow_source_state_transitions() {
    let sources = vec![
        InputSource::Audio(0),
        InputSource::Control(0),
        InputSource::Control(1),
    ];
    let signal = (0..16)
        .map(|sample| -0.75 + sample as f32 * 0.03125)
        .collect::<Vec<_>>();

    for invalid_rate in [f32::NAN, f32::NEG_INFINITY] {
        let mut poisoned =
            DirectCalc::new("Decimator", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
        let mut zero_rate =
            DirectCalc::new("Decimator", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
        assert_eq!(
            poisoned.process(&signal, &[invalid_rate, 8.0], signal.len()),
            zero_rate.process(&signal, &[0.0, 8.0], signal.len())
        );
        let poisoned_recovery = poisoned.process(&signal, &[48_000.0, 8.0], signal.len());
        let zero_recovery = zero_rate.process(&signal, &[48_000.0, 8.0], signal.len());
        assert_ne!(
            poisoned_recovery, zero_recovery,
            "a poisoned cadence must not be rewritten during recovery"
        );
    }

    let mut positive_infinity =
        DirectCalc::new("Decimator", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
    let mut sample_rate =
        DirectCalc::new("Decimator", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
    assert_eq!(
        positive_infinity.process(&signal, &[f32::INFINITY, 8.0], signal.len()),
        sample_rate.process(&signal, &[48_000.0, 8.0], signal.len())
    );

    let mut nan_bits = DirectCalc::new("Decimator", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
    assert!(
        nan_bits
            .process(&signal, &[48_000.0, f32::NAN], signal.len())
            .iter()
            .flatten()
            .all(|sample| sample.is_nan())
    );
    for infinite_bits in [f32::NEG_INFINITY, f32::INFINITY] {
        let mut infinite =
            DirectCalc::new("Decimator", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
        let mut thirty_one =
            DirectCalc::new("Decimator", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
        assert_eq!(
            infinite.process(&signal, &[48_000.0, infinite_bits], signal.len()),
            thirty_one.process(&signal, &[48_000.0, 31.0], signal.len())
        );
    }
}

/// Pins the Decimator pre-calculation to the constructor's actual first input sample.
#[test]
fn decimator_constructor_reads_input_sample_zero() {
    let mut unit = DirectCalc::new(
        "Decimator",
        vec![
            InputSource::Audio(0),
            InputSource::Control(0),
            InputSource::Control(1),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let input = 0.25_f32;
    let output = unit.construct_once(&vec![input; 64], &[48_000.0, 8.0], 64);
    let step = plyphon_dsp::math::powf(0.5, 8.0 - 0.999);
    let scaled = (input + step * 0.5) / step;
    let expected = input - (scaled - plyphon_dsp::math::trunc(scaled)) * step;

    assert_eq!(output[0][0].to_bits(), expected.to_bits());
    assert_ne!(output[0][0], 0.0);
}

/// Pins BMoog's raw resonance and mode branches while preserving safe cutoff lookup.
#[test]
fn bmoog_non_finite_q_poisons_and_non_finite_mode_selects_low_pass() {
    let sources = vec![
        InputSource::Audio(0),
        InputSource::Control(0),
        InputSource::Control(1),
        InputSource::Control(2),
    ];
    let signal = (0..64)
        .map(|sample| -0.8 + sample as f32 * 0.021875)
        .collect::<Vec<_>>();

    for mode in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let mut non_finite =
            DirectCalc::new("BMoog", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
        let mut low_pass = DirectCalc::new("BMoog", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
        assert_eq!(
            non_finite.process(&signal, &[440.0, 0.5, mode], signal.len()),
            low_pass.process(&signal, &[440.0, 0.5, 0.0], signal.len())
        );
    }

    for resonance in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let mut filter = DirectCalc::new("BMoog", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
        filter.process(&signal, &[440.0, 0.5, 0.0], signal.len());
        filter.process(&signal, &[440.0, resonance, 0.0], signal.len());
        let poisoned = filter.process(&signal, &[440.0, 0.25, 0.0], signal.len());
        assert!(
            poisoned.iter().flatten().any(|sample| sample.is_nan()),
            "non-finite resonance {resonance} must remain observable after recovery"
        );
        assert!(
            filter
                .process(&signal, &[440.0, 0.25, 0.0], signal.len())
                .iter()
                .flatten()
                .all(|sample| sample.is_nan()),
            "the poisoned feedback state must persist"
        );
    }
}

/// Preserves Perlin3's raw invalid-coordinate arithmetic around the safe index cast.
#[test]
fn perlin3_non_finite_coordinates_propagate_nan() {
    for coordinate in 0..3 {
        for invalid in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
            let mut controls = [0.125, 0.25, 0.375];
            controls[coordinate] = invalid;
            let mut unit = DirectCalc::new(
                "Perlin3",
                vec![
                    InputSource::Control(0),
                    InputSource::Control(1),
                    InputSource::Control(2),
                ],
                Rate::Audio,
                1,
                48_000.0,
                64,
            );
            assert!(
                unit.process(&[], &controls, 64)[0]
                    .iter()
                    .all(|sample| sample.is_nan()),
                "coordinate {coordinate}={invalid} must remain visible in raw Perlin arithmetic"
            );
        }
    }
}

/// Pins RosslerL's raw parameter poisoning and coordinate-reset recovery.
#[test]
fn rossler_l_non_finite_controls_follow_source_recovery_boundaries() {
    let sources = (0..8).map(InputSource::Control).collect::<Vec<_>>();
    let ordinary = [6_000.0, 0.2, 0.2, 5.7, 0.05, 0.1, 0.0, 0.0];

    let mut poisoned = DirectCalc::new("RosslerL", sources.clone(), Rate::Audio, 3, 48_000.0, 64);
    poisoned.process(&[], &ordinary, 64);
    let mut invalid_parameter = ordinary;
    invalid_parameter[1] = f32::NAN;
    assert!(
        poisoned
            .process(&[], &invalid_parameter, 64)
            .iter()
            .flatten()
            .any(|sample| sample.is_nan())
    );
    assert!(
        poisoned
            .process(&[], &ordinary, 64)
            .iter()
            .flatten()
            .all(|sample| sample.is_nan()),
        "a poisoned trajectory must not be silently restarted"
    );

    let mut coordinate_reset = DirectCalc::new("RosslerL", sources, Rate::Audio, 3, 48_000.0, 64);
    coordinate_reset.process(&[], &ordinary, 64);
    let mut invalid_coordinate = ordinary;
    invalid_coordinate[5] = f32::NAN;
    assert!(
        coordinate_reset
            .process(&[], &invalid_coordinate, 64)
            .iter()
            .flatten()
            .any(|sample| sample.is_nan())
    );
    let recovered = coordinate_reset.process(&[], &ordinary, 64);
    assert!(
        recovered.iter().flatten().any(|sample| sample.is_finite())
            && recovered
                .iter()
                .all(|output| { output.last().is_some_and(|sample| sample.is_finite()) }),
        "restoring a coordinate must recover after the retained interpolation anchor advances"
    );
}

/// Exercise one sample unit with actual direct callback lengths and ordered changing inputs.
fn check_sample_case(case: &SampleCase, sample_rate: f64, block_size: usize) {
    let mut whole = DirectCalc::new(
        case.name,
        case.sources(),
        Rate::Audio,
        case.outputs,
        sample_rate,
        block_size,
    );
    let mut partitioned = DirectCalc::new(
        case.name,
        case.sources(),
        Rate::Audio,
        case.outputs,
        sample_rate,
        block_size,
    );

    let (warm_audio, warm_controls) = case.input_block(0, WARM_SAMPLES);
    let warm_whole = whole.process(&warm_audio, &warm_controls, WARM_SAMPLES);
    let warm_partitioned = partitioned.process(&warm_audio, &warm_controls, WARM_SAMPLES);
    assert_outputs_equal(
        &warm_whole,
        &warm_partitioned,
        case.tolerance,
        &format!("{} warm-up at {sample_rate}/{block_size}", case.name),
    );
    assert_eq!(
        whole.state_snapshot(),
        partitioned.state_snapshot(),
        "{} byte-exact state after warm-up at {sample_rate}/{block_size}",
        case.name
    );
    whole.clear_observations();
    partitioned.clear_observations();

    let (whole_audio, controls) = case.input_block(WARM_SAMPLES, 64);
    let expected = whole.process(&whole_audio, &controls, 64);
    let mut actual = vec![Vec::with_capacity(64); case.outputs];
    let mut start = WARM_SAMPLES;
    for len in PARTITION {
        let (audio, split_controls) = case.input_block(start, len);
        append_outputs(
            &mut actual,
            partitioned.process(&audio, &split_controls, len),
        );
        start += len;
    }

    assert_eq!(
        whole.observed_callbacks,
        [64],
        "{} whole callback",
        case.name
    );
    assert_eq!(
        partitioned.observed_callbacks, PARTITION,
        "{} partition callbacks",
        case.name
    );
    assert_outputs_equal(
        &expected,
        &actual,
        case.tolerance,
        &format!(
            "{} direct partition at {sample_rate}/{block_size}",
            case.name
        ),
    );
    assert_eq!(
        whole.state_snapshot(),
        partitioned.state_snapshot(),
        "{} byte-exact state after direct partition at {sample_rate}/{block_size}",
        case.name
    );
    assert!(
        expected.iter().flatten().all(|sample| sample.is_finite()),
        "{} direct output must stay finite",
        case.name
    );
    assert!(
        expected
            .iter()
            .flatten()
            .any(|sample| sample.abs() > f32::EPSILON),
        "{} direct partition vector is vacuous",
        case.name
    );

    let (follow_audio, follow_controls) = case.input_block(WARM_SAMPLES + 64, 32);
    let follow_whole = whole.process(&follow_audio, &follow_controls, 32);
    let follow_partitioned = partitioned.process(&follow_audio, &follow_controls, 32);
    assert_outputs_equal(
        &follow_whole,
        &follow_partitioned,
        case.tolerance,
        &format!(
            "{} retained state after partition at {sample_rate}/{block_size}",
            case.name
        ),
    );
}

/// Exercise direct `DNoiseRing` pulls while recording both grouping and total pull count.
fn check_demand_partition(sample_rate: f64, block_size: usize) {
    let mut whole = DirectDemand::new(sample_rate, block_size);
    let mut partitioned = DirectDemand::new(sample_rate, block_size);
    assert_eq!(
        whole.pull_group(WARM_SAMPLES),
        partitioned.pull_group(WARM_SAMPLES),
        "DNoiseRing identical warm-up at {sample_rate}/{block_size}"
    );
    assert_eq!(
        whole.state_snapshot(),
        partitioned.state_snapshot(),
        "DNoiseRing byte-exact state after warm-up at {sample_rate}/{block_size}"
    );
    whole.clear_observations();
    partitioned.clear_observations();

    let expected = whole.pull_group(64);
    let actual = PARTITION
        .into_iter()
        .flat_map(|count| partitioned.pull_group(count))
        .collect::<Vec<_>>();
    assert_eq!(whole.observed_groups, [64]);
    assert_eq!(partitioned.observed_groups, PARTITION);
    assert_eq!(whole.pull_count, 64);
    assert_eq!(partitioned.pull_count, 64);
    assert_eq!(
        expected, actual,
        "DNoiseRing direct pull partition at {sample_rate}/{block_size}"
    );
    assert_eq!(
        whole.state_snapshot(),
        partitioned.state_snapshot(),
        "DNoiseRing byte-exact state after grouped pulls at {sample_rate}/{block_size}"
    );
    assert!(
        expected.windows(2).any(|pair| pair[0] != pair[1]),
        "DNoiseRing partition vector is vacuous"
    );
    let follow_whole = whole.pull_group(17);
    let follow_partitioned = partitioned.pull_group(17);
    assert_eq!(
        follow_whole, follow_partitioned,
        "DNoiseRing retained state after partition at {sample_rate}/{block_size}"
    );
    assert_eq!(
        whole.state_snapshot(),
        partitioned.state_snapshot(),
        "DNoiseRing byte-exact state after follow-up pulls at {sample_rate}/{block_size}"
    );
}

/// Create a deterministic packed spectrum whose every bin changes with `frame` and `lane`.
fn spectrum(fft_size: usize, frame: usize, lane: usize, sample_rate: f64) -> Buffer {
    let coord = if (frame + lane).is_multiple_of(2) {
        SpectrumCoord::Polar
    } else {
        SpectrumCoord::Complex
    };
    let mut data = vec![0.0; fft_size];
    data[0] = 0.25 + frame as f32 * 0.03125 + lane as f32 * 0.125;
    data[1] = -0.375 + frame as f32 * 0.015625 - lane as f32 * 0.0625;
    for (bin, pair) in data[2..].chunks_exact_mut(2).enumerate() {
        let bin = bin as f32 + 1.0;
        match coord {
            SpectrumCoord::Polar => {
                pair[0] = 0.5 + bin * 0.000_122_070_31 + frame as f32 * 0.0625;
                pair[1] = -0.75 + (bin % 97.0) * 0.0078125 + frame as f32 * 0.03125;
            }
            SpectrumCoord::Complex => {
                pair[0] = 0.25 + (bin % 113.0) * 0.00390625 + frame as f32 * 0.03125;
                pair[1] = -0.375 + (bin % 89.0) * 0.00390625 - lane as f32 * 0.03125;
            }
        }
    }
    let mut buffer = Buffer::from_interleaved(data, 1, sample_rate);
    buffer.set_coord(coord);
    buffer
}

/// Snapshot one installed spectrum's complete samples and coordinate tag.
fn spectrum_snapshot(harness: &DirectCalc, index: usize) -> SpectrumSnapshot {
    let buffer = harness
        .buffers
        .get(index)
        .unwrap_or_else(|| panic!("missing direct PV buffer {index}"));
    SpectrumSnapshot {
        data_bits: buffer
            .data()
            .iter()
            .map(|sample| sample.to_bits())
            .collect(),
        coord: buffer.coord(),
    }
}

/// Install the same fresh ready-frame inputs in both sides of a PV comparison.
fn install_pv_frames(
    name: &str,
    whole: &mut DirectCalc,
    partitioned: &mut DirectCalc,
    fft_size: usize,
    frame: usize,
) -> SpectrumSnapshot {
    let input = spectrum(fft_size, frame, 0, whole.sample_rate);
    let input_snapshot = SpectrumSnapshot {
        data_bits: input.data().iter().map(|sample| sample.to_bits()).collect(),
        coord: input.coord(),
    };
    whole.buffers.set(0, Box::new(input));
    partitioned
        .buffers
        .set(0, Box::new(spectrum(fft_size, frame, 0, whole.sample_rate)));
    if name == "PV_Morph" {
        whole
            .buffers
            .set(1, Box::new(spectrum(fft_size, frame, 1, whole.sample_rate)));
        partitioned
            .buffers
            .set(1, Box::new(spectrum(fft_size, frame, 1, whole.sample_rate)));
    }
    input_snapshot
}

/// Build ready-frame controls for one PV operator.
fn pv_controls(name: &str, ready: bool, frame: usize) -> Vec<f32> {
    let token = if ready { 0.0 } else { -1.0 };
    match name {
        "PV_Freeze" => vec![token, (frame >= 2) as u8 as f32],
        "PV_MagSmooth" => vec![token, 0.625],
        "PV_Morph" => vec![token, 1.0, 0.375],
        _ => unreachable!("unknown direct PV case"),
    }
}

/// One malformed packed-spectrum class used by the atomic PV rejection matrix.
#[derive(Clone, Copy)]
enum PvBufferFault {
    UnsupportedSize,
    MultipleChannels,
}

/// One invalid token, frame, or two-buffer ownership condition.
#[derive(Clone, Copy)]
enum PvRejection {
    PrimaryBuffer(PvBufferFault),
    SecondaryBuffer(PvBufferFault),
    LengthMismatch,
}

impl PvRejection {
    /// Stable assertion label for this rejection class.
    fn label(self) -> &'static str {
        match self {
            Self::PrimaryBuffer(fault) => match fault {
                PvBufferFault::UnsupportedSize => "primary unsupported packed size",
                PvBufferFault::MultipleChannels => "primary multichannel packed shape",
            },
            Self::SecondaryBuffer(fault) => match fault {
                PvBufferFault::UnsupportedSize => "secondary unsupported packed size",
                PvBufferFault::MultipleChannels => "secondary multichannel packed shape",
            },
            Self::LengthMismatch => "A/B length mismatch",
        }
    }
}

/// Construct a deterministic complex packed spectrum with one selected structural/numeric fault.
fn faulty_pv_buffer(fault: PvBufferFault, lane: usize, sample_rate: f64) -> Buffer {
    let (frames, channels) = match fault {
        PvBufferFault::UnsupportedSize => (65, 1),
        PvBufferFault::MultipleChannels => (64, 2),
    };
    let mut data = vec![0.0; frames * channels];
    data[0] = 0.25 + lane as f32 * 0.125;
    data[1] = -0.375 - lane as f32 * 0.0625;
    for (bin, pair) in data[2..].chunks_exact_mut(2).enumerate() {
        pair[0] = 0.5 + bin as f32 * 0.0078125;
        pair[1] = -0.25 + lane as f32 * 0.03125;
    }
    let mut buffer = Buffer::from_interleaved(data, channels, sample_rate);
    buffer.set_coord(SpectrumCoord::Complex);
    buffer
}

/// Snapshot both fixed global buffer slots, preserving sample bits and coordinate tags.
fn pv_buffer_snapshots(harness: &DirectCalc) -> [Option<SpectrumSnapshot>; 2] {
    [0, 1].map(|index| {
        harness.buffers.get(index).map(|buffer| SpectrumSnapshot {
            data_bits: buffer
                .data()
                .iter()
                .map(|sample| sample.to_bits())
                .collect(),
            coord: buffer.coord(),
        })
    })
}

/// Warm two direct PV instances through enough valid frames for the next frame to exercise state.
fn warm_pv_safety_pair(name: &str, subject: &mut DirectCalc, reference: &mut DirectCalc) {
    for frame in 0..2 {
        install_pv_frames(name, subject, reference, 64, frame);
        let controls = pv_controls(name, true, frame);
        assert_eq!(subject.process(&[], &controls, 1), vec![vec![0.0]]);
        assert_eq!(reference.process(&[], &controls, 1), vec![vec![0.0]]);
        assert_eq!(subject.state_snapshot(), reference.state_snapshot());
        assert_eq!(pv_buffer_snapshots(subject), pv_buffer_snapshots(reference));
    }
}

/// Apply one invalid input class and return the complete control vector plus expected output token.
fn install_pv_rejection(
    name: &str,
    rejection: PvRejection,
    subject: &mut DirectCalc,
) -> (Vec<f32>, f32) {
    let controls = pv_controls(name, true, 2);
    let expected = match rejection {
        PvRejection::PrimaryBuffer(fault) => {
            subject
                .buffers
                .set(0, Box::new(faulty_pv_buffer(fault, 0, subject.sample_rate)));
            0.0
        }
        PvRejection::SecondaryBuffer(fault) => {
            subject
                .buffers
                .set(1, Box::new(faulty_pv_buffer(fault, 1, subject.sample_rate)));
            0.0
        }
        PvRejection::LengthMismatch => {
            subject
                .buffers
                .set(1, Box::new(spectrum(128, 2, 1, subject.sample_rate)));
            0.0
        }
    };
    (controls, expected)
}

/// Check one PV rejection for byte-atomic state/buffers and exact recovery against a clean copy.
fn check_pv_rejection(name: &str, rejection: PvRejection) {
    let sources = match name {
        "PV_Morph" => vec![
            InputSource::Control(0),
            InputSource::Control(1),
            InputSource::Control(2),
        ],
        _ => vec![InputSource::Control(0), InputSource::Control(1)],
    };
    let mut subject = DirectCalc::new(name, sources.clone(), Rate::Control, 1, 48_000.0, 64);
    let mut reference = DirectCalc::new(name, sources, Rate::Control, 1, 48_000.0, 64);
    warm_pv_safety_pair(name, &mut subject, &mut reference);
    install_pv_frames(name, &mut subject, &mut reference, 64, 2);
    let (controls, expected_token) = install_pv_rejection(name, rejection, &mut subject);
    let before_state = subject.state_snapshot();
    let before_buffers = pv_buffer_snapshots(&subject);
    let output = subject.process(&[], &controls, 1);
    assert_eq!(
        output,
        vec![vec![expected_token]],
        "{name} {} output token",
        rejection.label()
    );
    assert_eq!(
        subject.state_snapshot(),
        before_state,
        "{name} {} changed state/aux",
        rejection.label()
    );
    assert_eq!(
        pv_buffer_snapshots(&subject),
        before_buffers,
        "{name} {} changed a buffer/tag",
        rejection.label()
    );

    install_pv_frames(name, &mut subject, &mut reference, 64, 3);
    let recovery_controls = pv_controls(name, true, 3);
    let subject_b = (name == "PV_Morph").then(|| spectrum_snapshot(&subject, 1));
    let reference_b = (name == "PV_Morph").then(|| spectrum_snapshot(&reference, 1));
    let actual = subject.process(&[], &recovery_controls, 1);
    let expected = reference.process(&[], &recovery_controls, 1);
    assert_eq!(
        actual,
        expected,
        "{name} {} recovery token",
        rejection.label()
    );
    assert_eq!(
        spectrum_snapshot(&subject, 0),
        spectrum_snapshot(&reference, 0),
        "{name} {} recovery owned buffer",
        rejection.label()
    );
    assert_eq!(
        subject.state_snapshot(),
        reference.state_snapshot(),
        "{name} {} recovery state/aux",
        rejection.label()
    );
    if let (Some(subject_b), Some(reference_b)) = (subject_b, reference_b) {
        assert_eq!(
            spectrum_snapshot(&subject, 1),
            subject_b,
            "{name} {} recovery mutated subject B",
            rejection.label()
        );
        assert_eq!(
            spectrum_snapshot(&reference, 1),
            reference_b,
            "{name} {} recovery mutated reference B",
            rejection.label()
        );
    }
}

/// Prove one PV operator ignores extra no-frame callbacks without losing any packed-bin state.
fn check_pv_partition(name: &str, sample_rate: f64, block_size: usize, fft_size: usize) {
    let inputs = match name {
        "PV_Morph" => vec![
            InputSource::Control(0),
            InputSource::Control(1),
            InputSource::Control(2),
        ],
        _ => vec![InputSource::Control(0), InputSource::Control(1)],
    };
    let mut whole = DirectCalc::new(
        name,
        inputs.clone(),
        Rate::Control,
        1,
        sample_rate,
        block_size,
    );
    let mut partitioned = DirectCalc::new(name, inputs, Rate::Control, 1, sample_rate, block_size);
    let mut mutated_full_spectrum = false;

    for frame in 0..6 {
        let input = install_pv_frames(name, &mut whole, &mut partitioned, fft_size, frame);
        let before_no_frame_a = spectrum_snapshot(&partitioned, 0);
        let before_no_frame_b = (name == "PV_Morph").then(|| spectrum_snapshot(&partitioned, 1));
        let before_no_frame_state = partitioned.state_snapshot();
        let no_frame = partitioned.process(&[], &pv_controls(name, false, frame), 1);
        assert_eq!(no_frame, vec![vec![-1.0]], "{name} no-frame token");
        assert_eq!(
            spectrum_snapshot(&partitioned, 0),
            before_no_frame_a,
            "{name} no-frame A at {sample_rate}/{block_size}/{fft_size}/{frame}"
        );
        if let Some(expected) = before_no_frame_b {
            assert_eq!(
                spectrum_snapshot(&partitioned, 1),
                expected,
                "{name} no-frame B at {sample_rate}/{block_size}/{fft_size}/{frame}"
            );
        }
        assert_eq!(
            partitioned.state_snapshot(),
            before_no_frame_state,
            "{name} no-frame retained state at {sample_rate}/{block_size}/{fft_size}/{frame}"
        );

        let controls = pv_controls(name, true, frame);
        let before_ready_b = (name == "PV_Morph").then(|| spectrum_snapshot(&partitioned, 1));
        assert_eq!(whole.process(&[], &controls, 1), vec![vec![0.0]]);
        assert_eq!(partitioned.process(&[], &controls, 1), vec![vec![0.0]]);
        let whole_a = spectrum_snapshot(&whole, 0);
        let partitioned_a = spectrum_snapshot(&partitioned, 0);
        assert_eq!(
            whole_a, partitioned_a,
            "{name} complete A spectrum at {sample_rate}/{block_size}/{fft_size}/{frame}"
        );
        if let Some(expected_b) = before_ready_b {
            let converted_b = spectrum_snapshot(&partitioned, 1);
            assert_eq!(
                spectrum_snapshot(&whole, 1),
                converted_b,
                "{name} B spectrum partition parity at {sample_rate}/{block_size}/{fft_size}/{frame}"
            );
            assert_eq!(converted_b.coord, SpectrumCoord::Polar);
            if expected_b.coord == SpectrumCoord::Complex {
                assert_ne!(
                    converted_b, expected_b,
                    "{name} source conversion must mutate complex B at {sample_rate}/{block_size}/{fft_size}/{frame}"
                );
            } else {
                assert_eq!(
                    converted_b, expected_b,
                    "{name} source conversion must preserve polar B at {sample_rate}/{block_size}/{fft_size}/{frame}"
                );
            }
        }
        assert_eq!(
            whole.state_snapshot(),
            partitioned.state_snapshot(),
            "{name} retained state at {sample_rate}/{block_size}/{fft_size}/{frame}"
        );
        mutated_full_spectrum |= whole_a != input;
    }

    assert!(
        mutated_full_spectrum,
        "{name} full-spectrum comparison was vacuous at {sample_rate}/{block_size}/{fft_size}"
    );
    assert_eq!(whole.observed_callbacks, vec![1; 6]);
    assert_eq!(partitioned.observed_callbacks, vec![1; 12]);
}

/// Proves PV modulation audio wires use their first sample rather than a later block sample.
#[test]
fn sc3_pv_audio_modulation_uses_first_sample_in0() {
    for name in ["PV_Freeze", "PV_MagSmooth", "PV_Morph"] {
        let (audio_sources, control_sources, token_controls) = match name {
            "PV_Freeze" | "PV_MagSmooth" => (
                vec![InputSource::Control(0), InputSource::Audio(0)],
                vec![InputSource::Control(0), InputSource::Control(1)],
                vec![0.0],
            ),
            "PV_Morph" => (
                vec![
                    InputSource::Control(0),
                    InputSource::Control(1),
                    InputSource::Audio(0),
                ],
                vec![
                    InputSource::Control(0),
                    InputSource::Control(1),
                    InputSource::Control(2),
                ],
                vec![0.0, 1.0],
            ),
            _ => unreachable!(),
        };
        let mut audio_modulation =
            DirectCalc::new(name, audio_sources, Rate::Control, 1, 48_000.0, 64);
        let mut first_sample_reference = DirectCalc::new(
            name,
            control_sources.clone(),
            Rate::Control,
            1,
            48_000.0,
            64,
        );
        let mut later_sample_reference =
            DirectCalc::new(name, control_sources, Rate::Control, 1, 48_000.0, 64);
        let (first_sample, later_sample) = match name {
            "PV_Freeze" => (1.0, 0.0),
            "PV_MagSmooth" | "PV_Morph" => (0.25, 0.75),
            _ => unreachable!(),
        };
        let mut audio_wire = vec![later_sample; 64];
        audio_wire[0] = first_sample;

        for frame in 0..4 {
            audio_modulation
                .buffers
                .set(0, Box::new(spectrum(64, frame, 0, 48_000.0)));
            first_sample_reference
                .buffers
                .set(0, Box::new(spectrum(64, frame, 0, 48_000.0)));
            later_sample_reference
                .buffers
                .set(0, Box::new(spectrum(64, frame, 0, 48_000.0)));
            if name == "PV_Morph" {
                for harness in [
                    &mut audio_modulation,
                    &mut first_sample_reference,
                    &mut later_sample_reference,
                ] {
                    harness
                        .buffers
                        .set(1, Box::new(spectrum(64, frame, 1, 48_000.0)));
                }
            }

            let mut first_controls = token_controls.clone();
            first_controls.push(first_sample);
            let mut later_controls = token_controls.clone();
            later_controls.push(later_sample);
            assert_eq!(
                audio_modulation.process(&audio_wire, &token_controls, 64),
                first_sample_reference.process(&[], &first_controls, 64),
                "{name} output token"
            );
            assert_eq!(
                audio_modulation.state_snapshot(),
                first_sample_reference.state_snapshot(),
                "{name} first-sample retained state"
            );
            assert_eq!(
                pv_buffer_snapshots(&audio_modulation),
                pv_buffer_snapshots(&first_sample_reference),
                "{name} first-sample spectra"
            );
            later_sample_reference.process(&[], &later_controls, 64);
        }

        assert_ne!(
            pv_buffer_snapshots(&first_sample_reference),
            pv_buffer_snapshots(&later_sample_reference),
            "{name} discriminator must distinguish its non-default first sample from later samples"
        );
    }
}

/// Pins the two-chain macro's negative-token passthrough rule.
#[test]
fn pv_morph_returns_negative_one_when_either_chain_is_not_ready() {
    let sources = vec![
        InputSource::Control(0),
        InputSource::Control(1),
        InputSource::Control(2),
    ];
    let mut primary_missing =
        DirectCalc::new("PV_Morph", sources.clone(), Rate::Control, 1, 48_000.0, 64);
    let mut secondary_missing =
        DirectCalc::new("PV_Morph", sources, Rate::Control, 1, 48_000.0, 64);
    assert_eq!(
        primary_missing.process(&[], &[-0.5, 1.0, 0.25], 1),
        vec![vec![-1.0]]
    );
    assert_eq!(
        secondary_missing.process(&[], &[0.0, -0.5, 0.25], 1),
        vec![vec![-1.0]]
    );
}

/// Unknown finite tokens fall back to world buffer zero; infinity passes through but misses.
#[test]
fn pv_morph_out_of_range_token_uses_buffer_zero() {
    let mut morph = DirectCalc::new(
        "PV_Morph",
        vec![
            InputSource::Control(0),
            InputSource::Control(1),
            InputSource::Control(2),
        ],
        Rate::Control,
        1,
        48_000.0,
        64,
    );
    let mut a = Buffer::from_interleaved(vec![1.0; 64], 1, 48_000.0);
    a.set_coord(SpectrumCoord::Polar);
    let mut b = Buffer::from_interleaved(vec![7.0; 64], 1, 48_000.0);
    b.set_coord(SpectrumCoord::Polar);
    morph.buffers.set(0, Box::new(a));
    morph.buffers.set(1, Box::new(b));

    assert_eq!(morph.process(&[], &[99.0, 1.0, 1.0], 1), vec![vec![99.0]]);
    assert!(
        morph
            .buffers
            .get(0)
            .expect("fallback buffer")
            .data()
            .iter()
            .all(|&value| value == 7.0)
    );

    morph
        .buffers
        .get_mut(0)
        .expect("fallback buffer")
        .data_mut()
        .fill(1.0);
    let output = morph.process(&[], &[f32::INFINITY, 1.0, 1.0], 1);
    assert_eq!(output[0][0], f32::INFINITY);
    assert!(
        morph
            .buffers
            .get(0)
            .expect("fallback buffer")
            .data()
            .iter()
            .all(|&value| value == 1.0)
    );
    let output = morph.process(&[], &[0.0, f32::INFINITY, 1.0], 1);
    assert_eq!(output[0][0], 0.0);
    assert!(
        morph
            .buffers
            .get(0)
            .expect("primary buffer")
            .data()
            .iter()
            .all(|&value| value == 1.0)
    );
}

/// Pins first-frame raw smoothing and the source's smaller-frame memory indexing.
#[test]
fn pv_mag_smooth_first_frame_and_smaller_frame_use_source_arithmetic() {
    let sources = vec![InputSource::Control(0), InputSource::Control(1)];
    let mut overflowing = DirectCalc::new(
        "PV_MagSmooth",
        sources.clone(),
        Rate::Control,
        1,
        48_000.0,
        64,
    );
    let mut extreme = vec![0.0; 64];
    extreme[0] = f32::MAX;
    extreme[1] = f32::MAX;
    extreme[2] = f32::MAX;
    let mut extreme = Buffer::from_interleaved(extreme, 1, 48_000.0);
    extreme.set_coord(SpectrumCoord::Polar);
    overflowing.buffers.set(0, Box::new(extreme));
    overflowing.process(&[], &[0.0, f32::MAX], 1);
    let overflowed = overflowing.buffers.get(0).expect("overflow spectrum");
    assert!(overflowed.data()[0].is_nan());
    assert!(overflowed.data()[2].is_nan());

    let mut smaller = DirectCalc::new("PV_MagSmooth", sources, Rate::Control, 1, 48_000.0, 64);
    let mut original = vec![0.0; 128];
    original[0] = 1.0;
    original[1] = 2.0;
    original[64] = 100.0;
    original[66] = 200.0;
    let mut original = Buffer::from_interleaved(original, 1, 48_000.0);
    original.set_coord(SpectrumCoord::Polar);
    smaller.buffers.set(0, Box::new(original));
    smaller.process(&[], &[0.0, 0.5], 1);

    let mut reduced = vec![0.0; 64];
    reduced[0] = 10.0;
    reduced[1] = 20.0;
    let mut reduced = Buffer::from_interleaved(reduced, 1, 48_000.0);
    reduced.set_coord(SpectrumCoord::Polar);
    smaller.buffers.set(0, Box::new(reduced));
    smaller.process(&[], &[0.0, 0.5], 1);
    let reduced = smaller.buffers.get(0).expect("reduced spectrum");
    assert_eq!(reduced.data()[0], 55.0);
    assert_eq!(reduced.data()[1], 110.0);
}

/// Proves malformed host buffer shapes are rejected atomically and valid processing resumes.
#[test]
fn sc3_pv_malformed_host_shapes_are_atomic_and_recover() {
    for name in ["PV_Freeze", "PV_MagSmooth", "PV_Morph"] {
        for rejection in [
            PvRejection::PrimaryBuffer(PvBufferFault::UnsupportedSize),
            PvRejection::PrimaryBuffer(PvBufferFault::MultipleChannels),
        ] {
            check_pv_rejection(name, rejection);
        }
        if name == "PV_Morph" {
            for rejection in [
                PvRejection::SecondaryBuffer(PvBufferFault::UnsupportedSize),
                PvRejection::SecondaryBuffer(PvBufferFault::MultipleChannels),
                PvRejection::LengthMismatch,
            ] {
                check_pv_rejection(name, rejection);
            }
        }
    }
}

/// Pins the source audio-cutoff/audio-resonance callback's retained-cutoff quirk.
#[test]
fn moog_ladder_audio_audio_retains_constructor_cutoff() {
    let mut ladder = DirectCalc::new(
        "MoogLadder",
        vec![
            InputSource::Audio(0),
            InputSource::Audio(1),
            InputSource::Audio(2),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let mut inputs = vec![0.25; 64];
    inputs.extend((0..64).map(|sample| 440.0 + sample as f32 * (1_560.0 / 63.0)));
    inputs.extend((0..64).map(|sample| 0.1 + sample as f32 * (0.6 / 63.0)));
    ladder.process(&inputs, &[], 64);

    let state = ladder.state_snapshot().0;
    let retained_cutoff = f32::from_ne_bytes(state[..4].try_into().expect("cutoff state bytes"));
    assert_eq!(
        retained_cutoff, 440.0,
        "the audio/audio source callback does not store its final cutoff"
    );
}

/// Pins the source's double-rate arithmetic before the ladder coefficient is stored as `f32`.
#[test]
fn moog_ladder_constructor_coefficient_matches_source_bits() {
    let mut ladder = DirectCalc::new(
        "MoogLadder",
        vec![
            InputSource::Audio(0),
            InputSource::Control(0),
            InputSource::Control(1),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    ladder.process(&[0.0], &[440.0, 0.0], 1);
    let state = ladder.state_snapshot().0;
    let coefficient = f32::from_ne_bytes(state[4..8].try_into().expect("coefficient state bytes"));
    assert_eq!(coefficient.to_bits(), 0x3d10_5310);
}

/// Proves oscillator, `MulAdd`, and binary-op constructor outputs reach both Moog constructors.
#[test]
fn moog_constructors_read_source_constructor_chain_outputs() {
    let mut sine = DirectCalc::new(
        "SinOsc",
        vec![InputSource::Constant(389.0), InputSource::Constant(0.4)],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let mut saw = DirectCalc::new(
        "LFSaw",
        vec![InputSource::Constant(131.0), InputSource::Constant(0.2)],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let mut triangle = DirectCalc::new(
        "LFTri",
        vec![InputSource::Constant(11.0), InputSource::Constant(0.3)],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let sine_ctor = sine.construct_once(&[], &[], 64)[0].clone();
    let saw_ctor = saw.construct_once(&[], &[], 64)[0].clone();
    let triangle_ctor = triangle.construct_once(&[], &[], 64)[0].clone();

    assert_eq!(sine_ctor[0].to_bits(), 0x3ec7_61d6);
    assert_eq!(saw_ctor[0].to_bits(), 0.2_f32.to_bits());
    assert_eq!(triangle_ctor[0].to_bits(), 0.3_f32.to_bits());

    let mut sine_scale = DirectCalc::new(
        "MulAdd",
        vec![
            InputSource::Audio(0),
            InputSource::Constant(0.2),
            InputSource::Constant(0.0),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let scaled_sine = sine_scale.construct_once(&sine_ctor, &[], 64)[0].clone();
    let mut saw_scale = DirectCalc::new(
        "MulAdd",
        vec![
            InputSource::Audio(0),
            InputSource::Constant(0.3),
            InputSource::Constant(0.0),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let scaled_saw = saw_scale.construct_once(&saw_ctor, &[], 64)[0].clone();

    let mut add_inputs = scaled_sine.clone();
    add_inputs.extend_from_slice(&scaled_saw);
    let mut add = DirectCalc::new(
        "BinaryOpUGen",
        vec![InputSource::Audio(0), InputSource::Audio(1)],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let add_ctor = add.construct_once(&add_inputs, &[], 64)[0].clone();
    assert_eq!(add_ctor[0], scaled_sine[0] + scaled_saw[0]);
    let mut multiply = DirectCalc::new_with_special(
        "BinaryOpUGen",
        vec![InputSource::Audio(0), InputSource::Audio(1)],
        Rate::Audio,
        1,
        48_000.0,
        64,
        2,
    );
    let multiply_ctor = multiply.construct_once(&add_inputs, &[], 64);
    assert_eq!(multiply_ctor[0][0], scaled_sine[0] * scaled_saw[0]);

    let mut cutoff_scale = DirectCalc::new(
        "MulAdd",
        vec![
            InputSource::Audio(0),
            InputSource::Constant(3_125.0),
            InputSource::Constant(3_375.0),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let cutoff_ctor = cutoff_scale.construct_once(&sine_ctor, &[], 64)[0].clone();
    let mut resonance_scale = DirectCalc::new(
        "MulAdd",
        vec![
            InputSource::Audio(0),
            InputSource::Constant(0.4),
            InputSource::Constant(0.5),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let resonance_ctor = resonance_scale.construct_once(&triangle_ctor, &[], 64)[0].clone();

    let mut wires = add_ctor;
    wires.extend_from_slice(&cutoff_ctor);
    wires.extend_from_slice(&resonance_ctor);
    for name in ["MoogLadder", "MoogVCF"] {
        let mut filter = DirectCalc::new(
            name,
            vec![
                InputSource::Audio(0),
                InputSource::Audio(1),
                InputSource::Audio(2),
            ],
            Rate::Audio,
            1,
            48_000.0,
            64,
        );
        filter.construct_once(&wires, &[], 64);
        let state = filter.state_snapshot().0;
        let cutoff = f32::from_ne_bytes(state[..4].try_into().expect("cutoff state bytes"));
        let resonance_offset = if name == "MoogVCF" { 4 } else { 8 };
        let resonance = f32::from_ne_bytes(
            state[resonance_offset..resonance_offset + 4]
                .try_into()
                .expect("resonance state bytes"),
        );
        let expected_cutoff = if name == "MoogVCF" {
            (cutoff_ctor[0] as f64 * 2.0 / 48_000.0) as f32
        } else {
            cutoff_ctor[0]
        };
        assert_eq!(cutoff, expected_cutoff, "{name} constructor cutoff");
        assert_eq!(resonance, resonance_ctor[0], "{name} constructor resonance");
    }
}

/// Pins the constructor draw and following process draw for audio-rate binary `rrand`.
#[test]
fn binary_random_constructor_consumes_one_source_draw() {
    let mut random = DirectCalc::new_with_special(
        "BinaryOpUGen",
        vec![InputSource::Audio(0), InputSource::Audio(1)],
        Rate::Audio,
        1,
        48_000.0,
        64,
        47,
    );
    let mut expected = Rng::new(UNIT_SEED);
    let expected_constructor = 100.0 + expected.next_bipolar() * 100.0;
    let expected_process = 100.0 + expected.next_bipolar() * 100.0;

    let mut constructor_inputs = vec![100.0; 64];
    constructor_inputs.extend([200.0; 64]);
    let constructor = random.construct_once(&constructor_inputs, &[], 64);
    let process = random.process(&[100.0, 200.0], &[], 1);
    assert_eq!(constructor[0][0].to_bits(), expected_constructor.to_bits());
    assert_eq!(process[0][0].to_bits(), expected_process.to_bits());
}

/// Pins the source min/max macros' second-operand result for equal signed zeros.
#[test]
fn binary_min_max_ties_return_the_second_operand() {
    for special_index in [12, 13] {
        for (a, b) in [(0.0, -0.0), (-0.0, 0.0)] {
            let mut unit = DirectCalc::new_with_special(
                "BinaryOpUGen",
                vec![InputSource::Constant(a), InputSource::Constant(b)],
                Rate::Audio,
                1,
                48_000.0,
                64,
                special_index,
            );
            let constructor = unit.construct_once(&[], &[], 1);
            assert_eq!(constructor[0][0].to_bits(), b.to_bits());
            let processed = unit.process(&[], &[], 1);
            assert_eq!(processed[0][0].to_bits(), b.to_bits());
        }
    }
}

/// Pins the double-precision source constant used by the `hypotx` constructor kernel.
#[test]
fn binary_hypotx_constructor_matches_source_bits() {
    let x = f32::from_bits(0x3f08_596c);
    let y = f32::from_bits(0x3f08_85db);
    let mut unit = DirectCalc::new_with_special(
        "BinaryOpUGen",
        vec![InputSource::Constant(x), InputSource::Constant(y)],
        Rate::Audio,
        1,
        48_000.0,
        64,
        24,
    );
    let constructor = unit.construct_once(&[], &[], 1);
    assert_eq!(constructor[0][0].to_bits(), 0x3f58_64fb);
}

/// Pins the source constructor's zero/one MulAdd specializations.
#[test]
fn mul_add_constructor_preserves_source_zero_specializations() {
    let cases: [(f32, f32, f32, f32); 4] = [
        (f32::NAN, 0.0, 1.0, 1.0),
        (-0.0, 2.0, 0.0, -0.0),
        (-0.0, 1.0, 0.0, -0.0),
        (f32::NAN, 0.0, -0.0, 0.0),
    ];
    for (input, mul, add, expected) in cases {
        let mut unit = DirectCalc::new(
            "MulAdd",
            vec![
                InputSource::Audio(0),
                InputSource::Constant(mul),
                InputSource::Constant(add),
            ],
            Rate::Audio,
            1,
            48_000.0,
            64,
        );
        let constructor = unit.construct_once(&[input], &[], 1);
        assert_eq!(constructor[0][0].to_bits(), expected.to_bits());
    }
}

/// Pins SinOsc's fixed-point source wavetable lookup during construction.
#[test]
fn sin_osc_constructor_matches_source_fixed_point_bits() {
    for (phase, expected) in [(0.1, 0x3dcc_7575), (-0.1, 0xbdcc_7575), (0.4, 0x3ec7_61d6)] {
        let mut oscillator = DirectCalc::new(
            "SinOsc",
            vec![InputSource::Constant(0.0), InputSource::Constant(phase)],
            Rate::Audio,
            1,
            48_000.0,
            64,
        );
        let constructor = oscillator.construct_once(&[], &[], 1);
        assert_eq!(constructor[0][0].to_bits(), expected);
    }
}

/// Restores LFSaw's raw initial phase after its hidden constructor sample.
#[test]
fn lfsaw_constructor_retains_out_of_range_initial_phase() {
    let mut oscillator = DirectCalc::new(
        "LFSaw",
        vec![InputSource::Constant(0.0), InputSource::Constant(2.5)],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let constructor = oscillator.construct_once(&[], &[], 1);
    assert_eq!(constructor[0][0], 2.5);
    let processed = oscillator.process(&[], &[], 1);
    assert_eq!(processed[0][0], 2.5);
}

/// Preserves the source wrap's positive zero for a negative exact triangle period.
#[test]
fn lftri_negative_period_constructor_wraps_to_positive_zero() {
    let mut triangle = DirectCalc::new(
        "LFTri",
        vec![InputSource::Constant(0.0), InputSource::Constant(-4.0)],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let constructor = triangle.construct_once(&[], &[], 1);
    assert_eq!(constructor[0][0].to_bits(), 0.0_f32.to_bits());
    let processed = triangle.process(&[], &[], 1);
    assert_eq!(processed[0][0].to_bits(), 0.0_f32.to_bits());
}

/// Selects `Line`'s exact terminal input once its one-frame counter reaches zero.
#[test]
fn line_one_frame_terminal_output_uses_end_bits() {
    let start = f32::from_bits(0x3ceb_3ffd);
    let end = f32::from_bits(0x97b7_5092);
    let mut line = DirectCalc::new(
        "Line",
        vec![
            InputSource::Constant(start),
            InputSource::Constant(end),
            InputSource::Constant(1.0 / 48_000.0),
            InputSource::Constant(0.0),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );

    assert_eq!(line.process(&[], &[], 1)[0][0].to_bits(), start.to_bits());
    assert_eq!(line.process(&[], &[], 1)[0][0].to_bits(), end.to_bits());
}

/// Pins the constructor count width shared by constant-duration `Duty` and gapped `TDuty`.
#[test]
fn duty_constructors_use_double_sample_rate_product() {
    let duration = 1.000_000_011_686_097_4e-7;
    for (name, sources) in [
        (
            "Duty",
            vec![
                InputSource::Constant(duration),
                InputSource::Constant(0.0),
                InputSource::Constant(0.0),
                InputSource::Constant(1.0),
            ],
        ),
        (
            "TDuty",
            vec![
                InputSource::Constant(duration),
                InputSource::Constant(0.0),
                InputSource::Constant(0.0),
                InputSource::Constant(1.0),
                InputSource::Constant(1.0),
            ],
        ),
    ] {
        let mut duty = DirectCalc::new(name, sources, Rate::Audio, 1, 44_100.1, 64);
        duty.construct_once(&[], &[], 64);
        let state = duty.state_snapshot().0;
        let count = f32::from_ne_bytes(state[..4].try_into().expect("count state bytes"));
        assert_eq!(count.to_bits(), 0x3b90_81d8, "{name} count");
    }
}

/// Pins the runtime coefficient's float intermediates before its double-precision exponential.
#[test]
fn moog_ladder_runtime_coefficient_matches_source_bits() {
    let mut ladder = DirectCalc::new(
        "MoogLadder",
        vec![
            InputSource::Audio(0),
            InputSource::Control(0),
            InputSource::Control(1),
        ],
        Rate::Audio,
        1,
        48_000.0,
        1,
    );
    ladder.process(&[0.0], &[440.0, 0.0], 1);
    ladder.process(&[0.0], &[779.400_024_414_062_5, 0.0], 1);

    let state = ladder.state_snapshot().0;
    let coefficient = f32::from_ne_bytes(state[4..8].try_into().expect("coefficient state bytes"));
    assert_eq!(coefficient.to_bits(), 0x3d7b_c73f);
}

/// Pins `CALCSLOPE`'s multiply-by-slope-factor evaluation order.
#[test]
fn moog_ladder_control_slope_matches_source_bits() {
    let mut ladder = DirectCalc::new(
        "MoogLadder",
        vec![
            InputSource::Audio(0),
            InputSource::Control(0),
            InputSource::Control(1),
        ],
        Rate::Audio,
        1,
        48_000.0,
        96,
    );
    ladder.process(&[0.0], &[440.0, 0.0], 1);
    ladder.process(&[0.0], &[440.0, -868.167_968_75], 1);

    let state = ladder.state_snapshot().0;
    let resonance = f32::from_ne_bytes(state[8..12].try_into().expect("resonance state bytes"));
    assert_eq!(resonance.to_bits(), 0xc110_b1d6);
}

/// Preserves a negative-zero control state when the source skips a zero slope.
#[test]
fn moog_ladder_zero_slope_preserves_signed_zero() {
    let mut ladder = DirectCalc::new(
        "MoogLadder",
        vec![
            InputSource::Audio(0),
            InputSource::Control(0),
            InputSource::Control(1),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    ladder.process(&[0.0], &[440.0, -0.0], 1);

    let state = ladder.state_snapshot().0;
    let resonance = f32::from_ne_bytes(state[8..12].try_into().expect("resonance state bytes"));
    assert_eq!(resonance.to_bits(), (-0.0_f32).to_bits());
}

/// Keeps the control cutoff multiplication in double precision until normalization is stored.
#[test]
fn moog_vcf_control_normalization_matches_source_bits() {
    let mut filter = DirectCalc::new(
        "MoogVCF",
        vec![
            InputSource::Audio(0),
            InputSource::Control(0),
            InputSource::Control(1),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    filter.process(&[0.0], &[f32::MAX, 0.0], 1);

    let state = filter.state_snapshot().0;
    let normalized = f32::from_ne_bytes(state[..4].try_into().expect("cutoff state bytes"));
    assert_eq!(normalized.to_bits(), 0x782e_c33d);
}

/// Preserves the constructor's `f32` cutoff multiplication before sample-duration scaling.
#[test]
fn moog_vcf_constructor_cutoff_overflow_matches_source() {
    let mut filter = DirectCalc::new(
        "MoogVCF",
        vec![
            InputSource::Audio(0),
            InputSource::Constant(f32::MAX),
            InputSource::Constant(0.0),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    filter.construct_once(&[0.0], &[], 1);

    let state = filter.state_snapshot().0;
    let normalized = f32::from_ne_bytes(state[..4].try_into().expect("cutoff state bytes"));
    assert_eq!(normalized, f32::INFINITY);
}

/// Pins the audio-cutoff callback's cached `f32` normalization multiplier.
#[test]
fn moog_vcf_audio_cutoff_uses_source_float_multiplier() {
    let mut filter = DirectCalc::new(
        "MoogVCF",
        vec![
            InputSource::Audio(0),
            InputSource::Audio(1),
            InputSource::Control(0),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    let cutoff = f32::from_bits(0x4595_6296);
    let output = filter.process(&[1.0, cutoff], &[0.0], 1);
    assert_eq!(output[0][0].to_bits(), 0x3c3a_d66f);
}

/// Matches the source's narrowing conversion at the `float sampleRate` filter call boundary.
#[test]
fn dfm1_sample_rate_narrows_before_coefficient_lookup() {
    let sources = vec![
        InputSource::Audio(0),
        InputSource::Control(0),
        InputSource::Control(1),
        InputSource::Control(2),
        InputSource::Control(3),
        InputSource::Control(4),
    ];
    let mut exact = DirectCalc::new("DFM1", sources.clone(), Rate::Audio, 1, 48_000.0, 64);
    let mut rounded = DirectCalc::new("DFM1", sources, Rate::Audio, 1, 48_000.000_1, 64);
    let input = vec![0.125; 64];
    let controls = [1_237.5, 0.7, 1.0, 0.0, 0.000_3];

    assert_eq!(
        exact.process(&input, &controls, 64),
        rounded.process(&input, &controls, 64)
    );
    assert_eq!(exact.state_snapshot(), rounded.state_snapshot());
}

/// Preserves the source's signed-zero and ordered-NaN branches in direct processing callbacks.
#[test]
fn translated_non_finite_boundaries_follow_source_arithmetic() {
    let mut envelope = DirectCalc::new(
        "EnvDetect",
        vec![
            InputSource::Audio(0),
            InputSource::Control(0),
            InputSource::Control(1),
        ],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    assert!(envelope.process(&[1.0], &[-0.0, 0.1], 1)[0][0].is_nan());

    let mut blit = DirectCalc::new(
        "BlitB3",
        vec![InputSource::Control(0)],
        Rate::Audio,
        1,
        48_000.0,
        64,
    );
    assert!(blit.process(&[], &[f32::NAN], 1)[0][0].is_nan());
}

/// Proves direct unit callbacks and demand pulls preserve ordered state across callback partitioning.
#[test]
fn sc3_processing_direct_partition_invariance() {
    for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
        for block_size in [1, 64, 256] {
            for case in sample_cases() {
                check_sample_case(&case, sample_rate, block_size);
            }
            check_demand_partition(sample_rate, block_size);

            for fft_size in [64, 1_024, 16_384] {
                for name in ["PV_Freeze", "PV_MagSmooth", "PV_Morph"] {
                    check_pv_partition(name, sample_rate, block_size, fft_size);
                }
            }
        }
    }
}
