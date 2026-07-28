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
            special_index: 0,
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
            initialized: false,
            observed_callbacks: Vec::new(),
            wavetables: Wavetables::new(),
            fft: FftTables::new(),
            buses: Buses::new(0, 0, 0, 0, configured_block_size),
            buffers: BufferTable::new(2),
            rgen: Rng::new(UNIT_SEED),
        }
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
        let mut state = AlignedBytes::from_bytes(&built.init_bytes, built.align);
        (built.reseed)(state.as_mut(), UNIT_SEED);
        let plan = vec![DemandVtbl {
            produce: built.produce,
            reset: built.reset,
            reseed: built.reseed,
            inputs: sources.into_boxed_slice(),
            state_offset: 0,
            state_size: built.size,
        }];
        Self {
            plan,
            state,
            buffers: BufferTable::new(0),
            rgen: Rng::new(UNIT_SEED),
            sample_rate,
            block_size,
            observed_groups: Vec::new(),
            pull_count: 0,
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
            .map(|_| access.produce(&mut world, 0))
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

/// One supplied inlet value and the explicit valid value a reference unit receives.
struct RuntimeSafetyProbe {
    inlet: usize,
    supplied: f32,
    reference: f32,
    label: String,
}

/// Return the immediately lower representable value for a finite non-negative boundary.
fn adjacent_below(value: f32) -> f32 {
    if value == 0.0 {
        -f32::from_bits(1)
    } else {
        f32::from_bits(value.to_bits() - 1)
    }
}

/// Return the immediately higher representable value for a finite non-negative boundary.
fn adjacent_above(value: f32) -> f32 {
    f32::from_bits(value.to_bits() + 1)
}

/// Add the finite boundaries, adjacent outside values, and three non-finite classes for a clamp.
fn add_bounded_probes(
    probes: &mut Vec<RuntimeSafetyProbe>,
    inlet: usize,
    minimum: f32,
    maximum: Option<f32>,
    remembered: f32,
    label: &str,
) {
    probes.push(RuntimeSafetyProbe {
        inlet,
        supplied: minimum,
        reference: minimum,
        label: format!("{label} lower boundary"),
    });
    probes.push(RuntimeSafetyProbe {
        inlet,
        supplied: adjacent_below(minimum),
        reference: minimum,
        label: format!("{label} immediately below lower boundary"),
    });
    if let Some(maximum) = maximum {
        probes.push(RuntimeSafetyProbe {
            inlet,
            supplied: maximum,
            reference: maximum,
            label: format!("{label} upper boundary"),
        });
        probes.push(RuntimeSafetyProbe {
            inlet,
            supplied: adjacent_above(maximum),
            reference: maximum,
            label: format!("{label} immediately above upper boundary"),
        });
    }
    add_nonfinite_probes(probes, inlet, remembered, label);
}

/// Add NaN and both infinities with an explicit remembered/default reference.
fn add_nonfinite_probes(
    probes: &mut Vec<RuntimeSafetyProbe>,
    inlet: usize,
    remembered: f32,
    label: &str,
) {
    for (class, supplied) in [
        ("NaN", f32::NAN),
        ("positive infinity", f32::INFINITY),
        ("negative infinity", f32::NEG_INFINITY),
    ] {
        probes.push(RuntimeSafetyProbe {
            inlet,
            supplied,
            reference: remembered,
            label: format!("{label} {class}"),
        });
    }
}

/// Add finite source-branch values whose valid behavior must not be mistaken for a safety clamp.
fn add_finite_probes(
    probes: &mut Vec<RuntimeSafetyProbe>,
    inlet: usize,
    values: &[f32],
    label: &str,
) {
    probes.extend(
        values
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| RuntimeSafetyProbe {
                inlet,
                supplied: value,
                reference: value,
                label: format!("{label} finite branch {index}"),
            }),
    );
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

/// Build the exact explicit-reference probes for one sample-processor configuration.
fn runtime_safety_probes(case: &RuntimeSafetyCase) -> Vec<RuntimeSafetyProbe> {
    let mut probes = Vec::new();
    let nyquist = 24_000.0;
    let signal_inlets: &[usize] = match case.name {
        "EnvDetect" | "Decimator" | "DFM1" | "BMoog" | "MoogLadder" | "MoogVCF" => &[0],
        "Perlin3" => &[0, 1, 2],
        _ => &[],
    };
    for &inlet in signal_inlets {
        add_nonfinite_probes(&mut probes, inlet, 0.0, "signal");
    }

    match case.name {
        "EnvDetect" => {
            add_bounded_probes(&mut probes, 1, 0.0, None, case.baseline[1], "attack");
            add_bounded_probes(&mut probes, 2, 0.0, None, case.baseline[2], "release");
        }
        "Decimator" => {
            add_bounded_probes(
                &mut probes,
                1,
                0.0,
                Some(48_000.0),
                case.baseline[1],
                "rate",
            );
            add_finite_probes(
                &mut probes,
                2,
                &[adjacent_below(1.0), 1.0, adjacent_below(31.0), 31.0],
                "bits pass-through boundary",
            );
            add_nonfinite_probes(&mut probes, 2, case.baseline[2], "bits");
        }
        "DFM1" => {
            add_bounded_probes(
                &mut probes,
                1,
                1.0,
                Some(nyquist),
                case.baseline[1],
                "frequency",
            );
            add_bounded_probes(
                &mut probes,
                2,
                0.0,
                Some(10.0),
                case.baseline[2],
                "resonance",
            );
            add_finite_probes(&mut probes, 3, &[-0.75, 1.25], "input gain");
            add_nonfinite_probes(&mut probes, 3, case.baseline[3], "input gain");
            add_finite_probes(
                &mut probes,
                4,
                &[adjacent_below(0.5), 0.5],
                "type threshold",
            );
            add_nonfinite_probes(&mut probes, 4, case.baseline[4], "type");
            add_bounded_probes(&mut probes, 5, 0.0, None, case.baseline[5], "noise level");
        }
        "BMoog" => {
            add_bounded_probes(
                &mut probes,
                1,
                20.0,
                Some(nyquist),
                case.baseline[1],
                "frequency",
            );
            add_bounded_probes(&mut probes, 2, 0.0, Some(1.0), case.baseline[2], "Q");
            add_finite_probes(
                &mut probes,
                3,
                &[
                    adjacent_below(1.0),
                    1.0,
                    adjacent_below(2.0),
                    2.0,
                    adjacent_below(3.0),
                    3.0,
                ],
                "mode threshold",
            );
            add_nonfinite_probes(&mut probes, 3, case.baseline[3], "mode");
        }
        "MoogLadder" | "MoogVCF" => {
            add_bounded_probes(
                &mut probes,
                1,
                0.0,
                Some(nyquist),
                case.baseline[1],
                "frequency",
            );
            add_bounded_probes(
                &mut probes,
                2,
                0.0,
                Some(1.0),
                case.baseline[2],
                "resonance",
            );
        }
        "BlitB3" | "BlitB3Saw" | "BlitB3Square" | "BlitB3Tri" => {
            add_bounded_probes(
                &mut probes,
                0,
                0.000_001,
                Some(nyquist),
                case.baseline[0],
                "frequency",
            );
            for inlet in 1..case.baseline.len() {
                add_bounded_probes(
                    &mut probes,
                    inlet,
                    0.0,
                    Some(1.0),
                    case.baseline[inlet],
                    "leak",
                );
            }
        }
        "Perlin3" => {}
        "RosslerL" => {
            for inlet in 0..case.baseline.len() {
                add_nonfinite_probes(&mut probes, inlet, case.baseline[inlet], "Rossler control");
            }
        }
        _ => unreachable!("unknown runtime safety case"),
    }
    probes
}

/// Compare one invalid/out-of-range callback and its valid successor to an explicit reference.
fn check_runtime_safety_probe(case: &RuntimeSafetyCase, probe: &RuntimeSafetyProbe) {
    let mut subject = DirectCalc::new(
        case.name,
        case.sources.clone(),
        case.rate,
        case.outputs,
        48_000.0,
        64,
    );
    let mut reference = DirectCalc::new(
        case.name,
        case.sources.clone(),
        case.rate,
        case.outputs,
        48_000.0,
        64,
    );
    let len = if case.rate == Rate::Audio { 16 } else { 1 };
    let (warm_audio, warm_controls) = case.input_block(&case.baseline, len);
    assert_eq!(
        subject.process(&warm_audio, &warm_controls, len),
        reference.process(&warm_audio, &warm_controls, len),
        "{} {} warm-up output",
        case.label,
        probe.label
    );
    assert_eq!(
        subject.state_snapshot(),
        reference.state_snapshot(),
        "{} {} warm-up state",
        case.label,
        probe.label
    );

    let mut supplied = case.baseline.clone();
    supplied[probe.inlet] = probe.supplied;
    let mut sanitized = case.baseline.clone();
    sanitized[probe.inlet] = probe.reference;
    let (subject_audio, subject_controls) = case.input_block(&supplied, len);
    let (reference_audio, reference_controls) = case.input_block(&sanitized, len);
    let actual = subject.process(&subject_audio, &subject_controls, len);
    let expected = reference.process(&reference_audio, &reference_controls, len);
    assert_eq!(
        actual, expected,
        "{} {} output differs from explicit reference",
        case.label, probe.label
    );
    assert!(
        actual.iter().flatten().all(|sample| sample.is_finite()),
        "{} {} emitted a non-finite sample",
        case.label,
        probe.label
    );
    assert_eq!(
        subject.state_snapshot(),
        reference.state_snapshot(),
        "{} {} state/aux differs from explicit reference",
        case.label,
        probe.label
    );

    let (recovery_audio, recovery_controls) = case.input_block(&case.recovery, len);
    let actual = subject.process(&recovery_audio, &recovery_controls, len);
    let expected = reference.process(&recovery_audio, &recovery_controls, len);
    assert_eq!(
        actual, expected,
        "{} {} recovery output",
        case.label, probe.label
    );
    assert!(
        actual.iter().flatten().all(|sample| sample.is_finite()),
        "{} {} recovery emitted a non-finite sample",
        case.label,
        probe.label
    );
    assert_eq!(
        subject.state_snapshot(),
        reference.state_snapshot(),
        "{} {} recovery state/aux",
        case.label,
        probe.label
    );
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

/// Proves every sample-processor runtime clamp and non-finite fallback is state-exact and recovers.
#[test]
fn sc3_processing_runtime_safety_matrix_is_exact_and_recovers() {
    for case in runtime_safety_cases() {
        let probes = runtime_safety_probes(&case);
        assert!(!probes.is_empty(), "{} safety matrix is empty", case.label);
        for probe in probes {
            check_runtime_safety_probe(&case, &probe);
        }
    }
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
    NonFiniteDc,
    NonFiniteNyquist,
    NonFiniteReal,
    NonFiniteImaginary,
    ComplexMagnitudeOverflow,
}

/// One invalid token, frame, or two-buffer ownership condition.
#[derive(Clone, Copy)]
enum PvRejection {
    PrimaryToken { supplied: f32, expected_output: f32 },
    SecondaryToken(f32),
    PrimaryBuffer(PvBufferFault),
    SecondaryBuffer(PvBufferFault),
    Alias,
    LengthMismatch,
}

impl PvRejection {
    /// Stable assertion label for this rejection class.
    fn label(self) -> &'static str {
        match self {
            Self::PrimaryToken {
                supplied,
                expected_output: _,
            } if supplied.is_nan() => "primary NaN token",
            Self::PrimaryToken {
                supplied: f32::INFINITY,
                expected_output: _,
            } => "primary positive-infinity token",
            Self::PrimaryToken {
                supplied: f32::NEG_INFINITY,
                expected_output: _,
            } => "primary negative-infinity token",
            Self::PrimaryToken {
                supplied: -1.0,
                expected_output: _,
            } => "primary negative token",
            Self::PrimaryToken {
                supplied: 0.5,
                expected_output: _,
            } => "primary fractional token",
            Self::PrimaryToken {
                supplied: 9.0,
                expected_output: _,
            } => "primary missing-buffer token",
            Self::PrimaryToken { .. } => "primary invalid token",
            Self::SecondaryToken(value) if value.is_nan() => "secondary NaN token",
            Self::SecondaryToken(f32::INFINITY) => "secondary positive-infinity token",
            Self::SecondaryToken(f32::NEG_INFINITY) => "secondary negative-infinity token",
            Self::SecondaryToken(-1.0) => "secondary negative token",
            Self::SecondaryToken(0.5) => "secondary fractional token",
            Self::SecondaryToken(9.0) => "secondary missing-buffer token",
            Self::SecondaryToken(_) => "secondary invalid token",
            Self::PrimaryBuffer(fault) => match fault {
                PvBufferFault::UnsupportedSize => "primary unsupported packed size",
                PvBufferFault::MultipleChannels => "primary multichannel packed shape",
                PvBufferFault::NonFiniteDc => "primary non-finite DC",
                PvBufferFault::NonFiniteNyquist => "primary non-finite Nyquist",
                PvBufferFault::NonFiniteReal => "primary non-finite real component",
                PvBufferFault::NonFiniteImaginary => "primary non-finite imaginary component",
                PvBufferFault::ComplexMagnitudeOverflow => {
                    "primary finite complex magnitude overflow"
                }
            },
            Self::SecondaryBuffer(fault) => match fault {
                PvBufferFault::UnsupportedSize => "secondary unsupported packed size",
                PvBufferFault::MultipleChannels => "secondary multichannel packed shape",
                PvBufferFault::NonFiniteDc => "secondary non-finite DC",
                PvBufferFault::NonFiniteNyquist => "secondary non-finite Nyquist",
                PvBufferFault::NonFiniteReal => "secondary non-finite real component",
                PvBufferFault::NonFiniteImaginary => "secondary non-finite imaginary component",
                PvBufferFault::ComplexMagnitudeOverflow => {
                    "secondary finite complex magnitude overflow"
                }
            },
            Self::Alias => "aliased A/B tokens",
            Self::LengthMismatch => "A/B length mismatch",
        }
    }
}

/// Construct a deterministic complex packed spectrum with one selected structural/numeric fault.
fn faulty_pv_buffer(fault: PvBufferFault, lane: usize, sample_rate: f64) -> Buffer {
    let (frames, channels) = match fault {
        PvBufferFault::UnsupportedSize => (65, 1),
        PvBufferFault::MultipleChannels => (64, 2),
        _ => (64, 1),
    };
    let mut data = vec![0.0; frames * channels];
    data[0] = 0.25 + lane as f32 * 0.125;
    data[1] = -0.375 - lane as f32 * 0.0625;
    for (bin, pair) in data[2..].chunks_exact_mut(2).enumerate() {
        pair[0] = 0.5 + bin as f32 * 0.0078125;
        pair[1] = -0.25 + lane as f32 * 0.03125;
    }
    match fault {
        PvBufferFault::NonFiniteDc => data[0] = f32::NAN,
        PvBufferFault::NonFiniteNyquist => data[1] = f32::INFINITY,
        PvBufferFault::NonFiniteReal => data[2] = f32::NEG_INFINITY,
        PvBufferFault::NonFiniteImaginary => data[3] = f32::NAN,
        PvBufferFault::ComplexMagnitudeOverflow => {
            data[2] = f32::MAX;
            data[3] = f32::MAX;
        }
        PvBufferFault::UnsupportedSize | PvBufferFault::MultipleChannels => {}
    }
    let mut buffer = Buffer::from_interleaved(data, channels, sample_rate);
    buffer.set_coord(SpectrumCoord::Complex);
    buffer
}

/// Return the complete invalid-token/frame matrix shared by every PV operator.
fn primary_pv_rejections() -> Vec<PvRejection> {
    let mut rejections = vec![
        PvRejection::PrimaryToken {
            supplied: -1.0,
            expected_output: -1.0,
        },
        PvRejection::PrimaryToken {
            supplied: f32::NAN,
            expected_output: -1.0,
        },
        PvRejection::PrimaryToken {
            supplied: f32::NEG_INFINITY,
            expected_output: -1.0,
        },
        PvRejection::PrimaryToken {
            supplied: f32::INFINITY,
            expected_output: -1.0,
        },
        PvRejection::PrimaryToken {
            supplied: 0.5,
            expected_output: 0.5,
        },
        PvRejection::PrimaryToken {
            supplied: 9.0,
            expected_output: 9.0,
        },
    ];
    rejections.extend(
        [
            PvBufferFault::UnsupportedSize,
            PvBufferFault::MultipleChannels,
            PvBufferFault::NonFiniteDc,
            PvBufferFault::NonFiniteNyquist,
            PvBufferFault::NonFiniteReal,
            PvBufferFault::NonFiniteImaginary,
            PvBufferFault::ComplexMagnitudeOverflow,
        ]
        .into_iter()
        .map(PvRejection::PrimaryBuffer),
    );
    rejections
}

/// Return every additional invalid B-frame and ownership class for `PV_Morph`.
fn secondary_pv_rejections() -> Vec<PvRejection> {
    let mut rejections = vec![
        PvRejection::SecondaryToken(-1.0),
        PvRejection::SecondaryToken(f32::NAN),
        PvRejection::SecondaryToken(f32::NEG_INFINITY),
        PvRejection::SecondaryToken(f32::INFINITY),
        PvRejection::SecondaryToken(0.5),
        PvRejection::SecondaryToken(9.0),
    ];
    rejections.extend(
        [
            PvBufferFault::UnsupportedSize,
            PvBufferFault::MultipleChannels,
            PvBufferFault::NonFiniteDc,
            PvBufferFault::NonFiniteNyquist,
            PvBufferFault::NonFiniteReal,
            PvBufferFault::NonFiniteImaginary,
            PvBufferFault::ComplexMagnitudeOverflow,
        ]
        .into_iter()
        .map(PvRejection::SecondaryBuffer),
    );
    rejections.extend([PvRejection::Alias, PvRejection::LengthMismatch]);
    rejections
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
    let mut controls = pv_controls(name, true, 2);
    let expected = match rejection {
        PvRejection::PrimaryToken {
            supplied,
            expected_output,
        } => {
            controls[0] = supplied;
            expected_output
        }
        PvRejection::SecondaryToken(supplied) => {
            controls[1] = supplied;
            0.0
        }
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
        PvRejection::Alias => {
            controls[1] = 0.0;
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
            assert_eq!(
                spectrum_snapshot(&whole, 1),
                expected_b,
                "{name} whole read-only B spectrum at {sample_rate}/{block_size}/{fft_size}/{frame}"
            );
            assert_eq!(
                spectrum_snapshot(&partitioned, 1),
                expected_b,
                "{name} partitioned read-only B spectrum at {sample_rate}/{block_size}/{fft_size}/{frame}"
            );
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

/// Proves every invalid SC3 PV token/frame is atomic and the next valid frame exactly recovers.
#[test]
fn sc3_pv_invalid_token_frame_matrix_is_atomic_and_recovers() {
    for name in ["PV_Freeze", "PV_MagSmooth", "PV_Morph"] {
        for rejection in primary_pv_rejections() {
            check_pv_rejection(name, rejection);
        }
        if name == "PV_Morph" {
            for rejection in secondary_pv_rejections() {
                check_pv_rejection(name, rejection);
            }
        }
    }
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
