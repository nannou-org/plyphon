//! `DNoiseRing` ABI, source-exact lifecycle, state transitions, and shared RNG cadence.

use plyphon::{
    AddAction, BuildContext, BuildError, GraphDef, InputRef, Options, ROOT_GROUP_ID, Rate,
    RateInfo, SynthDef, UnitRegistry, UnitSpec, World, engine,
};
use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::rng::Rng;
use plyphon_unit::graphdef::ConstructorUnit;
use plyphon_unit::unit::demand::{DemandAccess, DemandWorld};
use plyphon_unit::unit::{InputSource, LocalBufs, NodeMsg, NodeMsgSink};

const SR: f64 = 48_000.0;
const SEG_DUR: f32 = 0.002;
const SEG: usize = 96;
const FIXTURE_CHANNELS: usize = 10;
const FIXTURE_STOCHASTIC_CHANNEL: usize = 9;
const FIXTURE_PULL_FRAMES: [usize; 16] = [
    0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
];

/// Render `frames` mono samples.
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut output = vec![0.0; frames];
    world.fill(&mut output, 1);
    output
}

/// Read the middle of one held `Duty` segment.
fn segment(output: &[f32], index: usize) -> f32 {
    output[SEG / 2 + index * SEG]
}

/// Compute stable FNV-1a over little-endian `f32::to_bits` bytes.
fn fnv1a_bits(bits: &[u32]) -> u64 {
    bits.iter()
        .flat_map(|bits| bits.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// Decode a retained little-endian `f32` fixture.
fn fixture_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "fixture contains whole f32 values");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect()
}

/// A directly pulled compiled demand graph with explicit state and shared-RNG ownership.
struct DirectDemandGraph {
    definition: GraphDef,
    state: Vec<u8>,
    buffers: BufferTable,
    rng: Rng,
}

impl DirectDemandGraph {
    /// Compile demand-only `units`, run their constructors, and seed the graph RNG.
    fn new(units: Vec<UnitSpec>, seed: u64) -> Self {
        let audio = RateInfo::new(SR, 64);
        let control = RateInfo::new(SR / 64.0, 1);
        let definition = SynthDef {
            name: "direct-dnoise-lifecycle".to_string(),
            params: vec![],
            units,
        }
        .compile(
            &UnitRegistry::with_builtins(),
            &audio,
            &control,
            64,
            8,
            None,
            1,
        )
        .expect("compile direct demand graph");
        let state = definition.demand_state_image().to_vec();
        let mut graph = Self {
            definition,
            state,
            buffers: BufferTable::new(0),
            rng: Rng::new(seed),
        };
        graph.initialize();
        graph
    }

    /// Run every demand unit's source constructor callback once.
    fn initialize(&mut self) {
        let mut local_samples = [];
        let mut local_coords = [];
        let mut local_bufs = LocalBufs::new(&[], &mut local_samples, &mut local_coords, SR);
        let mut messages: Vec<NodeMsg> = Vec::new();
        let mut message_sink = NodeMsgSink::new(&mut messages, 0);
        let unit_count = self.definition.demand_units().len();
        let mut access = DemandAccess::new(
            self.definition.demand_units(),
            &mut self.state,
            &[],
            &[],
            64,
        );
        let mut world = DemandWorld {
            buffers: &mut self.buffers,
            local_bufs: &mut local_bufs,
            node_id: 1_000,
            node_msgs: &mut message_sink,
            rgen: &mut self.rng,
        };
        for unit in 0..unit_count {
            access.init(&mut world, unit);
        }
    }

    /// Produce one value from demand-plan `unit`.
    fn produce(&mut self, unit: usize) -> f32 {
        let mut local_samples = [];
        let mut local_coords = [];
        let mut local_bufs = LocalBufs::new(&[], &mut local_samples, &mut local_coords, SR);
        let mut messages: Vec<NodeMsg> = Vec::new();
        let mut message_sink = NodeMsgSink::new(&mut messages, 0);
        let mut access = DemandAccess::new(
            self.definition.demand_units(),
            &mut self.state,
            &[],
            &[],
            64,
        );
        let mut world = DemandWorld {
            buffers: &mut self.buffers,
            local_bufs: &mut local_bufs,
            node_id: 1_000,
            node_msgs: &mut message_sink,
            rgen: &mut self.rng,
        };
        access.produce(&mut world, unit, 1)
    }

    /// Reset demand-plan `unit` without producing a value.
    fn reset(&mut self, unit: usize) {
        let mut local_samples = [];
        let mut local_coords = [];
        let mut local_bufs = LocalBufs::new(&[], &mut local_samples, &mut local_coords, SR);
        let mut messages: Vec<NodeMsg> = Vec::new();
        let mut message_sink = NodeMsgSink::new(&mut messages, 0);
        let mut access = DemandAccess::new(
            self.definition.demand_units(),
            &mut self.state,
            &[],
            &[],
            64,
        );
        let mut world = DemandWorld {
            buffers: &mut self.buffers,
            local_bufs: &mut local_bufs,
            node_id: 1_000,
            node_msgs: &mut message_sink,
            rgen: &mut self.rng,
        };
        access.reset(&mut world, unit);
    }

    /// Return the exact pool-resident state bytes for demand-plan `unit`.
    fn unit_state(&self, unit: usize) -> Vec<u8> {
        let vtable = &self.definition.demand_units()[unit];
        self.state[vtable.state_offset..vtable.state_offset + vtable.state_size].to_vec()
    }

    /// Replace the exact pool-resident state bytes for demand-plan `unit`.
    fn set_unit_state<T: bytemuck::Pod>(&mut self, unit: usize, state: &T) {
        let vtable = &self.definition.demand_units()[unit];
        let bytes = bytemuck::bytes_of(state);
        assert_eq!(bytes.len(), vtable.state_size);
        self.state[vtable.state_offset..vtable.state_offset + vtable.state_size]
            .copy_from_slice(bytes);
    }

    /// Return the exact synth-shared RNG state bytes.
    fn rng_state(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.rng).to_vec()
    }
}

/// Exact byte layout of the registered `DNoiseRing` state.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct RingState {
    state: u32,
}

/// Exact byte layout shared by the registered `Drand` and `Dxrand` states.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct RandomListState {
    repeats: f64,
    repeat_count: u32,
    index: u32,
    need_reset_child: u32,
    _pad: u32,
}

/// Build a finite arithmetic demand stream.
fn finite_series(length: usize, start: f32, step: f32) -> UnitSpec {
    UnitSpec::new(
        "Dseries",
        Rate::Demand,
        vec![
            InputRef::Constant(length as f32),
            InputRef::Constant(start),
            InputRef::Constant(step),
        ],
        1,
    )
}

/// Build one constant-input `DNoiseRing` demand unit.
fn ring_unit(change: f32, chance: f32, shift: f32, bits: f32, reset: f32) -> UnitSpec {
    UnitSpec::new(
        "DNoiseRing",
        Rate::Demand,
        [change, chance, shift, bits, reset]
            .into_iter()
            .map(InputRef::Constant)
            .collect(),
        1,
    )
}

/// Build and drive one all-constant `DNoiseRing` through `Duty`.
fn drive_ring(inputs: [f32; 5]) -> World {
    let source = UnitSpec::new(
        "DNoiseRing",
        Rate::Demand,
        inputs.into_iter().map(InputRef::Constant).collect(),
        1,
    );
    let duty = UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Constant(SEG_DUR),
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 0, output: 0 },
        ],
        1,
    );
    let out = UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 1, output: 0 },
        ],
        0,
    );
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "ring".to_string(),
        params: vec![],
        units: vec![source, duty, out],
    });
    controller
        .synth_new("ring", ROOT_GROUP_ID, AddAction::Tail)
        .expect("DNoiseRing graph compiles");
    world
}

/// Invoke the registered constructor directly to cover every structural validation failure.
fn build_error(
    rate: Rate,
    input_rates: &[Rate],
    outputs: usize,
    special_index: i16,
) -> Option<BuildError> {
    let registry = UnitRegistry::with_builtins();
    let def = registry
        .get_demand("DNoiseRing")
        .expect("DNoiseRing demand registration");
    let input_units = vec![None; input_rates.len()];
    let input_sources = vec![InputSource::Constant(0.0); input_rates.len()];
    let audio = RateInfo::new(SR, 64);
    let control = RateInfo::new(SR / 64.0, 1);
    let ctx = BuildContext {
        input_rates,
        input_units: &input_units,
        input_sources: &input_sources,
        rate,
        num_outputs: outputs,
        audio: &audio,
        control: &control,
        special_index,
        seed: 1,
        local_bufs_so_far: 0,
    };
    def.build(&ctx).err()
}

/// Reject every invalid fixed-shape demand ABI while accepting all supported inlet rates.
#[test]
fn dnoise_ring_rejects_invalid_shapes_rates_and_special_index() {
    let five = [Rate::Scalar; 5];
    assert_eq!(
        build_error(Rate::Demand, &five[..4], 1, 0),
        Some(BuildError::WrongInputCount)
    );
    assert_eq!(
        build_error(Rate::Demand, &five, 2, 0),
        Some(BuildError::WrongOutputCount {
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        build_error(Rate::Demand, &five, 1, 7),
        Some(BuildError::UnsupportedOp(7))
    );
    assert_eq!(
        build_error(Rate::Control, &five, 1, 0),
        Some(BuildError::UnsupportedUnitRate)
    );

    let all_input_rates = [
        Rate::Scalar,
        Rate::Control,
        Rate::Audio,
        Rate::Demand,
        Rate::Scalar,
    ];
    assert_eq!(
        build_error(Rate::Demand, &all_input_rates, 1, 0),
        None,
        "constant, calc-wire, and nested-demand inputs are accepted"
    );
}

/// Pin source-domain truncation, rotation, and the absence of a numBits state mask.
#[test]
fn dnoise_ring_rotates_with_source_integer_rules() {
    let mut world = drive_ring([0.0, 0.0, 1.0, 4.0, 1.0]);
    let output = render(&mut world, SEG * 5);
    for (index, expected) in [8.0, 4.0, 2.0, 1.0, 8.0].into_iter().enumerate() {
        assert_eq!(segment(&output, index), expected, "segment {index}");
    }

    let mut high_bits = drive_ring([0.0, 0.0, 0.0, 8.9, 65_535.9]);
    let output = render(&mut high_bits, SEG * 2);
    assert_eq!(segment(&output, 0), 65_535.0);
    assert_eq!(segment(&output, 1), 65_535.0);
}

/// Render the first draw of a calc-rate random unit after one `DNoiseRing` pull.
fn calc_draw_after_ring(change: f32) -> f32 {
    let ring = UnitSpec::new(
        "DNoiseRing",
        Rate::Demand,
        vec![
            InputRef::Constant(change),
            InputRef::Constant(0.5),
            InputRef::Constant(1.0),
            InputRef::Constant(8.0),
            InputRef::Constant(1.0),
        ],
        1,
    );
    let duty = UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Constant(1.0),
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 0, output: 0 },
        ],
        1,
    );
    let random = UnitSpec::new(
        "TRand",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Constant(1.0),
            InputRef::Constant(0.0),
        ],
        1,
    );
    let out = UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 2, output: 0 },
        ],
        0,
    );
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "interleave".to_string(),
        params: vec![],
        units: vec![ring, duty, random, out],
    });
    controller
        .synth_new("interleave", ROOT_GROUP_ID, AddAction::Tail)
        .expect("interleaving graph compiles");
    render(&mut world, 64)[0]
}

/// Reset the shared RNG immediately before resetting and pulling a ring.
fn ring_sequence_after_seed(seed: f32) -> Vec<f32> {
    let units = vec![
        UnitSpec::new(
            "RandSeed",
            Rate::Control,
            vec![InputRef::Param(0), InputRef::Param(1)],
            1,
        ),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![
                InputRef::Constant(1.0),
                InputRef::Constant(0.5),
                InputRef::Constant(1.0),
                InputRef::Constant(8.0),
                InputRef::Constant(1.0),
            ],
            1,
        ),
        UnitSpec::new(
            "Duty",
            Rate::Audio,
            vec![
                InputRef::Constant(SEG_DUR),
                InputRef::Param(2),
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
            ],
            1,
        ),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 2, output: 0 },
            ],
            0,
        ),
    ];
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "seeded-ring".to_string(),
        params: vec![
            plyphon::Param::control("seedTrigger", 0.0),
            plyphon::Param::control("seed", seed),
            plyphon::Param::control("reset", 0.0),
        ],
        units,
    });
    let synth = controller
        .synth_new("seeded-ring", ROOT_GROUP_ID, AddAction::Tail)
        .expect("seeded DNoiseRing graph");

    render(&mut world, 64);
    controller.set_control(synth, 0, 1.0).expect("trigger seed");
    controller.set_control(synth, 2, 1.0).expect("reset ring");
    let reset_block = render(&mut world, 64);
    controller
        .set_control(synth, 0, 0.0)
        .expect("lower seed trigger");
    controller
        .set_control(synth, 2, 0.0)
        .expect("lower ring reset");
    let following = render(&mut world, SEG * 6);

    let mut sequence = vec![reset_block[0]];
    sequence.extend((0..6).map(|index| segment(&following, index)));
    sequence
}

/// Cover the one-or-two shared-RNG draws and deterministic graph-level reseeding.
#[test]
fn dnoise_ring_interleaves_and_reseeds_the_shared_rng() {
    assert_ne!(
        calc_draw_after_ring(0.0),
        calc_draw_after_ring(1.0),
        "the conditional second coin advances the following calc unit's stream"
    );

    let seeded_42_a = ring_sequence_after_seed(42.0);
    let seeded_42_b = ring_sequence_after_seed(42.0);
    let seeded_43 = ring_sequence_after_seed(43.0);
    assert_eq!(seeded_42_a, seeded_42_b, "the same seed replays");
    assert_ne!(seeded_42_a, seeded_43, "different seeds diverge");
}

/// Random-list repeat counters retain exact source comparisons above the `f32` integer boundary.
#[test]
fn random_list_repeat_counts_remain_exact_above_two_to_the_twenty_four() {
    for name in ["Drand", "Dxrand"] {
        let mut graph = DirectDemandGraph::new(
            vec![UnitSpec::new(
                name,
                Rate::Demand,
                vec![
                    InputRef::Constant(16_777_220.0),
                    InputRef::Constant(7.0),
                    InputRef::Constant(9.0),
                ],
                1,
            )],
            42,
        );
        graph.set_unit_state(
            0,
            &RandomListState {
                repeats: 16_777_220.0,
                repeat_count: 16_777_219,
                index: 1,
                need_reset_child: 1,
                _pad: 0,
            },
        );
        assert_eq!(graph.produce(0), 7.0, "{name} emits the final value");
        assert!(graph.produce(0).is_nan(), "{name} then exhausts");
    }
}

/// Pin the source `RGen::init` seed and DNoiseRing event sequence.
#[test]
fn dnoise_ring_seeded_host_golden_bits() {
    let bits: Vec<u32> = ring_sequence_after_seed(42.0)
        .into_iter()
        .map(f32::to_bits)
        .collect();
    let expected = [
        0x4300_0000,
        0x4280_0000,
        0x4200_0000,
        0x4180_0000,
        0x4110_0000,
        0x4304_0000,
        0x4284_0000,
    ];
    assert_eq!(bits, expected, "seeded host ring sequence changed");
    assert_eq!(
        fnv1a_bits(&bits),
        0xb81e_307c_ff80_22bf,
        "seeded host ring FNV-1a changed"
    );
}

/// Validate the retained scsynth stochastic lane against its pinned transitions.
#[test]
fn dnoise_ring_scsynth_transition_distribution_matches_manifest() {
    let manifest = include_str!("fixtures/sc3_processing/manifest.json");
    let section_start = manifest
        .find("\"dnoise_ring\": {")
        .expect("DNoiseRing vector metadata");
    let section_end = manifest[section_start..]
        .find("\n    \"env_detect\": {")
        .map(|offset| section_start + offset)
        .expect("DNoiseRing metadata terminator");
    let metadata = &manifest[section_start..section_end];
    assert!(
        metadata.contains("\"channels\": 10")
            && metadata.contains("\"frames\": 1024")
            && metadata.contains("\"path\": \"dnoise_ring.f32\"")
            && metadata.contains("\"pull_period_frames\": 64")
            && metadata.contains("\"seed\": 1956"),
        "retained DNoiseRing manifest contract changed"
    );

    let fixture = fixture_f32(include_bytes!("fixtures/sc3_processing/dnoise_ring.f32"));
    assert_eq!(fixture.len(), 1_024 * FIXTURE_CHANNELS);
    let states: Vec<u32> = FIXTURE_PULL_FRAMES
        .iter()
        .map(|&frame| {
            let value = fixture[frame * FIXTURE_CHANNELS + FIXTURE_STOCHASTIC_CHANNEL];
            assert!(
                value.is_finite() && value.fract() == 0.0 && (0.0..=15.0).contains(&value),
                "stochastic state at frame {frame} escaped the four-bit ring: {value}"
            );
            value as u32
        })
        .collect();
    let expected = [13, 14, 7, 11, 13, 14, 7, 11, 13, 14, 7, 11, 13, 14, 7, 11];
    assert_eq!(states, expected, "seed-1956 scsynth capture changed");

    for (pull, pair) in states.windows(2).enumerate() {
        let rotated = ((pair[0] >> 1) | (pair[0] << 3)) & 15;
        assert!(
            pair[1] == rotated || pair[1] == (rotated ^ 1),
            "pull {} is not rotate then optional bit-zero replacement: {} -> {}",
            pull + 1,
            pair[0],
            pair[1]
        );
    }
}

/// Pin constructor/reset retained-output semantics and aliased input order.
#[test]
fn dnoise_ring_constructor_and_reset_match_demand_input_zero_semantics() {
    let child = finite_series(16, 10.0, 1.0);
    let ring = UnitSpec::new(
        "DNoiseRing",
        Rate::Demand,
        (0..5)
            .map(|_| InputRef::Unit { unit: 0, output: 0 })
            .collect(),
        1,
    );
    let mut graph = DirectDemandGraph::new(vec![child, ring], 0x1234_5678_9abc_def0);

    assert_eq!(
        graph.unit_state(1),
        bytemuck::bytes_of(&RingState { state: 0 })
    );
    assert_eq!(
        graph.produce(1),
        1.0,
        "all five aliased inputs pull in order"
    );
    assert_eq!(graph.produce(0), 15.0, "five child values were consumed");

    graph.reset(1);
    assert_eq!(
        graph.unit_state(1),
        bytemuck::bytes_of(&RingState { state: 15 }),
        "reset reloads the retained input-four output"
    );
    assert_eq!(
        graph.produce(1),
        31.0,
        "inputs zero through three were reset again"
    );
}

/// Calc and demand constructors retain SynthDef order, including calc constructor outputs.
#[test]
fn dnoise_ring_constructor_reads_an_earlier_calc_constructor_output() {
    let units = vec![
        UnitSpec::new(
            "Line",
            Rate::Control,
            vec![
                InputRef::Constant(7.0),
                InputRef::Constant(9.0),
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
            ],
            1,
        ),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Constant(31.0),
                InputRef::Unit { unit: 0, output: 0 },
            ],
            1,
        ),
        UnitSpec::new(
            "Duty",
            Rate::Audio,
            vec![
                InputRef::Constant(SEG_DUR),
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
            ],
            1,
        ),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 2, output: 0 },
            ],
            0,
        ),
    ];
    let definition = SynthDef {
        name: "mixed-constructor-order".to_string(),
        params: vec![],
        units: units.clone(),
    }
    .compile(
        &UnitRegistry::with_builtins(),
        &RateInfo::new(SR, 64),
        &RateInfo::new(SR / 64.0, 1),
        64,
        8,
        None,
        1,
    )
    .expect("compile mixed constructor graph");
    assert_eq!(
        definition.constructor_units(),
        &[
            ConstructorUnit::Calc(0),
            ConstructorUnit::Demand(0),
            ConstructorUnit::Calc(1),
            ConstructorUnit::Calc(2),
        ]
    );

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "mixed-constructor-order".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("mixed-constructor-order", ROOT_GROUP_ID, AddAction::Tail)
        .expect("create mixed constructor graph");
    assert_eq!(render(&mut world, 1), vec![9.0]);
}

/// A `Duty` constructor pulls its demand inputs and publishes its held level for later constructors.
#[test]
fn dnoise_ring_constructor_reads_an_earlier_duty_constructor_output() {
    let dseq = |value| {
        UnitSpec::new(
            "Dseq",
            Rate::Demand,
            vec![InputRef::Constant(f32::INFINITY), InputRef::Constant(value)],
            1,
        )
    };
    let units = vec![
        dseq(SEG_DUR),
        dseq(7.0),
        UnitSpec::new(
            "Duty",
            Rate::Control,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
            ],
            1,
        ),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Constant(31.0),
                InputRef::Unit { unit: 2, output: 0 },
            ],
            1,
        ),
        UnitSpec::new(
            "Duty",
            Rate::Audio,
            vec![
                InputRef::Constant(SEG_DUR),
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 3, output: 0 },
            ],
            1,
        ),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 4, output: 0 },
            ],
            0,
        ),
    ];
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "duty-before-dnoise".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("duty-before-dnoise", ROOT_GROUP_ID, AddAction::Tail)
        .expect("create constructor-order graph");
    assert_eq!(render(&mut world, 1), vec![7.0]);
}

/// A demand callback samples an audio input at its one-based trigger offset.
#[test]
fn dnoise_ring_audio_inputs_use_the_consumer_callback_offset() {
    let definition = SynthDef {
        name: "dnoise-audio-offset".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Line",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "DNoiseRing",
                Rate::Demand,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(4.0),
                    InputRef::Constant(8.0),
                ],
                1,
            ),
        ],
    }
    .compile(
        &UnitRegistry::with_builtins(),
        &RateInfo::new(SR, 64),
        &RateInfo::new(SR / 64.0, 1),
        64,
        8,
        None,
        1,
    )
    .expect("compile audio-offset demand graph");

    let mut state = definition.demand_state_image().to_vec();
    let mut audio_wires = vec![0.0; 64];
    let InputSource::Audio(shift_wire) = definition.demand_units()[0].inputs[2] else {
        panic!("shift input must resolve to an audio wire");
    };
    audio_wires[shift_wire as usize * 64 + 5] = 1.0;
    let mut buffers = BufferTable::new(0);
    let mut local_samples = [];
    let mut local_coords = [];
    let mut local_bufs = LocalBufs::new(&[], &mut local_samples, &mut local_coords, SR);
    let mut messages = Vec::new();
    let mut message_sink = NodeMsgSink::new(&mut messages, 0);
    let mut rng = Rng::new(1);
    let mut world = DemandWorld {
        buffers: &mut buffers,
        local_bufs: &mut local_bufs,
        node_id: 1_000,
        node_msgs: &mut message_sink,
        rgen: &mut rng,
    };
    let mut access =
        DemandAccess::new(definition.demand_units(), &mut state, &audio_wires, &[], 64);
    access.init(&mut world, 0);
    assert_eq!(access.produce(&mut world, 0, 6), 4.0);
}

/// Prove exhausted float controls do not short-circuit later pulls or RNG draws.
#[test]
fn dnoise_ring_pulls_all_inputs_before_processing_exhausted_controls() {
    const SEED: u64 = 0x9988_7766_5544_3322;

    let change = finite_series(0, 0.0, 0.0);
    let reset = finite_series(16, 100.0, 1.0);
    let ring = UnitSpec::new(
        "DNoiseRing",
        Rate::Demand,
        vec![
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
            InputRef::Constant(8.0),
            InputRef::Unit { unit: 1, output: 0 },
        ],
        1,
    );
    let mut change_nan = DirectDemandGraph::new(vec![change, reset, ring], SEED);
    assert_eq!(change_nan.produce(2), 0.0);
    assert_eq!(change_nan.produce(1), 101.0, "input four was still pulled");
    let mut one_draw = Rng::new(SEED);
    one_draw.next_unipolar();
    assert_eq!(change_nan.rng_state(), bytemuck::bytes_of(&one_draw));

    let chance = finite_series(0, 0.0, 0.0);
    let reset = finite_series(16, 100.0, 1.0);
    let ring = UnitSpec::new(
        "DNoiseRing",
        Rate::Demand,
        vec![
            InputRef::Constant(1.0),
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Constant(0.0),
            InputRef::Constant(8.0),
            InputRef::Unit { unit: 1, output: 0 },
        ],
        1,
    );
    let mut chance_nan = DirectDemandGraph::new(vec![chance, reset, ring], SEED);
    assert_eq!(chance_nan.produce(2), 0.0);
    assert_eq!(chance_nan.produce(1), 101.0, "input four was still pulled");
    let mut two_draws = Rng::new(SEED);
    two_draws.next_unipolar();
    two_draws.next_unipolar();
    assert_eq!(chance_nan.rng_state(), bytemuck::bytes_of(&two_draws));
}

/// Pin safe Rust replacements only where the source would perform undefined casts or shifts.
#[test]
fn dnoise_ring_invalid_integer_domain_is_safe_and_deterministic() {
    let mut invalid_shift = DirectDemandGraph::new(vec![ring_unit(0.0, 0.0, 32.0, 8.0, 7.0)], 1);
    assert_eq!(invalid_shift.produce(0), 7.0);

    let mut invalid_reset =
        DirectDemandGraph::new(vec![ring_unit(0.0, 0.0, 0.0, 8.0, f32::NAN)], 1);
    assert_eq!(invalid_reset.produce(0), 0.0);
}
