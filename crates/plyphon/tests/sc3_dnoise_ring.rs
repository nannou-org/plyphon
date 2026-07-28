//! `DNoiseRing` ABI, deterministic state transitions, and synth-shared RNG interleaving.

use plyphon::{
    AddAction, BuildContext, BuildError, GraphDef, InputRef, Options, Param, ROOT_GROUP_ID, Rate,
    RateInfo, SynthDef, UnitRegistry, UnitSpec, World, engine,
};
use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::rng::Rng;
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
    control_wires: Vec<f32>,
    buffers: BufferTable,
    rng: Rng,
}

impl DirectDemandGraph {
    /// Compile demand-only `units` and seed their shared host RNG deterministically.
    fn new(units: Vec<UnitSpec>, seed: u64) -> Self {
        Self::with_params(units, vec![], seed)
    }

    /// Compile demand-only `units` with mutable control parameters.
    fn with_params(units: Vec<UnitSpec>, params: Vec<Param>, seed: u64) -> Self {
        let audio = RateInfo::new(SR, 64);
        let control = RateInfo::new(SR / 64.0, 1);
        let definition = SynthDef {
            name: "direct-dnoise-lifecycle".to_string(),
            params,
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
        let control_wires = definition.control_defaults().to_vec();
        Self {
            definition,
            state,
            control_wires,
            buffers: BufferTable::new(0),
            rng: Rng::new(seed),
        }
    }

    /// Replace one graph parameter before the next direct pull.
    fn set_param(&mut self, index: usize, value: f32) {
        self.control_wires[index] = value;
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
            &self.control_wires,
            64,
        );
        let mut world = DemandWorld {
            buffers: &mut self.buffers,
            local_bufs: &mut local_bufs,
            node_id: 1_000,
            node_msgs: &mut message_sink,
            rgen: &mut self.rng,
        };
        access.produce(&mut world, unit)
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
            &self.control_wires,
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

    /// Return the exact synth-shared RNG state bytes.
    fn rng_state(&self) -> Vec<u8> {
        bytemuck::bytes_of(&self.rng).to_vec()
    }

    /// Replace only the synth-shared RNG, mirroring a graph-level reseed event.
    fn reseed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }
}

/// Exact byte layout of the registered `DNoiseRing` demand state.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct RingState {
    state: u32,
    initialized: u32,
    last_change: f32,
    last_chance: f32,
    last_shift: f32,
    last_num_bits: f32,
    last_reset: f32,
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

/// Invoke the registered constructor directly so every structural error, including a wrong demand
/// node rate that the split registry rejects before normal graph compilation, is covered.
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

/// Rejects every invalid fixed-shape demand ABI while accepting all supported inlet rates.
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

/// Pins lazy initialization, truncation, masking, and signed rotation behavior.
#[test]
fn dnoise_ring_lazily_initializes_and_rotates_with_exact_integer_rules() {
    // change=0 still consumes one coin but never replaces bit zero. Reset 1 in four bits rotates:
    // 0001 -> 1000 -> 0100 -> 0010 -> 0001.
    let mut world = drive_ring([0.0, 0.0, 1.0, 4.0, 1.0]);
    let output = render(&mut world, SEG * 5);
    for (index, expected) in [8.0, 4.0, 2.0, 1.0, 8.0].into_iter().enumerate() {
        assert_eq!(segment(&output, index), expected, "segment {index}");
    }

    // Signed Euclidean remainder maps shift -1 to 3 at width 4, equivalent to a one-bit left
    // rotation: 0001 -> 0010 -> 0100.
    let mut negative = drive_ring([0.0, 0.0, -1.0, 4.9, 1.9]);
    let output = render(&mut negative, SEG * 2);
    assert_eq!(segment(&output, 0), 2.0);
    assert_eq!(segment(&output, 1), 4.0);
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

/// Drive a ring whose `change` input exhausts after one pull. A rising `Duty` reset must reset both
/// the ring and that nested source so production can resume.
fn exhausting_ring() -> (plyphon::Controller, World, i32) {
    let change = UnitSpec::new(
        "Dseries",
        Rate::Demand,
        vec![
            InputRef::Constant(1.0),
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
        ],
        1,
    );
    let ring = UnitSpec::new(
        "DNoiseRing",
        Rate::Demand,
        vec![
            InputRef::Unit { unit: 0, output: 0 },
            InputRef::Constant(0.0),
            InputRef::Constant(1.0),
            InputRef::Constant(4.0),
            InputRef::Constant(1.0),
        ],
        1,
    );
    let duty = UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Constant(SEG_DUR),
            InputRef::Param(0),
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 1, output: 0 },
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
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "exhausting-ring".to_string(),
        params: vec![Param::control("reset", 0.0)],
        units: vec![change, ring, duty, out],
    });
    let synth = controller
        .synth_new("exhausting-ring", ROOT_GROUP_ID, AddAction::Tail)
        .expect("exhausting DNoiseRing graph");
    (controller, world, synth)
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
            Param::control("seedTrigger", 0.0),
            Param::control("seed", seed),
            Param::control("reset", 0.0),
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

/// Covers shared-RNG draw order, nested exhaustion, reset, and deterministic reseeding.
#[test]
fn dnoise_ring_pull_reset_reseed_exhaust_and_interleave() {
    let after_one_coin = calc_draw_after_ring(0.0);
    let after_two_coins = calc_draw_after_ring(1.0);
    assert_ne!(
        after_one_coin, after_two_coins,
        "the conditional second coin must advance the stream observed by the following calc unit"
    );

    let (mut controller, mut world, synth) = exhausting_ring();
    let exhausted = render(&mut world, SEG * 3);
    assert_eq!(segment(&exhausted, 0), 8.0, "first nested pull");
    assert_eq!(
        segment(&exhausted, 1),
        8.0,
        "nested exhaustion holds the last completed outer value"
    );
    assert_eq!(
        segment(&exhausted, 2),
        8.0,
        "exhaustion does not partially rotate outer state"
    );
    controller.set_control(synth, 0, 1.0).expect("raise reset");
    assert_eq!(
        render(&mut world, 64)[0],
        8.0,
        "reset restarts the exhausted child before the retry"
    );

    let seeded_42_a = ring_sequence_after_seed(42.0);
    let seeded_42_b = ring_sequence_after_seed(42.0);
    let seeded_43 = ring_sequence_after_seed(43.0);
    assert_eq!(
        seeded_42_a, seeded_42_b,
        "the same reseed reproduces the ring stream"
    );
    assert_ne!(
        seeded_42_a, seeded_43,
        "RandSeed runs before the following DNoiseRing pulls on the shared stream"
    );
}

/// Pin the host RNG seed/event sequence independently of scsynth's different seed initialization.
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
        0x4188_0000,
        0x4309_0000,
        0x4344_0000,
        0x42c6_0000,
    ];
    assert_eq!(bits, expected, "seeded host ring sequence changed");
    assert_eq!(
        fnv1a_bits(&bits),
        0xe3ae_3453_9597_c9d2,
        "seeded host ring FNV-1a changed"
    );
}

/// Validate the retained scsynth stochastic lane against its manifest-pinned seed and transitions.
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
            "pull {} cannot be reconstructed as rotate then optional bit-zero replacement: {} -> {}",
            pull + 1,
            pair[0],
            pair[1]
        );
    }
    let set_bit_zero = states.iter().filter(|state| **state & 1 == 1).count();
    assert!(
        (4..=12).contains(&set_bit_zero),
        "seed-1956 bit-zero balance exceeded the fixed 4..=12 failure budget: {set_bit_zero}/16"
    );
    assert_eq!(
        states
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        [7, 11, 13, 14].into_iter().collect(),
        "seeded stochastic lane lost its pinned transition coverage"
    );
}

/// A control-triggered demand graph used by the reset-width lifecycle vectors.
struct ResetHarness {
    controller: plyphon::Controller,
    world: World,
    synth: i32,
}

impl ResetHarness {
    /// Build a graph from its demand-prefix units and the index of the ring under test.
    fn new(mut units: Vec<UnitSpec>, ring: u32, params: Vec<Param>) -> Self {
        let demand = units.len() as u32;
        units.push(UnitSpec::new(
            "Demand",
            Rate::Control,
            vec![
                InputRef::Param(0),
                InputRef::Param(1),
                InputRef::Unit {
                    unit: ring,
                    output: 0,
                },
            ],
            1,
        ));
        let audio = units.len() as u32;
        units.push(UnitSpec::new(
            "DC",
            Rate::Audio,
            vec![InputRef::Unit {
                unit: demand,
                output: 0,
            }],
            1,
        ));
        units.push(UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit {
                    unit: audio,
                    output: 0,
                },
            ],
            0,
        ));

        let (mut controller, _nrt, mut world) = engine(Options {
            sample_rate: SR,
            block_size: 64,
            output_channels: 1,
            input_channels: 0,
            ..Options::default()
        });
        controller.add_synthdef(SynthDef {
            name: "dnoise-reset-width".to_string(),
            params,
            units,
        });
        let synth = controller
            .synth_new("dnoise-reset-width", ROOT_GROUP_ID, AddAction::Tail)
            .expect("reset-width graph");
        let mut initial = [0.0; 64];
        world.fill(&mut initial, 1);
        Self {
            controller,
            world,
            synth,
        }
    }

    /// Change one graph control before the next block.
    fn set(&mut self, index: usize, value: f32) {
        self.controller
            .set_control(self.synth, index, value)
            .expect("set reset-vector control");
    }

    /// Render one control block and return its first held sample.
    fn block(&mut self) -> f32 {
        let mut output = [0.0; 64];
        self.world.fill(&mut output, 1);
        output[0]
    }

    /// Produce one rising edge on `parameter`, returning the high block's output.
    fn pulse(&mut self, parameter: usize) -> f32 {
        self.set(parameter, 1.0);
        let high = self.block();
        self.set(parameter, 0.0);
        self.block();
        high
    }
}

/// Return an infinite demand sequence over `values`.
fn repeating_sequence(values: &[f32]) -> UnitSpec {
    let mut inputs = vec![InputRef::Constant(f32::INFINITY)];
    inputs.extend(values.iter().copied().map(InputRef::Constant));
    UnitSpec::new("Dseq", Rate::Demand, inputs, 1)
}

/// Return a demand arithmetic series whose length/start are graph controls and whose step is zero.
fn controlled_constant_series(length_param: u32, start_param: u32) -> UnitSpec {
    UnitSpec::new(
        "Dseries",
        Rate::Demand,
        vec![
            InputRef::Param(length_param),
            InputRef::Param(start_param),
            InputRef::Constant(0.0),
        ],
        1,
    )
}

/// Pin reset's retained-width rule across nested, exhausted, changed, and failed-produce inputs.
#[test]
fn dnoise_ring_reset_retains_successful_num_bits_without_pulling_it() {
    // A reset propagates to the nested 8,4 sequence but must not pull it. The first post-reset
    // produce therefore still sees width 8 and preserves reset value 255. Pulling input 3 during
    // reset would consume 8, so the visible produce would use width 4 and emit 15.
    let mut nested = ResetHarness::new(
        vec![
            repeating_sequence(&[8.0, 4.0]),
            UnitSpec::new(
                "DNoiseRing",
                Rate::Demand,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(255.0),
                ],
                1,
            ),
        ],
        1,
        vec![Param::control("trigger", 0.0), Param::control("reset", 0.0)],
    );
    assert_eq!(nested.pulse(0), 255.0, "initial nested width");
    assert_eq!(nested.pulse(1), 255.0, "reset holds the prior output");
    assert_eq!(
        nested.pulse(0),
        255.0,
        "reset must not consume the nested numBits value"
    );

    // Exhaust width 8, then change the reset Dseries to two width-32 values. Reset must use the
    // last successful width 8 before it resets (but does not pull) that child. The following
    // width-32 produce exposes the retained-width mask as 255 rather than 65535.
    let mut exhausted = ResetHarness::new(
        vec![
            controlled_constant_series(2, 3),
            UnitSpec::new(
                "DNoiseRing",
                Rate::Demand,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(65_535.0),
                ],
                1,
            ),
        ],
        1,
        vec![
            Param::control("trigger", 0.0),
            Param::control("reset", 0.0),
            Param::control("length", 1.0),
            Param::control("start", 8.0),
        ],
    );
    assert_eq!(exhausted.pulse(0), 255.0, "width-8 initialization");
    assert_eq!(
        exhausted.pulse(0),
        255.0,
        "exhausted numBits holds the completed value"
    );
    exhausted.set(2, 2.0);
    exhausted.set(3, 32.0);
    assert_eq!(exhausted.pulse(1), 255.0, "reset holds prior output");
    assert_eq!(
        exhausted.pulse(0),
        255.0,
        "an exhausted and changed numBits child cannot alter reset width"
    );

    // The aliased finite child supplies input 0 and input 4. Its three values make the first
    // produce succeed, then make the second fail only after width 8 was pulled. Reset restarts the
    // alias through input 0 and pulls resetval 255, which must still be masked by the last
    // *successful* width 4. The next successful width-8 produce therefore emits 15, not 255.
    let mut failed = ResetHarness::new(
        vec![
            controlled_constant_series(3, 4),
            UnitSpec::new(
                "DNoiseRing",
                Rate::Demand,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Param(2),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                1,
            ),
        ],
        1,
        vec![
            Param::control("trigger", 0.0),
            Param::control("reset", 0.0),
            Param::control("numBits", 4.0),
            Param::control("length", 3.0),
            Param::control("value", 255.0),
        ],
    );
    assert_eq!(failed.pulse(0), 15.0, "successful width-4 produce");
    failed.set(2, 8.0);
    assert_eq!(
        failed.pulse(0),
        15.0,
        "late resetval exhaustion holds the completed outer value"
    );
    assert_eq!(failed.pulse(1), 15.0, "reset holds prior output");
    assert_eq!(
        failed.pulse(0),
        15.0,
        "a failed produce must not commit its successfully pulled width"
    );
}

/// Exhaust each inlet independently and pin ordered child advancement across repeated retries.
#[test]
fn dnoise_ring_each_inlet_exhaustion_commits_only_earlier_children() {
    const START: [f32; 5] = [0.0, 0.5, 1.0, 4.0, 1.0];
    const STEP: [f32; 5] = [0.25, 0.125, 1.0, 1.0, 1.0];

    for exhausted_inlet in 0..5 {
        let mut units = Vec::new();
        for inlet in 0..5 {
            units.push(finite_series(
                if inlet == exhausted_inlet { 1 } else { 16 },
                START[inlet],
                STEP[inlet],
            ));
        }
        units.push(UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            (0..5)
                .map(|unit| InputRef::Unit { unit, output: 0 })
                .collect(),
            1,
        ));
        let mut graph = DirectDemandGraph::new(units, 0x1020_3040_5060_7080);

        assert_eq!(
            graph.produce(5),
            8.0,
            "inlet {exhausted_inlet}: first complete transition"
        );
        let completed_outer = graph.unit_state(5);
        assert!(
            graph.produce(5).is_nan(),
            "inlet {exhausted_inlet}: first exhaustion"
        );
        assert_eq!(
            graph.unit_state(5),
            completed_outer,
            "inlet {exhausted_inlet}: failed pull changed outer state"
        );
        assert!(
            graph.produce(5).is_nan(),
            "inlet {exhausted_inlet}: repeated retry"
        );
        assert_eq!(
            graph.unit_state(5),
            completed_outer,
            "inlet {exhausted_inlet}: retry changed outer state"
        );

        for inlet in 0..5 {
            let next = graph.produce(inlet);
            if inlet == exhausted_inlet {
                assert!(
                    next.is_nan(),
                    "inlet {exhausted_inlet}: exhausted child revived"
                );
            } else {
                let consumed = if inlet < exhausted_inlet { 3.0 } else { 1.0 };
                assert_eq!(
                    next,
                    START[inlet] + consumed * STEP[inlet],
                    "inlet {exhausted_inlet}: child {inlet} next value"
                );
            }
        }
    }
}

/// Prove a nested ring and its shared RNG advance before a later inlet exhausts.
#[test]
fn dnoise_ring_nested_child_and_rng_advance_before_later_exhaustion() {
    const SEED: u64 = 0x9988_7766_5544_3322;
    let units = vec![
        ring_unit(1.0, 0.5, 1.0, 4.0, 1.0),
        finite_series(1, 4.0, 0.0),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.0),
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(1.0),
            ],
            1,
        ),
    ];
    let mut graph = DirectDemandGraph::new(units, SEED);

    let first_outer = graph.produce(2);
    let outer_after_success = graph.unit_state(2);
    let nested_after_success = graph.unit_state(0);
    assert!(
        graph.produce(2).is_nan(),
        "later numBits inlet must exhaust"
    );
    assert_eq!(
        graph.unit_state(2),
        outer_after_success,
        "failed outer transition must remain byte-identical"
    );
    assert_ne!(
        graph.unit_state(0),
        nested_after_success,
        "nested ring state must commit before the later exhaustion"
    );

    let mut expected_rng = Rng::new(SEED);
    for _ in 0..6 {
        expected_rng.next_unipolar();
    }
    assert_eq!(
        graph.rng_state(),
        bytemuck::bytes_of(&expected_rng),
        "first outer transition draws four coins and the failed retry's nested ring draws two"
    );

    assert_eq!(first_outer.to_bits(), 0, "initial outer value");
    let next_nested = graph.produce(0);
    assert_eq!(next_nested.to_bits(), 0x40e0_0000, "next nested value");
    for _ in 0..2 {
        expected_rng.next_unipolar();
    }
    assert_eq!(
        graph.rng_state(),
        bytemuck::bytes_of(&expected_rng),
        "the observable next nested value consumes exactly two more coins"
    );
}

/// Pin nested/aliased reset behavior and both reset/reseed event orders.
#[test]
fn dnoise_ring_nested_alias_reset_and_reseed_order_have_exact_next_values() {
    const INITIAL_SEED: u64 = 0x1234_5678_9abc_def0;
    const EVENT_SEED: u64 = 42;

    let alias_units = vec![
        finite_series(16, 10.0, 1.0),
        ring_unit(0.0, 0.0, 1.0, 4.0, 1.0),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Constant(4.0),
                InputRef::Unit { unit: 0, output: 0 },
            ],
            1,
        ),
    ];
    let mut aliased = DirectDemandGraph::new(alias_units, INITIAL_SEED);
    assert_eq!(
        aliased.produce(2).to_bits(),
        0x4150_0000,
        "initial aliased value"
    );
    aliased.reset(2);
    assert_eq!(
        aliased.produce(0),
        11.0,
        "reset aliases input 0/1 but pulls input 4 exactly once"
    );
    assert_eq!(
        aliased.produce(1),
        8.0,
        "nested ring's next value starts from its reset state"
    );
    assert_eq!(
        aliased.produce(2).to_bits(),
        0x4130_0000,
        "aliased value after nested reset"
    );

    let order_units = || vec![ring_unit(1.0, 0.5, 1.0, 4.0, 1.0)];
    let mut reset_then_reseed = DirectDemandGraph::new(order_units(), INITIAL_SEED);
    let initial_a = reset_then_reseed.produce(0);
    reset_then_reseed.reset(0);
    let reset_state = reset_then_reseed.unit_state(0);
    let rng_before_reseed = reset_then_reseed.rng_state();
    reset_then_reseed.reseed(EVENT_SEED);
    assert_eq!(
        reset_then_reseed.unit_state(0),
        reset_state,
        "reseed after reset changed ring state"
    );
    assert_ne!(
        reset_then_reseed.rng_state(),
        rng_before_reseed,
        "reseed after reset did not replace the shared RNG"
    );
    let next_a = reset_then_reseed.produce(0);

    let mut reseed_then_reset = DirectDemandGraph::new(order_units(), INITIAL_SEED);
    let initial_b = reseed_then_reset.produce(0);
    reseed_then_reset.reseed(EVENT_SEED);
    let reseeded_rng = reseed_then_reset.rng_state();
    let state_before_reset = reseed_then_reset.unit_state(0);
    reseed_then_reset.reset(0);
    assert_ne!(
        reseed_then_reset.unit_state(0),
        state_before_reset,
        "reset after reseed did not restore ring state"
    );
    assert_eq!(
        reseed_then_reset.rng_state(),
        reseeded_rng,
        "reset after reseed must not rewind the shared RNG"
    );
    let next_b = reseed_then_reset.produce(0);

    assert_eq!(initial_a.to_bits(), 0x4100_0000, "initial order value");
    assert_eq!(initial_b.to_bits(), 0x4100_0000, "initial order replay");
    assert_eq!(
        next_a.to_bits(),
        0x4100_0000,
        "reset then reseed next value"
    );
    assert_eq!(
        next_b.to_bits(),
        0x4100_0000,
        "reseed then reset next value"
    );
    assert_eq!(
        next_a.to_bits(),
        next_b.to_bits(),
        "reset and reseed commute because reset does not draw RNG"
    );
}

/// Pin every defined numeric conversion, fallback, and remembered-control byte.
#[test]
fn dnoise_ring_numeric_conversion_matrix_is_exact_and_atomic() {
    const SEED: u64 = 0x6d0f_76e1_894b_3a25;
    const MAX_BELOW_U32: f32 = 4_294_967_040.0;
    const FIRST_ABOVE_U32: f32 = 4_294_967_296.0;
    const TWO_TO_63: f32 = 9_223_372_036_854_775_808.0;

    let new_graph = |defaults: [f32; 5]| {
        let params = ["change", "chance", "shift", "numBits", "resetval"]
            .into_iter()
            .zip(defaults)
            .map(|(name, default)| Param::control(name, default))
            .collect();
        let ring = UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            (0..5).map(InputRef::Param).collect(),
            1,
        );
        DirectDemandGraph::with_params(vec![ring], params, SEED)
    };
    let assert_state = |graph: &DirectDemandGraph, expected: RingState, context: &str| {
        assert_eq!(
            graph.unit_state(0),
            bytemuck::bytes_of(&expected),
            "{context}: ring state/control bytes"
        );
    };
    let mask = |width: u32| {
        if width == 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        }
    };
    let rotate = |state: u32, shift: u32, width: u32| {
        let width_mask = mask(width);
        if shift == 0 {
            state & width_mask
        } else {
            ((state >> shift) | (state << (width - shift))) & width_mask
        }
    };
    let transition =
        |state: u32, shift: u32, width: u32, change: f32, chance: f32, rng: &mut Rng| {
            let mut next = rotate(state, shift, width);
            if rng.next_unipolar() < change {
                if rng.next_unipolar() < chance {
                    next |= 1;
                } else {
                    next &= !1;
                }
            }
            next & mask(width)
        };

    /// One finite conversion case and its expected exact output bits.
    struct IntegerCase {
        label: &'static str,
        value: f32,
        width: u32,
        shift_at_width_8: u32,
        reset: u32,
        width_output_bits: u32,
        shift_output_bits: u32,
        reset_output_bits: u32,
    }

    let finite = [
        IntegerCase {
            label: "-f32::MAX",
            value: -f32::MAX,
            width: 1,
            shift_at_width_8: 0,
            reset: 0,
            width_output_bits: 0x3f80_0000,
            shift_output_bits: 0x4301_0000,
            reset_output_bits: 0,
        },
        IntegerCase {
            label: "-1.75",
            value: -1.75,
            width: 1,
            shift_at_width_8: 7,
            reset: 0,
            width_output_bits: 0x3f80_0000,
            shift_output_bits: 0x4040_0000,
            reset_output_bits: 0,
        },
        IntegerCase {
            label: "0",
            value: 0.0,
            width: 1,
            shift_at_width_8: 0,
            reset: 0,
            width_output_bits: 0x3f80_0000,
            shift_output_bits: 0x4301_0000,
            reset_output_bits: 0,
        },
        IntegerCase {
            label: "3.75",
            value: 3.75,
            width: 3,
            shift_at_width_8: 3,
            reset: 3,
            width_output_bits: 0x40e0_0000,
            shift_output_bits: 0x4240_0000,
            reset_output_bits: 0x4040_0000,
        },
        IntegerCase {
            label: "greatest f32 below u32::MAX",
            value: MAX_BELOW_U32,
            width: 32,
            shift_at_width_8: 0,
            reset: 4_294_967_040,
            width_output_bits: 0x4f80_0000,
            shift_output_bits: 0x4301_0000,
            reset_output_bits: 0x4f7f_ffff,
        },
        IntegerCase {
            label: "first f32 above u32::MAX",
            value: FIRST_ABOVE_U32,
            width: 32,
            shift_at_width_8: 0,
            reset: u32::MAX,
            width_output_bits: 0x4f80_0000,
            shift_output_bits: 0x4301_0000,
            reset_output_bits: 0x4f80_0000,
        },
        IntegerCase {
            label: "2^63",
            value: TWO_TO_63,
            width: 32,
            shift_at_width_8: 7,
            reset: u32::MAX,
            width_output_bits: 0x4f80_0000,
            shift_output_bits: 0x4040_0000,
            reset_output_bits: 0x4f80_0000,
        },
        IntegerCase {
            label: "f32::MAX",
            value: f32::MAX,
            width: 32,
            shift_at_width_8: 7,
            reset: u32::MAX,
            width_output_bits: 0x4f80_0000,
            shift_output_bits: 0x4040_0000,
            reset_output_bits: 0x4f80_0000,
        },
    ];

    for case in &finite {
        let mut width_graph = new_graph([0.0, 0.0, 0.0, case.value, FIRST_ABOVE_U32]);
        let width_state = mask(case.width);
        assert_eq!(
            width_graph.produce(0).to_bits(),
            case.width_output_bits,
            "{}: numBits output",
            case.label
        );
        assert_state(
            &width_graph,
            RingState {
                state: width_state,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 0.0,
                last_num_bits: case.value,
                last_reset: FIRST_ABOVE_U32,
            },
            &format!("{} numBits {}", case.label, case.width),
        );

        let mut shift_graph = new_graph([0.0, 0.0, case.value, 8.0, 129.0]);
        let shifted = rotate(129, case.shift_at_width_8, 8);
        assert_eq!(
            shift_graph.produce(0).to_bits(),
            case.shift_output_bits,
            "{}: shift output",
            case.label
        );
        assert_state(
            &shift_graph,
            RingState {
                state: shifted,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: case.value,
                last_num_bits: 8.0,
                last_reset: 129.0,
            },
            &format!("{} Euclidean shift {}", case.label, case.shift_at_width_8),
        );

        let mut reset_graph = new_graph([0.0, 0.0, 0.0, 32.0, case.value]);
        assert_eq!(
            reset_graph.produce(0).to_bits(),
            case.reset_output_bits,
            "{}: width-32 reset output",
            case.label
        );
        assert_state(
            &reset_graph,
            RingState {
                state: case.reset,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 0.0,
                last_num_bits: 32.0,
                last_reset: case.value,
            },
            &format!("{} reset conversion {}", case.label, case.reset),
        );

        let mut masked_reset_graph = new_graph([0.0, 0.0, 0.0, 5.0, case.value]);
        let masked_reset = case.reset & 31;
        assert_eq!(
            masked_reset_graph.produce(0).to_bits(),
            (masked_reset as f32).to_bits(),
            "{}: width-5 reset mask",
            case.label
        );
        assert_state(
            &masked_reset_graph,
            RingState {
                state: masked_reset,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 0.0,
                last_num_bits: 5.0,
                last_reset: case.value,
            },
            &format!("{} width-5 reset mask", case.label),
        );
    }

    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut shift_graph = new_graph([0.0, 0.0, nonfinite, 8.0, 129.0]);
        assert_eq!(
            shift_graph.produce(0).to_bits(),
            0x4340_0000,
            "non-finite shift uses default one"
        );
        assert_state(
            &shift_graph,
            RingState {
                state: 192,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 1.0,
                last_num_bits: 8.0,
                last_reset: 129.0,
            },
            "non-finite shift default",
        );

        let mut width_graph = new_graph([0.0, 0.0, 0.0, nonfinite, 255.0]);
        assert_eq!(
            width_graph.produce(0).to_bits(),
            0x437f_0000,
            "non-finite numBits uses default eight"
        );
        assert_state(
            &width_graph,
            RingState {
                state: 255,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 0.0,
                last_num_bits: 8.0,
                last_reset: 255.0,
            },
            "non-finite numBits default",
        );

        let mut reset_graph = new_graph([0.0, 0.0, 0.0, 8.0, nonfinite]);
        assert_eq!(
            reset_graph.produce(0).to_bits(),
            0,
            "non-finite reset uses default zero"
        );
        assert_state(
            &reset_graph,
            RingState {
                state: 0,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 0.0,
                last_num_bits: 8.0,
                last_reset: 0.0,
            },
            "non-finite reset default",
        );
    }

    let mut shift_recovery = new_graph([0.0, 0.0, 3.75, 8.0, 129.0]);
    let mut shift_state = 129;
    let shift_recovery_cases: [(f32, f32); 5] = [
        (3.75, 3.75),
        (f32::NAN, 3.75),
        (f32::INFINITY, 3.75),
        (f32::NEG_INFINITY, 3.75),
        (-1.75, -1.75),
    ];
    for (input, remembered) in shift_recovery_cases {
        shift_recovery.set_param(2, input);
        let converted = (remembered.trunc() as i64).rem_euclid(8) as u32;
        shift_state = rotate(shift_state, converted, 8);
        assert_eq!(
            shift_recovery.produce(0).to_bits(),
            (shift_state as f32).to_bits(),
            "shift fallback/recovery output"
        );
        assert_state(
            &shift_recovery,
            RingState {
                state: shift_state,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: remembered,
                last_num_bits: 8.0,
                last_reset: 129.0,
            },
            "shift fallback/recovery memory",
        );
    }

    let mut width_recovery = new_graph([0.0, 0.0, 0.0, 3.75, 255.0]);
    for (input, remembered, expected_state) in [
        (3.75, 3.75, 7),
        (f32::NAN, 3.75, 7),
        (f32::INFINITY, 3.75, 7),
        (f32::NEG_INFINITY, 3.75, 7),
        (8.0, 8.0, 7),
    ] {
        width_recovery.set_param(3, input);
        assert_eq!(
            width_recovery.produce(0).to_bits(),
            (expected_state as f32).to_bits(),
            "numBits fallback/recovery output"
        );
        assert_state(
            &width_recovery,
            RingState {
                state: expected_state,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 0.0,
                last_num_bits: remembered,
                last_reset: 255.0,
            },
            "numBits fallback/recovery memory",
        );
    }

    let mut reset_recovery = new_graph([0.0, 0.0, 0.0, 8.0, 129.0]);
    for (input, remembered) in [
        (129.0, 129.0),
        (f32::NAN, 129.0),
        (f32::INFINITY, 129.0),
        (f32::NEG_INFINITY, 129.0),
        (3.75, 3.75),
    ] {
        reset_recovery.set_param(4, input);
        assert_eq!(
            reset_recovery.produce(0).to_bits(),
            0x4301_0000,
            "reset fallback/recovery output"
        );
        assert_state(
            &reset_recovery,
            RingState {
                state: 129,
                initialized: 1,
                last_change: 0.0,
                last_chance: 0.0,
                last_shift: 0.0,
                last_num_bits: 8.0,
                last_reset: remembered,
            },
            "reset fallback/recovery memory",
        );
    }
    reset_recovery.reset(0);
    assert_state(
        &reset_recovery,
        RingState {
            state: 3,
            initialized: 1,
            last_change: 0.0,
            last_chance: 0.0,
            last_shift: 0.0,
            last_num_bits: 8.0,
            last_reset: 3.75,
        },
        "valid reset recovery is used by the next reset",
    );

    let below_zero = f32::from_bits(0x8000_0001);
    let above_one = f32::from_bits(0x3f80_0001);
    for (label, value, sanitized) in [
        ("zero", 0.0, 0.0),
        ("immediately below zero", below_zero, 0.0),
        ("one", 1.0, 1.0),
        ("immediately above one", above_one, 1.0),
    ] {
        let mut change_graph = new_graph([value, 1.0, 0.0, 1.0, 0.0]);
        let expected = if sanitized == 0.0 { 0 } else { 1 };
        assert_eq!(
            change_graph.produce(0).to_bits(),
            (expected as f32).to_bits(),
            "{label}: change output"
        );
        assert_state(
            &change_graph,
            RingState {
                state: expected,
                initialized: 1,
                last_change: sanitized,
                last_chance: 1.0,
                last_shift: 0.0,
                last_num_bits: 1.0,
                last_reset: 0.0,
            },
            &format!("{label} change clamp"),
        );

        let mut chance_graph = new_graph([1.0, value, 0.0, 1.0, 0.0]);
        assert_eq!(
            chance_graph.produce(0).to_bits(),
            (expected as f32).to_bits(),
            "{label}: chance output"
        );
        assert_state(
            &chance_graph,
            RingState {
                state: expected,
                initialized: 1,
                last_change: 1.0,
                last_chance: sanitized,
                last_shift: 0.0,
                last_num_bits: 1.0,
                last_reset: 0.0,
            },
            &format!("{label} chance clamp"),
        );
    }

    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut change_graph = new_graph([nonfinite, 1.0, 0.0, 1.0, 0.0]);
        let mut expected_rng = Rng::new(SEED);
        let expected = transition(0, 0, 1, 0.5, 1.0, &mut expected_rng);
        assert_eq!(
            change_graph.produce(0).to_bits(),
            (expected as f32).to_bits(),
            "non-finite change default output"
        );
        assert_state(
            &change_graph,
            RingState {
                state: expected,
                initialized: 1,
                last_change: 0.5,
                last_chance: 1.0,
                last_shift: 0.0,
                last_num_bits: 1.0,
                last_reset: 0.0,
            },
            "non-finite change default",
        );

        let mut chance_graph = new_graph([1.0, nonfinite, 0.0, 1.0, 0.0]);
        let mut expected_rng = Rng::new(SEED);
        let expected = transition(0, 0, 1, 1.0, 0.5, &mut expected_rng);
        assert_eq!(
            chance_graph.produce(0).to_bits(),
            (expected as f32).to_bits(),
            "non-finite chance default output"
        );
        assert_state(
            &chance_graph,
            RingState {
                state: expected,
                initialized: 1,
                last_change: 1.0,
                last_chance: 0.5,
                last_shift: 0.0,
                last_num_bits: 1.0,
                last_reset: 0.0,
            },
            "non-finite chance default",
        );
    }

    let mut change_recovery = new_graph([0.25, 1.0, 0.0, 1.0, 0.0]);
    let mut change_rng = Rng::new(SEED);
    let mut change_state = 0;
    for (input, remembered) in [
        (0.25, 0.25),
        (f32::NAN, 0.25),
        (f32::INFINITY, 0.25),
        (f32::NEG_INFINITY, 0.25),
        (0.75, 0.75),
    ] {
        change_recovery.set_param(0, input);
        change_state = transition(change_state, 0, 1, remembered, 1.0, &mut change_rng);
        assert_eq!(
            change_recovery.produce(0).to_bits(),
            (change_state as f32).to_bits(),
            "change fallback/recovery output"
        );
        assert_state(
            &change_recovery,
            RingState {
                state: change_state,
                initialized: 1,
                last_change: remembered,
                last_chance: 1.0,
                last_shift: 0.0,
                last_num_bits: 1.0,
                last_reset: 0.0,
            },
            "change fallback/recovery memory",
        );
    }

    let mut chance_recovery = new_graph([1.0, 0.25, 0.0, 1.0, 0.0]);
    let mut chance_rng = Rng::new(SEED);
    let mut chance_state = 0;
    for (input, remembered) in [
        (0.25, 0.25),
        (f32::NAN, 0.25),
        (f32::INFINITY, 0.25),
        (f32::NEG_INFINITY, 0.25),
        (0.75, 0.75),
    ] {
        chance_recovery.set_param(1, input);
        chance_state = transition(chance_state, 0, 1, 1.0, remembered, &mut chance_rng);
        assert_eq!(
            chance_recovery.produce(0).to_bits(),
            (chance_state as f32).to_bits(),
            "chance fallback/recovery output"
        );
        assert_state(
            &chance_recovery,
            RingState {
                state: chance_state,
                initialized: 1,
                last_change: 1.0,
                last_chance: remembered,
                last_shift: 0.0,
                last_num_bits: 1.0,
                last_reset: 0.0,
            },
            "chance fallback/recovery memory",
        );
    }
}
