//! Process-only benchmarks for the SC3 processing and spectral unit families.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};
use plyphon_dsp::buffer::SpectrumCoord;

/// Sample rate shared by every benchmark case.
const SAMPLE_RATE: f64 = 48_000.0;
/// Control block size shared by every benchmark case.
const BLOCK_SIZE: usize = 64;
/// FFT size used by all three spectral benchmark cases.
const FFT_SIZE: usize = 1_024;
/// Voice counts used to expose per-voice scaling.
const VOICE_COUNTS: [usize; 3] = [1, 16, 64];

/// One required process/pull benchmark case.
#[derive(Clone, Copy)]
enum Case {
    /// `EnvDetect` envelope tracking.
    EnvDetect,
    /// `DFM1` nonlinear filter processing.
    Dfm1,
    /// `MoogLadder` filter processing.
    MoogLadder,
    /// `BlitB3Saw` oscillator processing.
    BlitB3Saw,
    /// Three-output `RosslerL` integration.
    RosslerL,
    /// `DNoiseRing` demand pulls through `Duty`.
    DNoiseRing,
    /// `PV_Freeze` ready-frame processing.
    PvFreeze,
    /// `PV_MagSmooth` ready-frame processing.
    PvMagSmooth,
    /// `PV_Morph` ready-frame processing.
    PvMorph,
}

impl Case {
    /// Returns the stable Criterion ID required for this case.
    const fn id(self) -> &'static str {
        match self {
            Self::EnvDetect => "env_detect",
            Self::Dfm1 => "dfm1",
            Self::MoogLadder => "moog_ladder",
            Self::BlitB3Saw => "blit_b3_saw",
            Self::RosslerL => "rossler_l",
            Self::DNoiseRing => "dnoise_ring",
            Self::PvFreeze => "pv_freeze",
            Self::PvMagSmooth => "pv_mag_smooth",
            Self::PvMorph => "pv_morph",
        }
    }

    /// Builds and warms this case at `voices` concurrent graph instances.
    fn harness(self, voices: usize) -> Harness {
        match self {
            Self::EnvDetect => calc_harness(
                self.id(),
                vec![
                    sine(),
                    UnitSpec::new(
                        "EnvDetect",
                        Rate::Audio,
                        vec![wire(0), constant(0.001), constant(0.01)],
                        1,
                    ),
                    out(1),
                ],
                voices,
            ),
            Self::Dfm1 => calc_harness(
                self.id(),
                vec![
                    sine(),
                    UnitSpec::new(
                        "DFM1",
                        Rate::Audio,
                        vec![
                            wire(0),
                            constant(1_000.0),
                            constant(0.2),
                            constant(1.0),
                            constant(0.0),
                            constant(0.0),
                        ],
                        1,
                    ),
                    out(1),
                ],
                voices,
            ),
            Self::MoogLadder => calc_harness(
                self.id(),
                vec![
                    sine(),
                    UnitSpec::new(
                        "MoogLadder",
                        Rate::Audio,
                        vec![wire(0), constant(1_000.0), constant(0.2)],
                        1,
                    ),
                    out(1),
                ],
                voices,
            ),
            Self::BlitB3Saw => calc_harness(
                self.id(),
                vec![
                    UnitSpec::new(
                        "BlitB3Saw",
                        Rate::Audio,
                        vec![constant(440.0), constant(0.99)],
                        1,
                    ),
                    out(0),
                ],
                voices,
            ),
            Self::RosslerL => calc_harness(
                self.id(),
                vec![
                    UnitSpec::new(
                        "RosslerL",
                        Rate::Audio,
                        vec![
                            constant(6_000.0),
                            constant(0.2),
                            constant(0.2),
                            constant(5.7),
                            constant(0.05),
                            constant(0.1),
                            constant(0.0),
                            constant(0.0),
                        ],
                        3,
                    ),
                    out(0),
                ],
                voices,
            ),
            Self::DNoiseRing => demand_harness(voices),
            Self::PvFreeze => pv_harness("PV_Freeze", self.id(), voices),
            Self::PvMagSmooth => pv_harness("PV_MagSmooth", self.id(), voices),
            Self::PvMorph => pv_harness("PV_Morph", self.id(), voices),
        }
    }
}

/// The complete stable benchmark-ID matrix.
const CASES: [Case; 9] = [
    Case::EnvDetect,
    Case::Dfm1,
    Case::MoogLadder,
    Case::BlitB3Saw,
    Case::RosslerL,
    Case::DNoiseRing,
    Case::PvFreeze,
    Case::PvMagSmooth,
    Case::PvMorph,
];

/// A prebuilt engine and output block used by a timed callback.
struct Harness {
    /// Real-time engine side containing all warmed voices.
    world: World,
    /// Preallocated mono hardware-output block.
    output: Vec<f32>,
}

impl Harness {
    /// Drains graph/buffer installation and processes one untimed warm-up block.
    fn warm(mut world: World) -> Self {
        let mut output = vec![0.0; BLOCK_SIZE];
        world.fill(&mut output, 1);
        Self { world, output }
    }

    /// Processes one block without graph construction or buffer allocation.
    fn process(&mut self) {
        self.world.fill(&mut self.output, 1);
        black_box(self.output[0]);
    }
}

/// Shorthand for a constant UGen input.
fn constant(value: f32) -> InputRef {
    InputRef::Constant(value)
}

/// Shorthand for output zero of an earlier UGen.
fn wire(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// Builds the shared audio source used by filter benchmarks.
fn sine() -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![constant(220.0), constant(0.0)],
        1,
    )
}

/// Builds a mono hardware output connected to `unit`.
fn out(unit: u32) -> UnitSpec {
    UnitSpec::new("Out", Rate::Audio, vec![constant(0.0), wire(unit)], 0)
}

/// Installs `voices` copies of one calc-rate benchmark graph and warms it.
fn calc_harness(id: &str, units: Vec<UnitSpec>, voices: usize) -> Harness {
    let (mut controller, _nrt, world) = engine(benchmark_options(false));
    let def_name = format!("bench-{id}");
    controller.add_synthdef(SynthDef {
        name: def_name.clone(),
        params: vec![],
        units,
    });
    spawn_voices(&mut controller, &def_name, voices);
    Harness::warm(world)
}

/// Installs `voices` copies of a graph that pulls `DNoiseRing` every sample.
fn demand_harness(voices: usize) -> Harness {
    let (mut controller, _nrt, world) = engine(benchmark_options(false));
    let def_name = "bench-dnoise-ring";
    controller.add_synthdef(SynthDef {
        name: def_name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "DNoiseRing",
                Rate::Demand,
                vec![
                    constant(0.5),
                    constant(0.5),
                    constant(1.0),
                    constant(8.0),
                    constant(1.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Duty",
                Rate::Audio,
                vec![
                    constant(1.0 / SAMPLE_RATE as f32),
                    constant(0.0),
                    constant(0.0),
                    wire(0),
                ],
                1,
            ),
            out(1),
        ],
    });
    spawn_voices(&mut controller, def_name, voices);
    Harness::warm(world)
}

/// Creates one valid 1024-point packed spectrum.
fn spectrum(seed: f32) -> Buffer {
    let mut samples = vec![0.0; FFT_SIZE];
    samples[0] = seed;
    samples[1] = seed + 0.5;
    for (bin, pair) in samples[2..].chunks_exact_mut(2).enumerate() {
        pair[0] = seed + (bin + 1) as f32 * 0.001;
        pair[1] = (bin as f32 * 0.01).sin();
    }
    let mut buffer = Buffer::from_interleaved(samples, 1, SAMPLE_RATE);
    buffer.set_coord(SpectrumCoord::Polar);
    buffer
}

/// Installs `voices` copies of one ready-frame spectral benchmark graph.
fn pv_harness(unit: &str, id: &str, voices: usize) -> Harness {
    let (mut controller, _nrt, world) = engine(benchmark_options(true));
    controller
        .buffer_set(0, Box::new(spectrum(1.0)))
        .expect("install benchmark spectrum A");
    let inputs = if unit == "PV_Morph" {
        controller
            .buffer_set(1, Box::new(spectrum(2.0)))
            .expect("install benchmark spectrum B");
        vec![constant(0.0), constant(1.0), constant(0.5)]
    } else {
        vec![constant(0.0), constant(0.5)]
    };

    let def_name = format!("bench-{id}");
    controller.add_synthdef(SynthDef {
        name: def_name.clone(),
        params: vec![],
        units: vec![
            UnitSpec::new(unit, Rate::Control, inputs, 1),
            UnitSpec::new("K2A", Rate::Audio, vec![wire(0)], 1),
            out(1),
        ],
    });
    spawn_voices(&mut controller, &def_name, voices);
    Harness::warm(world)
}

/// Returns engine options with enough RT-pool space for 64 fixed-size PV states.
fn benchmark_options(spectral: bool) -> Options {
    Options {
        sample_rate: SAMPLE_RATE,
        block_size: BLOCK_SIZE,
        output_channels: 1,
        pool_bytes: if spectral {
            16 * 1024 * 1024
        } else {
            Options::default().pool_bytes
        },
        ..Options::default()
    }
}

/// Queues `voices` copies of `def_name` before the untimed warm-up block.
fn spawn_voices(controller: &mut plyphon::Controller, def_name: &str, voices: usize) {
    for _ in 0..voices {
        controller
            .synth_new(def_name, ROOT_GROUP_ID, AddAction::Tail)
            .expect("benchmark voice compiles");
    }
}

/// Registers all required process/pull benchmarks.
fn sc3_processing(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sc3_processing");

    for case in CASES {
        for voices in VOICE_COUNTS {
            let mut harness = case.harness(voices);
            group.throughput(Throughput::Elements(voices as u64));
            group.bench_function(BenchmarkId::new(case.id(), voices), |bencher| {
                bencher.iter(|| harness.process());
            });
        }
    }

    group.finish();
}

criterion_group!(benches, sc3_processing);
criterion_main!(benches);
