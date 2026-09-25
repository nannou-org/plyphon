//! Synth processing benchmarks: the cost of running, creating, and freeing synths through a real
//! engine, for synths with and without per-unit memory (delay lines, reverb lines, local buffers).
//!
//! - `steady` - one control block of an already-running population, the audio thread's hot path.
//!   `plain` has no per-unit memory; `delays` and `local_buf` exercise units that own sized memory.
//! - `spawn` - create a population and run its first block, where every unit is initialised.
//! - `free` - free a running population and run the block that tears it down.
//! - `churn` - replace voices one at a time, a block each, as a voice allocator would.
//!
//! Every scenario uses constant sizes and only built-in units, so the same code runs against any
//! engine revision and before/after numbers compare like for like.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

const BLOCK: usize = 64;
const CHANNELS: usize = 2;
/// Voices in each population.
const VOICES: usize = 16;
/// Voices replaced per timed `churn` run.
const CHURN_CYCLES: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: 48_000.0,
        block_size: BLOCK,
        output_channels: CHANNELS,
        // Room for every population below at once, so no scenario measures pool exhaustion.
        pool_bytes: 64 * 1024 * 1024,
        ..Options::default()
    }
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

fn unit(name: &str, rate: Rate, inputs: Vec<InputRef>, outputs: usize) -> UnitSpec {
    UnitSpec::new(name, rate, inputs, outputs)
}

fn def(name: &str, units: Vec<UnitSpec>) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units,
    }
}

/// `Out.ar(0, SinOsc.ar(440))`: no per-unit memory.
fn plain() -> SynthDef {
    def(
        "plain",
        vec![
            unit("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            unit("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    )
}

/// A sine through `DelayC` (0.5 s line), `CombC` (0.5 s), `PitchShift` (0.2 s window) and `GVerb`
/// (max room 30): four units that each own a sized line.
fn delays() -> SynthDef {
    def(
        "delays",
        vec![
            unit("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            unit("DelayC", Rate::Audio, vec![u(0), c(0.5), c(0.2)], 1),
            unit("CombC", Rate::Audio, vec![u(1), c(0.5), c(0.3), c(2.0)], 1),
            unit(
                "PitchShift",
                Rate::Audio,
                vec![u(2), c(0.2), c(1.5), c(0.0), c(0.0)],
                1,
            ),
            unit(
                "GVerb",
                Rate::Audio,
                vec![
                    u(3),
                    c(10.0),
                    c(3.0),
                    c(0.5),
                    c(0.5),
                    c(15.0),
                    c(1.0),
                    c(0.7),
                    c(0.5),
                    c(30.0),
                ],
                2,
            ),
            unit("Out", Rate::Audio, vec![c(0.0), u(4)], 0),
        ],
    )
}

/// A one-second `LocalBuf` written with `BufWr` and read back with `BufRd` at a `Phasor`.
fn local_buf() -> SynthDef {
    const FRAMES: f32 = 48_000.0;
    def(
        "local_buf",
        vec![
            unit("LocalBuf", Rate::Scalar, vec![c(1.0), c(FRAMES)], 1),
            unit("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            unit(
                "Phasor",
                Rate::Audio,
                vec![c(0.0), c(1.0), c(0.0), c(FRAMES), c(0.0)],
                1,
            ),
            unit("BufWr", Rate::Audio, vec![u(0), u(2), c(1.0), u(1)], 1),
            unit("BufRd", Rate::Audio, vec![u(0), u(2), c(1.0), c(1.0)], 1),
            unit("Out", Rate::Audio, vec![c(0.0), u(4)], 0),
        ],
    )
}

/// An engine with every scenario def resident and compiled, so no timed run pays for compilation.
struct Engine {
    controller: Controller,
    nrt: Nrt,
    world: World,
    out: Vec<f32>,
}

impl Engine {
    fn new() -> Self {
        let (mut controller, nrt, world) = engine(opts());
        for d in [plain(), delays(), local_buf()] {
            let name = d.name.clone();
            controller.add_synthdef(d);
            controller
                .ensure_compiled(&name)
                .expect("scenario def compiles");
        }
        let mut engine = Engine {
            controller,
            nrt,
            world,
            out: vec![0.0; BLOCK * CHANNELS],
        };
        engine.block();
        engine
    }

    /// Run one control block, then drain the notifications it produced as a host would, failing
    /// loudly if any synth could not be created: a benchmark of failed spawns measures nothing.
    fn block(&mut self) {
        self.world.fill(&mut self.out, CHANNELS);
        black_box(&self.out);
        self.nrt.process();
        while let Some(event) = self.nrt.poll() {
            assert!(
                !matches!(event, Event::SynthFailed { .. }),
                "a scenario synth failed to start: {event:?}"
            );
        }
    }

    fn spawn(&mut self, def: &str) -> i32 {
        self.controller
            .synth_new(def, ROOT_GROUP_ID, AddAction::Tail)
            .expect("synth_new")
    }

    fn spawn_voices(&mut self, def: &str) -> Vec<i32> {
        (0..VOICES).map(|_| self.spawn(def)).collect()
    }

    /// An engine already running a warmed-up population of `def`.
    fn running(def: &str) -> (Self, Vec<i32>) {
        let mut engine = Engine::new();
        let voices = engine.spawn_voices(def);
        for _ in 0..4 {
            engine.block();
        }
        (engine, voices)
    }
}

const SCENARIOS: [&str; 3] = ["plain", "delays", "local_buf"];

fn steady(c: &mut Criterion) {
    let mut group = c.benchmark_group("steady");
    group.throughput(Throughput::Elements(VOICES as u64));
    for def in SCENARIOS {
        let (mut engine, _voices) = Engine::running(def);
        group.bench_function(def, |b| b.iter(|| engine.block()));
    }
    group.finish();
}

fn spawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn");
    group.throughput(Throughput::Elements(VOICES as u64));
    for def in SCENARIOS {
        group.bench_function(def, |b| {
            b.iter_batched(
                Engine::new,
                |mut engine| {
                    let voices = engine.spawn_voices(def);
                    engine.block();
                    (engine, voices)
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn free(c: &mut Criterion) {
    let mut group = c.benchmark_group("free");
    group.throughput(Throughput::Elements(VOICES as u64));
    for def in SCENARIOS {
        group.bench_function(def, |b| {
            b.iter_batched(
                || Engine::running(def),
                |(mut engine, voices)| {
                    for &node in &voices {
                        engine.controller.free(node).expect("free");
                    }
                    engine.block();
                    engine
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

fn churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("churn");
    group.throughput(Throughput::Elements(CHURN_CYCLES as u64));
    for def in SCENARIOS {
        group.bench_function(def, |b| {
            b.iter_batched(
                || Engine::running(def),
                |(mut engine, mut voices)| {
                    for i in 0..CHURN_CYCLES {
                        let slot = i % VOICES;
                        engine.controller.free(voices[slot]).expect("free");
                        voices[slot] = engine.spawn(def);
                        engine.block();
                    }
                    (engine, voices)
                },
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, steady, spawn, free, churn);
criterion_main!(benches);
