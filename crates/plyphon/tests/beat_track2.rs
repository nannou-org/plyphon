//! `BeatTrack2`, the template-matching beat tracker, driven over deterministic control-bus features.
//!
//! Three feature buses carry low-level noise and a click train (one feature adds an off-beat
//! click) whose period changes from 360 to 280 control blocks partway through; one run locks the
//! outputs for a stretch. Every feature value is built from integer arithmetic and exact float
//! operations, and the tracker itself uses only arithmetic, so the expected values hold on every
//! platform. They are scsynth's: its `BeatTrack2.cpp` was compiled against the plugin headers and
//! fed the same features, block by block, at 48 kHz with 64-sample blocks.

use plyphon::{
    AddAction, Buffer, Controller, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

const BLOCK: usize = 64;
const SR: f64 = 48_000.0;
/// The first of the feature buses.
const FEATURE_BUS: u32 = 10;
/// The number of feature buses.
const NF: usize = 3;
/// The control bus carrying the `lock` input.
const LOCK_BUS: u32 = 1;

/// A 32-bit linear congruential generator yielding exact floats in `[0, 1)`.
struct Lcg(u32);

impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 * (1.0 / 16777216.0)
    }
}

/// FNV-1a over the little-endian bytes of each output's bit pattern.
struct Fnv(u64);

impl Fnv {
    fn add(&mut self, v: f32) {
        for b in v.to_bits().to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

/// The features at block `b`.
fn features(g: &mut Lcg, b: usize) -> [f32; NF] {
    let period = if b < 4000 { 360 } else { 280 };
    let mut out = [0.0; NF];
    for (k, v) in out.iter_mut().enumerate() {
        *v = g.unit() * 0.1;
        let pos = (b + 17 * k) % period;
        if pos < 3 {
            *v += 1.0 - 0.25 * pos as f32;
        }
        if k == 1 && (b + period / 2) % period < 2 {
            *v += 0.5;
        }
    }
    out
}

/// `BeatTrack2.kr(FEATURE_BUS, NF, 2, 0.02, In.kr(LOCK_BUS), weightingscheme)`, each of its six
/// outputs held into `Out.ar(i)` by `DC.ar`.
fn def(numfeatures: f32, weightingscheme: f32) -> SynthDef {
    let mut units = vec![
        UnitSpec::new(
            "In",
            Rate::Control,
            vec![InputRef::Constant(LOCK_BUS as f32)],
            1,
        ),
        UnitSpec::new(
            "BeatTrack2",
            Rate::Control,
            vec![
                InputRef::Constant(FEATURE_BUS as f32),
                InputRef::Constant(numfeatures),
                InputRef::Constant(2.0),
                InputRef::Constant(0.02),
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(weightingscheme),
            ],
            6,
        ),
    ];
    for ch in 0..6u32 {
        units.push(UnitSpec::new(
            "DC",
            Rate::Audio,
            vec![InputRef::Unit {
                unit: 1,
                output: ch,
            }],
            1,
        ));
        units.push(UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(ch as f32),
                InputRef::Unit {
                    unit: 2 + 2 * ch,
                    output: 0,
                },
            ],
            0,
        ));
    }
    SynthDef {
        name: "bt2".to_string(),
        params: vec![],
        units,
    }
}

fn setup(numfeatures: f32, weights: Option<Vec<f32>>) -> (Controller, World) {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 6,
        ..Options::default()
    });
    if let Some(weights) = weights {
        controller
            .buffer_set(0, Box::new(Buffer::from_interleaved(weights, 1, SR)))
            .unwrap();
    }
    controller.add_synthdef(def(numfeatures, -2.1));
    controller
        .synth_new("bt2", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    (controller, world)
}

/// What a run produced: the tempo output's changes as `(block, bits)`, the beat, quaver and
/// semiquaver trigger counts, the final phase output's bits, and the hash of every output.
#[derive(Debug, PartialEq)]
struct Run {
    tempo_changes: Vec<(usize, u32)>,
    triggers: [usize; 3],
    last_phase: u32,
    hash: u64,
}

fn render(weights: Option<Vec<f32>>, blocks: usize, lock: core::ops::Range<usize>) -> Run {
    let (mut controller, mut world) = setup(NF as f32, weights);
    let mut g = Lcg(4242);
    let mut fnv = Fnv(0xcbf29ce484222325);
    let mut tempo_changes = Vec::new();
    let mut triggers = [0; 3];
    let mut last_tempo = 2.0f32;
    let mut out = vec![0.0f32; BLOCK * 6];
    for b in 0..blocks {
        controller
            .set_control_bus_n(FEATURE_BUS, &features(&mut g, b))
            .unwrap();
        let locked = if lock.contains(&b) { 1.0 } else { 0.0 };
        controller.set_control_bus(LOCK_BUS, locked).unwrap();
        world.fill(&mut out, 6);
        let frame = &out[..6];
        for &v in frame {
            fnv.add(v);
        }
        for (count, &v) in triggers.iter_mut().zip(frame) {
            *count += (v == 1.0) as usize;
        }
        if frame[3] != last_tempo {
            tempo_changes.push((b, frame[3].to_bits()));
            last_tempo = frame[3];
        }
    }
    Run {
        tempo_changes,
        triggers,
        last_phase: out[4].to_bits(),
        hash: fnv.0,
    }
}

#[test]
fn beat_track2_follows_scsynth() {
    // The default weighting scheme (-2.1) names buffer 0, which does not exist: flat weights.
    let run = render(None, 8000, 6000..6500);
    assert_eq!(
        run,
        Run {
            tempo_changes: vec![
                (990, 0x40055556),
                (3990, 0x40266666),
                (5490, 0x402ccccd),
                (7365, 0x402bbbbc),
            ],
            triggers: [24, 49, 98],
            last_phase: 0x3f1160e8,
            hash: 0xd2013fffe21f5ba7,
        }
    );
}

#[test]
fn beat_track2_weights_tempi_by_buffer_zero() {
    // scsynth compares the default weighting scheme (-2.1) with the unsigned buffer count, so it
    // reads its tempo weights from buffer 0 once that exists.
    let mut wg = Lcg(999);
    let weights = (0..128).map(|_| 0.5 + wg.unit()).collect();
    let run = render(Some(weights), 8000, 0..0);
    assert_eq!(
        run,
        Run {
            tempo_changes: vec![(990, 0x40055556), (7365, 0x40011111)],
            triggers: [21, 42, 84],
            last_phase: 0x3e4409d2,
            hash: 0x7511914c9a650b28,
        }
    );
}

#[test]
fn beat_track2_without_features_is_cleared() {
    // No feature buses: scsynth's allocation fails (a negative count) or reads memory it never
    // allocated (zero), and plyphon clears the unit's outputs.
    let (mut controller, mut world) = setup(0.0, None);
    let mut out = vec![0.0f32; BLOCK * 6];
    for _ in 0..4 {
        controller.set_control_bus(LOCK_BUS, 0.0).unwrap();
        world.fill(&mut out, 6);
        assert_eq!(&out[..6], &[0.0; 6]);
    }
}
