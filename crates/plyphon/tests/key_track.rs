//! `KeyTrack`, the chroma-histogram key tracker, driven over a deterministic chord sequence.
//!
//! The chain buffer is rewritten before each frame with a synthetic spectrum (low-level noise plus
//! three harmonics of each note of a triad whose root and quality change every 40 frames), and the
//! chain signal comes from a control bus, so a frame is ready on every fourth block. Every value is
//! built from integer arithmetic and exact float operations, so the sequence is the same in any
//! implementation. One run puts `PV_MagAbove(chain, 0)` ahead of the tracker, so each frame arrives
//! in polar form and `KeyTrack` converts it back with scsynth's lookup tables; that run also pins
//! the whole chain buffer as the tracker leaves it. The expected values are scsynth's: its
//! `KeyTrack.cpp` was compiled against the plugin headers (with `SC_Complex.h`'s conversions) and
//! fed the same frames.
//!
//! Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, Controller, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, Reply, SynthDef,
    UnitSpec, World, engine,
};

const BLOCK: usize = 64;
/// The chain buffer's size: the 4096-point FFT `KeyTrack`'s tables assume.
const FFT_SIZE: usize = 4096;
/// The control bus carrying the chain signal.
const CHAIN_BUS: u32 = 0;
/// Blocks per frame.
const HOP_BLOCKS: usize = 4;

/// FFT bin of each note from C3 up to B4 (4096 points at 48 kHz, rounded).
const NOTE_BIN: [usize; 24] = [
    11, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 30, 32, 33, 35, 38, 40, 42,
];
/// The chord roots, in semitones above C3, cycled every 40 frames.
const ROOTS: [usize; 8] = [0, 7, 2, 9, 4, 11, 5, 10];

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

/// Frame `f`'s packed spectrum: the first 2048 floats (the bins `KeyTrack` reads) are filled.
fn spectrum(g: &mut Lcg, f: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; FFT_SIZE];
    for d in &mut data[..2048] {
        *d = (g.unit() - 0.5) * 0.02;
    }
    let seg = f / 40;
    let root = ROOTS[seg % 8];
    let third = if seg % 3 == 1 { 3 } else { 4 };
    for note in [root, root + third, root + 7] {
        for h in 1..=3 {
            let b = NOTE_BIN[note] * h;
            let amp = 1.0 + g.unit();
            data[2 * b] += amp / h as f32;
            data[2 * b + 1] += amp * 0.5 / h as f32;
        }
    }
    data
}

/// `KeyTrack.kr(chain, keydecay, chromaleak)`, held into `Out.ar(0)` by `DC.ar`, where `chain` is
/// `In.kr(CHAIN_BUS)`, or `PV_MagAbove(In.kr(CHAIN_BUS), 0)` when `polar`: a zero threshold keeps
/// every bin but leaves the frame in polar form, so `KeyTrack` converts it back.
fn key_track_def(keydecay: f32, chromaleak: f32, polar: bool) -> SynthDef {
    let mut units = vec![UnitSpec::new(
        "In",
        Rate::Control,
        vec![InputRef::Constant(CHAIN_BUS as f32)],
        1,
    )];
    if polar {
        units.push(UnitSpec::new(
            "PV_MagAbove",
            Rate::Control,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.0),
            ],
            1,
        ));
    }
    let chain = units.len() as u32 - 1;
    units.push(UnitSpec::new(
        "KeyTrack",
        Rate::Control,
        vec![
            InputRef::Unit {
                unit: chain,
                output: 0,
            },
            InputRef::Constant(keydecay),
            InputRef::Constant(chromaleak),
        ],
        1,
    ));
    units.push(UnitSpec::new(
        "DC",
        Rate::Audio,
        vec![InputRef::Unit {
            unit: chain + 1,
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
                unit: chain + 2,
                output: 0,
            },
        ],
        0,
    ));
    SynthDef {
        name: "kt".to_string(),
        params: vec![],
        units,
    }
}

fn setup(
    sample_rate: f64,
    keydecay: f32,
    chromaleak: f32,
    polar: bool,
) -> (Controller, Nrt, World) {
    let (mut controller, nrt, world) = engine(Options {
        sample_rate,
        output_channels: 1,
        // Room in the reply ring for a whole-buffer `/b_getn`.
        max_nodes: 2048,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(
                vec![0.0; FFT_SIZE],
                1,
                sample_rate,
            )),
        )
        .unwrap();
    controller.set_control_bus(CHAIN_BUS, -1.0).unwrap();
    controller.add_synthdef(key_track_def(keydecay, chromaleak, polar));
    controller
        .synth_new("kt", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    (controller, nrt, world)
}

/// What a run produced: the output's changes as `(block, key)`, the hash of every block's output,
/// and (for a polar run) the hash of the whole chain buffer as `KeyTrack` left it on every 200th
/// block.
#[derive(Debug, PartialEq)]
struct Run {
    changes: Vec<(usize, f32)>,
    hash: u64,
    snapshots: Vec<(usize, u64)>,
}

/// Render `blocks` blocks, rewriting the chain buffer before every fourth.
fn render(sample_rate: f64, keydecay: f32, chromaleak: f32, blocks: usize, polar: bool) -> Run {
    let (mut controller, mut nrt, mut world) = setup(sample_rate, keydecay, chromaleak, polar);
    let mut g = Lcg(12345);
    let mut fnv = Fnv(0xcbf29ce484222325);
    let mut changes = Vec::new();
    let mut snapshots = Vec::new();
    let mut last = -1.0f32;
    let mut out = vec![0.0f32; BLOCK];
    for b in 0..blocks {
        if b % HOP_BLOCKS == 0 {
            let data = spectrum(&mut g, b / HOP_BLOCKS);
            controller
                .buffer_write_region(
                    0,
                    0,
                    Box::new(Buffer::from_interleaved(data, 1, sample_rate)),
                )
                .unwrap();
            controller.set_control_bus(CHAIN_BUS, 0.0).unwrap();
        } else {
            controller.set_control_bus(CHAIN_BUS, -1.0).unwrap();
        }
        // Read the whole chain buffer back (`/b_getn`, answered before this block runs) one block
        // after every 200th, so it holds the frame as `KeyTrack` left it.
        let snapshot = polar && b % 200 == 1;
        if snapshot {
            for start in (0..FFT_SIZE).step_by(256) {
                controller.query_buffer_range(0, start, 256).unwrap();
            }
        }
        world.fill(&mut out, 1);
        nrt.process();
        fnv.add(out[0]);
        if out[0] != last {
            changes.push((b, out[0]));
            last = out[0];
        }
        if snapshot {
            let mut s = Fnv(0xcbf29ce484222325);
            let mut count = 0;
            while let Some(reply) = nrt.poll_reply() {
                if let Reply::RangeValue { value } = reply {
                    s.add(value);
                    count += 1;
                }
            }
            assert_eq!(count, FFT_SIZE);
            snapshots.push((b - 1, s.0));
        }
    }
    Run {
        changes,
        hash: fnv.0,
        snapshots,
    }
}

#[test]
fn key_track_follows_scsynth_at_48k() {
    let run = render(48_000.0, 1.0, 0.7, 1600, false);
    let expected: &[(usize, f32)] = &[
        (0, 16.0),
        (4, 0.0),
        (180, 19.0),
        (204, 3.0),
        (216, 19.0),
        (304, 3.0),
        (320, 19.0),
        (484, 2.0),
        (504, 9.0),
        (656, 16.0),
        (836, 11.0),
        (976, 16.0),
        (980, 18.0),
        (984, 21.0),
        (1000, 5.0),
        (1132, 22.0),
        (1292, 17.0),
        (1320, 0.0),
        (1460, 7.0),
    ];
    assert_eq!(run.changes, expected);
    assert_eq!(run.hash, 0xe4e9f0e6173be485);
}

#[test]
fn key_track_uses_the_44100_tables_at_44100() {
    let run = render(44_100.0, 2.0, 0.5, 1600, false);
    let expected: &[(usize, f32)] = &[
        (0, 15.0),
        (188, 18.0),
        (360, 1.0),
        (388, 17.0),
        (516, 8.0),
        (640, 12.0),
        (660, 3.0),
        (664, 19.0),
        (688, 15.0),
        (820, 10.0),
        (972, 14.0),
        (992, 3.0),
        (1008, 20.0),
        (1132, 17.0),
        (1148, 13.0),
        (1152, 21.0),
        (1304, 19.0),
        (1316, 15.0),
        (1476, 22.0),
    ];
    assert_eq!(run.changes, expected);
    assert_eq!(run.hash, 0x03ca27029cb5f825);
}

#[test]
fn key_track_halves_a_double_rate_to_pick_its_tables() {
    // 88.2 kHz is taken as a doubled 44.1 kHz: the 44100 tables and frame period apply.
    let run = render(88_200.0, 3.0, 0.9, 1200, false);
    let expected: &[(usize, f32)] = &[
        (0, 15.0),
        (232, 18.0),
        (416, 1.0),
        (484, 17.0),
        (568, 8.0),
        (676, 12.0),
        (692, 3.0),
        (708, 19.0),
        (720, 15.0),
        (852, 10.0),
        (1040, 3.0),
        (1068, 20.0),
        (1164, 17.0),
        (1188, 13.0),
        (1192, 21.0),
    ];
    assert_eq!(run.changes, expected);
    assert_eq!(run.hash, 0xa28c23540389bc85);
}

#[test]
fn key_track_converts_a_polar_frame_with_scsynths_tables() {
    // `PV_MagAbove(chain, 0)` keeps every bin but leaves each frame polar (scsynth's
    // `ToPolarApx`); `KeyTrack` converts it back in place with `ToComplexApx`, so both its keys
    // and the buffer it leaves are scsynth's.
    let run = render(48_000.0, 1.0, 0.7, 1600, true);
    let expected: &[(usize, f32)] = &[
        (0, 16.0),
        (4, 0.0),
        (180, 19.0),
        (204, 3.0),
        (216, 19.0),
        (304, 3.0),
        (320, 19.0),
        (484, 2.0),
        (504, 9.0),
        (656, 16.0),
        (836, 11.0),
        (976, 16.0),
        (980, 18.0),
        (984, 21.0),
        (1000, 5.0),
        (1132, 22.0),
        (1292, 17.0),
        (1320, 0.0),
        (1460, 7.0),
    ];
    assert_eq!(run.changes, expected);
    assert_eq!(run.hash, 0xe4e9f0e6173be485);
    assert_eq!(
        run.snapshots,
        [
            (0, 0xde8279b4558de6b7),
            (200, 0x9cafd3cc9c5f4642),
            (400, 0x466fb7b04f63557b),
            (600, 0x453b9e0ffd0ed283),
            (800, 0xa96e0792626d8616),
            (1000, 0xd87a2b2f234cc770),
            (1200, 0x9bf6a372e2417027),
            (1400, 0x0f09a9798e031f66),
        ]
    );
}
