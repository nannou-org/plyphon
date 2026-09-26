//! `BeatTrack`, the autocorrelation beat tracker, driven over a deterministic sequence of onsets.
//!
//! The chain buffer is rewritten before each frame with a synthetic spectrum: quiet noise, a loud
//! broadband burst every `interval` frames and a softer one halfway between, with the interval
//! changing partway through. The chain signal comes from a control bus, so a frame is ready every
//! `hop` blocks, and one run locks the outputs for a stretch. Every value is built from integer
//! arithmetic and exact float operations. The expected values are scsynth's: its `BeatTrack.cpp`
//! was compiled against the plugin headers and fed the same frames, block by block, with 64-sample
//! blocks.
//!
//! Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, Controller, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

const BLOCK: usize = 64;
/// The chain buffer's size: the 1024-point FFT `BeatTrack` reads.
const FFT_SIZE: usize = 1024;
/// The control bus carrying the chain signal.
const CHAIN_BUS: u32 = 0;
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

/// The onset pattern: a burst every `interval1` frames until frame `change`, then every
/// `interval2`.
struct Pattern {
    interval1: usize,
    interval2: usize,
    change: usize,
}

/// Frame `f`'s packed spectrum.
fn spectrum(g: &mut Lcg, f: usize, p: &Pattern) -> Vec<f32> {
    let (interval, since) = if f < p.change {
        (p.interval1, f)
    } else {
        (p.interval2, f - p.change)
    };
    let onset = since % interval == 0;
    let offbeat = since % interval == interval / 2;
    let mut data = vec![0.0f32; FFT_SIZE];
    for d in &mut data {
        let mut v = (g.unit() - 0.5) * 0.02;
        if onset {
            v += (g.unit() - 0.5) * 4.0;
        } else if offbeat {
            v += (g.unit() - 0.5) * 1.0;
        }
        *d = v;
    }
    data
}

/// `BeatTrack.kr(In.kr(CHAIN_BUS), In.kr(LOCK_BUS))`, each of its four outputs held into
/// `Out.ar(i)` by `DC.ar`.
fn def() -> SynthDef {
    let mut units = vec![
        UnitSpec::new(
            "In",
            Rate::Control,
            vec![InputRef::Constant(CHAIN_BUS as f32)],
            1,
        ),
        UnitSpec::new(
            "In",
            Rate::Control,
            vec![InputRef::Constant(LOCK_BUS as f32)],
            1,
        ),
        UnitSpec::new(
            "BeatTrack",
            Rate::Control,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
            ],
            4,
        ),
    ];
    for ch in 0..4u32 {
        units.push(UnitSpec::new(
            "DC",
            Rate::Audio,
            vec![InputRef::Unit {
                unit: 2,
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
                    unit: 3 + 2 * ch,
                    output: 0,
                },
            ],
            0,
        ));
    }
    SynthDef {
        name: "bt".to_string(),
        params: vec![],
        units,
    }
}

fn setup(sample_rate: f64) -> (Controller, Nrt, World) {
    let (mut controller, nrt, world) = engine(Options {
        sample_rate,
        output_channels: 4,
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
    controller.add_synthdef(def());
    controller
        .synth_new("bt", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    (controller, nrt, world)
}

/// What a run produced: the tempo output's changes as `(block, bits)`, the beat, quaver and
/// semiquaver trigger counts, and the hash of every output.
#[derive(Debug, PartialEq)]
struct Run {
    tempo_changes: Vec<(usize, u32)>,
    triggers: [usize; 3],
    hash: u64,
}

fn render(
    sample_rate: f64,
    hop: usize,
    blocks: usize,
    pattern: Pattern,
    lock: core::ops::Range<usize>,
) -> Run {
    let (mut controller, mut nrt, mut world) = setup(sample_rate);
    let mut g = Lcg(777);
    let mut fnv = Fnv(0xcbf29ce484222325);
    let mut tempo_changes = Vec::new();
    let mut triggers = [0; 3];
    let mut last_tempo = 2.0f32;
    let mut out = vec![0.0f32; BLOCK * 4];
    for b in 0..blocks {
        if b % hop == 0 {
            let data = spectrum(&mut g, b / hop, &pattern);
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
        let locked = if lock.contains(&b) { 1.0 } else { 0.0 };
        controller.set_control_bus(LOCK_BUS, locked).unwrap();
        world.fill(&mut out, 4);
        // Drop the spliced-in frames the audio thread hands back.
        nrt.process();
        let frame = &out[..4];
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
        hash: fnv.0,
    }
}

#[test]
fn beat_track_follows_scsynth_at_44100() {
    let pattern = Pattern {
        interval1: 40,
        interval2: 32,
        change: 700,
    };
    let run = render(44_100.0, 8, 12000, pattern, 9000..9600);
    assert_eq!(
        run,
        Run {
            tempo_changes: vec![(3482, 0x40088841), (8594, 0x402a451a)],
            triggers: [39, 78, 155],
            hash: 0x7374da940f05ec15,
        }
    );
}

#[test]
fn beat_track_scales_its_frame_period_at_48000() {
    let pattern = Pattern {
        interval1: 45,
        interval2: 36,
        change: 700,
    };
    let run = render(48_000.0, 8, 12000, pattern, 0..0);
    assert_eq!(
        run,
        Run {
            tempo_changes: vec![(3487, 0x40043b2d), (8598, 0x4024f2ba)],
            triggers: [36, 72, 143],
            hash: 0x168ba13a60458b04,
        }
    );
}

#[test]
fn beat_track_halves_a_double_rate() {
    // 96 kHz is taken as a doubled 48 kHz with a 2048-point FFT (a frame every 16 blocks), whose
    // lower half the tracker reads.
    let pattern = Pattern {
        interval1: 40,
        interval2: 30,
        change: 400,
    };
    let run = render(96_000.0, 16, 12000, pattern, 0..0);
    assert_eq!(
        run,
        Run {
            tempo_changes: vec![(6538, 0x40149b45), (10624, 0x404587ce)],
            triggers: [17, 35, 70],
            hash: 0xbb471802a9393035,
        }
    );
}
