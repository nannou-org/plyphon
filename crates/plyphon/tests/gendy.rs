//! `Gendy1` against scsynth's own `GendynUGens.cpp`, compiled from source and run on
//! a fresh stream 0 (`RGen::init(0)`) at 48 kHz with 64-sample blocks. Each case pins
//! the FNV-1a hash of every output sample's bit pattern over the render, plus the bit patterns at a
//! few sample indices so a mismatch shows where it starts.
//!
//! The distributions call the platform's libm (`tan`, `atan`, `log`, `sin`, and `acos` for
//! `pi_f`), so scsynth's own output differs between macOS and glibc in the last bits, and so does
//! plyphon's, which calls the same functions. Each case holds both platforms' values.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// Render `blocks` blocks of `name` at `rate` with constant `inputs`. An audio-rate unit's every
/// sample; a control-rate unit's one value per block (held across the block by `Select.ar`).
fn render(name: &str, rate: Rate, inputs: &[f32], blocks: usize) -> Vec<f32> {
    let mut units = vec![UnitSpec::new(
        name,
        rate,
        inputs.iter().map(|&v| c(v)).collect(),
        1,
    )];
    let src = if rate == Rate::Control {
        units.push(UnitSpec::new("Select", Rate::Audio, vec![c(0.0), u(0)], 1));
        1
    } else {
        0
    };
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0));
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * blocks];
    world.fill(&mut buf, 1);
    if rate == Rate::Control {
        buf.into_iter().step_by(BLOCK).collect()
    } else {
        buf
    }
}

/// FNV-1a over each sample's bit pattern, as little-endian bytes.
fn fnv(samples: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for s in samples {
        for b in s.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Sample indices pinned for an audio-rate render of 32 blocks.
const AUDIO_PICKS: [usize; 10] = [0, 1, 2, 63, 64, 100, 257, 511, 1000, 2047];
/// Block indices pinned for a control-rate render of 64 blocks.
const CONTROL_PICKS: [usize; 7] = [0, 1, 2, 5, 17, 40, 63];

/// scsynth's render on one platform: the bits at the pinned indices, and the hash of all of it.
type Expected<'a> = (&'a [u32], u64);

/// `macos` on macOS, `glibc` elsewhere (scsynth and plyphon on Linux).
fn platform<'a>(macos: Expected<'a>, glibc: Expected<'a>) -> Expected<'a> {
    if cfg!(target_os = "macos") {
        macos
    } else {
        glibc
    }
}

fn check(name: &str, rate: Rate, inputs: &[f32], (bits, hash): Expected<'_>) {
    let (blocks, picks): (usize, &[usize]) = if rate == Rate::Control {
        (64, &CONTROL_PICKS)
    } else {
        (32, &AUDIO_PICKS)
    };
    let out = render(name, rate, inputs, blocks);
    let got: Vec<u32> = picks.iter().map(|&i| out[i].to_bits()).collect();
    assert_eq!(
        got, bits,
        "{name} {inputs:?}: picked samples differ from scsynth"
    );
    assert_eq!(
        fnv(&out),
        hash,
        "{name} {inputs:?}: the render differs from scsynth"
    );
}

#[test]
fn gendy1_matches_scsynth() {
    // Cauchy walks.
    check(
        "Gendy1",
        Rate::Audio,
        &[1.0, 1.0, 1.0, 1.0, 220.0, 440.0, 0.5, 0.5, 12.0, 12.0],
        platform(
            (
                &[
                    0x00000000, 0xbd386edd, 0xbdb86edd, 0xbee65667, 0xbf06c2b9, 0x3e198c55,
                    0x3f63e2c8, 0xbe8555fc, 0xbd85a3f3, 0xbf640fab,
                ],
                0xe0f26d4d34a79061,
            ),
            (
                &[
                    0x00000000, 0xbd386edd, 0xbdb86edd, 0xbee65667, 0xbf06c2b9, 0x3e198c55,
                    0x3f63e2c8, 0xbe8555f4, 0xbd85a3f8, 0xbf640fac,
                ],
                0x9cc84efe7e4f183a,
            ),
        ),
    );
    // Arcsine amplitude and logistic duration walks.
    check(
        "Gendy1",
        Rate::Audio,
        &[4.0, 2.0, 0.7, 0.4, 200.0, 800.0, 0.6, 0.4, 9.0, 9.0],
        platform(
            (
                &[
                    0x00000000, 0xbca15015, 0xbd215015, 0xbed9582a, 0xbea00c09, 0x3ed3fcdd,
                    0xbe985c83, 0x3e9663e8, 0xbe7fe25f, 0xbe50cf4c,
                ],
                0x189c05b5a3a461a0,
            ),
            (
                &[
                    0x00000000, 0xbca15011, 0xbd215011, 0xbed95825, 0xbea00c05, 0x3ed3fcdd,
                    0xbe985c83, 0x3e9663e6, 0xbe7fe23d, 0xbe50cf47,
                ],
                0x97e8868c0043628f,
            ),
        ),
    );
}
