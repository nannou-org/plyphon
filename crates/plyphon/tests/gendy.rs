//! `Gendy1`, `Gendy2` and `Gendy3` against scsynth's own `GendynUGens.cpp`, compiled from source
//! and run on a fresh stream 0 (`RGen::init(0)`) at 48 kHz with 64-sample blocks. Each case pins
//! the FNV-1a hash of every output sample's bit pattern over the render, plus the bit patterns at a
//! few sample indices so a mismatch shows where it starts.
//!
//! The distributions call the platform's libm (`tan`, `atan`, `log`, `sin`, and `acos` for
//! `pi_f`), so scsynth's own output differs between libms in the last bits, and so does plyphon's,
//! which calls the same functions. Each case holds the macOS values and the values for the glibc CI
//! builds against (older glibc versions round some of these functions differently).

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

/// `macos` on macOS, `glibc` elsewhere (scsynth and plyphon on Linux, with CI's glibc).
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

#[test]
fn gendy2_matches_scsynth() {
    // Cauchy walks with the language's default Lehmer constants.
    check(
        "Gendy2",
        Rate::Audio,
        &[
            1.0, 1.0, 1.0, 1.0, 440.0, 660.0, 0.5, 0.5, 12.0, 12.0, 1.17, 0.31,
        ],
        platform(
            (
                &[
                    0x00000000, 0xbca9a615, 0xbd29a615, 0xbefde4b8, 0xbedb0b94, 0x3f821b80,
                    0xbeefc7f3, 0xbf3cf190, 0x3c587d26, 0x3f705399,
                ],
                0xb84c57ae68632f06,
            ),
            (
                &[
                    0x00000000, 0xbca9a615, 0xbd29a615, 0xbefde4b8, 0xbedb0b94, 0x3f821b80,
                    0xbeefc7f3, 0xbf3cf190, 0x3c58842b, 0x3f6bd97a,
                ],
                0xfb2622ad0da14d7d,
            ),
        ),
    );
    // Hyperbolic-cosine amplitude and exponential duration walks, over 7 of 10 breakpoints.
    check(
        "Gendy2",
        Rate::Audio,
        &[
            3.0, 5.0, 0.3, 0.7, 100.0, 1000.0, 0.8, 0.3, 10.0, 7.0, 1.5, 0.2,
        ],
        platform(
            (
                &[
                    0x00000000, 0xbd31263a, 0xbdb1263a, 0xbf41b678, 0xbf35f40c, 0x3e9dfbe8,
                    0x3f22c669, 0x3f68782b, 0x3e9836f4, 0x3e9e68f5,
                ],
                0xb4355200d7bece2b,
            ),
            (
                &[
                    0x00000000, 0xbd31263a, 0xbdb1263a, 0xbf41b678, 0xbf35f40c, 0x3e9dfbe8,
                    0x3f22c669, 0x3f68782b, 0x3e9836f4, 0x3f6df718,
                ],
                0x71fcab50de77655d,
            ),
        ),
    );
    // Logistic and arcsine walks; a `knum` of 0 means every breakpoint.
    check(
        "Gendy2",
        Rate::Audio,
        &[
            2.0, 4.0, 0.5, 0.9, 50.0, 3000.0, 1.0, 1.0, 6.0, 0.0, 1.17, 0.31,
        ],
        platform(
            (
                &[
                    0x00000000, 0xbdc1836c, 0xbe41836c, 0x39a60906, 0xbeb4bc82, 0x3ed61ca9,
                    0x3f5da80b, 0x3f37d2ce, 0x3eff07a2, 0x3e818aa0,
                ],
                0x0a740756e484b85a,
            ),
            (
                &[
                    0x00000000, 0xbdc1836b, 0xbe41836b, 0x39a617d3, 0xbeb4bc7f, 0x3ed61c9a,
                    0x3f5da821, 0x3f37d15e, 0x3eff0976, 0x3e818c2c,
                ],
                0xdace6207ce07bc09,
            ),
        ),
    );
    // Linear and "sinus" walks: no libm, so both platforms agree.
    check(
        "Gendy2",
        Rate::Audio,
        &[
            0.0, 6.0, 0.6, 0.2, 300.0, 900.0, 0.3, 0.6, 8.0, 8.0, 0.9, 0.5,
        ],
        (
            &[
                0x00000000, 0xbc6db0de, 0xbcedb0de, 0x3f5c3e75, 0x3f4d106b, 0x3e340827, 0x3f273c2e,
                0x3e5aa97c, 0x3f7cd8e7, 0x3f43e3e2,
            ],
            0x721a9c681f391611,
        ),
    );
}

#[test]
fn gendy2_kr_matches_scsynth() {
    check(
        "Gendy2",
        Rate::Control,
        &[
            1.0, 1.0, 1.0, 1.0, 20.0, 40.0, 0.5, 0.5, 12.0, 12.0, 1.17, 0.31,
        ],
        platform(
            (
                &[
                    0x00000000, 0xbd9a4da0, 0xbe47450f, 0xbf771500, 0xbebc50b7, 0xbee0ba6d,
                    0xbe39286b,
                ],
                0xa8903ff14b6442e6,
            ),
            (
                &[
                    0x00000000, 0xbd9a4da0, 0xbe47450f, 0xbf771500, 0xbebc50b7, 0xbee0ba6d,
                    0xbe39286e,
                ],
                0xf9b93acd94436c29,
            ),
        ),
    );
}

#[test]
fn gendy3_matches_scsynth() {
    check(
        "Gendy3",
        Rate::Audio,
        &[1.0, 1.0, 1.0, 1.0, 440.0, 0.5, 0.5, 12.0, 12.0],
        platform(
            (
                &[
                    0x00000000, 0x3d532b63, 0x3dac041a, 0x3f1326ec, 0x3f06cba8, 0xbf0b6161,
                    0x3e8b8144, 0x3f1e71a3, 0xbea38b02, 0x3f395450,
                ],
                0x9f306552eb46e245,
            ),
            (
                &[
                    0x00000000, 0x3d532b63, 0x3dac041a, 0x3f1326ec, 0x3f06cba8, 0xbf0b6160,
                    0x3e8b8144, 0x3f1e71a3, 0xbea38b04, 0x3f395450,
                ],
                0xbfd7c437d0cdaed7,
            ),
        ),
    );
    check(
        "Gendy3",
        Rate::Audio,
        &[3.0, 2.0, 0.4, 0.8, 1234.0, 0.8, 1.0, 10.0, 6.0],
        platform(
            (
                &[
                    0x00000000, 0xbe08f3ef, 0xbe88f3ef, 0x3f2b10d2, 0x3f3a9c57, 0xbd0839ac,
                    0xbf5b658a, 0xbeed60bc, 0x3f52c7a6, 0x3ef88f9d,
                ],
                0x46c4839fc81164d8,
            ),
            (
                &[
                    0x00000000, 0xbe08f3ef, 0xbe88f3ef, 0x3f2b10d2, 0x3f3a9c57, 0xbd0839ac,
                    0xbf5b658a, 0xbeed60bc, 0x3f52c7a6, 0x3ef88f9d,
                ],
                0x46c4839fc81164d8,
            ),
        ),
    );
    // 3000 breakpoints at 90 Hz: some normalised durations fall under a sample and are dropped,
    // and the regions are shorter than a sample's phase step, so the region index falls behind the
    // phase and scsynth's truncated `interpmult` interpolation overshoots far past the breakpoints.
    check(
        "Gendy3",
        Rate::Audio,
        &[4.0, 5.0, 0.9, 0.2, 90.0, 0.3, 0.9, 3000.0, 3000.0],
        platform(
            (
                &[
                    0x00000000, 0x3fcd6e04, 0x408e7abf, 0x42280abc, 0x4359344d, 0xc3fbf074,
                    0x4449dc30, 0xc5800431, 0xc434a2dc, 0xc40ef7a1,
                ],
                0x656f1c3fc9c29ed7,
            ),
            (
                &[
                    0x00000000, 0x3fcd6e00, 0x408e7abf, 0x42280abc, 0x4359344d, 0xc3fbf074,
                    0x4449dc30, 0xc5800431, 0xc434a2de, 0xc40ef7a0,
                ],
                0x1ee06e0595e04cc9,
            ),
        ),
    );
}

#[test]
fn gendy3_kr_matches_scsynth() {
    check(
        "Gendy3",
        Rate::Control,
        &[1.0, 1.0, 1.0, 1.0, 10.0, 0.5, 0.5, 12.0, 12.0],
        platform(
            (
                &[
                    0x00000000, 0x3d8a04bb, 0x3de4adb8, 0x3e7a5459, 0x3f4ccb45, 0x3f4eb01c,
                    0x3e873f77,
                ],
                0x5a7a7f31d74f228a,
            ),
            (
                &[
                    0x00000000, 0x3d8a04bb, 0x3de4adb8, 0x3e7a5459, 0x3f4ccb45, 0x3f4eb01c,
                    0x3e873f78,
                ],
                0xde22814c58b179ad,
            ),
        ),
    );
}
