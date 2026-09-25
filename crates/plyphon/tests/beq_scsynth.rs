//! The BEQSuite against scsynth's own code: each expected block is the bit pattern
//! scsynth's `FilterUGens.cpp`, compiled from source, produces at 48 kHz with 64-sample blocks, at
//! samples 0, 1, 2, 33, 61, 62 and 63 of each of the first five blocks. The input is `WhiteNoise.ar`
//! on a fresh stream 0.
//!
//! - `_kk`: control-rate parameters, each switching to a second value on its own block (`freq`
//!   on block 1, the width on block 2, `db` on block 3), so those blocks ramp the coefficients.
//! - `_ii`: `BLowPass`/`BHiPass` with constant parameters.
//! - `_aa`: every parameter audio-rate and changing every sample (`WhiteNoise.ar * scale + offset`,
//!   each from its own `WhiteNoise`), which exercises scsynth's one-parameter-sample-per-group reads
//!   and its per-unit bookkeeping of the compared values.
//! - A `BPeakEQ` with only `freq` audio-rate, which takes the `_kkk` calc.

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const BLOCK: usize = 64;
const BLOCKS: usize = 5;
const PICKS: [usize; 7] = [0, 1, 2, 33, 61, 62, 63];

/// The [`PICKS`] of each of the [`BLOCKS`] blocks, as bit patterns.
type Expected = [[u32; 7]; BLOCKS];

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `BinaryOpUGen` `op` (2 multiply, 0 add) of unit `a` and constant `b`.
fn binop(op: i16, a: u32, b: f32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![u(a), c(b)],
        num_outputs: 1,
        special_index: op,
    }
}

/// How a test feeds the parameters.
enum Params<'a> {
    /// Control parameters starting at the first values, parameter `k` switching to the second on
    /// block `k + 1`.
    Control(&'a [f32], &'a [f32]),
    /// Constants.
    Scalar(&'a [f32]),
    /// Audio rate: parameter `k` is its own `WhiteNoise.ar * SCALE[k] + OFFSET[k]`.
    Audio(usize),
    /// `freq` audio-rate as in `Audio`, the other parameters control parameters starting at the
    /// first values, parameter `k` (from 1) switching to the second on block `k + 1`.
    Mixed(&'a [f32], &'a [f32]),
}

const SCALE: [f32; 3] = [300.0, 0.25, 6.0];
const OFFSET: [f32; 3] = [1000.0, 1.0, 0.0];

/// Run `name.ar(WhiteNoise.ar, params...)` for [`BLOCKS`] blocks and pick [`PICKS`] of each.
fn run(name: &str, params: Params<'_>) -> Vec<[u32; 7]> {
    let mut units = vec![UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1)];
    let mut def_params = vec![];
    let mut inputs = vec![u(0)];
    let mut changes: Vec<(usize, usize, f32)> = vec![];
    match params {
        Params::Control(first, second) => {
            for (k, (&a, &b)) in first.iter().zip(second).enumerate() {
                def_params.push(Param::control(format!("p{k}"), a));
                inputs.push(InputRef::Param(k as u32));
                changes.push((k + 1, k, b));
            }
        }
        Params::Scalar(values) => inputs.extend(values.iter().map(|&v| c(v))),
        Params::Audio(n) => {
            for _ in 0..n {
                units.push(UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1));
            }
            for k in 0..n {
                let mul = units.len() as u32;
                units.push(binop(2, 1 + k as u32, SCALE[k]));
                units.push(binop(0, mul, OFFSET[k]));
                inputs.push(u(mul + 1));
            }
        }
        Params::Mixed(first, second) => {
            units.push(UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1));
            units.push(binop(2, 1, SCALE[0]));
            units.push(binop(0, 2, OFFSET[0]));
            inputs.push(u(3));
            for (k, (&a, &b)) in first.iter().zip(second).enumerate() {
                def_params.push(Param::control(format!("p{k}"), a));
                inputs.push(InputRef::Param(k as u32));
                changes.push((k + 2, k, b));
            }
        }
    }
    let filter = units.len() as u32;
    units.push(UnitSpec::new(name, Rate::Audio, inputs, 1));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(filter)],
        0,
    ));
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: 48_000.0,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: def_params,
        units,
    });
    let node = controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = vec![0.0f32; BLOCK];
    (0..BLOCKS)
        .map(|b| {
            for &(block, param, value) in &changes {
                if block == b {
                    controller.set_control(node, param, value).unwrap();
                }
            }
            world.fill(&mut buf, 1);
            PICKS.map(|i| buf[i].to_bits())
        })
        .collect()
}

const PASS: ([f32; 2], [f32; 2]) = ([1000.0, 1.0], [2500.0, 0.4]);
const BAND: ([f32; 2], [f32; 2]) = ([1000.0, 1.0], [2500.0, 2.0]);
const PEAK: ([f32; 3], [f32; 3]) = ([1000.0, 1.0, 6.0], [2500.0, 0.5, -9.0]);
const SHELF: ([f32; 3], [f32; 3]) = ([800.0, 1.0, 6.0], [3000.0, 0.6, -9.0]);

#[test]
fn control_rate_parameters_ramp_the_coefficients_as_scsynth() {
    let cases: [(&str, &[f32], &[f32], Expected); 8] = [
        ("BLowPass", &PASS.0, &PASS.1, LOW_PASS_KK),
        ("BHiPass", &PASS.0, &PASS.1, HI_PASS_KK),
        ("BAllPass", &PASS.0, &PASS.1, ALL_PASS_KK),
        ("BBandPass", &BAND.0, &BAND.1, BAND_PASS_KK),
        ("BBandStop", &BAND.0, &BAND.1, BAND_STOP_KK),
        ("BPeakEQ", &PEAK.0, &PEAK.1, PEAK_EQ_KK),
        ("BLowShelf", &SHELF.0, &SHELF.1, LOW_SHELF_KK),
        ("BHiShelf", &SHELF.0, &SHELF.1, HI_SHELF_KK),
    ];
    for (name, first, second, want) in cases {
        assert_eq!(run(name, Params::Control(first, second)), want, "{name}");
    }
}

#[test]
fn scalar_parameters_filter_with_fixed_coefficients_as_scsynth() {
    assert_eq!(run("BLowPass", Params::Scalar(&PASS.1)), LOW_PASS_II);
    assert_eq!(run("BHiPass", Params::Scalar(&PASS.1)), HI_PASS_II);
}

#[test]
fn a_mix_of_audio_and_control_rate_parameters_takes_the_kk_calc() {
    // `_aaa` only when every parameter is audio-rate; otherwise `_kkk`, reading the audio-rate
    // `freq` at each block's first sample.
    let (first, second) = (&PEAK.0[1..], &PEAK.1[1..]);
    assert_eq!(run("BPeakEQ", Params::Mixed(first, second)), PEAK_EQ_MIXED);
}

#[test]
fn audio_rate_parameters_follow_scsynths_aa_calcs() {
    let cases: [(&str, usize, Expected); 8] = [
        ("BLowPass", 2, LOW_PASS_AA),
        ("BHiPass", 2, HI_PASS_AA),
        ("BAllPass", 2, ALL_PASS_AA),
        ("BBandPass", 2, BAND_PASS_AA),
        ("BBandStop", 2, BAND_STOP_AA),
        ("BPeakEQ", 3, PEAK_EQ_AA),
        ("BLowShelf", 3, LOW_SHELF_AA),
        ("BHiShelf", 3, HI_SHELF_AA),
    ];
    for (name, n, want) in cases {
        assert_eq!(run(name, Params::Audio(n)), want, "{name}");
    }
}

const LOW_PASS_KK: [[u32; 7]; 5] = [
    [
        0x3bfad6fe, 0x3ba2623c, 0xbbca26c2, 0x3db5b19d, 0xbe77cfbd, 0xbe815a16, 0xbe85f589,
    ],
    [
        0xbe880f2f, 0xbe887045, 0xbe87e184, 0x3e4f0303, 0xbe23f552, 0xbe1958a0, 0xbdf6f10b,
    ],
    [
        0xbe06fd18, 0xbe31fee9, 0xbe558a80, 0xbe10cb61, 0x3d677a56, 0x3d9753b0, 0x3da4bba7,
    ],
    [
        0x3da5cfb1, 0x3df321a2, 0x3e2f2036, 0xbf3a547c, 0xbf240ddc, 0xbf054e2c, 0xbeac9f99,
    ],
    [
        0xbda3d2fb, 0x3e56c8bd, 0x3ee11be7, 0xbd9c45c6, 0x3e0eb4e6, 0x3e7651d7, 0x3e898b0b,
    ],
];

const HI_PASS_KK: [[u32; 7]; 5] = [
    [
        0xbf644ab5, 0xbf207395, 0x3e3c7f52, 0xbecb6c4f, 0xbd7824cb, 0xbe1bdeeb, 0x3f3f165f,
    ],
    [
        0xbe4da9f9, 0x3ee01600, 0xbe7e838b, 0x3f0174ad, 0x3f3c0c7b, 0x3f0ce608, 0xbf6d22bb,
    ],
    [
        0xbecdef62, 0x3f1a441b, 0xbf0920cf, 0x3f68898e, 0x3f566179, 0xbef1707a, 0xbe9beb23,
    ],
    [
        0x3f5f575e, 0xbd98c430, 0xbe41f8c8, 0x3f8cd6a0, 0x3f3f6e93, 0x3eb1f506, 0x3f6292b7,
    ],
    [
        0x3f150b58, 0xbf5aafa0, 0xbf88c56e, 0x3f4b213d, 0x3cc2be13, 0xbf4d3ba0, 0xbf8d09ea,
    ],
];

const ALL_PASS_KK: [[u32; 7]; 5] = [
    [
        0xbf69ccd5, 0xbf0d2bda, 0x3e8df80c, 0xbe6718f1, 0xbe69e570, 0xbea22ba7, 0x3f08b21d,
    ],
    [
        0xbee80647, 0x3e2d222d, 0xbf074851, 0x3ef96590, 0x3f253570, 0x3e8842fa, 0xbf8fade9,
    ],
    [
        0xbec7d665, 0x3f0a17fa, 0xbf25042f, 0x3f285cf3, 0x3f6266a9, 0xbeddafa3, 0xbe52ebe5,
    ],
    [
        0x3f6f1a98, 0xbce502ec, 0xbd9599e9, 0x3f0fc772, 0xbb1c8ca6, 0xbeb5ab45, 0x3e91d442,
    ],
    [
        0x3e11a786, 0xbf7c8b26, 0xbf5808d9, 0x3f19dec5, 0x3c73b1ff, 0xbf2874b4, 0xbf4ded3b,
    ],
];

const BAND_PASS_KK: [[u32; 7]; 5] = [
    [
        0x3cb54edf, 0xbd494cc6, 0xbd937e65, 0xbd6e6e98, 0xbd537615, 0xbd796ed1, 0xbd04c3a4,
    ],
    [
        0xbbaf285a, 0x3bffec69, 0x3c9a2014, 0x3e0a431f, 0xbdd3e2a5, 0x3d5c3abf, 0x3cd73207,
    ],
    [
        0xbdfea834, 0xbdc8d91d, 0xbdae61d0, 0x3e18a29e, 0x3e94cb21, 0x3e74cebc, 0xbd0fbc40,
    ],
    [
        0x3d787515, 0x3e3ca166, 0x3d6eb7bf, 0xbe2876fc, 0x3d9b6515, 0x3e95db65, 0x3f0431dd,
    ],
    [
        0x3f46be83, 0x3f25c667, 0x3e8660b3, 0x3e8c0c89, 0x3e324ec4, 0x3e39f6c3, 0xbd7a5403,
    ],
];

const BAND_STOP_KK: [[u32; 7]; 5] = [
    [
        0xbf6087af, 0xbf249cfc, 0x3e1a2b53, 0xbeaaa585, 0xbea67775, 0xbedd3996, 0x3eefa2a4,
    ],
    [
        0xbef3056f, 0x3e29755d, 0xbf04abf9, 0x3f4b274c, 0x3f1b6531, 0x3ef67530, 0xbf7f5f78,
    ],
    [
        0xbf0dadb5, 0x3ed517d7, 0xbf423676, 0x3f3a27d1, 0x3f1ce604, 0xbf19f76b, 0xbe5415f0,
    ],
    [
        0x3f6980bf, 0xbd8b472c, 0xbcb490fe, 0x3eb20fa6, 0x3e0fc565, 0xbe917ec5, 0x3e96d3ae,
    ],
    [
        0x3dafcc69, 0xbf733623, 0xbf2d2151, 0x3f0f4bff, 0x3e0c8f6c, 0xbf255751, 0xbf4d0a24,
    ],
];

const PEAK_EQ_KK: [[u32; 7]; 5] = [
    [
        0xbf553bc7, 0xbf3db15e, 0x3be4c69a, 0xbee60e2a, 0xbedb2640, 0xbf0daf18, 0x3ece9116,
    ],
    [
        0xbef8756c, 0x3e3972c3, 0xbef61cb4, 0x3f87c2b0, 0x3ecc89db, 0x3f15cfb7, 0xbf728d10,
    ],
    [
        0xbf4d0285, 0x3e625bc4, 0xbf6d8641, 0x3f7d5bf7, 0x3f614fc7, 0xbeb681a8, 0xbe8c16bc,
    ],
    [
        0x3f7ba8a2, 0x3e3a2e37, 0x3dc1806a, 0x3e621935, 0x3e20c80b, 0xbe0fe798, 0x3f0b47c2,
    ],
    [
        0x3ef37442, 0xbf28aed3, 0xbf1db272, 0x3f32955c, 0x3e426ab3, 0xbf116e89, 0xbf5a49ad,
    ],
];

const LOW_SHELF_KK: [[u32; 7]; 5] = [
    [
        0xbf56f695, 0xbf37926d, 0x3d178a54, 0xbebcb75a, 0xbf15ceab, 0xbf37aa01, 0x3e5dcc94,
    ],
    [
        0xbf314711, 0xbd1f2b34, 0xbf361be7, 0x3fa40e7a, 0x3ea5b476, 0x3ef941f4, 0xbf843701,
    ],
    [
        0xbf613257, 0x3dd024c7, 0xbf8a91b1, 0x3f464cf1, 0x3f89b043, 0xbe0ac7d9, 0xbd901721,
    ],
    [
        0x3f967648, 0x3ec5842a, 0x3e99123f, 0x3e84c4d9, 0x3f01e67d, 0x3e1048a9, 0x3f3d9ac2,
    ],
    [
        0x3f072e0b, 0xbf356584, 0xbf2c50f5, 0x3f144591, 0x3ec58a82, 0xbecfff63, 0xbf2eabf2,
    ],
];

const HI_SHELF_KK: [[u32; 7]; 5] = [
    [
        0xbfdde710, 0xbfaa2735, 0x3e73dd4c, 0xbf48c772, 0xbefb071a, 0xbf335216, 0x3f90d860,
    ],
    [
        0xbf377754, 0x3f143d08, 0xbf451839, 0x3fb565dd, 0x3f9a0489, 0x3f80aa49, 0xbff99b13,
    ],
    [
        0xbf8a99b5, 0x3f6821ed, 0xbfb299fa, 0x3fe6930a, 0x3fc2bd42, 0xbf89feba, 0xbf30bba2,
    ],
    [
        0x3fd9f61c, 0xbdd1a9fa, 0xbe57ca38, 0x3e76e51b, 0xbe6042c6, 0xbe7ccf74, 0x3df7acb5,
    ],
    [
        0x3e920d5f, 0xbce06a6a, 0xbd3258e6, 0x3eca9e25, 0xbcaf138f, 0xbe84936e, 0xbed843b1,
    ],
];

const LOW_PASS_II: [[u32; 7]; 5] = [
    [
        0x3d3cb28a, 0x3cc69a43, 0xbd51db2f, 0xbde61cbd, 0xbe7f8eb0, 0xbdd38028, 0x3c4d6092,
    ],
    [
        0x3de2701b, 0x3e353873, 0x3e4ec702, 0x3e8e5e02, 0xbe6b3f60, 0xbe9c04ab, 0xbea685fc,
    ],
    [
        0xbeb9c53e, 0xbed1ceb1, 0xbedd5160, 0x3d666475, 0x3d1185b0, 0x3c5a1b52, 0xbc1590e8,
    ],
    [
        0xbccb3777, 0x3c17879d, 0x3d8dc28f, 0xbf379ba1, 0xbf24a00d, 0xbf05c3c8, 0xbead411a,
    ],
    [
        0xbda51552, 0x3e56c5ed, 0x3ee16040, 0xbd9c9f61, 0x3e0eb8d6, 0x3e765807, 0x3e898ed1,
    ],
];

const HI_PASS_II: [[u32; 7]; 5] = [
    [
        0xbf6d1d5d, 0xbf23683e, 0x3e702a7d, 0xbe455d8b, 0xbeaa96a5, 0xbf09c470, 0x3e94f95e,
    ],
    [
        0xbf322934, 0xbd726e0a, 0xbf34d18a, 0x3eddfc3f, 0x3f61cf79, 0x3f5f6c7b, 0xbf211501,
    ],
    [
        0xbe7674ec, 0x3f44736b, 0xbecb0776, 0x3f33ed9e, 0x3f69b4b7, 0xbebbcc8f, 0xbe4016eb,
    ],
    [
        0x3f7d75db, 0x3d02e061, 0xbdd3800c, 0x3f8c0c98, 0x3f3fe492, 0x3eb28e25, 0x3f62b337,
    ],
    [
        0x3f1501dd, 0xbf5add11, 0xbf88e9b2, 0x3f4b2f72, 0x3cc284ac, 0xbf4d3db9, 0xbf8d0aff,
    ],
];

const LOW_PASS_AA: [[u32; 7]; 5] = [
    [
        0x3c0b5b10, 0x3c84074b, 0x3caac3f2, 0x3c78796d, 0xbe90eba8, 0xbe9120bb, 0xbe8f7201,
    ],
    [
        0xbe3d2a30, 0xbe32dd55, 0xbe22a7bc, 0x3be5d60b, 0xbed3dab1, 0xbeca544d, 0xbeb801ec,
    ],
    [
        0xbe0d88c6, 0xbdf487d1, 0xbdc91bb9, 0xbe8f7a3e, 0xbc1e02ed, 0xbcdecf05, 0xbd0e7a95,
    ],
    [
        0xbcda55ea, 0xbce4cf76, 0xbd0bc9f4, 0x3d8143e1, 0x3f03404b, 0x3f07079c, 0x3f062e16,
    ],
    [
        0x3e9835df, 0x3e8e37c6, 0x3e7edf0a, 0x3de5a4a0, 0xbdc80d39, 0xbde599f9, 0xbde679bc,
    ],
];

const HI_PASS_AA: [[u32; 7]; 5] = [
    [
        0x3c1ce830, 0xbe9d68c1, 0xbea5c4f9, 0xbd0159a4, 0x3f3e18ea, 0xbe5b41ec, 0x3ed6e99f,
    ],
    [
        0x3e31c9ed, 0x3f471d01, 0x3cbcd817, 0xbcd601c1, 0x3f97cc22, 0x3f8813a4, 0xbe49ee11,
    ],
    [
        0x3e939c73, 0x3d4d25fa, 0x3f202561, 0xbedeb23c, 0x3f8822b8, 0x3f3c3ee5, 0xbb8e9c02,
    ],
    [
        0x3f0886da, 0xbf567a9a, 0xbf2585ea, 0x3f06d595, 0x3e534b7c, 0xbf85ce31, 0xbf3433a0,
    ],
    [
        0xbe224709, 0xbf217e37, 0xbf5ff3ab, 0x3f4c4d7d, 0x3f872792, 0x3db682bd, 0x3f7b3f60,
    ],
];

const ALL_PASS_AA: [[u32; 7]; 5] = [
    [
        0xbd2d8ee0, 0xbead6859, 0xbea5248d, 0x3e3f3eac, 0x3ef78227, 0xbf03e0a9, 0x3dd8d36f,
    ],
    [
        0xbd6f7156, 0x3f034eeb, 0xbe862aa1, 0x3e38d850, 0x3f3bba23, 0x3ef28463, 0xbf4f8938,
    ],
    [
        0xbd594c41, 0xbe94ae62, 0x3e8a7b47, 0xbf1a08c3, 0x3f9feed4, 0x3f4a8c29, 0xbae9f53a,
    ],
    [
        0x3f029634, 0xbf5965a3, 0xbf19b363, 0x3ee01df3, 0x3f1c0680, 0xbf10afd8, 0xbdc716b5,
    ],
    [
        0x3e873135, 0xbe36aab5, 0xbeb6beb1, 0x3f754302, 0x3f8d3527, 0x3ccde7a4, 0x3f539459,
    ],
];

const BAND_PASS_AA: [[u32; 7]; 5] = [
    [
        0x3d327446, 0x3d10562c, 0x3c8470f3, 0xbe09df8c, 0xbd6d918c, 0xbd14aeeb, 0xbcd025e9,
    ],
    [
        0x3ad9afce, 0x3d15d6ff, 0x3d8a3d39, 0xbdd9d4fc, 0xbcc92f89, 0x3da3ffcd, 0x3dfe9baa,
    ],
    [
        0x3de24bf6, 0x3e07449e, 0x3e2a8f41, 0xbe01b6ae, 0xbe64fe98, 0xbde6d629, 0xbd830018,
    ],
    [
        0xbc7e7eec, 0xbcc27d9d, 0xbd8bb9e5, 0x3d1c36c7, 0x3d95ae54, 0x3d68fe2a, 0xbc800779,
    ],
    [
        0xbd378340, 0xbdad7547, 0xbe222791, 0xbdeb9f08, 0xbdc51b62, 0xbd2146ee, 0x3c923ea5,
    ],
];

const BAND_STOP_AA: [[u32; 7]; 5] = [
    [
        0x3d0f3c3a, 0xbe8ef306, 0xbe9953f7, 0xbdadf988, 0x3efcbc62, 0xbee32c33, 0x3e4b797d,
    ],
    [
        0x3d0da1c2, 0x3f281efc, 0xbd9dc599, 0xbde621c4, 0x3f55e530, 0x3f483816, 0xbeda8663,
    ],
    [
        0x3e788d55, 0x3ca11a90, 0x3f1e2260, 0xbf33b60c, 0x3f8a7e63, 0x3f3cdf75, 0xbc4d3140,
    ],
    [
        0x3f04cc1c, 0xbf5bc89b, 0xbf315a73, 0x3f3429bc, 0x3f416a09, 0xbf070997, 0xbe7ca001,
    ],
    [
        0x3d6f1f40, 0xbee2d6a6, 0xbf3c8e94, 0x3f7a3501, 0x3f68d418, 0xbd0209d2, 0x3f64ba03,
    ],
];

const PEAK_EQ_AA: [[u32; 7]; 5] = [
    [
        0xbe5ef4f9, 0xbe8b3c2f, 0x3e8e364a, 0xbdfdf8b4, 0xbef80656, 0x3e2df84e, 0xbf008d9c,
    ],
    [
        0xbea5ca64, 0xbf122150, 0x3ebad41f, 0xbf2e33b0, 0xbe82c127, 0xbf331ca9, 0xbf63d1fa,
    ],
    [
        0xbec59825, 0x3f4f5367, 0xbf5020cc, 0xbf2e15a1, 0x3f50fb5a, 0x3f6037f2, 0x3f091e9a,
    ],
    [
        0xbf0bfe0b, 0xbf651436, 0x3f2b702f, 0xbe530940, 0xbdb0ddba, 0x3f6b4086, 0xbd6ddeb8,
    ],
    [
        0x3f025452, 0x3dcc1b9b, 0x3f39f138, 0x3da0d47a, 0xbf167321, 0xbeb8c467, 0x3ec60b1d,
    ],
];

const LOW_SHELF_AA: [[u32; 7]; 5] = [
    [
        0xbe62c9d1, 0xbe8a5bde, 0x3e8fdc53, 0xbe379c25, 0xbf0c9cc7, 0x3dd374d3, 0xbf120879,
    ],
    [
        0xbe762411, 0xbefc16f6, 0x3ee427d9, 0xbf2d0dac, 0xbe8cfb69, 0xbf38e1e0, 0xbf6a0fcb,
    ],
    [
        0xbecae3b4, 0x3f4bd4ce, 0xbf543658, 0xbf2debb8, 0x3f5dea9c, 0x3f6d965b, 0x3f177637,
    ],
    [
        0xbf142e87, 0xbf6ea6e5, 0x3f218ed7, 0xbe5dc630, 0xbdc2731e, 0x3f6744ce, 0xbd9ff2ab,
    ],
    [
        0x3f038d55, 0x3dd541e6, 0x3f3af8ff, 0x3de9917f, 0xbf17aa6a, 0xbebd6c7b, 0x3ec101f6,
    ],
];

const HI_SHELF_AA: [[u32; 7]; 5] = [
    [
        0xbedb7c24, 0xbef3b0fb, 0x3ed4f15c, 0xbe822dd6, 0xbeb32092, 0x3ca43ccc, 0xbeb8b172,
    ],
    [
        0xbe81948f, 0xbed15265, 0x3def931e, 0xbfb61d46, 0xbe5af3df, 0xbeededa9, 0xbf14a04d,
    ],
    [
        0xbebca695, 0x3f6d8dbd, 0xbf59b0e8, 0xbedeb98e, 0x3f70d9e8, 0x3f7cb8bf, 0x3f0984a4,
    ],
    [
        0xbec16a22, 0xbf2ccb1f, 0x3f0a8f00, 0xbe0dd1b4, 0xbd9470ec, 0x3f8e9a5e, 0xbddbf7a6,
    ],
    [
        0x3eeede03, 0x3dd7e0c4, 0x3f2c6483, 0x3d11cdb3, 0xbf6a2459, 0xbf12d307, 0x3f02eb6e,
    ],
];

/// scsynth's `BPeakEQ` with an audio-rate `freq` and control-rate `rq`/`db`, which is `_kkk`.
const PEAK_EQ_MIXED: [[u32; 7]; 5] = [
    [
        0xbf2b665e, 0x3da1a8dd, 0xbe7f7a4d, 0xbf736d59, 0xbf0b05f2, 0x3ed510d0, 0xbef0c786,
    ],
    [
        0x3eb2e2c8, 0xbf4f7213, 0xbf091137, 0x3edded9a, 0xbe0e342f, 0xbd8cffee, 0x3f943c48,
    ],
    [
        0xbde5b6de, 0xbe920c0b, 0xbe8c7cd7, 0x3f871f46, 0xbf15f1f2, 0xbf7d9574, 0xbeb53188,
    ],
    [
        0xbf42ee81, 0x3e80a503, 0x3da3f116, 0xbe7768a7, 0xbe93a9e4, 0x3f730500, 0x3f297f80,
    ],
    [
        0x3e82ef05, 0x3eb3463a, 0xbec8602e, 0xbecc7ebd, 0x3e155532, 0x3da967cf, 0x3f483dd4,
    ],
];
