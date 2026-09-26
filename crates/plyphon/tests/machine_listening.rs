//! The machine-listening units against scsynth's own code (`ML_SpecStats.cpp`, `Loudness.cpp`,
//! `MFCC.cpp`, `Onsets.cpp` with `onsetsds.c`), compiled from source and run over the same packed
//! frames. Each expected block is the bit pattern of every output on that block, at 48 kHz with
//! 64-sample blocks.
//!
//! The frames are deterministic pseudo-random packed spectra held in buffers `0..`, and the chain
//! input is `In.kr` of a control bus the test sets before each block - a buffer number for a frame,
//! `-1` between frames - so every unit sees exactly the frames the reference saw. Both sides
//! convert the chain buffers between Cartesian and polar form with scsynth's lookup-table
//! `ToPolarApx`/`ToComplexApx` (`SC_Complex.h`).
//!
//! Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
/// The control bus carrying the chain signal.
const CHAIN_BUS: u32 = 0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
}

/// A packed frame of `n` floats: slot `i` holds `(u * 2 - 1) * scale / (1 + k * tilt)`, where `u`
/// is the next draw of a 32-bit LCG in `[0, 1)` and `k` the slot's bin (DC `0`, Nyquist `n / 2`).
/// Every step is an exactly rounded float operation, so the reference computes the same bits.
fn frame(n: usize, seed: u32, scale: f32, tilt: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (s >> 8) as f32 / 16_777_216.0;
            let k = match i {
                0 => 0,
                1 => n / 2,
                _ => (i - 2) / 2 + 1,
            };
            (u * 2.0 - 1.0) * scale / (1.0 + k as f32 * tilt)
        })
        .collect()
}

/// Run `name.kr(chain, args...)` with `outputs` outputs over `frames` (buffers `0..`), the chain
/// reading `chain[b]` on block `b` (the constructor reads `chain[0]`), and return every output's
/// bits on each block.
fn run(
    name: &str,
    args: &[f32],
    outputs: usize,
    frames: &[Vec<f32>],
    chain: &[f32],
    sample_rate: f64,
) -> Vec<Vec<u32>> {
    run_behind(false, name, args, outputs, frames, chain, sample_rate)
}

/// [`run`], with the unit behind a `PV_MagAbove(chain, 0)` when `polar` - an identity that leaves
/// each frame in polar form, so the unit finds it that way.
fn run_behind(
    polar: bool,
    name: &str,
    args: &[f32],
    outputs: usize,
    frames: &[Vec<f32>],
    chain: &[f32],
    sample_rate: f64,
) -> Vec<Vec<u32>> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate,
        block_size: BLOCK,
        output_channels: outputs,
        ..Options::default()
    });
    for (i, f) in frames.iter().enumerate() {
        controller
            .buffer_set(
                i,
                Box::new(Buffer::from_interleaved(f.clone(), 1, sample_rate)),
            )
            .expect("buffer_set");
    }
    let mut units = vec![UnitSpec::new(
        "In",
        Rate::Control,
        vec![c(CHAIN_BUS as f32)],
        1,
    )];
    if polar {
        units.push(UnitSpec::new(
            "PV_MagAbove",
            Rate::Control,
            vec![u(0, 0), c(0.0)],
            1,
        ));
    }
    let unit = units.len() as u32;
    let mut inputs = vec![u(unit - 1, 0)];
    inputs.extend(args.iter().map(|&a| c(a)));
    units.push(UnitSpec::new(name, Rate::Control, inputs, outputs));
    // Adding zero at audio rate carries each control output into its block exactly.
    for o in 0..outputs {
        units.push(UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![u(unit, o as u32), c(0.0)],
            num_outputs: 1,
            special_index: 0,
        });
    }
    let mut out = vec![c(0.0)];
    out.extend((0..outputs).map(|o| u(unit + 1 + o as u32, 0)));
    units.push(UnitSpec::new("Out", Rate::Audio, out, 0));
    controller.add_synthdef(SynthDef {
        name: "ml".to_string(),
        params: vec![],
        units,
    });
    controller
        .set_control_bus(CHAIN_BUS, chain[0])
        .expect("set bus");
    controller
        .synth_new("ml", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    chain
        .iter()
        .map(|&value| {
            controller
                .set_control_bus(CHAIN_BUS, value)
                .expect("set bus");
            block(&mut world, outputs)
        })
        .collect()
}

/// Render one block and return the bits of each channel's first sample.
fn block(world: &mut World, channels: usize) -> Vec<u32> {
    let mut buf = vec![0.0f32; BLOCK * channels];
    world.fill(&mut buf, channels);
    buf[..channels].iter().map(|s| s.to_bits()).collect()
}

/// Assert `got` matches the reference `want`, block by block.
fn check<const N: usize>(label: &str, got: &[Vec<u32>], want: &[[u32; N]]) {
    assert_eq!(got.len(), want.len(), "{label}: block count");
    for (b, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.as_slice(),
            w.as_slice(),
            "{label}: block {b} is {:?}, scsynth gives {:?}",
            g.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>(),
            w.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>()
        );
    }
}

// ---- SpecCentroid, SpecFlatness, SpecPcile ----

/// Two ordinary 256-sample frames, one with every third bin zeroed, and a silent one.
fn spec_frames() -> Vec<Vec<f32>> {
    let mut f = vec![
        frame(256, 1, 1.0, 0.25),
        frame(256, 2, 0.5, 0.0),
        frame(256, 3, 2.0, 1.0),
        vec![0.0; 256],
    ];
    for i in (2..256).step_by(6) {
        f[2][i] = 0.0;
        f[2][i + 1] = 0.0;
    }
    f
}

/// Frames, gaps between them (the result holds), a zeroed-bin frame, the silent frame, and a
/// repeat of the first frame (now already in the unit's coordinate form).
const SPEC_CHAIN: [f32; 9] = [-1.0, 0.0, -1.0, -1.0, 1.0, 2.0, -1.0, 3.0, 0.0];

fn spec(name: &str, args: &[f32]) -> Vec<Vec<u32>> {
    run(name, args, 1, &spec_frames(), &SPEC_CHAIN, SR)
}

#[test]
fn spec_centroid_matches_scsynth() {
    const WANT: [[u32; 1]; 9] = [
        [0x00000000],
        [0x45c19a7b],
        [0x45c19a7b],
        [0x45c19a7b],
        [0x463612cf],
        [0x45b865a6],
        [0x45b865a6],
        [0x00000000],
        [0x45c19a7b],
    ];
    check("SpecCentroid", &spec("SpecCentroid", &[]), &WANT);
}

#[test]
fn spec_flatness_matches_scsynth() {
    // The silent frame yields scsynth's stand-in value 0.8.
    const WANT: [[u32; 1]; 9] = [
        [0x00000000],
        [0x3f11da73],
        [0x3f11da73],
        [0x3f11da73],
        [0x3f5fd67b],
        [0x401949db],
        [0x401949db],
        [0x3f4ccccd],
        [0x3f11da73],
    ];
    check("SpecFlatness", &spec("SpecFlatness", &[]), &WANT);
}

#[test]
fn spec_pcile_matches_scsynth() {
    const FREQ: [[u32; 1]; 9] = [
        [0x00000000],
        [0x455cee24],
        [0x455cee24],
        [0x455cee24],
        [0x46315359],
        [0x44d14d65],
        [0x44d14d65],
        [0x433a0be8],
        [0x455cee24],
    ];
    check("SpecPcile", &spec("SpecPcile", &[0.5, 0.0, 0.0]), &FREQ);
    const FREQ_INTERP: [[u32; 1]; 9] = [
        [0x00000000],
        [0x44a1b7c4],
        [0x44a1b7c4],
        [0x44a1b7c4],
        [0x45cee37e],
        [0x439ecfe4],
        [0x439ecfe4],
        [0x433a0be8],
        [0x44a1b7c4],
    ];
    check(
        "SpecPcile interpolated",
        &spec("SpecPcile", &[0.3, 1.0, 0.0]),
        &FREQ_INTERP,
    );
    const BIN: [[u32; 1]; 9] = [
        [0x00000000],
        [0x42640000],
        [0x42640000],
        [0x42640000],
        [0x42c40000],
        [0x423c0000],
        [0x423c0000],
        [0x00000000],
        [0x42640000],
    ];
    check("SpecPcile bin", &spec("SpecPcile", &[0.8, 0.0, 1.0]), &BIN);
    const BIN_INTERP: [[u32; 1]; 9] = [
        [0x00000000],
        [0x4264a820],
        [0x4264a820],
        [0x4264a820],
        [0x42c5c1f1],
        [0x423f47c7],
        [0x423f47c7],
        [0x00000000],
        [0x4264a820],
    ];
    check(
        "SpecPcile bin interpolated",
        &spec("SpecPcile", &[0.8, 1.0, 1.0]),
        &BIN_INTERP,
    );
}

#[test]
fn spec_pcile_ignores_a_frame_of_another_size() {
    // The first frame fixes the bin count. A 128-sample frame is then skipped, the unit outputting
    // the chain value (4) for that block, and it holds its previous result again after that.
    let mut frames = spec_frames();
    frames.push(frame(128, 5, 1.0, 0.0));
    const WANT: [[u32; 1]; 5] = [
        [0x00000000],
        [0x46314e4d],
        [0x40800000],
        [0x46314e4d],
        [0x455a00e5],
    ];
    let got = run(
        "SpecPcile",
        &[0.5, 1.0, 0.0],
        1,
        &frames,
        &[-1.0, 1.0, 4.0, -1.0, 0.0],
        SR,
    );
    check("SpecPcile size change", &got, &WANT);
}

// ---- Loudness ----

/// Five 1024-sample frames - loud, quiet, louder with a steep tilt, silent, near-silent - and a
/// 512-sample frame, too small for the band layout.
fn loudness_frames() -> Vec<Vec<f32>> {
    vec![
        frame(1024, 11, 1.0, 0.05),
        frame(1024, 12, 0.05, 0.0),
        frame(1024, 13, 4.0, 0.2),
        vec![0.0; 1024],
        frame(1024, 14, 0.0001, 0.0),
        frame(512, 15, 1.0, 0.0),
    ]
}

#[test]
fn loudness_matches_scsynth() {
    // The constructor analyses frame 0 (the chain already carries it). Silence then decays each
    // band by `tmask` phons per frame.
    const WANT: [[u32; 1]; 10] = [
        [0x4061a2dd],
        [0x4061a2dd],
        [0x40529657],
        [0x40942ebd],
        [0x40942ebd],
        [0x408a425d],
        [0x4081001b],
        [0x4070b92a],
        [0x40609a50],
        [0x406907aa],
    ];
    let chain = [0.0, -1.0, 1.0, 2.0, -1.0, 3.0, 3.0, 3.0, 4.0, 0.0];
    let got = run("Loudness", &[0.25, 1.0], 1, &loudness_frames(), &chain, SR);
    check("Loudness", &got, &WANT);

    const MASKS: [[u32; 1]; 9] = [
        [0x00000000],
        [0x40959396],
        [0x407a23e8],
        [0x407a23e8],
        [0x404b4ab5],
        [0x40251fd3],
        [0x40061f70],
        [0x3fd9e237],
        [0x40959396],
    ];
    let chain = [-1.0, 2.0, 0.0, -1.0, 1.0, 3.0, 3.0, 4.0, 2.0];
    let got = run("Loudness", &[0.5, 3.0], 1, &loudness_frames(), &chain, SR);
    check("Loudness smask/tmask", &got, &MASKS);
}

#[test]
fn loudness_skips_a_frame_too_small_for_its_bands() {
    // The reference reads past the end of a buffer under 1024 samples; plyphon skips the frame and
    // holds the previous loudness.
    let chain = [0.0, 5.0, 5.0];
    let got = run("Loudness", &[0.25, 1.0], 1, &loudness_frames(), &chain, SR);
    assert_eq!(got, vec![vec![0x4061a2dd]; 3]);
}

// ---- MFCC ----

/// Three 1024-sample frames and a silent one.
fn mfcc_frames() -> Vec<Vec<f32>> {
    vec![
        frame(1024, 21, 1.0, 0.05),
        frame(1024, 22, 0.05, 0.0),
        frame(1024, 23, 4.0, 0.2),
        vec![0.0; 1024],
    ]
}

/// Frames, gaps, the silent frame (every coefficient `0.25`) and a repeat of the first frame.
const MFCC_CHAIN: [f32; 7] = [0.0, -1.0, 1.0, 2.0, -1.0, 3.0, 0.0];

// Each frame's coefficients from scsynth's MFCC.cpp. MFCC calls the platform's `log10f` for the band
// levels; macOS and the glibc CI builds against agree on these frames (some older glibc versions
// round a few band levels differently in the last bit, as scsynth's own output does there).

/// The silent frame: every band is floored at -50 dB, so every coefficient is `0.25`.
const MFCC_SILENT: [u32; 13] = [0x3e800000; 13];
/// Frame 0 at 48 kHz.
const MFCC_48K_0: [u32; 13] = [
    0x3ef68375, 0x3e66b4ce, 0x3e8f1245, 0x3e7a937a, 0x3e8d50f9, 0x3e8b0d55, 0x3e813a11, 0x3e6f253c,
    0x3e734058, 0x3e7b7973, 0x3e655b21, 0x3e57c4f1, 0x3e79b388,
];
/// Frame 1 at 48 kHz.
const MFCC_48K_1: [u32; 13] = [
    0xbcf9fd40, 0x3e80c5fb, 0x3e71211a, 0x3e8d6b68, 0x3e847254, 0x3e89f667, 0x3e622b46, 0x3e7aa6ca,
    0x3e736601, 0x3e7168fc, 0x3e798b82, 0x3e70a97d, 0x3e8befbd,
];
/// Frame 2 at 48 kHz, all 42 coefficients.
const MFCC_48K_2_ALL: [u32; 42] = [
    0x3f2a2f32, 0x3e94320f, 0x3e9dea02, 0x3e84cc9d, 0x3e8a6f7e, 0x3e921697, 0x3e864536, 0x3e6dd5df,
    0x3e82c909, 0x3e7bac7c, 0x3e7a6b41, 0x3e75a391, 0x3e751dac, 0x3e8879e4, 0x3e7dad6c, 0x3e81108b,
    0x3e840d4b, 0x3e8793ad, 0x3e7f6212, 0x3e778882, 0x3e735ded, 0x3e6eb60c, 0x3e728216, 0x3e879e76,
    0x3e817413, 0x3e748b8c, 0x3e783a9f, 0x3e78efb7, 0x3e8561ff, 0x3e7d64df, 0x3e7b7291, 0x3e827a28,
    0x3e792f79, 0x3e83facd, 0x3e8824b8, 0x3e8630e6, 0x3e7b8168, 0x3e7978f8, 0x3e830550, 0x3e815b31,
    0x3e8329f4, 0x3e800000,
];
/// Frame 0 at 48 kHz, all 42 coefficients.
const MFCC_48K_0_ALL: [u32; 42] = [
    0x3ef68375, 0x3e66b4ce, 0x3e8f1245, 0x3e7a937a, 0x3e8d50f9, 0x3e8b0d55, 0x3e813a11, 0x3e6f253c,
    0x3e734058, 0x3e7b7973, 0x3e655b21, 0x3e57c4f1, 0x3e79b388, 0x3e859a7c, 0x3e8596ec, 0x3e864fb7,
    0x3e7d521c, 0x3e83eac7, 0x3e8dbcc5, 0x3e828d87, 0x3e893f68, 0x3e7c0722, 0x3e6d97e1, 0x3e728364,
    0x3e7c2d6a, 0x3e851a5b, 0x3e81e665, 0x3e84e057, 0x3e83da20, 0x3e7a59e0, 0x3e77eebc, 0x3e74802e,
    0x3e79ab18, 0x3e7ae4ac, 0x3e7a1a89, 0x3e85bcda, 0x3e837b2f, 0x3e79fba6, 0x3e7c9333, 0x3e7d27c3,
    0x3e7a0b8d, 0x3e800000,
];
/// Frame 0 at 44.1 kHz.
const MFCC_44K_0: [u32; 13] = [
    0x3efb2b48, 0x3e6682a8, 0x3e8eced9, 0x3e7dde18, 0x3e8d4d98, 0x3e8fed1b, 0x3e898b97, 0x3e7db6b2,
    0x3e72eccd, 0x3e7a6b97, 0x3e714dc8, 0x3e530dc0, 0x3e5c9562,
];
/// Frame 1 at 44.1 kHz.
const MFCC_44K_1: [u32; 13] = [
    0xbce6a6c0, 0x3e825885, 0x3e6e81ef, 0x3e8f1086, 0x3e83f1cf, 0x3e8de3d1, 0x3e6cea41, 0x3e763c28,
    0x3e78fac1, 0x3e6c4548, 0x3e789963, 0x3e6a453d, 0x3e7aa917,
];
/// Frame 2 at 44.1 kHz.
const MFCC_44K_2: [u32; 13] = [
    0x3f2c4322, 0x3e97b1e9, 0x3e9e606b, 0x3e8883e3, 0x3e884fe1, 0x3e9210ce, 0x3e8cf815, 0x3e75923a,
    0x3e7d3836, 0x3e810536, 0x3e77d8dd, 0x3e7d7324, 0x3e6689bf,
];

/// The first `N` coefficients of `row`.
fn first<const N: usize>(row: &[u32]) -> [u32; N] {
    row[..N].try_into().expect("row is long enough")
}

/// The expected blocks of [`MFCC_CHAIN`], given frames 0 to 2's first `N` coefficients.
fn mfcc_blocks<const N: usize>(frames: [&[u32]; 3]) -> Vec<[u32; N]> {
    let [f0, f1, f2] = frames.map(first::<N>);
    let silent = first::<N>(&MFCC_SILENT);
    vec![f0, f0, f1, f2, f2, silent, f0]
}

#[test]
fn mfcc_matches_scsynth_at_48k() {
    let want = mfcc_blocks::<13>([&MFCC_48K_0, &MFCC_48K_1, &MFCC_48K_2_ALL]);
    let got = run("MFCC", &[13.0], 13, &mfcc_frames(), &MFCC_CHAIN, SR);
    check("MFCC 48 kHz", &got, &want);
}

#[test]
fn mfcc_matches_scsynth_at_44k() {
    let want = mfcc_blocks::<13>([&MFCC_44K_0, &MFCC_44K_1, &MFCC_44K_2]);
    let got = run("MFCC", &[13.0], 13, &mfcc_frames(), &MFCC_CHAIN, 44_100.0);
    check("MFCC 44.1 kHz", &got, &want);
}

#[test]
fn mfcc_halves_a_double_rate() {
    // 96 kHz uses the 48 kHz filterbank and 88.2 kHz the 44.1 kHz one.
    let want = mfcc_blocks::<5>([&MFCC_48K_0, &MFCC_48K_1, &MFCC_48K_2_ALL]);
    let got = run("MFCC", &[5.0], 5, &mfcc_frames(), &MFCC_CHAIN, 96_000.0);
    check("MFCC 96 kHz", &got, &want);
    let want = mfcc_blocks::<5>([&MFCC_44K_0, &MFCC_44K_1, &MFCC_44K_2]);
    let got = run("MFCC", &[5.0], 5, &mfcc_frames(), &MFCC_CHAIN, 88_200.0);
    check("MFCC 88.2 kHz", &got, &want);
}

#[test]
fn mfcc_clamps_its_coefficient_count() {
    // All 42 coefficients the DCT table holds, for frame 2 and then frame 0.
    let got = run("MFCC", &[42.0], 42, &mfcc_frames(), &[2.0, 0.0], SR);
    check(
        "MFCC 42 coefficients",
        &got,
        &[MFCC_48K_2_ALL, MFCC_48K_0_ALL],
    );
    // A count below one computes one coefficient.
    let got = run("MFCC", &[0.0], 1, &mfcc_frames(), &[2.0, 0.0], SR);
    check(
        "MFCC one coefficient",
        &got,
        &[first::<1>(&MFCC_48K_2_ALL), first::<1>(&MFCC_48K_0)],
    );
}

/// Frames 0, 1 and 1 again at 48 kHz behind a polar unit: each block the frame goes to polar form
/// with `ToPolarApx` and back with `ToComplexApx` before the filterbank reads it, so the repeat of
/// frame 1 has made two lookup-table round trips.
const MFCC_48K_POLAR: [[u32; 13]; 3] = [
    [
        0x3ef68408, 0x3e66b964, 0x3e8f123e, 0x3e7a95bd, 0x3e8d533e, 0x3e8b0e47, 0x3e813a51,
        0x3e6f2111, 0x3e733fd8, 0x3e7b7b44, 0x3e655b7e, 0x3e57c46e, 0x3e79afc3,
    ],
    [
        0xbcfa0980, 0x3e80c49b, 0x3e711ee1, 0x3e8d6aea, 0x3e84736c, 0x3e89f5bf, 0x3e622876,
        0x3e7aa7de, 0x3e73618a, 0x3e716418, 0x3e7987dd, 0x3e70ab91, 0x3e8bf11f,
    ],
    [
        0xbcfa1260, 0x3e80c2b4, 0x3e711c3e, 0x3e8d6a51, 0x3e847336, 0x3e89f23c, 0x3e62213b,
        0x3e7aa4a6, 0x3e73593b, 0x3e715d6a, 0x3e798434, 0x3e70ada8, 0x3e8bf0fb,
    ],
];

#[test]
fn mfcc_converts_a_polar_frame() {
    // Behind a polar predecessor each frame makes a lookup-table round trip through polar form
    // before the filterbank reads it, which moves its coefficients away from the Cartesian frame's.
    let got = run_behind(
        true,
        "MFCC",
        &[13.0],
        13,
        &mfcc_frames(),
        &[0.0, 1.0, 1.0],
        SR,
    );
    check("MFCC behind a polar unit", &got, &MFCC_48K_POLAR);
}

#[test]
fn mfcc_skips_a_frame_too_small_for_its_filterbank() {
    // The reference reads past the end of a 512-sample buffer; plyphon skips the frame and holds
    // the previous coefficients.
    let mut frames = mfcc_frames();
    frames.push(frame(512, 24, 1.0, 0.0));
    let got = run("MFCC", &[3.0], 3, &frames, &[2.0, 4.0, 4.0], SR);
    assert_eq!(got[1], got[0]);
    assert_eq!(got[2], got[0]);
    assert_eq!(got[0], vec![0x3f2a2f32, 0x3e94320f, 0x3e9dea02]);
}

// ---- Onsets ----

/// Eight 512-sample frames: quiet ones, two loud ones (2 and 5) and one with a steep tilt.
fn onsets_frames() -> Vec<Vec<f32>> {
    vec![
        frame(512, 31, 0.1, 0.0),
        frame(512, 32, 0.12, 0.01),
        frame(512, 33, 3.0, 0.02),
        frame(512, 34, 0.1, 0.0),
        frame(512, 35, 0.08, 0.0),
        frame(512, 36, 5.0, 0.0),
        frame(512, 37, 0.5, 0.5),
        frame(512, 38, 0.1, 0.0),
    ]
}

/// A gap, then the frames in a varied order with two more gaps; each loud frame is an onset.
const ONSETS_CHAIN: [f32; 24] = [
    -1.0, 0.0, 1.0, 3.0, 4.0, -1.0, 7.0, 2.0, 0.0, 1.0, 3.0, -1.0, 4.0, 7.0, 0.0, 5.0, 1.0, 3.0,
    4.0, 7.0, 6.0, 0.0, 2.0, 1.0,
];

/// `(label, [threshold, odftype, relaxtime, floor, mingap, medianspan, whtype, rawodf], output
/// bits per block of `ONSETS_CHAIN`)`, from scsynth's `Onsets.cpp` and `onsetsds.c`:
///
/// - `ODF_n`: the raw detection function of each type with the default parameters;
/// - `DETECT_*`: detections with the defaults, shorter gaps, a negative `mingap` (a gap that never
///   ends), the MKL function, an even median span and a span of one;
/// - `RAW_*`: no whitening, no relaxation, and a fast relaxation with a high floor.
#[rustfmt::skip]
const ONSETS_CASES: [(&str, [f32; 8], [u32; 24]); 17] = [
    ("ODF_0", [0.5, 0.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x403fb1ce, 0x3f97cc01, 0x4032b195, 0x3fee1b4e, 0x3fee1b4e, 0x4039831f, 0x409d8809, 0x3e53e78c, 0x3d65dc38, 0x3e84052c, 0x3e84052c, 0x3e0ae151, 0x3e98bff3, 0x3e630c4a, 0x409f2e08, 0x3adbbb67, 0x3bab713b, 0x3b80c1d3, 0x3bc22fdb, 0x3ab8afb4, 0x3bd5c15f, 0x3f1e7d87, 0x3af81976]),
    ("ODF_1", [0.5, 1.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x4068e122, 0x400bac34, 0x4060fd58, 0x4037276e, 0x4037276e, 0x40655860, 0x409e7647, 0x3f3f4de1, 0x3ec92d22, 0x3f4551a9, 0x3f4551a9, 0x3f1a1f40, 0x3f5387ed, 0x3f47f069, 0x409f8297, 0x3d94d091, 0x3e0043b3, 0x3dd092a3, 0x3e077343, 0x3cd94421, 0x3e07beb2, 0x3fb039f5, 0x3d9fc5cc]),
    ("ODF_2", [0.5, 2.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x406db22f, 0x40918e1b, 0x408affd0, 0x4094c284, 0x4094c284, 0x4099a64e, 0x40c4e0eb, 0x40a2e074, 0x3f5ac473, 0x3f687ed9, 0x3f687ed9, 0x3f7ee783, 0x3f883170, 0x3f94108d, 0x40a58c96, 0x4039eee2, 0x3e0d5523, 0x3e11b19f, 0x3e231f1d, 0x3cda24f7, 0x3e06e83f, 0x3fb37a25, 0x3f8b09f5]),
    ("ODF_3", [0.5, 3.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x406db22f, 0x3f5f02ab, 0x4065714e, 0x3fba6040, 0x3fba6040, 0x405ea327, 0x40bfec12, 0x3d4b0ffc, 0x3de024b7, 0x3f4d2cfc, 0x3f4d2cfc, 0x3ea78ae2, 0x3f4efeea, 0x3f0865eb, 0x40a58c96, 0x00000000, 0x3df5d9ec, 0x3d4f22e0, 0x3dfa8ef9, 0x3c258a21, 0x3dfc074a, 0x3fb34bc2, 0x00000000]),
    ("ODF_4", [0.5, 4.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x4071914a, 0x4089ca3a, 0x40847699, 0x40778f2a, 0x40778f2a, 0x40808275, 0x4073f427, 0x407aacc6, 0x40838380, 0x40736a83, 0x40736a83, 0x4081ea61, 0x407bd3a5, 0x40822cea, 0x4077ac47, 0x4007ea5f, 0x40653f2b, 0x404be60d, 0x40674235, 0x3ea56821, 0x406a7175, 0x4088b056, 0x4016f354]),
    ("ODF_5", [0.5, 5.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x4031f56d, 0x3fed66a0, 0x403963c6, 0x400d0bab, 0x400d0bab, 0x403957a0, 0x4070e920, 0x3f19d911, 0x3e91ae57, 0x3f293640, 0x3f293640, 0x3ee6eabe, 0x3f25bb7b, 0x3f10afe9, 0x40774dfd, 0x3d6b4c1e, 0x3db3cf13, 0x3d94d61b, 0x3dd0de97, 0x3c8a17f2, 0x3dbf4cc5, 0x3f8ee61f, 0x3d525f55]),
    ("ODF_6", [0.5, 6.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x40805499, 0x3f0120f0, 0x3f7d9c59, 0x3f1ef14b, 0x3f1ef14b, 0x3f4f0e36, 0x3f68d28b, 0x3e058ae4, 0x3ee71b28, 0x3f6ab9a5, 0x3f6ab9a5, 0x3f128a83, 0x3f42fd35, 0x3f290a5a, 0x4008d41a, 0x3c5ff240, 0x3f233e53, 0x3edab19b, 0x3f132365, 0x3dd41efc, 0x3f70c4aa, 0x3ff67ce2, 0x3d852ffe]),
    ("DETECT_DEFAULT", [0.5, 3.0, 1.0, 0.1, 10.0, 11.0, 1.0, 0.0], [0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]),
    ("DETECT_GAP0", [0.3, 3.0, 1.0, 0.1, 0.0, 11.0, 1.0, 0.0], [0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000]),
    ("DETECT_GAP2_POWER", [0.3, 0.0, 1.0, 0.1, 2.0, 11.0, 1.0, 0.0], [0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000]),
    ("DETECT_GAP_NEG", [0.3, 1.0, 1.0, 0.1, -1.0, 11.0, 1.0, 0.0], [0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]),
    ("DETECT_MKL", [0.2, 6.0, 1.0, 0.1, 1.0, 5.0, 1.0, 0.0], [0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000]),
    ("RAW_NOWHITEN", [0.5, 3.0, 1.0, 0.1, 10.0, 11.0, 0.0, 1.0], [0x00000000, 0x3ec429a2, 0x3db67f4a, 0x3ec037af, 0x3e195b56, 0x3e195b56, 0x3ebd94bd, 0x4085f520, 0x3ba72f13, 0x3db0dd5e, 0x3ec037af, 0x3ec037af, 0x3e195b56, 0x3ebd94bd, 0x3e90541c, 0x419d0c82, 0x00000000, 0x3eba364a, 0x3e195b56, 0x3ebd94bd, 0x3cf137b4, 0x3ebe0b14, 0x4087351f, 0x00000000]),
    ("DETECT_EVEN", [0.3, 2.0, 1.0, 0.1, 1.0, 4.0, 1.0, 0.0], [0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f800000, 0x00000000]),
    ("DETECT_SPAN1", [0.5, 5.0, 1.0, 0.1, 1.0, 1.0, 1.0, 0.0], [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]),
    ("RAW_RELAX0", [0.5, 4.0, 0.0, 0.1, 10.0, 11.0, 1.0, 1.0], [0x00000000, 0x4071914a, 0x4089ca3a, 0x40847699, 0x40778f2a, 0x40778f2a, 0x40808275, 0x4073f427, 0x407ec6e7, 0x407c4e4c, 0x40847699, 0x40847699, 0x40778f2a, 0x40808275, 0x40798ed8, 0x4077ac47, 0x407afed8, 0x4068c0ef, 0x40778f2a, 0x40808275, 0x407f61c9, 0x4067b24b, 0x4088b056, 0x406b53c7]),
    ("RAW_FAST_FLOOR", [0.5, 1.0, 0.05, 1.0, 10.0, 7.0, 1.0, 1.0], [0x00000000, 0x3ec07d18, 0x3e660650, 0x3ebd3faf, 0x3e9834dd, 0x3e9834dd, 0x3ec41c12, 0x4055e21a, 0x3eb75f68, 0x3e5d1139, 0x3ebbe36d, 0x3ebbe36d, 0x3e982ea0, 0x3ec41c12, 0x3ec07d18, 0x409e9233, 0x3db51a03, 0x3e39a7d7, 0x3e33283d, 0x3e8a2542, 0x3d818932, 0x3eb237ae, 0x4050f2e8, 0x3e545f03]),
];

#[test]
fn onsets_matches_scsynth() {
    for (label, args, want) in ONSETS_CASES {
        let got = run("Onsets", &args, 1, &onsets_frames(), &ONSETS_CHAIN, SR);
        let want: Vec<[u32; 1]> = want.iter().map(|&w| [w]).collect();
        check(label, &got, &want);
    }
}

#[test]
fn onsets_reads_the_start_of_a_larger_frame() {
    // After the first frame fixes the detector at 512 samples, a 1024-sample frame contributes its
    // first 512 (its DC, Nyquist and lowest 255 bins).
    let mut frames = onsets_frames();
    frames.push(frame(1024, 39, 1.0, 0.0));
    let chain = [
        0.0, 1.0, 8.0, 2.0, 8.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 8.0, 0.0, 1.0,
        2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ];
    const WANT: [u32; 24] = [
        0x406db22f, 0x3f5f02ab, 0x40b1c806, 0x404aed66, 0x404f17e9, 0x3bf8ae87, 0x3ea486be,
        0x3e41fefc, 0x40a284cd, 0x00000000, 0x3df66fcc, 0x3c967098, 0x3fab9d40, 0x3b0a4cf4,
        0x3d4c69ad, 0x3faa2c6e, 0x3b084983, 0x3cebc789, 0x3fb52ac4, 0x3b147f97, 0x3d593825,
        0x40a20a86, 0x00000000, 0x3df66e23,
    ];
    let args = [0.5, 3.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0];
    let got = run("Onsets", &args, 1, &frames, &chain, SR);
    let want: Vec<[u32; 1]> = WANT.iter().map(|&w| [w]).collect();
    check("Onsets larger frame", &got, &want);
}

#[test]
fn onsets_skips_a_frame_smaller_than_its_first() {
    // The reference would read past the end of the smaller frame; plyphon skips it and holds the
    // previous output.
    let mut frames = onsets_frames();
    frames.push(frame(256, 40, 1.0, 0.0));
    let args = [0.5, 3.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0];
    let got = run("Onsets", &args, 1, &frames, &[0.0, 8.0, 8.0, 1.0], SR);
    let want = run("Onsets", &args, 1, &frames, &[0.0, -1.0, -1.0, 1.0], SR);
    assert_eq!(got, want);
}

#[test]
fn onsets_skips_frames_it_cannot_set_up_for() {
    // An unknown detection function or a median span under one frame crashes scsynth; plyphon
    // never sets the detector up, and outputs its held zero.
    for args in [
        [0.5, 7.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0],
        [0.5, -1.0, 1.0, 0.1, 10.0, 11.0, 1.0, 1.0],
        [0.5, 3.0, 1.0, 0.1, 10.0, 0.0, 1.0, 1.0],
    ] {
        let got = run("Onsets", &args, 1, &onsets_frames(), &[0.0, 2.0, 5.0], SR);
        assert_eq!(got, vec![vec![0]; 3], "{args:?}");
    }
}
