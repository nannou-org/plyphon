//! Onset detection (`FeatureDetection.cpp`): `PV_JensenAndersen`, `PV_HainsworthFoote` and
//! `RunningSum`.
//!
//! The detectors fire (output `1` for the block) when their weighted feature sum exceeds
//! `threshold`, so each frame's sum is pinned exactly by driving the threshold per block: at the sum
//! itself nothing fires, and one float below it every frame fires. The sums, fire patterns and
//! `RunningSum` outputs are bit patterns from scsynth's own `FeatureDetection.cpp` functions,
//! compiled against the real headers and driven block by block with the same inputs. Both sides convert
//! the frames to polar with scsynth's `SC_Complex.h` lookup tables (`ToPolarApx`), over real-only
//! frames and over general spectra with bins of mixed signs.
//!
//! The detectors require the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
/// The detectors' chain-buffer frames: 127 bins.
const ONSET_FRAME: usize = 256;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// The detector frame for block `f`: real bins cycling through thirteen levels in `[-0.5, 1]`.
fn onset_frame(f: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; ONSET_FRAME];
    data[0] = f as f32 * 0.25;
    data[1] = 0.5;
    for k in 0..(ONSET_FRAME - 2) / 2 {
        data[2 + 2 * k] = ((k * 7 + f * 11) % 13) as f32 / 8.0 - 0.5;
    }
    data
}

/// A general Cartesian frame for block `f`: both parts of every bin, and DC and Nyquist, of mixed
/// signs.
fn general_frame(f: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; ONSET_FRAME];
    data[0] = f as f32 * 0.25 - 0.5;
    data[1] = 0.5 - f as f32 * 0.125;
    for k in 0..(ONSET_FRAME - 2) / 2 {
        data[2 + 2 * k] = ((k * 7 + f * 11) % 13) as f32 / 8.0 - 0.5;
        data[3 + 2 * k] = ((k * 5 + f * 3) % 11) as f32 / 8.0 - 0.625;
    }
    data
}

/// The largest float below `x`.
fn next_down(x: f32) -> f32 {
    if x > 0.0 {
        f32::from_bits(x.to_bits() - 1)
    } else if x < 0.0 {
        f32::from_bits(x.to_bits() + 1)
    } else {
        -f32::from_bits(1)
    }
}

fn engine_with(
    units: Vec<UnitSpec>,
    channels: usize,
    buffer: Vec<f32>,
) -> (plyphon::Controller, World) {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: channels,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(buffer, 1, SR)))
        .expect("buffer_set");
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (controller, world)
}

/// Run detector `name` over `frame`s - its threshold read from control bus 0, its weights
/// `props`, its wait `waittime` - and return each block's output, checking the block is uniform.
fn run_detector(
    name: &str,
    frame: fn(usize) -> Vec<f32>,
    props: &[f32],
    waittime: f32,
    thresholds: &[f32],
) -> Vec<f32> {
    let mut inputs = vec![c(0.0)];
    inputs.extend(props.iter().map(|&p| c(p)));
    inputs.push(u(0));
    inputs.push(c(waittime));
    let units = vec![
        UnitSpec::new("In", Rate::Control, vec![c(0.0)], 1),
        UnitSpec::new(name, Rate::Audio, inputs, 1),
        UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
    ];
    // The synth starts with the first frame's buffer, which sizes the detector.
    let (mut controller, mut world) = engine_with(units, 1, frame(0));
    let mut outs = Vec::new();
    for (f, &threshold) in thresholds.iter().enumerate() {
        controller
            .buffer_set(0, Box::new(Buffer::from_interleaved(frame(f), 1, SR)))
            .expect("buffer_set");
        controller.set_control_bus(0, threshold).expect("set bus");
        let mut buf = vec![0.0f32; BLOCK];
        world.fill(&mut buf, 1);
        assert!(
            buf.iter().all(|&s| s == buf[0]),
            "{name}: block {f} is not uniform: {buf:?}"
        );
        outs.push(buf[0]);
    }
    outs
}

/// Pin every frame's sum: nothing fires at the sums, everything fires just below them (a zero wait
/// lets consecutive frames fire).
fn assert_sums(name: &str, frame: fn(usize) -> Vec<f32>, props: &[f32], sums: &[u32]) {
    let at: Vec<f32> = sums.iter().map(|&s| f32::from_bits(s)).collect();
    let below: Vec<f32> = at.iter().map(|&s| next_down(s)).collect();
    assert_eq!(
        run_detector(name, frame, props, 0.0, &at),
        vec![0.0; sums.len()],
        "{name}: a frame fired at its own sum"
    );
    assert_eq!(
        run_detector(name, frame, props, 0.0, &below),
        vec![1.0; sums.len()],
        "{name}: a frame did not fire just below its sum"
    );
}

/// With every sum over the threshold, a wait of 144 samples (3 ms) outlasts the next block but not
/// the one after, so the detector fires on every other block.
fn assert_waits(name: &str, props: &[f32], waits: &[u32]) {
    let got = run_detector(name, onset_frame, props, 0.003, &[-1.0e30; 6]);
    let got: Vec<u32> = got.iter().map(|s| s.to_bits()).collect();
    assert_eq!(got, waits, "{name}: fire pattern");
}

const JA_PROPS: [f32; 4] = [0.5, 0.25, 2.0, 1.0];
const HF_PROPS: [f32; 2] = [1.0, 0.5];

#[test]
fn pv_jensen_andersen_matches_scsynth() {
    assert_sums("PV_JensenAndersen", onset_frame, &JA_PROPS, &JA_SUMS);
    assert_sums(
        "PV_JensenAndersen",
        general_frame,
        &JA_PROPS,
        &JA_GENERAL_SUMS,
    );
    assert_waits("PV_JensenAndersen", &JA_PROPS, &JA_WAITS);
}

#[test]
fn pv_hainsworth_foote_matches_scsynth() {
    assert_sums("PV_HainsworthFoote", onset_frame, &HF_PROPS, &HF_SUMS);
    assert_sums(
        "PV_HainsworthFoote",
        general_frame,
        &HF_PROPS,
        &HF_GENERAL_SUMS,
    );
    assert_waits("PV_HainsworthFoote", &HF_PROPS, &HF_WAITS);
}

#[test]
fn onset_detectors_size_themselves_from_an_fft_chain() {
    // `FFT(LocalBuf(256), Impulse(20))`: the detector reads the chain buffer from the FFT's
    // constructor output, as scsynth's does, so it keeps a previous frame and fires on the clicks.
    // Silence between clicks gives nothing to fire on.
    for (name, props) in [
        ("PV_HainsworthFoote", &[1.0f32, 0.0][..]),
        ("PV_JensenAndersen", &[0.0, 0.0, 0.0, 1.0][..]),
    ] {
        let mut detector = vec![u(2)];
        detector.extend(props.iter().map(|&p| c(p)));
        detector.push(c(0.01));
        detector.push(c(0.04));
        let units = vec![
            UnitSpec::new(
                "LocalBuf",
                Rate::Scalar,
                vec![c(1.0), c(ONSET_FRAME as f32)],
                1,
            ),
            UnitSpec::new("Impulse", Rate::Audio, vec![c(20.0), c(0.0)], 1),
            UnitSpec::new(
                "FFT",
                Rate::Control,
                vec![u(0), u(1), c(0.5), c(0.0), c(1.0), c(0.0)],
                1,
            ),
            UnitSpec::new(name, Rate::Audio, detector, 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(3)], 0),
        ];
        let (_c, mut world) = engine_with(units, 1, vec![0.0; 1]);
        let mut buf = vec![0.0f32; BLOCK * 200];
        world.fill(&mut buf, 1);
        let fired = buf.chunks(BLOCK).filter(|b| b[0] == 1.0).count();
        assert!(
            (2..20).contains(&fired),
            "{name}: fired on {fired} blocks over 200"
        );
    }
}

/// `RunningSum`'s input: values of every sign that do not sum exactly, so the running sum rounds.
fn rs_input(i: usize) -> f32 {
    ((i * 37) % 101) as f32 / 101.0 - 0.4
}

/// `RunningSum(in, numsamp)` at `rate` over [`rs_input`] (read from a buffer by a counter), for
/// `blocks` blocks.
fn running_sum(rate: Rate, numsamp: f32, blocks: usize) -> Vec<f32> {
    let len = BLOCK * blocks;
    let mut units = vec![
        UnitSpec::new(
            "Phasor",
            Rate::Audio,
            vec![c(0.0), c(1.0), c(0.0), c(len as f32), c(0.0)],
            1,
        ),
        UnitSpec::new("BufRd", Rate::Audio, vec![c(0.0), u(0), c(1.0), c(1.0)], 1),
        UnitSpec::new("RunningSum", rate, vec![u(1), c(numsamp)], 1),
    ];
    // A control-rate result is carried into the block exactly by adding zero at audio rate.
    units.push(UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![u(2), c(0.0)],
        num_outputs: 1,
        special_index: 0,
    });
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(3)], 0));
    let (_c, mut world) = engine_with(units, 1, (0..len).map(rs_input).collect());
    let mut buf = vec![0.0f32; len];
    world.fill(&mut buf, 1);
    buf
}

#[test]
fn running_sum_matches_scsynth() {
    // Audio rate, a 40-sample window that wraps mid-block.
    let got: Vec<u32> = running_sum(Rate::Audio, 40.0, 4)
        .iter()
        .map(|s| s.to_bits())
        .collect();
    assert_eq!(got, RUNNING_SUM_AR, "audio rate");

    // Control rate: one input (the block's first sample) per block, a 3-block window.
    let got: Vec<u32> = running_sum(Rate::Control, 3.0, 8)
        .chunks(BLOCK)
        .map(|b| b[0].to_bits())
        .collect();
    assert_eq!(got, RUNNING_SUM_KR, "control rate");
}

#[test]
fn running_sum_without_a_window_is_silent() {
    // A negative window fails scsynth's allocation (the unit is cleared); a zero one would loop
    // forever there, and outputs zero here.
    for numsamp in [-1.0, 0.0] {
        let out = running_sum(Rate::Audio, numsamp, 2);
        assert!(out.iter().all(|&s| s == 0.0), "numsamp {numsamp}: {out:?}");
    }
}

// Bit patterns from scsynth's `FeatureDetection.cpp`, as described in the module docs.

const JA_SUMS: [u32; 6] = [
    0x3f8d3ec3, 0xbe300c19, 0xb9b04c00, 0xbc1d0d50, 0x3b2057c0, 0x3be732c0,
];
const JA_WAITS: [u32; 6] = [
    0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000,
];
const HF_SUMS: [u32; 6] = [
    0x41386a19, 0x3fa8b726, 0x3fa89288, 0x3fa860d0, 0x3fa8bb6a, 0x3fa8b662,
];
const HF_WAITS: [u32; 6] = [
    0x3f800000, 0x00000000, 0x3f800000, 0x00000000, 0x3f800000, 0x00000000,
];
const JA_GENERAL_SUMS: [u32; 6] = [
    0x3fb8d69c, 0xbeb808e4, 0x3b34dec0, 0xbcb58e78, 0xbc837b88, 0x3c60f260,
];
const HF_GENERAL_SUMS: [u32; 6] = [
    0x414d56fe, 0x3ef947a7, 0x3eb51b57, 0x3e6ceb55, 0x3e657f32, 0x3e8d350e,
];
const RUNNING_SUM_AR: [u32; 256] = [
    0xbecccccd, 0xbede0920, 0xbdced3e4, 0xbecdd059, 0xbeac5b3f, 0x3dc2a950, 0xbddafe7c, 0x3d6b3740,
    0x3f168ef4, 0x3ef86562, 0x3f3f9eaa, 0x3ec1a5c1, 0x3ebf9ea8, 0x3f3c9404, 0x3eee41e2, 0x3f0f761b,
    0x3f82c9c1, 0x3f597926, 0x3f859383, 0x3fcd4e92, 0x3fc3edbf, 0x3fe9710a, 0x3fbdd874, 0x3fc123fc,
    0x3ff353a3, 0x3fd46768, 0x3fe45f4c, 0x40119da7, 0x40087db7, 0x4016cfd6, 0x403c9404, 0x4039ca42,
    0x404e728f, 0x403a8ceb, 0x403e1956, 0x405917d1, 0x404b885b, 0x40556af4, 0x4076bf9c, 0x406f8654,
    0x408cac5a, 0x4081958a, 0x406cfd74, 0x408b67ea, 0x4080511a, 0x406a7495, 0x408a237a, 0x407e1955,
    0x4067ebb5, 0x4088df0b, 0x407b9077, 0x4092b16c, 0x40879a9d, 0x4079079a, 0x40916cfd, 0x4086562d,
    0x40767eba, 0x4090288e, 0x408511be, 0x4073f5dc, 0x408ee41e, 0x4083cd4e, 0x4098b67f, 0x408d9faf,
    0x408288df, 0x4097720f, 0x408c5b3f, 0x4081446f, 0x40962d9f, 0x408b16cf, 0x407fffff, 0x4069d260,
    0x4053a4c0, 0x407d7721, 0x40674982, 0x40511be2, 0x407aee42, 0x4064c0a2, 0x404e9303, 0x40786564,
    0x406237c0, 0x40860510, 0x4075dc7f, 0x405faee0, 0x4084c0a0, 0x407353a1, 0x405d2601, 0x40837c31,
    0x4070cac3, 0x405a9d24, 0x408237c2, 0x406e41e4, 0x408c0a22, 0x4080f352, 0x406bb904, 0x408ac5b2,
    0x407f5dc4, 0x40693025, 0x40898143, 0x407cd4e6, 0x4066a746, 0x405079a6, 0x403a4c07, 0x40641e67,
    0x404df0c7, 0x4037c327, 0x40619587, 0x404b67e8, 0x40353a49, 0x405f0ca9, 0x4048df09, 0x4072b169,
    0x405c83ca, 0x4046562a, 0x4070288a, 0x4059faea, 0x4043cd4a, 0x406d9fab, 0x4057720b, 0x4041446b,
    0x406b16ce, 0x4054e92e, 0x407ebb8f, 0x40688def, 0x4052604f, 0x407c32af, 0x40660510, 0x404fd771,
    0x4079a9d1, 0x40637c31, 0x404d4e91, 0x407720f2, 0x4060f352, 0x408562d9, 0x40749812, 0x405e6a72,
    0x40841e69, 0x40720f32, 0x405be192, 0x4082d9f9, 0x406f8652, 0x408cac59, 0x40819589, 0x406cfd72,
    0x408b67e9, 0x40805119, 0x406a7493, 0x408a2379, 0x407e1953, 0x4067ebb3, 0x4088df0a, 0x407b9075,
    0x4092b16b, 0x40879a9c, 0x40790798, 0x40916cfc, 0x4086562c, 0x40767eb8, 0x4090288d, 0x408511bd,
    0x4073f5dc, 0x408ee41e, 0x4083cd4e, 0x4098b67f, 0x408d9faf, 0x408288df, 0x4097720f, 0x408c5b3f,
    0x4081446f, 0x40962d9f, 0x408b16cf, 0x407fffff, 0x4069d260, 0x4053a4c0, 0x407d7721, 0x40674982,
    0x40511be2, 0x407aee42, 0x4064c0a2, 0x404e9303, 0x40786564, 0x406237c4, 0x40860512, 0x4075dc83,
    0x405faee4, 0x4084c0a2, 0x407353a5, 0x405d2605, 0x40837c33, 0x4070cac7, 0x405a9d28, 0x408237c4,
    0x406e41e8, 0x408c0a24, 0x4080f354, 0x406bb908, 0x408ac5b4, 0x407f5dc8, 0x40693029, 0x40898145,
    0x407cd4e8, 0x4066a748, 0x405079a8, 0x403a4c09, 0x40641e69, 0x404df0c9, 0x4037c329, 0x40619589,
    0x404b67ea, 0x40353a4b, 0x405f0cab, 0x4048df0b, 0x4072b16b, 0x405c83cc, 0x4046562c, 0x4070288c,
    0x4059faec, 0x4043cd4c, 0x406d9fad, 0x4057720d, 0x4041446d, 0x406b16ce, 0x4054e92e, 0x407ebb8f,
    0x40688def, 0x4052604f, 0x407c32af, 0x40660510, 0x404fd771, 0x4079a9d1, 0x40637c31, 0x404d4e91,
    0x407720f2, 0x4060f352, 0x408562d9, 0x40749812, 0x405e6a72, 0x40841e69, 0x40720f32, 0x405be192,
    0x4082d9f9, 0x406f8652, 0x408cac59, 0x40819589, 0x406cfd72, 0x408b67e9, 0x40805119, 0x406a7493,
    0x408a2379, 0x407e1953, 0x4067ebb3, 0x4088df0a, 0x407b9075, 0x4092b16b, 0x40879a9c, 0x40790798,
];
const RUNNING_SUM_KR: [u32; 8] = [
    0xbecccccd, 0xbeb57b30, 0x3e0be9ae, 0x3ef25016, 0x3f4f55ab, 0x3e160d2d, 0x3ef761d6, 0xbe3885d6,
];
