//! The buffer-kernel convolvers of scsynth's `Convolution.cpp` and the partitioned convolver of its
//! `PartitionedConvolution.cpp`, rendered through the engine and checked against those source files
//! run over the same inputs.
//!
//! The reference values come from scsynth's own code compiled against a minimal plugin environment
//! whose `scfft` layer is `SC_fftlib.cpp`'s FFTW path (the Linux build) with the transform computed as
//! a double-precision DFT. The engine transforms in single precision, so the FFT-based units are
//! compared to a tolerance; `Convolution3` has no transform and is pinned to the bit.
//!
//! The FFT-based units require the default `fft` feature.

use plyphon::{
    AddAction, Buffer, BuildError, Event, InputRef, Options, Param, ROOT_GROUP_ID, Rate, RateInfo,
    SynthDef, UnitRegistry, UnitSpec, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
/// The input signal's buffer.
const SIGNAL_BUF: usize = 0;

/// A deterministic, noise-like signal in `[-0.5, 0.5)` from a linear congruential generator.
fn test_signal(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// The input every case convolves (the reference's `test_signal(4096)`).
fn signal() -> Vec<f32> {
    test_signal(4096, 0x1234_5678)
}

/// A kernel of `len` noise samples decaying by `decay` per sample.
fn kernel(len: usize, seed: u32, decay: f32) -> Vec<f32> {
    let mut k = test_signal(len, seed);
    let mut g = 1.0f32;
    for x in &mut k {
        *x *= g;
        g *= decay;
    }
    k
}

/// Where a case's unit input comes from.
#[derive(Clone, Copy)]
enum Src {
    /// The test signal, played from its buffer at audio rate.
    Signal,
    /// A constant.
    Const(f32),
    /// A control parameter set before every block from `f(block)`.
    Sched(fn(usize) -> f32),
}

/// One rendering: the unit under test with its inputs, at `rate`, with `outputs` outputs.
struct Case {
    unit: &'static str,
    rate: Rate,
    inputs: Vec<Src>,
    outputs: usize,
    buffers: Vec<(usize, Vec<f32>)>,
    blocks: usize,
}

/// What a rendering produced: each output channel, and whether the synth freed itself because the
/// unit reported done.
struct Rendered {
    channels: Vec<Vec<f32>>,
    ended: bool,
}

/// Render `case`: `PlayBuf -> unit -> [K2A] -> Out.ar(0, ...)` plus a `FreeSelfWhenDone` on the unit,
/// setting every scheduled parameter before each block. A control-rate unit's outputs pass through
/// `K2A`, whose block `b` starts at the unit's block `b - 1` value.
fn render(case: &Case) -> Rendered {
    let mut units = vec![UnitSpec::new(
        "PlayBuf",
        Rate::Audio,
        vec![
            InputRef::Constant(SIGNAL_BUF as f32),
            InputRef::Constant(1.0),
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
            InputRef::Constant(1.0),
            InputRef::Constant(0.0),
        ],
        1,
    )];
    let mut params = vec![];
    let mut schedules = vec![];
    let inputs = case
        .inputs
        .iter()
        .map(|src| match *src {
            Src::Signal => InputRef::Unit { unit: 0, output: 0 },
            Src::Const(v) => InputRef::Constant(v),
            Src::Sched(f) => {
                let index = params.len() as u32;
                params.push(Param::control(format!("p{index}"), f(0)));
                schedules.push(f);
                InputRef::Param(index)
            }
        })
        .collect();
    units.push(UnitSpec::new(case.unit, case.rate, inputs, case.outputs));
    let mut outs = vec![InputRef::Constant(0.0)];
    for o in 0..case.outputs as u32 {
        if case.rate == Rate::Control {
            units.push(UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 1, output: o }],
                1,
            ));
            outs.push(InputRef::Unit {
                unit: units.len() as u32 - 1,
                output: 0,
            });
        } else {
            outs.push(InputRef::Unit { unit: 1, output: o });
        }
    }
    units.push(UnitSpec::new("Out", Rate::Audio, outs, 0));
    units.push(UnitSpec::new(
        "FreeSelfWhenDone",
        Rate::Control,
        vec![InputRef::Unit { unit: 1, output: 0 }],
        1,
    ));
    let def = SynthDef {
        name: "conv-family".to_string(),
        params,
        units,
    };

    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: case.outputs,
        block_size: BLOCK,
        ..Options::default()
    });
    controller
        .buffer_set(
            SIGNAL_BUF,
            Box::new(Buffer::from_interleaved(signal(), 1, SR)),
        )
        .unwrap();
    for (index, data) in &case.buffers {
        controller
            .buffer_set(
                *index,
                Box::new(Buffer::from_interleaved(data.clone(), 1, SR)),
            )
            .unwrap();
    }
    controller.add_synthdef(def);
    let node = controller
        .synth_new("conv-family", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut channels = vec![Vec::with_capacity(case.blocks * BLOCK); case.outputs];
    let mut buf = vec![0.0f32; BLOCK * case.outputs];
    for b in 0..case.blocks {
        for (i, f) in schedules.iter().enumerate() {
            controller.set_control(node, i, f(b)).unwrap();
        }
        world.fill(&mut buf, case.outputs);
        for frame in buf.chunks_exact(case.outputs) {
            for (ch, &x) in channels.iter_mut().zip(frame) {
                ch.push(x);
            }
        }
    }
    let ended = std::iter::from_fn(|| nrt.poll()).any(|e| matches!(e, Event::NodeEnded(_)));
    Rendered { channels, ended }
}

/// Assert `got` equals the reference's bit patterns exactly.
fn assert_bits(label: &str, got: &[f32], expected: &[u32]) {
    assert_eq!(got.len(), expected.len(), "{label}: length");
    for (i, (g, &want)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            want,
            "{label}: sample {i} is {g}, the reference gives {}",
            f32::from_bits(want)
        );
    }
}

/// Assert `r` was silent from sample `from` on, and that the unit reported done.
fn assert_silenced(label: &str, r: &Rendered, from: usize) {
    assert!(r.ended, "{label}: the unit reports done");
    for ch in &r.channels {
        assert!(
            ch[from..].iter().all(|&s| s == 0.0),
            "{label}: silent from sample {from}"
        );
    }
}

/// Compile `def` at the World block, for the build-time contract tests.
fn compile(def: &SynthDef) -> Result<(), BuildError> {
    let rate_info = RateInfo::new(SR, BLOCK);
    def.compile(
        &UnitRegistry::with_builtins(),
        &rate_info,
        &rate_info,
        64,
        32,
        None,
        1,
    )
    .map(|_| ())
}

/// Compile a def of `unit` alone, over constant inputs.
fn compile_unit(unit: &str, rate: Rate, inputs: usize, outputs: usize) -> Result<(), BuildError> {
    compile(&SynthDef {
        name: "conv-build".to_string(),
        params: vec![],
        units: vec![UnitSpec::new(
            unit,
            rate,
            vec![InputRef::Constant(1.0); inputs],
            outputs,
        )],
    })
}

#[test]
fn convolution3_matches_scsynth_bit_for_bit() {
    // Audio rate over an audio input; `framesize` 0 takes the first kernel's 100 frames, which the
    // 64-sample block does not divide. The trigger at block 3 re-reads 100 samples of the 120-frame
    // kernel 2.
    let r = render(&Case {
        unit: "Convolution3",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Sched(|b| if b >= 3 { 2.0 } else { 1.0 }),
            Src::Sched(|b| if b == 3 { 1.0 } else { 0.0 }),
            Src::Const(0.0),
        ],
        outputs: 1,
        buffers: vec![
            (1, kernel(100, 0xabcd_ef01, 0.95)),
            (2, kernel(120, 0x5eed_1234, 0.96)),
        ],
        blocks: 8,
    });
    assert!(!r.ended, "Convolution3.ar runs");
    assert_bits("Convolution3.ar", &r.channels[0], &CONV3_AR);

    // Control rate over a control-rate input: one sample per block, the position running up to the
    // frame size itself before wrapping. `K2A` delays each value by a block.
    let r = render(&Case {
        unit: "Convolution3",
        rate: Rate::Control,
        inputs: vec![
            Src::Sched(|b| signal()[b]),
            Src::Sched(|b| if b >= 25 { 4.0 } else { 3.0 }),
            Src::Sched(|b| if b == 25 { 1.0 } else { 0.0 }),
            Src::Const(10.0),
        ],
        outputs: 1,
        buffers: vec![
            (3, kernel(10, 0x0bad_f00d, 0.8)),
            (4, kernel(10, 0x1357_9bdf, 0.85)),
        ],
        blocks: 41,
    });
    let per_block: Vec<f32> = (1..41).map(|b| r.channels[0][b * BLOCK]).collect();
    assert_bits("Convolution3.kr", &per_block, &CONV3_KR);

    // Audio rate over a control-rate input runs the one-sample calc, which writes each block's first
    // sample.
    let r = render(&Case {
        unit: "Convolution3",
        rate: Rate::Audio,
        inputs: vec![
            Src::Sched(|b| signal()[b]),
            Src::Const(3.0),
            Src::Const(0.0),
            Src::Const(10.0),
        ],
        outputs: 1,
        buffers: vec![(3, kernel(10, 0x0bad_f00d, 0.8))],
        blocks: 24,
    });
    let firsts: Vec<f32> = (0..24).map(|b| r.channels[0][b * BLOCK]).collect();
    assert_bits(
        "Convolution3.ar over a control input",
        &firsts,
        &CONV3_AR_KR_IN,
    );
}

#[test]
fn convolution3_without_a_kernel_or_a_usable_frame_is_silenced() {
    // No kernel buffer (`ConvGetBuffer` fails), and a frame smaller than the block over an audio
    // input (the reference's block calc copies the block into its `framesize`-sample input frame,
    // overrunning it).
    for (label, kernel_buf, framesize) in [("no buffer", 9.0, 16.0), ("small frame", 1.0, 50.0)] {
        let r = render(&Case {
            unit: "Convolution3",
            rate: Rate::Audio,
            inputs: vec![
                Src::Signal,
                Src::Const(kernel_buf),
                Src::Const(0.0),
                Src::Const(framesize),
            ],
            outputs: 1,
            buffers: vec![(1, kernel(64, 1, 0.9))],
            blocks: 4,
        });
        assert_silenced(&format!("Convolution3, {label}"), &r, 0);
    }
}

#[test]
fn convolution3_kr_over_an_audio_input_is_rejected() {
    // The reference runs its block calc for an audio-rate input whatever the unit's rate, writing a
    // whole block into a control-rate unit's single output sample.
    let def = SynthDef {
        name: "conv3-kr-audio".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(
                "Convolution3",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(8.0),
                ],
                1,
            ),
        ],
    };
    assert_eq!(
        compile(&def),
        Err(BuildError::UnsupportedRate(Rate::Control))
    );
    assert_eq!(compile_unit("Convolution3", Rate::Control, 4, 1), Ok(()));
    assert_eq!(
        compile_unit("Convolution3", Rate::Audio, 5, 1),
        Err(BuildError::WrongInputCount)
    );
}

// Reference values: scsynth's own code over the same inputs (see the module docs).

#[rustfmt::skip]
const CONV3_AR: [u32; 512] = [
    0x3d64_d30c, 0xbc7b_c3cd, 0xbd62_6e68, 0x3e55_6a8e, 0x3df2_92b5, 0xbe96_0968, 0xbe79_bf87,
    0xbe39_213b, 0xbca0_ddd9, 0x3dd4_5a2e, 0x3e52_264a, 0x3e34_99ae, 0x3e89_b4e3, 0x3e4a_4083,
    0x3dbb_4bc6, 0xbebc_1cfd, 0xbd56_d9dc, 0xbda3_472c, 0xbec7_9a4b, 0xbe03_cd18, 0x3df8_7a99,
    0xbe43_f548, 0xbcb1_c56b, 0x3bab_a41e, 0xbe09_d6cb, 0x3cfc_d203, 0x3e61_0dd5, 0xbd95_c2b8,
    0x3dd1_fd54, 0x3ed7_eb4b, 0x3d23_c6ac, 0xbe94_aabe, 0xbd2d_f2d4, 0xbe09_efb2, 0xbc59_c7b2,
    0xbdd8_b24b, 0xbdde_cb1b, 0xbdfb_ee54, 0x3d7c_bb58, 0xbdc4_20c2, 0x3d8b_5fb6, 0x3e0b_bcda,
    0xbde9_2453, 0xbea8_e002, 0xbdbf_a57d, 0x3d66_c40f, 0x3d8f_f9a7, 0x3e59_db59, 0x3d9d_79fc,
    0x3db1_3bae, 0x3da0_0bab, 0xbd8b_1c03, 0xbeec_3bc9, 0xbe7d_e440, 0x3e99_cc41, 0x3e19_7422,
    0xbf11_76e5, 0xbe10_122e, 0x3dff_322f, 0xbdd2_2714, 0x3dff_10ed, 0x3f04_52d8, 0x3e96_d823,
    0x3edf_6519, 0x3e1a_5879, 0xbe4d_2f1f, 0x3e5b_b510, 0x3f10_037a, 0xbf08_a4db, 0xbec1_9300,
    0xbd3f_cc33, 0xbd14_3db5, 0xbe48_5cd3, 0xbdc9_871e, 0xbe09_8b7a, 0x3f0c_5e3e, 0x3ea8_5f1f,
    0x3ba2_935c, 0xbe8d_3171, 0x3eb1_9726, 0x3e5d_dcde, 0x3ec0_c4cc, 0x3e13_c1bb, 0x3e71_6f90,
    0xbc8f_7e92, 0xbe84_6174, 0xbed1_5fb5, 0xbc0c_c7d8, 0xbe16_9df5, 0xbe6c_def4, 0x3e5a_4157,
    0x3dbf_9119, 0xbbb0_39f8, 0xbe00_3ba5, 0xbd90_f0de, 0x3e92_9a88, 0x3e45_c464, 0x3e7e_6849,
    0x3db9_a791, 0xbd67_5758, 0x3daa_d4ea, 0x3e43_0dc4, 0xbead_396c, 0x3f01_c2cc, 0x3e90_af1a,
    0xbeeb_179c, 0xbed3_b443, 0xbeb3_11fb, 0xbe8e_b69c, 0x3e8d_93c6, 0x3e08_1157, 0xbe6d_170c,
    0x3e01_6ba0, 0x3eca_050f, 0x3ea6_aba5, 0xbe57_58f4, 0x3de8_9c90, 0x3dbd_a101, 0xbdee_f32f,
    0x3e07_0f01, 0xbda2_b53f, 0xbeb0_be41, 0xbe83_d890, 0x3d55_f1fd, 0xbe79_f8e9, 0xbe72_1d0b,
    0x3cea_55df, 0x3c5b_b936, 0x3e94_166f, 0x3f52_e70a, 0x3e29_0052, 0xbf21_7e21, 0x3e88_9ff2,
    0xbe87_b2b9, 0xbc98_3afa, 0x3b16_229c, 0xbe88_a412, 0xbee9_7155, 0x3eba_f346, 0xbc51_af40,
    0xbea5_71eb, 0x3c88_0d30, 0xbe53_30bd, 0xbe79_e8f2, 0xbe09_0b52, 0x3ead_1134, 0x3e37_fe1b,
    0x3e86_e4a0, 0x3ea5_e2a3, 0xbe81_4cb0, 0xbe94_bc2f, 0xbe79_bfe1, 0xbebc_6d12, 0xbee8_23e5,
    0x3f0a_be76, 0x3f16_5010, 0xbf42_f315, 0xbe6e_bec7, 0x3e5d_f9a8, 0xbe84_1d5a, 0x3df1_8533,
    0x3f39_6dfd, 0xbe58_dc38, 0x3dc0_d587, 0x3e7b_7259, 0xbee1_02b8, 0x3e88_df06, 0x3f4a_f439,
    0xbf32_c399, 0xbf45_fae3, 0xbe00_dda2, 0x3d2e_2184, 0xbc91_342d, 0xbeb0_7413, 0xbe64_6b9c,
    0x3efe_5ddd, 0x3e05_138a, 0xbe21_7435, 0xbf0f_cd82, 0x3ece_0720, 0x3f06_98d9, 0x3ee5_70b7,
    0x3dca_7cf5, 0x3de1_f52b, 0x3db2_c26c, 0xbeb0_7145, 0xbe8e_0241, 0xbc7b_b889, 0xbe3a_a0d6,
    0xbeb7_0529, 0x3e6b_ba56, 0xbe2d_e091, 0x3e43_fb80, 0x3e8d_df25, 0xbef7_e3b4, 0x3f14_7d89,
    0xbc8b_9930, 0x3ef2_4da2, 0x3e55_af4e, 0xbe08_a829, 0xbe47_f752, 0x3f09_3928, 0xbea5_591e,
    0x3f5c_babc, 0x3e62_8c55, 0x3e02_5489, 0xbf18_255c, 0xbe95_fa14, 0xbedd_6f54, 0x3f37_9346,
    0x3c9b_023e, 0xbf20_d1cc, 0xbe9e_ae96, 0x3f91_2840, 0xbda7_4669, 0xbeba_1fcb, 0xbe3b_1b02,
    0x3f0f_82b7, 0xbdec_d091, 0x3e30_9a36, 0x3d6d_ce0c, 0xbea0_9d0f, 0xbe70_6f34, 0xbd67_c0d8,
    0x3ca3_c492, 0xbe7b_6712, 0x3df5_24c8, 0xbd29_078a, 0x3f06_128f, 0x3ef4_9eb7, 0xbdb9_3148,
    0xbe10_4976, 0x3e43_6ebc, 0xbf0a_04e2, 0xbd28_73c2, 0x3e3a_5859, 0xbedb_a8a5, 0xbf0f_b61e,
    0x3f16_c7d2, 0xbe4c_fb24, 0xbe9b_8d3f, 0x3dab_7c57, 0x3e86_152f, 0xbe23_9261, 0xbea5_ce91,
    0x3f43_45e8, 0x3f3c_7d20, 0x3e10_daa6, 0x3ecb_db59, 0xbe3a_d1a2, 0xbe51_b00a, 0xbf8f_57c6,
    0xbeaf_c4ea, 0xbe75_1017, 0x3e61_6a97, 0x3d54_cb06, 0xbf69_54ee, 0xbeb6_5ba5, 0x3e40_2e1e,
    0xbf01_e2a7, 0x3f0c_b017, 0x3f19_1e82, 0x3e36_bcda, 0xbe86_3220, 0x3f76_f3f6, 0xbf64_8dcd,
    0x3ea2_a7f5, 0x3f44_7a82, 0xbed6_c150, 0xbf92_7055, 0xbd7f_7a45, 0xbe88_fbc9, 0x3e40_42f9,
    0xbf1f_0c21, 0xbee0_4bbb, 0x3ecd_8994, 0x3e8e_d2f6, 0x3eb6_8f39, 0xbf15_3d8b, 0x3e64_004d,
    0x3f5b_8d65, 0x3f0a_6a53, 0xbe8d_0c22, 0xbde9_51c7, 0x3ed5_0f23, 0xbf1a_116a, 0xbeba_c57a,
    0xbe97_2c27, 0xbe94_0bc7, 0xbefe_035d, 0x3eca_a3fe, 0xbea5_5a5b, 0x3e17_9f27, 0x3d07_471c,
    0xbe85_6ebb, 0x3f40_2044, 0xbd60_b868, 0x3d34_ce87, 0x3f16_7871, 0xbf1a_cbbe, 0xbeda_ab05,
    0x3dbf_0f02, 0xbe83_6dc0, 0x3f29_9d4b, 0xbe7c_f04c, 0x3e43_0415, 0xbf99_29de, 0xbee7_b847,
    0xbf4b_66c0, 0x3fa1_f100, 0xbf2c_e8c9, 0xbf26_3b96, 0xbf76_dc2f, 0x3fca_7832, 0xbf12_bc1a,
    0xbdc9_22f6, 0xbed0_a287, 0x3f6d_97c3, 0xbecd_adaa, 0xbee6_8ea2, 0x3e70_25bc, 0xbed0_dabc,
    0x3d3b_d744, 0xbeea_e18d, 0x3d8d_85a8, 0xbf31_e4a0, 0x3f2b_d4a7, 0xbe65_10e4, 0x3f4d_e9f8,
    0x3d26_5c4e, 0xbed9_0b0f, 0xbc4d_c304, 0x3eda_c9f1, 0xbdd2_7fee, 0xbe1e_1fa0, 0x3e0c_58fa,
    0xbf2d_1a63, 0xbf53_e622, 0x3e63_9c7e, 0xbd8a_a3bc, 0xbea7_9fb2, 0x3ea1_9b9a, 0x3e25_eccb,
    0xbea9_e40a, 0xbf1e_c10d, 0x3ea0_db23, 0x3f2b_3767, 0xbd7b_8172, 0x3f4a_52a8, 0xbe05_3521,
    0x3d86_805d, 0xbfa7_0336, 0xbf7e_5ab1, 0xbe9a_0448, 0x3d8b_b98f, 0x3e98_e8dd, 0xbfaa_5eb8,
    0xbdde_f9a8, 0x3e11_2504, 0x3da4_3612, 0x3f3b_7373, 0x3f85_9256, 0xbdb2_58d2, 0xbec2_a2b9,
    0x3f48_4bb2, 0x3e3f_295b, 0x3df7_9af4, 0x3f3b_2194, 0xbea4_5f6b, 0xbf98_cf56, 0xbf33_3c0a,
    0xbea4_167c, 0x3eba_9261, 0xbea4_84c4, 0xbf8e_0968, 0x3bb8_5818, 0x3ee7_1927, 0x3f35_5f64,
    0xbeb8_c2fc, 0xbd0d_4bba, 0x3f5d_a6ad, 0x3ef0_8651, 0x3e14_19d6, 0xbee3_61d2, 0x3e6c_05c4,
    0xbf90_04b1, 0xbec4_94ea, 0xbeb8_bad0, 0xbe9a_3af1, 0xbf34_02d2, 0x3f4a_2522, 0xbf53_e368,
    0xbefc_3af6, 0xbd27_f86b, 0xbdd2_a099, 0x3f7f_ebd5, 0xbea7_5d5c, 0x3e4b_d72a, 0x3f2c_e9a5,
    0xbf7d_1e3f, 0xbe15_e740, 0xbbcb_6c20, 0xbf03_096f, 0x3efb_e408, 0xbdfe_b2fc, 0x3ec2_956b,
    0xbf9e_7825, 0xbf3a_56e6, 0xbf6e_fb7f, 0x3fae_2548, 0xbf37_8683, 0xbf60_9a5b, 0x3cf6_3eb7,
    0x3fce_efc2, 0xbf86_e224, 0xbd9c_5a68, 0xbf46_fb17, 0x3f5d_8da0, 0xbd57_5291, 0xbef2_727d,
    0x3ed6_39d7, 0xbee0_f63c, 0xbe5a_9748, 0xbf4e_34c5, 0x3dc3_ad60, 0xbf45_e1dc, 0x3f81_90fc,
    0xbec6_9eae, 0x3f7b_4ffc, 0x3e20_45d2, 0x3e8b_ae06, 0xbe5e_2a15, 0x3d51_436b, 0xbb2f_a7a8,
    0x3dc0_c899, 0x3dc9_2920, 0xbf69_67dd, 0xbeeb_a8f9, 0xbe19_cb58, 0x3d59_c9f1, 0xbf1d_8fbb,
    0xbdff_0d39, 0x3ed9_9c88, 0xbd0b_4f1b, 0xbf7f_c0ba, 0xbd90_6775, 0x3ee8_2d8d, 0xbe51_b730,
    0x3f6c_0a3c, 0xbebd_4b13, 0x3e92_fe43, 0xbf9d_060e, 0xbfab_c7ae, 0xbeb5_9567, 0xba75_64c8,
    0x3e93_493e, 0xbfc4_57ae, 0xbea3_85cb, 0xbd20_8004, 0x3ee4_d23a, 0x3e71_da51, 0x3fa0_4a9d,
    0xbe21_1b52, 0xbf14_c9d8, 0x3eb0_1127, 0x3ded_dd9d, 0xbe5e_0845, 0x3f1e_2d31, 0xbf27_2774,
    0xbf39_abb6, 0xbf46_68ae, 0xbec7_906b, 0xbd39_3d25, 0xbd07_a2cd, 0xbf92_dc85, 0x3e2e_97c7,
    0x3e5c_1bb8, 0x3f6a_bdd1, 0xbeaa_cdc5, 0xbe8f_7ffd, 0x3f96_7402, 0x3f2e_0233, 0xbf42_f6d2,
    0xbf33_9762, 0x3e8b_aada, 0xbf55_b30b, 0xbf24_0312, 0x3e41_91cf, 0xbf71_b8a1, 0xbf5c_e8dc,
    0x3e87_43d1, 0xbf2e_7115, 0xbec7_a6c4, 0x3e05_1d40, 0xbe4b_c788, 0x3f84_04f7, 0xbf04_ce57,
    0xbe80_95e2, 0x3ecc_d2a1, 0xbf0d_48ec, 0xbf37_2978, 0xbe83_b49f, 0xbebe_b118, 0x3f7d_f4af,
    0xbeba_b5e2, 0x3eae_3a8f, 0xbf2b_f40f, 0xbf86_5ba4, 0xbf56_5f77, 0x3f9b_dd95, 0xbdab_bb73,
    0xbf84_f307,
];

#[rustfmt::skip]
const CONV3_KR: [u32; 40] = [
    0xbb4d_f57a, 0x3d08_f618, 0xbde3_7bc5, 0x3e06_94c9, 0xbd7d_be8c, 0x3dda_36bf, 0xbe04_7ce5,
    0x3cdf_fcec, 0xbddf_1f64, 0x3c6b_8bc8, 0x0000_0000, 0xbda5_f943, 0x3dfe_2fd7, 0xbe27_e92a,
    0x3e9e_7d82, 0xbe0c_795b, 0x3e8b_685d, 0xbde1_5c5a, 0xbcc3_726e, 0xbe0c_77bb, 0xbd43_09a7,
    0x0000_0000, 0xbd85_43ea, 0x3e00_02b8, 0xbe7f_ffd9, 0x3ea5_b754, 0xbe47_9a6d, 0x3ea8_a398,
    0xbd84_0588, 0xbe15_2029, 0xbde5_cc68, 0x3ceb_d058, 0x0000_0000, 0xbd29_56da, 0x3d6c_4d6b,
    0xbe6e_433c, 0x3e9a_3a71, 0xbe3c_2d4f, 0x3e72_d77b, 0xbdf5_4d17,
];

#[rustfmt::skip]
const CONV3_AR_KR_IN: [u32; 24] = [
    0xbb4d_f57a, 0x3d08_f618, 0xbde3_7bc5, 0x3e06_94c9, 0xbd7d_be8c, 0x3dda_36bf, 0xbe04_7ce5,
    0x3cdf_fcec, 0xbddf_1f64, 0x3c6b_8bc8, 0x0000_0000, 0xbda5_f943, 0x3dfe_2fd7, 0xbe27_e92a,
    0x3e9e_7d82, 0xbe0c_795b, 0x3e8b_685d, 0xbde1_5c5a, 0xbcc3_726e, 0xbe0c_77bb, 0xbd43_09a7,
    0x0000_0000, 0xbd85_43ea, 0x3e00_02b8,
];
