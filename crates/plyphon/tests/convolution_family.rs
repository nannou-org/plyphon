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

/// Tolerance for the FFT-based units, relative to the larger of `1` and the reference value: the
/// engine's single-precision transforms against the reference's double-precision ones.
const FFT_TOL: f32 = 2e-5;

/// Assert every `stride`-th sample of `out` from `first` matches `expected` to `tol`, relative to the
/// larger of `1` and the expected magnitude. `expected` holds the reference values' bit patterns.
fn assert_points(
    label: &str,
    out: &[f32],
    first: usize,
    stride: usize,
    expected: &[u32],
    tol: f32,
) {
    let points: Vec<(usize, f32)> = (first..out.len())
        .step_by(stride)
        .map(|i| (i, out[i]))
        .collect();
    assert_eq!(points.len(), expected.len(), "{label}: point count");
    for ((i, got), &want) in points.into_iter().zip(expected) {
        let want = f32::from_bits(want);
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tol * scale,
            "{label}: sample {i} is {got}, the reference gives {want}"
        );
    }
}

/// Kernel switch for the triggered cases: buffer `2` for blocks `from..to`, else buffer `1`.
fn kernel_1_2(from: usize, to: usize, b: usize) -> f32 {
    if (from..to).contains(&b) { 2.0 } else { 1.0 }
}

#[test]
fn convolution2_matches_scsynth_across_kernel_swaps() {
    // A 100-frame kernel zero-padded into a 128-sample frame, swapped on a trigger for a 200-frame
    // kernel truncated to the frame, then back. Each trigger re-reads the buffer the kernel input
    // names in that block, and the frame completed in that block already uses it.
    let r = render(&Case {
        unit: "Convolution2",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Sched(|b| kernel_1_2(9, 20, b)),
            Src::Sched(|b| if b == 9 || b == 20 { 1.0 } else { 0.0 }),
            Src::Const(128.0),
        ],
        outputs: 1,
        buffers: vec![
            (1, kernel(100, 0xabcd_ef01, 0.97)),
            (2, kernel(200, 0x5eed_1234, 0.985)),
        ],
        blocks: 32,
    });
    assert!(!r.ended, "Convolution2 runs");
    assert_points("Convolution2", &r.channels[0], 3, 17, &CONV2, FFT_TOL);

    // A `framesize` of `0` takes the kernel buffer's frame count.
    let r = render(&Case {
        unit: "Convolution2",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Const(3.0),
            Src::Const(0.0),
            Src::Const(0.0),
        ],
        outputs: 1,
        buffers: vec![(3, kernel(256, 0x0bad_f00d, 0.99))],
        blocks: 24,
    });
    assert!(!r.ended, "Convolution2 with the buffer's frame count runs");
    assert_points(
        "Convolution2 (framesize from the buffer)",
        &r.channels[0],
        5,
        19,
        &CONV2_BUFFER_FRAMESIZE,
        FFT_TOL,
    );
}

#[test]
fn convolution2l_matches_scsynth_through_crossfades() {
    // A trigger at block 5 loads kernel 2 into the idle spectrum and crossfades to it over three
    // frames; a trigger at block 18 loads kernel 1 back with a one-frame crossfade (whose second
    // half is taken whole from the new kernel's result).
    let r = render(&Case {
        unit: "Convolution2L",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Sched(|b| kernel_1_2(5, 18, b)),
            Src::Sched(|b| if b == 5 || b == 18 { 1.0 } else { 0.0 }),
            Src::Const(128.0),
            Src::Sched(|b| if b < 18 { 3.0 } else { 1.0 }),
        ],
        outputs: 1,
        buffers: vec![
            (1, kernel(128, 0xabcd_ef01, 0.97)),
            (2, kernel(128, 0x5eed_1234, 0.985)),
        ],
        blocks: 32,
    });
    assert!(!r.ended, "Convolution2L runs");
    assert_points("Convolution2L", &r.channels[0], 2, 13, &CONV2L, FFT_TOL);
}

#[test]
fn convolution2_without_a_kernel_or_a_usable_frame_is_silenced() {
    let conv2 = |kernel_buf: f32, framesize: f32| Case {
        unit: "Convolution2",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Const(kernel_buf),
            Src::Const(0.0),
            Src::Const(framesize),
        ],
        outputs: 1,
        buffers: vec![(1, kernel(128, 1, 0.9))],
        blocks: 8,
    };
    // No kernel buffer when the synth starts (`ConvGetBuffer` fails).
    assert_silenced("Convolution2, no buffer", &render(&conv2(9.0, 128.0)), 0);
    // A frame smaller than the block.
    assert_silenced("Convolution2, small frame", &render(&conv2(1.0, 32.0)), 0);
    // A frame the transform sizes cannot hold.
    assert_silenced("Convolution2, odd frame", &render(&conv2(1.0, 192.0)), 0);

    // A trigger naming a missing buffer silences the unit from that block on.
    let r = render(&Case {
        unit: "Convolution2",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Sched(|b| if b >= 4 { 9.0 } else { 1.0 }),
            Src::Sched(|b| if b == 4 { 1.0 } else { 0.0 }),
            Src::Const(64.0),
        ],
        outputs: 1,
        buffers: vec![(1, kernel(64, 1, 0.9))],
        blocks: 8,
    });
    assert!(
        r.channels[0][..4 * BLOCK].iter().any(|&s| s != 0.0),
        "Convolution2 plays before the failed trigger"
    );
    assert_silenced("Convolution2, failed re-read", &r, 4 * BLOCK);

    assert_silenced(
        "Convolution2L, no buffer",
        &render(&Case {
            unit: "Convolution2L",
            rate: Rate::Audio,
            inputs: vec![
                Src::Signal,
                Src::Const(9.0),
                Src::Const(0.0),
                Src::Const(128.0),
                Src::Const(1.0),
            ],
            outputs: 1,
            buffers: vec![],
            blocks: 4,
        }),
        0,
    );
}

#[test]
fn convolution2_rates_and_arities() {
    // `Convolution2`'s calc always runs a whole audio block; `Convolution2L`'s runs the unit's own
    // calc length.
    assert_eq!(
        compile_unit("Convolution2", Rate::Control, 4, 1),
        Err(BuildError::UnsupportedRate(Rate::Control))
    );
    assert_eq!(compile_unit("Convolution2L", Rate::Control, 5, 1), Ok(()));
    assert_eq!(
        compile_unit("Convolution2", Rate::Audio, 5, 1),
        Err(BuildError::WrongInputCount)
    );
    assert_eq!(
        compile_unit("Convolution2L", Rate::Audio, 4, 1),
        Err(BuildError::WrongInputCount)
    );
}

/// Tolerance for `StereoConvolution2L`, whose repeated unnormalized in-place transforms amplify the
/// single- versus double-precision difference.
const STEREO_TOL: f32 = 1e-3;

#[test]
fn stereo_convolution2l_matches_scsynth_transform_for_transform() {
    // The reference's `scfft` wiring (see the unit's docs) makes the left output a product spectrum
    // and repeatedly inverse-transforms kernel set B in place, so after the first trigger's two-frame
    // crossfade the output grows by orders of magnitude, until the second trigger reloads set A.
    // Both channels follow the reference throughout.
    let r = render(&Case {
        unit: "StereoConvolution2L",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Sched(|b| if (6..20).contains(&b) { 3.0 } else { 1.0 }),
            Src::Sched(|b| if (6..20).contains(&b) { 4.0 } else { 2.0 }),
            Src::Sched(|b| if b == 6 || b == 20 { 1.0 } else { 0.0 }),
            Src::Const(128.0),
            Src::Const(2.0),
        ],
        outputs: 2,
        buffers: vec![
            (1, kernel(128, 0xabcd_ef01, 0.97)),
            (2, kernel(128, 0x5eed_1234, 0.985)),
            (3, kernel(128, 0x0bad_f00d, 0.95)),
            (4, kernel(128, 0x1357_9bdf, 0.96)),
        ],
        blocks: 32,
    });
    assert!(!r.ended, "StereoConvolution2L runs");
    assert_points(
        "StereoConvolution2L left",
        &r.channels[0],
        1,
        11,
        &STEREO_L,
        STEREO_TOL,
    );
    assert_points(
        "StereoConvolution2L right",
        &r.channels[1],
        1,
        11,
        &STEREO_R,
        STEREO_TOL,
    );
}

#[test]
fn stereo_convolution2l_without_its_kernels_is_silenced() {
    let case = |kernel_r: fn(usize) -> f32, trigger: fn(usize) -> f32| Case {
        unit: "StereoConvolution2L",
        rate: Rate::Audio,
        inputs: vec![
            Src::Signal,
            Src::Const(1.0),
            Src::Sched(kernel_r),
            Src::Sched(trigger),
            Src::Const(64.0),
            Src::Const(1.0),
        ],
        outputs: 2,
        buffers: vec![(1, kernel(64, 1, 0.9))],
        blocks: 8,
    };
    // No right kernel when the synth starts.
    assert_silenced(
        "StereoConvolution2L, no buffer",
        &render(&case(|_| 9.0, |_| 0.0)),
        0,
    );
    // A missing right kernel at a trigger: the reference clears the outputs and marks the unit
    // done, then dereferences the missing buffer; here the unit stops there.
    let r = render(&case(
        |b| if b >= 4 { 9.0 } else { 1.0 },
        |b| if b == 4 { 1.0 } else { 0.0 },
    ));
    assert!(
        r.channels[1][..4 * BLOCK].iter().any(|&s| s != 0.0),
        "StereoConvolution2L plays before the failed trigger"
    );
    assert_silenced("StereoConvolution2L, failed re-read", &r, 4 * BLOCK);
}

#[test]
fn stereo_convolution2l_rates_and_arities() {
    // The reference's calc always runs a whole audio block.
    assert_eq!(
        compile_unit("StereoConvolution2L", Rate::Control, 6, 2),
        Err(BuildError::UnsupportedRate(Rate::Control))
    );
    assert_eq!(
        compile_unit("StereoConvolution2L", Rate::Audio, 5, 2),
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

#[rustfmt::skip]
const CONV2: [u32; 121] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0xbe30_bd05, 0xbdda_3107, 0x3dc1_7698,
    0x3d98_726e, 0x3f51_0ed1, 0x3e0a_cf3b, 0x3e84_d563, 0xbec8_acbd, 0x3e95_3a95, 0xbdea_7486,
    0xbe94_a2f5, 0xbe92_e2f4, 0x3deb_57c5, 0xbda7_f1a1, 0xbd59_10f2, 0xbe76_9123, 0xbe96_e6e6,
    0xbca4_a495, 0x3e15_beda, 0x3d90_4c0a, 0xbeb3_23bb, 0xbeff_fe69, 0x3e83_2c15, 0xbd4a_6000,
    0xbe90_ba89, 0x3f1c_426b, 0x3e29_e50d, 0xbd9c_1138, 0x3f03_26bf, 0x3b0c_de74, 0xbe7e_881c,
    0xbdd9_3bf8, 0xbf06_fd43, 0x3eb1_2c81, 0x3f17_3f03, 0xbcc0_cbdd, 0xbeeb_ea30, 0xbe76_f6c1,
    0x3dfd_8c42, 0x3f59_3f96, 0xbeef_2607, 0xbe8a_137e, 0x3f2a_0cee, 0xbf2a_c152, 0xbba0_9df0,
    0xbea9_c853, 0x3e0d_66b2, 0xbdbb_fb10, 0xbeb3_7e20, 0x3faf_fd46, 0x3f87_7325, 0x3efe_00a2,
    0x3dff_a60f, 0x3f30_2331, 0x3ee5_d28d, 0x3f06_e0b5, 0xbe24_fc39, 0x3da4_4417, 0x3f20_0b33,
    0xbf86_4621, 0xbe20_4d31, 0xbdd1_2cfc, 0x3f49_bf63, 0x3e9f_a8aa, 0x3e6d_48d8, 0x3ede_57c2,
    0xbee1_e09c, 0xbcaa_e404, 0x3d57_a80f, 0xbdb1_ca98, 0xbee2_c8e6, 0xbf50_f72a, 0xbc74_7290,
    0xbf3a_0177, 0xbf20_d1be, 0x3e11_315e, 0xbe89_2791, 0xbefd_d87e, 0xbebd_69cf, 0x3ea8_28d6,
    0xbee9_2498, 0xbead_f308, 0xbd87_06c1, 0xbee4_a188, 0x3cfe_51d6, 0x3d05_31a8, 0x3daa_c660,
    0xbe1d_a812, 0x3edf_b592, 0xbdb1_1f5a, 0x3cfd_5dd8, 0xb783_4000, 0x3ed4_c7e7, 0xbe93_5242,
    0xbea3_b4eb, 0x3e39_98eb, 0xbe63_9b9a, 0xbef5_a86a, 0xbdc8_845a, 0xbe3b_c000, 0x3c83_c708,
    0xbf27_7301, 0x3d80_8036, 0xbe6b_d3ba, 0xbee6_a5bb, 0x3c82_03b2, 0x3f0b_3042, 0x3ecb_bd39,
    0x3eed_c12d, 0x3e19_c7a2, 0xbdef_9249, 0xbd54_c422, 0xbea8_2cfd, 0xbcce_60d0, 0xbecd_6f11,
    0x3ea8_9842, 0x3cbb_3068,
];

#[rustfmt::skip]
const CONV2_BUFFER_FRAMESIZE: [u32; 81] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x3e3d_da01, 0xbf2b_3ccd, 0xbf18_ea74, 0x3bad_c5b5,
    0xbf46_de48, 0x3f5c_6e48, 0xbede_9efa, 0xbeac_8f42, 0xbe46_215d, 0xbf47_bdab, 0xbf70_1e56,
    0xbeb8_b023, 0x3c6f_f98d, 0xbe86_b66c, 0xbf22_769b, 0xbeb3_dc37, 0xbf44_79b1, 0xbebd_5220,
    0xbe76_068c, 0x3f7e_151e, 0xbd88_b89e, 0xbece_7d0d, 0x3ea2_8a8a, 0xbf20_ff90, 0x3f1c_cf10,
    0xbe83_bafa, 0xbec6_99ef, 0xbf58_65f2, 0x3f73_c9ae, 0x3f92_1a75, 0xbe1a_47b3, 0xbe2e_5fc5,
    0x3eb5_1366, 0x3f27_b19f, 0xbf29_fe10, 0xbe99_4cf4, 0x3daa_f126, 0x3f3e_91c6, 0x3ef5_78c7,
    0xbf0c_8e43, 0xbdeb_72f2, 0xbf6b_a69e, 0xbd1c_ea38, 0x3cd1_9608, 0xbef9_f6f0, 0xbbdc_39a0,
    0xbf7d_e879, 0x3f05_f924, 0xbd66_01ca, 0x3e80_2c00, 0x3e06_31f6, 0xbe9b_1e5f, 0x3e94_8318,
    0x3f2a_e102, 0x3f0c_d243, 0x3e9e_b2d8, 0x3e7e_09d8, 0x3e8e_6504, 0xbf07_cd74, 0xbe86_c18c,
    0x3e51_1b8d, 0x3f16_a8f2, 0x3e81_f52e, 0xbf4b_f0aa, 0xbc89_1b7c, 0x3f78_5b52, 0x3ead_5204,
    0xbeec_6746, 0x3ed3_079e, 0xbebf_1883, 0xbfa6_64e1,
];

#[rustfmt::skip]
const CONV2L: [u32; 158] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x3e60_a2df, 0xbd81_c140,
    0x3f0a_b9a1, 0xbe0b_0052, 0x3e36_b737, 0xbf2f_8a6f, 0x3ee9_c961, 0xbe8e_e012, 0xbe19_7d5f,
    0xbeb6_6e4d, 0xbd96_55a7, 0x3dbb_1df0, 0xbeb4_fe0a, 0x3ead_b8c3, 0xbded_3f27, 0xbde9_b6f3,
    0x3de5_f151, 0xbe75_b891, 0x3e4d_9ef0, 0xbe98_7b77, 0x3d13_3704, 0x3d98_4fd3, 0x3e58_ea64,
    0x3e86_cdce, 0x3ea6_e777, 0x3eed_a9e6, 0x3f03_173a, 0x3ef5_af5e, 0xbe53_9c9f, 0x3e98_81ca,
    0xbd0a_a04c, 0x3e3b_9383, 0xbe4e_cf02, 0xbdbc_585c, 0xbc91_f6d8, 0xbe1c_978e, 0xbdf4_bf6c,
    0x3dc3_fc2c, 0xbeaf_b96d, 0x3e92_b9b7, 0xbf03_1d5d, 0x3eeb_1bb8, 0xbe31_61c0, 0xbedd_2264,
    0x3e39_4c26, 0x3ceb_a630, 0xbe44_da88, 0x3ee7_aa30, 0xbf3d_033e, 0x3b80_3ad0, 0x3e08_715c,
    0x3f24_6289, 0xbe42_fac6, 0x3f28_f78d, 0x3ef1_4f5e, 0x3ebd_652c, 0xbe8a_b36d, 0x3e7a_ca7e,
    0xbf8b_5190, 0x3f25_cda8, 0xbee6_81e8, 0x3eb0_f70f, 0xbeb8_c0d4, 0x3e60_492f, 0xbf3d_d338,
    0x3ea1_7ac1, 0x3eb2_45b5, 0x3efe_00a2, 0x3e9f_bd26, 0xbeaa_9af0, 0xbf93_99c4, 0xbe97_89b3,
    0x3d1d_21b8, 0xbdbb_5718, 0x3e66_a514, 0xbe22_f4eb, 0xbf14_2d82, 0x3eeb_a226, 0xbf64_abb1,
    0x3ef3_f0ec, 0x3e5a_45a1, 0x3f2e_7a5b, 0x3e87_2741, 0x3dd7_5ba7, 0x3e6d_48d8, 0xbea5_fb3e,
    0xbdb8_f018, 0x3e25_5419, 0xbf14_1346, 0xbf27_fef6, 0xbe2c_9924, 0x3e89_ad38, 0x3e91_f799,
    0xbde5_df7e, 0xbe9d_a60a, 0xbe85_0d7a, 0x3e2f_2991, 0xbe74_f355, 0xbe2e_e18d, 0x3d44_cd61,
    0xbed0_e6d1, 0xbf06_dc95, 0x3e71_a5ae, 0xbe7a_98bc, 0xbeef_8a4f, 0x3ccf_3ce3, 0x3bcd_9109,
    0xbaeb_2bbc, 0x3f04_8519, 0x3dd3_a1c4, 0xbe64_d518, 0xbee3_ed17, 0x3dcc_c7b1, 0xbe13_0ea2,
    0xbd53_0cda, 0xbe30_0ce8, 0xbe54_5059, 0xbe84_6e49, 0x3bac_1020, 0x3e94_94d8, 0xbd3f_32ec,
    0xbe98_bf83, 0x3dc7_70c3, 0x3e90_e399, 0x3e14_623f, 0xbe1a_ac5d, 0x3dd6_7d0a, 0xbeb6_5c9b,
    0x3e97_52cf, 0x3f30_e94c, 0xbe08_bf6d, 0x3eb5_0c54, 0xbdb4_a1ec, 0x3ed5_08b1, 0xbe97_aa55,
    0xbe4e_a1cb, 0xbe18_5366, 0x3efd_3819, 0x3eca_ad42, 0x3e73_70b5, 0x3e82_f350, 0xbe54_507f,
    0xbeca_7010, 0xbe3e_1205, 0xbeaa_608c, 0xbdac_1f07, 0x3e4e_4493, 0x3e48_4bc7, 0x3dfd_3d58,
    0x3e22_d81c, 0x3def_9893, 0xbdf7_44b2, 0xbc3d_a3b0,
];

#[rustfmt::skip]
const STEREO_L: [u32; 187] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0xbfd9_2783,
    0x3e1e_1d3e, 0x3f82_978e, 0x3e98_2e1d, 0x3f9a_4c15, 0xc002_6abe, 0xbeeb_c26e, 0xbf96_27d1,
    0x3f9a_27c4, 0x3d78_0ed4, 0x4025_da8f, 0x40b5_ab9a, 0x3f2c_caa0, 0x3ef9_ae78, 0xbf36_4c72,
    0x401f_b2cf, 0xbf77_b724, 0xbe53_35e6, 0xc05e_d79c, 0xbebb_15b4, 0xbe8c_c51f, 0xbf28_2666,
    0x3f68_bcd1, 0x4044_90a0, 0xbdf2_5450, 0x3e00_a93c, 0xc165_c391, 0x3fcc_4924, 0xc036_70b5,
    0xc06f_472b, 0x409c_d104, 0xc008_9dd1, 0x3fd5_4708, 0x4029_8ac9, 0xc088_f715, 0xc05b_0210,
    0x3ff0_7daf, 0x3f07_a48c, 0xbf13_5eb6, 0x3f65_9ff0, 0x3fbc_64c1, 0x3fd3_b83e, 0xbd93_1df0,
    0x3fc1_c7fd, 0xbee0_3ef0, 0x3f2d_3982, 0x40b4_4780, 0xc285_a5fc, 0x4274_0aa2, 0xc21f_e52b,
    0x41ce_5b32, 0xc188_ad5e, 0xc24b_6718, 0xbf35_23f0, 0x4110_b349, 0x41bf_f467, 0x404e_aae4,
    0x407a_ea9e, 0xbfa3_4aea, 0xc1a9_bcfb, 0x4395_d2fd, 0xc207_cc28, 0x4116_cf3b, 0xc171_a468,
    0xc09d_e469, 0x41a0_e7b0, 0x42b0_4693, 0xc094_cb62, 0xc358_63db, 0xc3dd_0351, 0x4574_fe33,
    0x4528_6d82, 0x4483_7928, 0x437c_45e1, 0xc471_dc81, 0x42fa_2fd0, 0xc2b4_8e97, 0x4400_a256,
    0x43e5_bebe, 0xc31a_ec8f, 0xc505_1e89, 0xc3a3_b252, 0x4766_cc96, 0xc61f_c45f, 0x4664_af21,
    0xc69d_2eb0, 0x45a4_1ccd, 0x44b1_70e2, 0xc52e_7ce0, 0x4634_b85c, 0x449a_ab36, 0x43e1_7f40,
    0x4596_4792, 0x4973_4e77, 0x48c9_1c66, 0xc921_6d9f, 0x48a9_fab5, 0x475b_eaf8, 0xc8d6_e0e6,
    0x4997_dc2f, 0xc74a_7e40, 0x47aa_993e, 0x4939_45ed, 0xc905_d307, 0x49a1_415c, 0x4bb5_36c0,
    0x4b19_647a, 0x4c27_e33a, 0xcaca_f646, 0x4b54_5e8f, 0x4985_86bc, 0xc9cc_e849, 0x49d5_dd5e,
    0x49a5_0c68, 0x498a_fe1e, 0x4942_9798, 0x49e1_0d81, 0x4e07_3520, 0x4d3d_c10f, 0xcd56_ded3,
    0x4e5c_fd3b, 0x4d6e_273c, 0x4c37_90fa, 0x4c65_01e2, 0xcd2c_38b8, 0x4cd2_f2ad, 0x4c6a_83e1,
    0xce1a_c438, 0x50a2_a397, 0x4f4b_55f1, 0x4f2d_a389, 0x4e60_d9b3, 0xcd34_7670, 0xce3c_9834,
    0xcd26_c26a, 0x4c56_1050, 0x4c88_0258, 0xccb0_933c, 0xcca3_7a0f, 0x4bbc_1bae, 0x4c2c_ba5a,
    0x4c7d_a195, 0xca2e_af50, 0x4cbe_05b1, 0xcd62_4e65, 0xce70_2d05, 0x4e1e_f82d, 0xcf3b_f3c8,
    0x4f11_8e42, 0xcc8a_81fb, 0xd036_c503, 0xc046_caa0, 0xbfde_08bc, 0xc037_e46a, 0xc122_4b66,
    0x3f8f_acfe, 0xc10a_7813, 0x3f0e_5eee, 0xc0c1_6d6a, 0xbff7_be0b, 0x40ba_e173, 0xbf9d_1d16,
    0xbf54_e82e, 0xbf73_f2c0, 0x3fd0_28df, 0xc08f_0075, 0xbf9f_acd4, 0xc0d7_8e0f, 0x3eda_f614,
    0x3e3a_2d2c, 0xbfe1_f33a, 0x3dc8_5aaa, 0xc056_595c, 0xbf1b_d772, 0xbf54_ffe1, 0xbf39_a658,
    0xbfce_62ba, 0x3fdd_d822, 0xbf86_a44f, 0xbf05_9f12, 0xbf84_c560,
];

#[rustfmt::skip]
const STEREO_R: [u32; 187] = [
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x3c6e_c1be,
    0x3cd8_e90d, 0xbd49_63b4, 0x3d61_254b, 0xbc8c_fe50, 0x3b9f_3a06, 0x3d2b_64de, 0xbc35_56e7,
    0x3cde_225b, 0xbc2b_09fc, 0xbbc6_bb21, 0x3ac7_7ca9, 0xbcf6_06ac, 0x3b02_bbe2, 0xbb94_ba04,
    0xbc43_0cbc, 0xbdc2_14d6, 0x3d45_d077, 0xbc2f_1ba0, 0x3d20_4273, 0xbcc0_64d0, 0x3d3e_c4bb,
    0x3b82_548c, 0x3c9b_7479, 0xbb86_9890, 0x3d5e_c9f2, 0xbb8f_c5b4, 0x3c9c_7b22, 0x3d8c_e71f,
    0x38ce_8980, 0x3d6d_acee, 0x3ce7_3ac5, 0x3d6a_39c0, 0x3c06_d2da, 0xbca8_6fb3, 0x3e58_27f0,
    0xbe90_a7a6, 0x3d1f_0b2c, 0x3b80_8350, 0xbeb5_1605, 0x3eaf_accd, 0xbe08_0894, 0xbdf5_3082,
    0xbeb9_cefc, 0x3f1b_346c, 0xbeaa_5005, 0xbb01_b6fc, 0x394e_3400, 0x3f03_140d, 0xbde4_5a45,
    0x3e9c_a0fe, 0xbed8_6609, 0x3e82_6527, 0x3e90_21eb, 0xbc68_d678, 0x3f0e_040d, 0xbf01_b838,
    0x3dfe_09df, 0xc0ca_c77b, 0xc142_1776, 0xc074_55db, 0x3fcf_25bd, 0xc05c_8562, 0x3faf_81f6,
    0x411a_024e, 0xc133_0668, 0xc12e_0073, 0x416a_e0a9, 0x4050_c0f7, 0xbf40_c46a, 0xc156_274c,
    0x4142_5d13, 0x40cd_01ac, 0x416f_180c, 0xbf81_614b, 0x40f2_9324, 0xc17c_9dd2, 0x40ca_b335,
    0x419a_2a2c, 0xbf8c_e220, 0xc103_7039, 0x4183_80fb, 0x4164_0973, 0xbc41_5c00, 0xc1f8_9a3a,
    0xc13c_cce5, 0x4185_496d, 0x412e_0a17, 0x40ce_44b8, 0x4101_0831, 0xc0dd_bdc4, 0xc134_16e4,
    0xc0c6_0631, 0xc072_9132, 0xc060_2514, 0xc10a_efc5, 0x4085_e29e, 0x41ac_af00, 0x3f02_8f6c,
    0x3f86_e298, 0x3fe4_e1c6, 0xc12a_fe2f, 0xc0c6_2526, 0x41ea_ffc3, 0x41ce_a13a, 0xc017_4164,
    0x40a0_13b4, 0x4086_46cb, 0x4190_578c, 0x4108_a89e, 0x40e8_5a56, 0xc12d_bfc9, 0xc122_9b0e,
    0xc00d_1ab5, 0x417e_cdfe, 0xbe6e_ed60, 0xc161_27c5, 0x40bc_29c0, 0x4102_3ca7, 0x40cc_e84b,
    0xc0d4_a182, 0xc195_62c8, 0xc11a_fe56, 0xbf49_1138, 0xc0b8_414f, 0x407f_dbe6, 0xc049_b162,
    0x3f81_c18e, 0xc2f9_65fc, 0x4256_cf59, 0x41bc_5b51, 0xc29d_b2ef, 0x4223_11c6, 0x41d1_ce77,
    0x40ac_5043, 0x4223_f992, 0x41e8_9ba8, 0xc165_32cd, 0xc0d8_d6d6, 0x404a_0395, 0x3fb9_1f06,
    0x406e_5b94, 0xc035_b7c3, 0x3fae_f07d, 0xc174_cc32, 0x40a2_035f, 0x4050_37b6, 0x4006_8c17,
    0xc122_3451, 0xc1c3_2c94, 0x420f_14eb, 0x3dc6_096f, 0xbcce_2604, 0xbc19_f52e, 0xbd86_c0b2,
    0xbd56_8a56, 0xbc6e_2c6d, 0xbcc0_fee7, 0xbc84_9ab5, 0xbc95_3b98, 0xbccb_f1e5, 0xbcff_9230,
    0xbdc9_d808, 0x3d42_29cb, 0x3bc2_d094, 0x3c8a_33af, 0x3d6f_63b9, 0x3cd6_f632, 0x3caa_2122,
    0xbd3b_c22c, 0x3d91_eb88, 0xbd2c_934c, 0x3ce8_ad0c, 0xbd36_753e, 0xbd3d_3ed2, 0xbd25_f0f4,
    0xbce8_fed6, 0x3d88_849a, 0xbdc1_1368, 0x3c85_0119, 0xbcda_2870,
];
