//! Two-buffer spectral ops on two FFT chains: `PV_Add` sums two spectra (both tones survive), and
//! `PV_Copy` overwrites its target buffer with the source (only the source tone survives). Requires
//! the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const FFT_SIZE: usize = 1024;
const BIN: f32 = SR as f32 / FFT_SIZE as f32; // 46.875 Hz

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 128, 512, 256];
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR as f32).floor();
    let w = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in samples {
        let s = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
}

fn sin(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

/// `<sine unit> * 0.5`, referencing the SinOsc at index `src`.
fn half(src: u32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![
            InputRef::Unit {
                unit: src,
                output: 0,
            },
            InputRef::Constant(0.5),
        ],
        num_outputs: 1,
        special_index: 2,
    }
}

fn fft(bufnum: f32, in_unit: u32) -> UnitSpec {
    UnitSpec::new(
        "FFT",
        Rate::Control,
        vec![
            InputRef::Constant(bufnum),
            InputRef::Unit {
                unit: in_unit,
                output: 0,
            },
            InputRef::Constant(0.5),
            InputRef::Constant(0.0),
            InputRef::Constant(1.0),
            InputRef::Constant(FFT_SIZE as f32),
        ],
        1,
    )
}

/// Two FFT chains (tone A on buf 0, tone B on buf 1), combined by `pv`, resynthesised. Returns the
/// steady-state tail.
fn combined(pv: &str, freq_a: f32, freq_b: f32) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    for b in 0..2 {
        controller
            .buffer_set(
                b,
                Box::new(Buffer::from_interleaved(vec![0.0; FFT_SIZE], 1, SR)),
            )
            .unwrap();
    }
    // Units: 0,1 = sine A -> 2 = FFT(buf0); 3,4 = sine B -> 5 = FFT(buf1); 6 = pv(fftA, fftB);
    // 7 = IFFT; 8 = Out.
    let units = vec![
        sin(freq_a), // 0
        half(0),     // 1
        fft(0.0, 1), // 2
        sin(freq_b), // 3
        half(3),     // 4
        fft(1.0, 4), // 5
        // 6: pv(fftA=unit2, fftB=unit5).
        UnitSpec::new(
            pv,
            Rate::Control,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 5, output: 0 },
            ],
            1,
        ),
        // 7: IFFT(the pv output).
        UnitSpec::new(
            "IFFT",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 6, output: 0 },
                InputRef::Constant(0.0),
                InputRef::Constant(FFT_SIZE as f32),
            ],
            1,
        ),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 7, output: 0 },
            ],
            0,
        ),
    ];
    controller.add_synthdef(SynthDef {
        name: "pvc".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("pvc", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, 12_288);
    out[8_192..].to_vec()
}

#[test]
fn pv_add_sums_two_spectra() {
    // Tones at bins 20 and 40; PV_Add keeps both.
    let (fa, fb) = (20.0 * BIN, 40.0 * BIN);
    let out = combined("PV_Add", fa, fb);
    assert!(goertzel(&out, fa) > 0.05, "PV_Add should keep tone A");
    assert!(goertzel(&out, fb) > 0.05, "PV_Add should keep tone B");
}

#[test]
fn pv_max_keeps_the_louder_bins() {
    // Both tones live in different bins, so the per-bin max keeps each.
    let (fa, fb) = (20.0 * BIN, 40.0 * BIN);
    let out = combined("PV_Max", fa, fb);
    assert!(goertzel(&out, fa) > 0.05, "PV_Max should keep tone A");
    assert!(goertzel(&out, fb) > 0.05, "PV_Max should keep tone B");
}

#[test]
fn pv_copy_overwrites_the_target() {
    // PV_Copy copies A (bin 20) into B and continues with B, so only tone A survives; B's own tone
    // (bin 40) is discarded.
    let (fa, fb) = (20.0 * BIN, 40.0 * BIN);
    let out = combined("PV_Copy", fa, fb);
    assert!(
        goertzel(&out, fa) > 5.0 * goertzel(&out, fb),
        "PV_Copy should keep only the source tone"
    );
}

// The frame tests below run one two-buffer op for one control block over two small pre-filled chain
// buffers and read both buffers, and the op's chain output, straight back out. The expected bit
// patterns are scsynth's, from a C++ harness that includes scsynth's `FFT_UGens.h` (`ToPolarApx`/
// `ToComplexApx` over `SC_Complex.h`'s tables) and runs each unit's calc body from `PV_UGens.cpp`
// after `PV_GET_BUF2`. They are the same on macOS and glibc Linux.

/// Samples per control block.
const BLOCK: usize = 64;
/// Frames in each small chain buffer: `[dc, nyq]` and three bins.
const SMALL: usize = 8;

/// Buffer `A`: a Cartesian frame.
const FRAME_A: [f32; SMALL] = [0.8125, -0.4375, 0.75, -0.3, -1.25, 0.6, 0.2, 1.7];
/// Buffer `B`: a Cartesian frame.
const FRAME_B: [f32; SMALL] = [-0.5, 1.5, 0.4, 0.9, -0.35, -0.8, 1.1, -0.05];
/// Buffer `B` as a polar frame: `(mag, phase)` pairs, one phase past a full turn.
const FRAME_B_POLAR: [f32; SMALL] = [-0.5, 1.5, 0.9, 2.5, 1.2, -0.7, 0.3, 7.1];

/// [`FRAME_B`] after `ToPolarApx`.
const B_TO_POLAR: [u32; SMALL] = [
    0xbf000000, 0x3fc00000, 0x3f7c1edf, 0x3f938a73, 0x3f5f8ada, 0x4089996d, 0x3f8cf2bf, 0xbd3bde3f,
];
/// [`FRAME_B_POLAR`] after `ToComplexApx`.
const B_POLAR_TO_COMPLEX: [u32; SMALL] = [
    0xbf000000, 0x3fc00000, 0xbf388804, 0x3f09f533, 0x3f6b0f34, 0xbf45c8fa, 0x3e52752d, 0x3e5fc8c5,
];

/// One two-buffer op and what scsynth leaves in buffer `A`.
struct Op {
    name: &'static str,
    /// Whether the op works in polar form (else Cartesian).
    polar: bool,
    /// `A` after the op with `B` = [`FRAME_B`].
    a: [u32; SMALL],
    /// `A` after the op with `B` = [`FRAME_B_POLAR`].
    a_polar_b: [u32; SMALL],
    /// `A` after the op with both inputs naming `A`.
    same: [u32; SMALL],
}

const OPS: [Op; 7] = [
    Op {
        name: "PV_Add",
        polar: false,
        a: [
            0x3ea00000, 0x3f880000, 0x3f933333, 0x3f199999, 0xbfcccccd, 0xbe4ccccc, 0x3fa66667,
            0x3fd33334,
        ],
        a_polar_b: [
            0x3ea00000, 0x3f880000, 0x3ceeff80, 0x3e74a198, 0xbea9e198, 0xbe30bd80, 0x3ecfa0fd,
            0x3ff592b3,
        ],
        same: [
            0x3fd00000, 0xbf600000, 0x3fc00000, 0xbf19999a, 0xc0200000, 0x3f99999a, 0x3ecccccd,
            0x4059999a,
        ],
    },
    Op {
        name: "PV_Mul",
        polar: false,
        a: [
            0xbed00000, 0xbf280000, 0x3f11eb86, 0x3f0e147a, 0x3f6ae148, 0x3f4a3d70, 0x3e9c28f6,
            0x3fee147d,
        ],
        a_polar_b: [
            0xbed00000, 0xbf280000, 0xbec205b4, 0x3f1ed3e8, 0xbf2f2736, 0x3fc2222c, 0xbea92bbd,
            0x3ec9447b,
        ],
        same: [
            0x3f290000, 0x3e440000, 0x3ef1eb85, 0xbf132b02, 0x3f99eb85, 0xc0460626, 0xc0366667,
            0xc0a3ae15,
        ],
    },
    Op {
        name: "PV_Div",
        polar: false,
        a: [
            0xbfd00000, 0xbe955555, 0x3cfd5c60, 0xbf51d07e, 0xbd644d3e, 0xbfcb1f0e, 0x3de40657,
            0x3fc6774a,
        ],
        a_polar_b: [
            0xbfd00000, 0xbe955555, 0xbf5df532, 0xbe6d9339, 0xbf8f3a3f, 0xbe937e76, 0x4092b5c2,
            0x4059603b,
        ],
        same: [
            0x3f800000, 0x3f800000, 0x3f800000, 0xbdeb66fd, 0x3f800000, 0x3f33c414, 0x3f800000,
            0x3eeda6e1,
        ],
    },
    Op {
        name: "PV_Max",
        polar: true,
        a: [
            0x3f500000, 0x3fc00000, 0x3f7c1edf, 0x3f938a73, 0x3fb18289, 0x402c6572, 0x3fdb16cd,
            0x3fba214a,
        ],
        a_polar_b: [
            0x3f500000, 0x3fc00000, 0x3f666666, 0x40200000, 0x3fb18289, 0x402c6572, 0x3fdb16cd,
            0x3fba214a,
        ],
        same: [
            0x3f500000, 0xbee00000, 0x3f4ed176, 0xbec2fddd, 0x3fb18289, 0x402c6572, 0x3fdb16cd,
            0x3fba214a,
        ],
    },
    Op {
        name: "PV_Min",
        polar: true,
        a: [
            0xbf000000, 0xbee00000, 0x3f4ed176, 0xbec2fddd, 0x3f5f8ada, 0x4089996d, 0x3f8cf2bf,
            0xbd3bde3f,
        ],
        a_polar_b: [
            0xbf000000, 0xbee00000, 0x3f4ed176, 0xbec2fddd, 0x3f99999a, 0xbf333333, 0x3e99999a,
            0x40e33333,
        ],
        same: [
            0x3f500000, 0xbee00000, 0x3f4ed176, 0xbec2fddd, 0x3fb18289, 0x402c6572, 0x3fdb16cd,
            0x3fba214a,
        ],
    },
    Op {
        name: "PV_CopyPhase",
        polar: true,
        a: [
            0xbf500000, 0x3ee00000, 0x3f4ed176, 0x3f938a73, 0x3fb18289, 0x4089996d, 0x3fdb16cd,
            0xbd3bde3f,
        ],
        a_polar_b: [
            0xbf500000, 0x3ee00000, 0x3f4ed176, 0x40200000, 0x3fb18289, 0xbf333333, 0x3fdb16cd,
            0x40e33333,
        ],
        same: [
            0x3f500000, 0xbee00000, 0x3f4ed176, 0xbec2fddd, 0x3fb18289, 0x402c6572, 0x3fdb16cd,
            0x3fba214a,
        ],
    },
    Op {
        name: "PV_MagMul",
        polar: true,
        a: [
            0xbed00000, 0xbf280000, 0x3f4baf21, 0xbec2fddd, 0x3f9b00fd, 0x402c6572, 0x3ff1406e,
            0x3fba214a,
        ],
        a_polar_b: [
            0xbed00000, 0xbf280000, 0x3f3a22ea, 0xbec2fddd, 0x3fd5030b, 0x402c6572, 0x3f037415,
            0x3fba214a,
        ],
        same: [
            0x3f290000, 0x3e440000, 0x3f2715c5, 0xbec2fddd, 0x3ff62b88, 0x402c6572, 0x403b8005,
            0x3fba214a,
        ],
    },
];

/// What one block of a two-buffer op leaves behind.
struct After {
    /// Buffer `A` (buffer 0).
    a: Vec<f32>,
    /// Buffer `B` (buffer 1).
    b: Vec<f32>,
    /// The op's chain output.
    chain: f32,
}

/// Run `op(chain_a, chain_b)` for one block over buffer 0 = `a` (Cartesian) and buffer 1 = `b`
/// (polar when `b_polar`), then read both buffers and the chain output back.
fn run_pair(op: &str, chain_a: f32, chain_b: f32, a: &[f32], b: &[f32], b_polar: bool) -> After {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 3,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(a.to_vec(), 1, SR)))
        .unwrap();
    let mut buf_b = Buffer::from_interleaved(b.to_vec(), 1, SR);
    if b_polar {
        buf_b.set_coord(plyphon_dsp::SpectrumCoord::Polar);
    }
    controller.buffer_set(1, Box::new(buf_b)).unwrap();
    let c = InputRef::Constant;
    let u = |unit| InputRef::Unit { unit, output: 0 };
    let bufrd = |bufnum: f32| {
        UnitSpec::new(
            "BufRd",
            Rate::Audio,
            vec![c(bufnum), u(1), c(1.0), c(1.0)],
            1,
        )
    };
    controller.add_synthdef(SynthDef {
        name: "pair".to_string(),
        params: vec![],
        units: vec![
            // 0: the op under test.
            UnitSpec::new(op, Rate::Control, vec![c(chain_a), c(chain_b)], 1),
            // 1: a sample counter over the smaller buffer's frames.
            UnitSpec::new(
                "Phasor",
                Rate::Audio,
                vec![
                    c(0.0),
                    c(1.0),
                    c(0.0),
                    c(a.len().min(b.len()) as f32),
                    c(0.0),
                ],
                1,
            ),
            // 2, 3: the two buffers.
            bufrd(0.0),
            bufrd(1.0),
            // 4: the chain output, carried into the block exactly by adding zero at audio rate.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![u(0), c(0.0)],
                num_outputs: 1,
                special_index: 0,
            },
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2), u(3), u(4)], 0),
        ],
    });
    controller
        .synth_new("pair", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut out = vec![0.0f32; BLOCK * 3];
    world.fill(&mut out, 3);
    let channel = |ch: usize, n: usize| out.iter().skip(ch).step_by(3).take(n).copied().collect();
    After {
        a: channel(0, a.len()),
        b: channel(1, a.len().min(b.len())),
        chain: out[2],
    }
}

/// Assert `got` matches the expected bit patterns slot by slot.
fn assert_bits(got: &[f32], want: &[u32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: frame length");
    for (i, (g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w,
            "{what}: slot {i} is {g}, scsynth has {}",
            f32::from_bits(w)
        );
    }
}

/// The bit patterns of `frame`.
fn bits(frame: &[f32]) -> Vec<u32> {
    frame.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn two_buffer_ops_convert_both_buffers_and_match_scsynth() {
    for op in &OPS {
        // Two Cartesian buffers: `A` gets the op's result and `B` is left converted to the op's
        // form, so a chain continuing from `B` sees it that way too.
        let got = run_pair(op.name, 0.0, 1.0, &FRAME_A, &FRAME_B, false);
        assert_eq!(got.chain, 0.0, "{}: passes A's chain on", op.name);
        assert_bits(&got.a, &op.a, &format!("{} A", op.name));
        let b_want = if op.polar {
            B_TO_POLAR.to_vec()
        } else {
            bits(&FRAME_B)
        };
        assert_bits(&got.b, &b_want, &format!("{} B", op.name));

        // `B` arrives polar: a Cartesian op converts it back first.
        let got = run_pair(op.name, 0.0, 1.0, &FRAME_A, &FRAME_B_POLAR, true);
        assert_bits(&got.a, &op.a_polar_b, &format!("{} A, polar B", op.name));
        let b_want = if op.polar {
            bits(&FRAME_B_POLAR)
        } else {
            B_POLAR_TO_COMPLEX.to_vec()
        };
        assert_bits(&got.b, &b_want, &format!("{} polar B", op.name));
    }
}

#[test]
fn two_buffer_ops_run_when_both_inputs_name_one_buffer() {
    // scsynth runs the op with both pointers on the one buffer, so its reads of `B` see what it has
    // already written to `A`: `PV_Mul` and `PV_Div` read the real part they just wrote.
    for op in &OPS {
        let got = run_pair(op.name, 0.0, 0.0, &FRAME_A, &FRAME_B, false);
        assert_eq!(got.chain, 0.0, "{}: passes the chain on", op.name);
        assert_bits(&got.a, &op.same, &format!("{} on one buffer", op.name));
        assert_bits(
            &got.b,
            &bits(&FRAME_B),
            &format!("{} leaves buffer 1", op.name),
        );
    }
}

#[test]
fn two_buffer_ops_need_a_frame_on_both_inputs() {
    // Without a frame on either input the output is -1 and neither buffer is touched, whichever
    // input lacks it.
    for op in &OPS {
        for (chain_a, chain_b) in [(0.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let got = run_pair(op.name, chain_a, chain_b, &FRAME_A, &FRAME_B, false);
            let what = format!("{} ({chain_a}, {chain_b})", op.name);
            assert_eq!(got.chain, -1.0, "{what}: no frame");
            assert_bits(&got.a, &bits(&FRAME_A), &format!("{what}: A"));
            assert_bits(&got.b, &bits(&FRAME_B), &format!("{what}: B"));
        }
    }
}

#[test]
fn two_buffer_ops_skip_buffers_of_different_sizes() {
    // Buffers holding different numbers of samples: scsynth still passes A's chain on, then stops
    // before converting either buffer.
    let wide: Vec<f32> = FRAME_B.iter().chain(&FRAME_B).copied().collect();
    for op in &OPS {
        let got = run_pair(op.name, 0.0, 1.0, &FRAME_A, &wide, false);
        assert_eq!(got.chain, 0.0, "{}: passes A's chain on", op.name);
        assert_bits(&got.a, &bits(&FRAME_A), &format!("{} A", op.name));
        assert_bits(&got.b, &bits(&FRAME_B), &format!("{} B", op.name));
    }
}
