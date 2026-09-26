//! Spectral (`PV_*`) operators.
//!
//! The first two are inserted into an FFT -> PV -> IFFT chain and judged by what comes out:
//! `PV_MagAbove` gates the whole spectrum away above a huge threshold (and passes at threshold 0),
//! and `PV_BrickWall` high/low-passes by zeroing a fraction of the bins. `PV_BinShift`,
//! `PV_MagSmear` and `PV_RectComb` rewrite the packed frame slot by slot, so they are driven over a
//! pre-filled chain buffer and the frame is read straight back out of it.
//!
//! Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const FFT_SIZE: usize = 1024;

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

fn rms(s: &[f32]) -> f32 {
    (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt()
}

/// `SinOsc(freq)*0.5 -> FFT(buf 0) -> pv(fbufnum, ..extra) -> IFFT -> Out`, rendered; returns the
/// steady-state tail's RMS.
fn tail_rms(freq: f32, pv: &str, extra: Vec<InputRef>) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(vec![0.0; FFT_SIZE], 1, SR)),
        )
        .unwrap();

    let mut pv_inputs = vec![InputRef::Unit { unit: 2, output: 0 }];
    pv_inputs.extend(extra);
    controller.add_synthdef(SynthDef {
        name: "pvc".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.5),
                ],
                num_outputs: 1,
                special_index: 2,
            },
            // 2: FFT(buf 0, in, 0.5, 0, 1, FFT_SIZE).
            UnitSpec::new(
                "FFT",
                Rate::Control,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.5),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(FFT_SIZE as f32),
                ],
                1,
            ),
            // 3: the PV op under test.
            UnitSpec::new(pv, Rate::Control, pv_inputs, 1),
            // 4: IFFT(fbufnum, 0, FFT_SIZE).
            UnitSpec::new(
                "IFFT",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
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
                    InputRef::Unit { unit: 4, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("pvc", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, 12_288);
    rms(&out[8_192..])
}

#[test]
fn pv_mag_above_gates_the_spectrum() {
    // A bin-aligned tone (20 bins).
    let freq = 20.0 * (SR as f32 / FFT_SIZE as f32);
    // thresh 0: nothing is below 0, so every bin passes -> the tone survives.
    let passed = tail_rms(freq, "PV_MagAbove", vec![InputRef::Constant(0.0)]);
    // A huge threshold zeroes every bin (all magnitudes are below it) -> silence.
    let gated = tail_rms(freq, "PV_MagAbove", vec![InputRef::Constant(1.0e9)]);
    assert!(
        passed > 0.05,
        "PV_MagAbove(0) should pass the tone, rms {passed}"
    );
    assert!(
        gated < 0.01,
        "PV_MagAbove(huge) should gate everything, rms {gated}"
    );
}

#[test]
fn pv_brick_wall_high_and_low_passes() {
    // A low-ish tone (bin 20). BrickWall(+0.5) zeroes the lower half of bins (high-pass) -> removed;
    // BrickWall(-0.5) zeroes the upper half (low-pass) -> the low tone survives.
    let freq = 20.0 * (SR as f32 / FFT_SIZE as f32);
    let highpassed = tail_rms(freq, "PV_BrickWall", vec![InputRef::Constant(0.5)]);
    let lowpassed = tail_rms(freq, "PV_BrickWall", vec![InputRef::Constant(-0.5)]);
    assert!(
        lowpassed > 0.05,
        "a low-pass should keep the low tone, rms {lowpassed}"
    );
    assert!(
        highpassed < 0.2 * lowpassed,
        "a high-pass should remove the low tone (high={highpassed}, low={lowpassed})"
    );
}

// The ops below rewrite the packed frame in ways an RMS-of-the-resynthesis check cannot pin down, so
// they are driven over a pre-filled chain buffer with the chain signal supplied directly - a
// constant buffer number, or a control bus where the test needs to change it between blocks - and
// the frame is read straight back out of the buffer by a counter driving a non-interpolating
// `BufRd`. One rendered control block then carries the whole packed frame as it stands after that
// block's spectral units ran.

/// Samples per control block for the frame-inspection tests, chosen equal to [`FRAME`] so one block
/// reads the whole frame.
const BLOCK: usize = 64;
/// Chain-buffer frames: the smallest FFT size plyphon plans for.
const FRAME: usize = 64;
/// Bins in a packed [`FRAME`] frame, which holds `[dc, nyq, bins...]`.
const BINS: usize = (FRAME - 2) / 2;
/// A second chain buffer of a different supported size, for the frame-size-change case.
const WIDE_FRAME: usize = 128;
/// The control bus carrying the chain signal where a test changes it between blocks.
const CHAIN_BUS: u32 = 0;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// A packed spectrum with pairwise-distinct, non-zero terms throughout, so a dropped, swapped or
/// silently converted slot cannot pass unnoticed. The imaginary parts alternate in sign and grow
/// faster than the real parts, so the bins spread over two quadrants and both of the polar
/// conversion's table branches.
fn test_frame() -> Vec<f32> {
    let mut data = vec![0.0f32; FRAME];
    data[0] = 0.8125;
    data[1] = -0.4375;
    for i in 0..BINS {
        data[2 + 2 * i] = 0.5 + i as f32 / 32.0;
        let im = 0.25 + (i * i) as f32 / 256.0;
        data[3 + 2 * i] = if i % 2 == 0 { -im } else { im };
    }
    data
}

// The expected frames below are scsynth's, from a C++ harness that includes scsynth's `FFT_UGens.h`
// (`ToPolarApx`/`ToComplexApx`, over `SC_Complex.h`'s lookup tables) and runs the calc bodies of
// `PV_UGens.cpp` over [`test_frame`]. They are the same on macOS and glibc Linux.

/// [`test_frame`] in polar form: `ToPolarApx`, which `PV_MagAbove` at threshold 0 leaves behind.
const POLAR_FRAME: [u32; FRAME] = [
    0x3f500000, 0xbee00000, 0x3f0f1bbd, 0xbeed6338, 0x3f16b617, 0x3ee41aee, 0x3f1f4662, 0xbee210c9,
    0x3f2891f7, 0x3ee4eb3b, 0x3f32e2ac, 0xbeed6338, 0x3f3e14f3, 0x3ef93f18, 0x3f4a70d4, 0xbf044eee,
    0x3f57d8b4, 0x3f0cd52f, 0x3f66ca3a, 0xbf16961b, 0x3f76fc8d, 0x3f20867d, 0x3f847a35, 0xbf2b0469,
    0x3f8e3205, 0x3f354e2b, 0x3f98d9b5, 0xbf3f9a9d, 0x3fa45aa9, 0x3f49b03f, 0x3fb0fbfc, 0x40aea9b0,
    0x3fbe5810, 0x3f5caca3, 0x3fccfa64, 0x40ac6745, 0x3fdc50ad, 0x3f6dde45, 0x3fecf009, 0x40aa5d66,
    0x3ffe6666, 0x3f7d19df, 0x40088d98, 0x40a894e2, 0x40125377, 0x3f853851, 0x401cbac7, 0x40a706b8,
    0x4027773d, 0x3f8b243b, 0x4032e2ac, 0x40a5a217, 0x403ebcdf, 0x3f903d47, 0x404b2e4f, 0x40a46fcf,
    0x40581570, 0x3f94b194, 0x40659d17, 0x40a36979, 0x40737749, 0x3f98a335, 0x4080fc50, 0x40a277ed,
];

/// `PV_BinShift(stretch 1, shift 1, interp 0)` behind the polar frame: `ToComplexApx`, then every
/// bin moved up one.
const SHIFTED_AFTER_POLAR: [u32; FRAME] = [
    0x3f500000, 0xbee00000, 0x00000000, 0x00000000, 0x3f000650, 0xbe7fcd7b, 0x3f080b06, 0x3e81b5ca,
    0x3f1008f7, 0xbe87fa0a, 0x3f180d67, 0x3e918c0e, 0x3f2007e4, 0xbe9fe06c, 0x3f280c3c, 0x3eb1a9c1,
    0x3f301080, 0xbec7d5e3, 0x3f3805a7, 0x3ee19d8e, 0x3f40175f, 0xbeffd9dc, 0x3f480f9c, 0x3f10d671,
    0x3f501faf, 0xbf23f7d1, 0x3f580dd3, 0x3f38eddc, 0x3f6021ee, 0xbf4fe372, 0x3f67e57c, 0x3f68f750,
    0x3f701d58, 0xbf82098a, 0x3f77fdba, 0x3f906a75, 0x3f80151f, 0xbfa008b6, 0x3f83eb54, 0x3fb07445,
    0x3f87fc0b, 0xbfc207b0, 0x3f8bdfa2, 0x3fd47f18, 0x3f8ffb3a, 0xbfe811db, 0x3f9422f7, 0x3ffc63e1,
    0x3f98326b, 0xc00903ea, 0x3f9bd7c8, 0x40143ba9, 0x3f9fe06c, 0xc02007e4, 0x3fa3e516, 0x402c3c9f,
    0x3fa7cd90, 0xc0390c76, 0x3fac18c0, 0x40463630, 0x3fb063a8, 0xc0540014, 0x3fb41cc9, 0x406232b2,
];

/// `PV_MagSmear(bins 1)` over [`test_frame`].
const SMEARED_1: [u32; FRAME] = [
    0x3f500000, 0xbee00000, 0x3ec3e138, 0xbeed6338, 0x3f170812, 0x3ee41aee, 0x3f1f84d0, 0xbee210c9,
    0x3f28e902, 0x3ee4eb3b, 0x3f332dde, 0xbeed6338, 0x3f3e7827, 0x3ef93f18, 0x3f4aca2a, 0xbf044eee,
    0x3f585beb, 0x3f0cd52f, 0x3f67352a, 0xbf16961b, 0x3f7793bb, 0x3f20867d, 0x3f84b8d6, 0xbf2b0469,
    0x3f8e81fb, 0x3f354e2b, 0x3f992222, 0xbf3f9a9d, 0x3fa4bac9, 0x3f49b03f, 0x3fb13a3c, 0x40aea9b0,
    0x3fbec4d0, 0x3f5caca3, 0x3fcd3660, 0x40ac6745, 0x3fdcbe5e, 0x3f6dde45, 0x3fed37b4, 0x40aa5d66,
    0x3ffed08b, 0x3f7d19df, 0x4008b16c, 0x40a894e2, 0x40128948, 0x3f853851, 0x401cd72a, 0x40a706b8,
    0x4027b190, 0x3f8b243b, 0x40330798, 0x40a5a217, 0x403eef4a, 0x3f903d47, 0x404b558b, 0x40a46fcf,
    0x40584af3, 0x3f94b194, 0x4065b89b, 0x40a36979, 0x4073af00, 0x3f98a335, 0x4027254e, 0x40a277ed,
];

/// `PV_MagSmear(bins 1e9)` over [`test_frame`]: the width clamps to the whole spectrum.
const SMEARED_ALL: [u32; FRAME] = [
    0x3f500000, 0xbee00000, 0x3f67e896, 0xbeed6338, 0x3f67e896, 0x3ee41aee, 0x3f67e896, 0xbee210c9,
    0x3f67e896, 0x3ee4eb3b, 0x3f67e896, 0xbeed6338, 0x3f67e896, 0x3ef93f18, 0x3f67e896, 0xbf044eee,
    0x3f67e896, 0x3f0cd52f, 0x3f67e896, 0xbf16961b, 0x3f67e896, 0x3f20867d, 0x3f67e896, 0xbf2b0469,
    0x3f67e896, 0x3f354e2b, 0x3f67e896, 0xbf3f9a9d, 0x3f67e896, 0x3f49b03f, 0x3f67e896, 0x40aea9b0,
    0x3f67e896, 0x3f5caca3, 0x3f67e896, 0x40ac6745, 0x3f67e896, 0x3f6dde45, 0x3f67e896, 0x40aa5d66,
    0x3f67e896, 0x3f7d19df, 0x3f67e896, 0x40a894e2, 0x3f67e896, 0x3f853851, 0x3f67e896, 0x40a706b8,
    0x3f67e896, 0x3f8b243b, 0x3f67e896, 0x40a5a217, 0x3f67e896, 0x3f903d47, 0x3f67e896, 0x40a46fcf,
    0x3f67e896, 0x3f94b194, 0x3f67e896, 0x40a36979, 0x3f67e896, 0x3f98a335, 0x3f67e896, 0x40a277ed,
];

/// The polar form of bin `i` of [`test_frame`], from [`POLAR_FRAME`].
fn polar(i: usize) -> (f32, f32) {
    (
        f32::from_bits(POLAR_FRAME[2 + 2 * i]),
        f32::from_bits(POLAR_FRAME[3 + 2 * i]),
    )
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

/// An engine holding `frame` in buffer 0 and a wider, distinctly filled buffer 1, running `units` as
/// one synth over `channels` outputs.
fn frame_engine(
    units: Vec<UnitSpec>,
    frame: &[f32],
    channels: usize,
) -> (plyphon::Controller, World) {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: channels,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(frame.to_vec(), 1, SR)))
        .expect("buffer_set");
    let wide: Vec<f32> = (0..WIDE_FRAME).map(|i| 1.0 + i as f32).collect();
    controller
        .buffer_set(1, Box::new(Buffer::from_interleaved(wide, 1, SR)))
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

/// Render one control block of `channels`-channel output.
fn one_block(world: &mut World, channels: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; BLOCK * channels];
    world.fill(&mut buf, channels);
    buf
}

/// Channel `ch` of an interleaved block.
fn channel(buf: &[f32], channels: usize, ch: usize) -> Vec<f32> {
    buf.iter().skip(ch).step_by(channels).copied().collect()
}

/// A frame counter driving a non-interpolating `BufRd` over buffer `bufnum`, then `Out` - appended
/// after the units under test, so it reads the frame they just wrote. `extra` goes into the output
/// channels after the frame.
fn read_frame(
    mut units: Vec<UnitSpec>,
    bufnum: f32,
    end: usize,
    extra: Vec<InputRef>,
) -> Vec<UnitSpec> {
    let phasor = units.len() as u32;
    units.push(UnitSpec::new(
        "Phasor",
        Rate::Audio,
        vec![c(0.0), c(1.0), c(0.0), c(end as f32), c(0.0)],
        1,
    ));
    units.push(UnitSpec::new(
        "BufRd",
        Rate::Audio,
        vec![c(bufnum), u(phasor), c(1.0), c(1.0)],
        1,
    ));
    let mut out = vec![c(0.0), u(phasor + 1)];
    out.extend(extra);
    units.push(UnitSpec::new("Out", Rate::Audio, out, 0));
    units
}

/// The frame left behind by a single control block of `op` over the test spectrum, optionally
/// chained behind `predecessor`.
fn frame_after(op: UnitSpec, predecessor: Option<UnitSpec>) -> Vec<f32> {
    let frame = test_frame();
    let mut units = Vec::new();
    if let Some(unit) = predecessor {
        units.push(unit);
    }
    units.push(op);
    let (_c, mut world) = frame_engine(read_frame(units, 0.0, FRAME, vec![]), &frame, 1);
    one_block(&mut world, 1)
}

/// `PV_BinShift(chain, stretch, shift, interp)`.
fn bin_shift(chain: InputRef, stretch: f32, shift: f32, interp: f32) -> UnitSpec {
    UnitSpec::new(
        "PV_BinShift",
        Rate::Control,
        vec![chain, c(stretch), c(shift), c(interp)],
        1,
    )
}

/// `PV_MagAbove(chain, 0)`: an identity that leaves the frame in polar form, for the coordinate-form
/// cases.
fn polar_identity(chain: InputRef) -> UnitSpec {
    UnitSpec::new("PV_MagAbove", Rate::Control, vec![chain, c(0.0)], 1)
}

#[test]
fn pv_conj_subtracts_the_imaginary_part_from_zero() {
    // scsynth's `PV_Conj` writes `0.f - imag`, so a `+0` imaginary part stays `+0` where a plain
    // negation would flip it to `-0`; `-0` becomes `+0` either way.
    let mut frame = test_frame();
    frame[3] = 0.0;
    frame[5] = -0.0;
    let conj = UnitSpec::new("PV_Conj", Rate::Control, vec![c(0.0)], 1);
    let (_c, mut world) = frame_engine(read_frame(vec![conj], 0.0, FRAME, vec![]), &frame, 1);
    let got = one_block(&mut world, 1);
    let mut want = frame.clone();
    for k in 0..BINS {
        want[3 + 2 * k] = 0.0 - frame[3 + 2 * k];
    }
    assert_eq!(got[3].to_bits(), 0.0f32.to_bits(), "+0 stays +0");
    assert_eq!(got[5].to_bits(), 0.0f32.to_bits(), "-0 becomes +0");
    assert_eq!(got, want, "every imaginary part is subtracted from zero");
}

#[test]
fn pv_bin_shift_maps_bins_and_leaves_complex() {
    let frame = test_frame();

    // Nearest-bin mapping (interp <= 0): bin `i` moves to `round(shift + i * stretch)`, whole. The
    // two bins that would land past the top are dropped and the two bins below the shift stay at the
    // zero the destination frame starts from, so the accumulation is visibly onto a cleared frame.
    let got = frame_after(bin_shift(c(0.0), 1.0, 2.0, 0.0), None);
    let mut want = frame.clone();
    for k in 0..BINS {
        let (re, im) = match k.checked_sub(2) {
            Some(src) if src < BINS => (frame[2 + 2 * src], frame[3 + 2 * src]),
            _ => (0.0, 0.0),
        };
        want[2 + 2 * k] = re;
        want[3 + 2 * k] = im;
    }
    assert_eq!(got, want, "nearest-bin shift by two");
    assert_eq!(got[0], frame[0], "the DC term passes through a bin shift");
    assert_eq!(
        got[1], frame[1],
        "the Nyquist term passes through a bin shift"
    );

    // Linear interpolation (interp > 0): a half-bin shift splits each bin evenly between its two
    // neighbours, and the two halves accumulate into the same destination.
    let got = frame_after(bin_shift(c(0.0), 1.0, 0.5, 1.0), None);
    for k in 0..BINS {
        let lower = if k == 0 {
            0.0
        } else {
            0.5 * frame[2 + 2 * (k - 1)]
        };
        let lower_im = if k == 0 {
            0.0
        } else {
            0.5 * frame[3 + 2 * (k - 1)]
        };
        let want_re = lower + 0.5 * frame[2 + 2 * k];
        let want_im = lower_im + 0.5 * frame[3 + 2 * k];
        assert!(
            (got[2 + 2 * k] - want_re).abs() < 1e-6 && (got[3 + 2 * k] - want_im).abs() < 1e-6,
            "interpolated bin {k} is ({}, {}), expected ({want_re}, {want_im})",
            got[2 + 2 * k],
            got[3 + 2 * k]
        );
    }

    // A non-finite position places no bin, so a NaN stretch silences every bin past the first.
    let got = frame_after(bin_shift(c(0.0), f32::NAN, 0.0, 0.0), None);
    assert_eq!(
        (got[2], got[3]),
        (frame[2], frame[3]),
        "the first bin's position is still finite"
    );
    assert!(
        got[4..].iter().all(|&s| s == 0.0),
        "bins at non-finite positions are dropped"
    );

    // Behind a polar predecessor the frame arrives as magnitude/phase pairs. The op converts before
    // it maps, so the shifted frame reads back as Cartesian bins - which the polar pairs are not -
    // after scsynth's table round trip.
    let got = frame_after(bin_shift(u(0), 1.0, 1.0, 0.0), Some(polar_identity(c(0.0))));
    assert_bits(
        &got,
        &SHIFTED_AFTER_POLAR,
        "shift behind a polar predecessor",
    );
    let (polar_mag, polar_phase) = polar(0);
    assert!(
        (polar_mag - frame[2]).abs() > 1e-3 || (polar_phase - frame[3]).abs() > 1e-3,
        "the polar and Cartesian forms must differ for the conversion check to discriminate"
    );
}

#[test]
fn pv_mag_smear_averages_and_leaves_polar() {
    let frame = test_frame();
    let smear = |bins: f32| UnitSpec::new("PV_MagSmear", Rate::Control, vec![c(0.0), c(bins)], 1);

    // A width of one averages each magnitude with its two neighbours. The window is truncated at the
    // edges but the divisor is not, so the outermost bins are attenuated - the reference's shape.
    // Phases, and the DC and Nyquist terms, survive.
    let got = frame_after(smear(1.0), None);
    assert_bits(&got, &SMEARED_1, "smear of width 1");
    for j in 0..BINS {
        assert_eq!(got[3 + 2 * j], polar(j).1, "phase {j} must survive a smear");
    }
    assert_eq!(got[0], frame[0], "the DC term passes through a smear");
    assert_eq!(got[1], frame[1], "the Nyquist term passes through a smear");

    // The frame is left in polar form: the stored pairs are magnitude/phase, not the Cartesian pairs
    // that went in.
    assert!(
        (got[2] - frame[2]).abs() > 1e-3 || (got[3] - frame[3]).abs() > 1e-3,
        "a polar end state must be visibly different from the Cartesian input"
    );

    // The width is truncated to an integer and clamped to the spectrum, so a huge width averages the
    // whole spectrum into every bin.
    let got = frame_after(smear(1.0e9), None);
    assert_bits(&got, &SMEARED_ALL, "smear of width 1e9");

    // A negative width, and a NaN (which reads as zero), both clamp to a window of one: the
    // magnitudes are untouched, though the frame still ends polar. An infinite width saturates
    // the cast instead and clamps to the widest window - the full-smear case above.
    let inf = frame_after(smear(f32::INFINITY), None);
    let full = frame_after(smear(BINS as f32), None);
    assert_eq!(
        inf, full,
        "an infinite width must smear like the widest window"
    );
    for width in [-5.0f32, f32::NAN] {
        let got = frame_after(smear(width), None);
        assert_bits(&got, &POLAR_FRAME, &format!("smear of width {width}"));
    }
}

#[test]
fn pv_rect_comb_zeroes_teeth_without_conversion() {
    let frame = test_frame();
    let (num_teeth, start_phase, width) = (4.0f32, 0.0f32, 0.5f32);
    let comb = |chain: InputRef| {
        UnitSpec::new(
            "PV_RectComb",
            Rate::Control,
            vec![chain, c(num_teeth), c(start_phase), c(width)],
            1,
        )
    };

    // The reference walks a phase across the frame, one step of `numTeeth / (numbins + 1)` per slot
    // from the DC term through every bin to the Nyquist term, zeroing a slot whenever the phase is
    // past `width`. Replaying that here gives the tooth pattern.
    let step = num_teeth / (BINS + 1) as f32;
    let wrap = |p: f32| {
        if p >= 1.0 {
            p - 1.0
        } else if p < 0.0 {
            p + 1.0
        } else {
            p
        }
    };
    let mut phase = start_phase;
    let dc_zeroed = phase > width;
    phase = wrap(phase + step);
    let mut zeroed = Vec::with_capacity(BINS);
    for _ in 0..BINS {
        zeroed.push(phase > width);
        phase = wrap(phase + step);
    }
    let nyq_zeroed = phase > width;
    assert!(
        zeroed.iter().any(|&z| z) && zeroed.iter().any(|&z| !z),
        "the comb must both keep and zero bins for this to be discriminating"
    );

    let got = frame_after(comb(c(0.0)), None);
    assert_eq!(got[0] == 0.0, dc_zeroed, "the DC term follows the comb");
    assert_eq!(
        got[1] == 0.0,
        nyq_zeroed,
        "the Nyquist term follows the comb"
    );
    for (i, &zero) in zeroed.iter().enumerate() {
        let (re, im) = (got[2 + 2 * i], got[3 + 2 * i]);
        if !zero {
            // Bit-identical, not merely close: the op edits the packed frame without converting it,
            // so a surviving bin cannot have been through a polar round trip.
            assert_eq!(
                (re, im),
                (frame[2 + 2 * i], frame[3 + 2 * i]),
                "bin {i} is inside a tooth and must be untouched"
            );
        } else {
            assert_eq!((re, im), (0.0, 0.0), "bin {i} is outside every tooth");
        }
    }
    assert!(
        polar(0) != (frame[2], frame[3]),
        "a polar conversion must change the stored pair for the bit-identity check to discriminate"
    );

    // Behind a polar predecessor the surviving slots keep the magnitude/phase pairs they arrived
    // with, so the op left the frame's coordinate form exactly as it found it.
    let got = frame_after(comb(u(0)), Some(polar_identity(c(0.0))));
    for (i, &zero) in zeroed.iter().enumerate() {
        if !zero {
            assert_eq!(
                (got[2 + 2 * i], got[3 + 2 * i]),
                polar(i),
                "bin {i} should still be the polar pair"
            );
        }
    }
}

#[test]
fn pv_new_units_pass_through_on_missing_chain() {
    let frame = test_frame();
    let ops = [
        bin_shift(u(0), 1.0, 1.0, 0.0),
        UnitSpec::new("PV_MagSmear", Rate::Control, vec![u(0), c(1.0)], 1),
        UnitSpec::new(
            "PV_RectComb",
            Rate::Control,
            vec![u(0), c(4.0), c(0.0), c(0.5)],
            1,
        ),
    ];
    for op in ops {
        let name = op.name.clone();
        // Channel 0 reads buffer 1 - the wider frame the size-change case points the op at - and
        // channel 1 carries the chain index the op passes on. Adding zero at audio rate carries a
        // control value into a block exactly, where `K2A` would interpolate across the step.
        let units = vec![
            UnitSpec::new("In", Rate::Control, vec![c(CHAIN_BUS as f32)], 1),
            op,
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![u(1), c(0.0)],
                num_outputs: 1,
                special_index: 0,
            },
        ];
        // The counter wraps every block, so channel 0 always reads the wide frame's first `FRAME`
        // slots - enough to see any edit, since both sizing ops rewrite from bin 0 up.
        let units = read_frame(units, 1.0, FRAME, vec![u(2)]);
        let (mut controller, mut world) = frame_engine(units, &frame, 2);

        // No frame ready: the chain index is normalised to -1, which ends the chain downstream.
        controller
            .set_control_bus(CHAIN_BUS, -1.0)
            .expect("set bus");
        let out = one_block(&mut world, 2);
        assert_eq!(
            channel(&out, 2, 1)[0],
            -1.0,
            "{name} must emit -1 between frames"
        );

        // A buffer number with nothing behind it: the index passes through unchanged, because
        // writing -1 there would end the chain for every unit downstream too.
        controller.set_control_bus(CHAIN_BUS, 5.0).expect("set bus");
        let out = one_block(&mut world, 2);
        assert_eq!(
            channel(&out, 2, 1)[0],
            5.0,
            "{name} must pass a missing buffer's index through"
        );

        // A first real frame, then one of a different size. `PV_BinShift` and `PV_MagSmear` size
        // their per-bin state from the first frame they see and pass any other size through
        // untouched; `PV_RectComb` keeps no per-bin state, so it edits either size.
        controller.set_control_bus(CHAIN_BUS, 0.0).expect("set bus");
        one_block(&mut world, 2);
        controller.set_control_bus(CHAIN_BUS, 1.0).expect("set bus");
        let out = one_block(&mut world, 2);
        assert_eq!(
            channel(&out, 2, 1)[0],
            1.0,
            "{name} must pass a differently sized frame's index through"
        );
        if name != "PV_RectComb" {
            let wide = channel(&out, 2, 0);
            for (i, &s) in wide.iter().enumerate() {
                assert_eq!(
                    s,
                    1.0 + i as f32,
                    "{name} must leave a differently sized frame untouched (slot {i})"
                );
            }
        }
    }
}

#[test]
fn pv_bin_shift_and_mag_smear_size_their_scratch_from_the_frame() {
    // A 16384-frame chain: the scratch is allocated on the first frame at the chain buffer's own
    // size (scsynth's `MAKE_TEMP_BUF`), so no size ceiling applies. Reading back the frame's head:
    // a one-bin shift clears bin 0 and moves the old bin 0 into bin 1.
    const BIG: usize = 16_384;
    let big: Vec<f32> = (0..BIG).map(|i| 1.0 + (i % 97) as f32 / 97.0).collect();
    let (_c, mut world) = frame_engine(
        read_frame(vec![bin_shift(c(0.0), 1.0, 1.0, 0.0)], 0.0, FRAME, vec![]),
        &big,
        1,
    );
    let got = one_block(&mut world, 1);
    assert_eq!((got[2], got[3]), (0.0, 0.0), "bin 0 is cleared");
    assert_eq!(
        (got[4], got[5]),
        (big[2], big[3]),
        "old bin 0 lands in bin 1"
    );

    let smear = UnitSpec::new("PV_MagSmear", Rate::Control, vec![c(0.0), c(1.0)], 1);
    let (_c, mut world) = frame_engine(read_frame(vec![smear], 0.0, FRAME, vec![]), &big, 1);
    let got = one_block(&mut world, 1);
    // scsynth's `PV_MagSmear(bins 1)` over the whole 16384-frame buffer, from the same harness as
    // the frames above: the head of the frame, bins 0..=2 included.
    const BIG_SMEARED_HEAD: [u32; 8] = [
        0x3f800000, 0x3f8151d0, 0x3f79f743, 0x3f4a71c0, 0x3fbd5cae, 0x3f4a71c0, 0x3fc11b3e,
        0x3f4a516c,
    ];
    assert_bits(
        &got[..8],
        &BIG_SMEARED_HEAD,
        "head of a smeared 16384-frame chain",
    );
}

#[test]
fn a_failed_scratch_allocation_outputs_no_frame() {
    // A unit pool too small for the scratch: the op is silenced with `-1` on its chain output
    // (scsynth's `FFT_ClearUnitOutputs`), so nothing downstream sees a ready frame.
    for op in ["PV_BinShift", "PV_MagSmear"] {
        let (mut controller, _nrt, mut world) = engine(Options {
            sample_rate: SR,
            block_size: BLOCK,
            output_channels: 1,
            unit_pool_bytes: 64,
            ..Options::default()
        });
        controller
            .buffer_set(0, Box::new(Buffer::from_interleaved(test_frame(), 1, SR)))
            .unwrap();
        controller.add_synthdef(SynthDef {
            name: "t".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new(op, Rate::Control, vec![c(0.0), c(1.0), c(1.0), c(0.0)], 1),
                UnitSpec::new("K2A", Rate::Audio, vec![u(0)], 1),
                UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
            ],
        });
        controller
            .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();
        one_block(&mut world, 1);
        let got = one_block(&mut world, 1);
        assert!(
            got.iter().all(|&s| s == -1.0),
            "{op}: no frame, got {got:?}"
        );
    }
}
