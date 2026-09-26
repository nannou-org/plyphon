//! The spectral units' polar/Cartesian conversion, pinned against scsynth's lookup tables
//! (`SC_Complex.h`'s `ToPolarApx`/`ToComplexApx`).
//!
//! A packed frame is put in a chain buffer, a unit that converts it without otherwise changing it
//! runs for one control block, and the converted frame is read straight back out of the buffer. The
//! expected bit patterns come from a C++ harness that includes scsynth's `FFT_UGens.h` and
//! `SC_Complex.h` and runs the same buffer-level conversions; they are the same on macOS and glibc
//! Linux.
//!
//! Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};
use plyphon_dsp::SpectrumCoord;

const SR: f64 = 48_000.0;
/// Samples per control block, equal to [`FRAME`] so one block reads the whole frame.
const BLOCK: usize = 64;
/// Chain-buffer frames.
const FRAME: usize = 64;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// Run `units` over `frame` (in buffer 0, flagged `coord`) for one block, then read the frame back
/// with a non-interpolating `BufRd` driven by a sample counter.
fn converted(frame: &[f32], coord: SpectrumCoord, mut units: Vec<UnitSpec>) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 1,
        ..Options::default()
    });
    let mut buffer = Buffer::from_interleaved(frame.to_vec(), 1, SR);
    buffer.set_coord(coord);
    controller
        .buffer_set(0, Box::new(buffer))
        .expect("buffer_set");
    let phasor = units.len() as u32;
    units.push(UnitSpec::new(
        "Phasor",
        Rate::Audio,
        vec![c(0.0), c(1.0), c(0.0), c(FRAME as f32), c(0.0)],
        1,
    ));
    units.push(UnitSpec::new(
        "BufRd",
        Rate::Audio,
        vec![c(0.0), u(phasor), c(1.0), c(1.0)],
        1,
    ));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(phasor + 1)],
        0,
    ));
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; BLOCK];
    world.fill(&mut out, 1);
    out
}

/// Assert `got` matches the expected bit patterns slot by slot.
fn assert_bits(got: &[f32], want: &[u32], what: &str) {
    for (i, (g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w,
            "{what}: slot {i} is {g} ({:#010x}), scsynth has {} ({w:#010x})",
            g.to_bits(),
            f32::from_bits(w)
        );
    }
}

/// `[dc, nyq]` then Cartesian `(re, im)` pairs covering every quadrant, both axes in both
/// directions, zero (of either sign), equal components, tiny and huge ratios, a subnormal, an
/// infinite component and a NaN one.
#[rustfmt::skip]
const POLAR_INPUT: [f32; FRAME] = [
    0.8125, -0.4375,
    3.0, 4.0, -3.0, 4.0, 3.0, -4.0, -3.0, -4.0,
    4.0, 3.0, -4.0, 3.0, 4.0, -3.0, -4.0, -3.0,
    1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0,
    0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -2.0, -0.0,
    1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    1e-3, 0.7, 0.3, -1e-6, 2.5, 2.4999, -0.1, 5e4,
    1e-40, 0.0, f32::INFINITY, 1.0, 1.0, f32::NEG_INFINITY, f32::NAN, 0.0,
    0.123456, -0.654321, -7.5, 0.0001, -1e-30, 1e-30,
];

/// scsynth's `ToPolarApx` of [`POLAR_INPUT`].
const POLAR_SPREAD: [u32; FRAME] = [
    0x3f500000, 0xbee00000, 0x40a00000, 0x3f6d6338, 0x40a00000, 0x400db70d, 0x40a00000, 0x40ab6374,
    0x40a00000, 0x40823454, 0x40a00000, 0x3f24bc7d, 0x40a00000, 0x401fe0bb, 0x40a00000, 0xbf24bc7d,
    0x40a00000, 0x40723efa, 0x3f800000, 0x00000000, 0x3f800000, 0x40490fdb, 0x3f800000, 0x3fc90fdb,
    0x3f800000, 0x4096cbe4, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x40000000, 0x40490fdb, 0x3fb504f3, 0x3f490fda, 0x3fb504f3, 0x4016cbe4, 0x3fb504f3, 0x40afeddf,
    0x3fb504f3, 0x407b53d1, 0x3f333339, 0x3fc8efdb, 0x3e99999f, 0xba7ffffb, 0x406229e9, 0x3f48efd7,
    0x47435006, 0x3fc92fdb, 0x000116c2, 0x00000000, 0x7f800000, 0x00000000, 0x7f800000, 0x4096cbe4,
    0x00000000, 0x00000000, 0x3f2a7c5d, 0x409cc9b6, 0x40f00008, 0x4048ffdb, 0x0de57822, 0x4016cbe4,
];

/// `[dc, nyq]` then polar `(mag, phase)` pairs: phases on the axes, at and past a full turn,
/// negative ones (which wrap through the table's two's-complement index), tiny and large ones, a
/// negative and a zero magnitude, and the ends of the `[-pi/4, 7pi/4]` range `ToPolarApx` itself
/// produces. The literals near multiples of pi are deliberate: they straddle table entries.
#[rustfmt::skip]
#[allow(clippy::approx_constant)]
fn complex_input() -> [f32; FRAME] {
    use core::f32::consts::PI;
    [
        -0.5, 1.5,
        1.0, 0.0, 1.0, 0.5, 1.0, PI / 2.0, 1.0, PI,
        1.0, 3.0 * PI / 2.0, 1.0, 2.0 * PI, 1.0, 7.0, 1.0, 100.0,
        1.0, -0.5, 1.0, -PI, 1.0, -7.0, 1.0, -100.0,
        1.0, -1e-7, 1.0, 1e5, 1.0, -1e5, 1.0, 1e6,
        -2.0, 0.25, 0.0, 1.3, 0.7, 6.2831, 0.7, 6.2832,
        2.5, -0.0001, 2.5, 0.0001, 3.0, 1.5707, 3.0, 4.7124,
        1.0, -0.7853982, 1.0, 5.497787, 1.0, 1e-38, 1.0, -0.0,
        1.75, 3.3, 0.5, 12.566371, 0.5, -12.566371,
    ]
}

/// scsynth's `ToComplexApx` of [`complex_input`].
const COMPLEX_SPREAD: [u32; FRAME] = [
    0xbf000000, 0x3fc00000, 0x3f800000, 0x00000000, 0x3f60bee5, 0x3ef527f8, 0x250d3132, 0x3f800000,
    0xbf800000, 0x250d3132, 0x00000000, 0xbf800000, 0x3f800000, 0x00000000, 0x3f4112ec, 0x3f281a40,
    0x3f5cae5b, 0xbf01c0ca, 0x3f60bee5, 0xbef527f8, 0xbf800000, 0x250d3132, 0x3f4112ec, 0xbf281a40,
    0x3f5cae5b, 0x3f01c0ca, 0x3f800000, 0x00000000, 0xbf7fd56c, 0x3d139f75, 0xbf7fd56c, 0xbd139f75,
    0x3f6fc0bf, 0xbeb37e82, 0xbff8166f, 0xbefc9e82, 0x00000000, 0x00000000, 0x3f333330, 0xba0cbe4b,
    0x3f333333, 0x00000000, 0x40200000, 0x00000000, 0x40200000, 0x00000000, 0x3b16cbe3, 0x403ffffc,
    0x00000000, 0xc0400000, 0x3f3504f3, 0xbf3504f3, 0x3f34e165, 0xbf35287b, 0x3f800000, 0x00000000,
    0x3f800000, 0x00000000, 0xbfdd35b8, 0xbe8cfad0, 0x3f000000, 0x00000000, 0x3f000000, 0x00000000,
];

#[test]
fn to_polar_matches_scsynth_tables() {
    // `PV_MagAbove` at threshold 0 converts the frame to polar and zeroes nothing, so the frame read
    // back is exactly the conversion. DC and Nyquist are real and pass through.
    let mag_above = UnitSpec::new("PV_MagAbove", Rate::Control, vec![c(0.0), c(0.0)], 1);
    let got = converted(&POLAR_INPUT, SpectrumCoord::Complex, vec![mag_above]);
    assert_bits(&got, &POLAR_SPREAD, "ToPolarApx");
}

#[test]
fn to_complex_matches_scsynth_tables() {
    // A polar frame through `PV_PhaseShift90` then `PV_PhaseShift270`: the first converts it to
    // Cartesian form, and the two quarter turns cancel exactly (they only swap and negate), so the
    // frame read back is exactly the conversion.
    let units = vec![
        UnitSpec::new("PV_PhaseShift90", Rate::Control, vec![c(0.0)], 1),
        UnitSpec::new("PV_PhaseShift270", Rate::Control, vec![u(0)], 1),
    ];
    let got = converted(&complex_input(), SpectrumCoord::Polar, units);
    assert_bits(&got, &COMPLEX_SPREAD, "ToComplexApx");
}
