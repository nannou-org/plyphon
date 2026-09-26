//! Randomised spectral operators: `PV_MagNoise`, `PV_RandComb`, `PV_RandWipe` and `PV_BinScramble`.
//!
//! Each is driven over pre-filled chain buffers, with its inputs on control buses, and the frame it
//! leaves is read back out of the buffer by a counter driving a non-interpolating `BufRd` in the
//! same block. The expected frames are bit patterns from scsynth's own `PV_UGens.cpp` functions,
//! compiled against the real headers and driven block by block with the same inputs and a random
//! stream seeded `RGen::init(0)` (stream 0 of a fresh engine); the polar case converts with
//! scsynth's `SC_Complex.h` lookup tables on both sides. No step calls a libm transcendental function,
//! so the values hold on every platform.
//!
//! Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Samples per control block, equal to [`FRAME`] so one block reads the whole frame back.
const BLOCK: usize = 64;
/// Chain-buffer frames.
const FRAME: usize = 64;
/// Bins in a packed [`FRAME`] frame, which holds `[dc, nyq, bins...]`.
const BINS: usize = (FRAME - 2) / 2;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// `In.kr(bus)`.
fn bus_in(bus: u32) -> UnitSpec {
    UnitSpec::new("In", Rate::Control, vec![c(bus as f32)], 1)
}

/// A packed Cartesian spectrum with pairwise-distinct, non-zero terms.
fn test_frame() -> Vec<f32> {
    let mut data = vec![0.0f32; FRAME];
    data[0] = 0.8125;
    data[1] = -0.4375;
    for i in 0..BINS {
        data[2 + 2 * i] = 0.5 + i as f32 / 32.0;
        data[3 + 2 * i] = -0.25 - i as f32 / 64.0;
    }
    data
}

/// A second spectrum, distinct from [`test_frame`] in every slot.
fn b_frame() -> Vec<f32> {
    (0..FRAME).map(|j| 3.0 + j as f32 / 16.0).collect()
}

/// Append a frame counter driving a non-interpolating `BufRd` over buffer 0, then `Out` with the
/// frame on channel 0 and `extra` after it.
fn read_frame(mut units: Vec<UnitSpec>, extra: Vec<InputRef>) -> Vec<UnitSpec> {
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
    let mut out = vec![c(0.0), u(phasor + 1)];
    out.extend(extra);
    units.push(UnitSpec::new("Out", Rate::Audio, out, 0));
    units
}

/// Carry a control value into an audio block exactly (adding zero at audio rate).
fn to_audio(src: u32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![u(src), c(0.0)],
        num_outputs: 1,
        special_index: 0,
    }
}

/// An engine running `units` as one synth over `channels` outputs, with buffers `1..` installed.
fn frame_engine(
    units: Vec<UnitSpec>,
    channels: usize,
    extra_buffers: &[Vec<f32>],
    unit_pool_bytes: Option<usize>,
) -> (plyphon::Controller, World) {
    let mut options = Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: channels,
        ..Options::default()
    };
    if let Some(bytes) = unit_pool_bytes {
        options.unit_pool_bytes = bytes;
    }
    let (mut controller, _nrt, world) = engine(options);
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(test_frame(), 1, SR)))
        .expect("buffer_set");
    for (i, data) in extra_buffers.iter().enumerate() {
        controller
            .buffer_set(
                1 + i,
                Box::new(Buffer::from_interleaved(data.clone(), 1, SR)),
            )
            .expect("buffer_set");
    }
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

/// Install `frame` in buffer 0, set the control buses from `0`, and render one block; returns
/// each channel's samples.
fn step(
    controller: &mut plyphon::Controller,
    world: &mut World,
    frame: &[f32],
    buses: &[f32],
    channels: usize,
) -> Vec<Vec<f32>> {
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(frame.to_vec(), 1, SR)))
        .expect("buffer_set");
    for (bus, &v) in buses.iter().enumerate() {
        controller.set_control_bus(bus as u32, v).expect("set bus");
    }
    let mut buf = vec![0.0f32; BLOCK * channels];
    world.fill(&mut buf, channels);
    (0..channels)
        .map(|ch| buf.iter().skip(ch).step_by(channels).copied().collect())
        .collect()
}

/// Assert that `got` is bit-identical to the scsynth frame `want`.
fn assert_frame(got: &[f32], want: &[u32], what: &str) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w,
            "{what}: slot {i} is {g} ({:#010x}), scsynth has {} ({w:#010x})",
            g.to_bits(),
            f32::from_bits(w)
        );
    }
}

#[test]
fn pv_mag_noise_matches_scsynth_in_both_forms() {
    // A Cartesian frame, twice: each frame draws one value per bin, then DC, then Nyquist, and
    // scales both halves of a bin by its draw. The second frame scales the first's result.
    let units = read_frame(
        vec![UnitSpec::new("PV_MagNoise", Rate::Control, vec![c(0.0)], 1)],
        vec![],
    );
    let (_c, mut world) = frame_engine(units, 1, &[], None);
    for (block, want) in MAG_NOISE_COMPLEX.iter().enumerate() {
        let mut buf = vec![0.0f32; BLOCK];
        world.fill(&mut buf, 1);
        assert_frame(&buf, want, &format!("complex frame {block}"));
    }

    // Behind `PV_MagAbove(0)`, which converts the frame to polar with scsynth's lookup tables
    // (`ToPolarApx`): only the magnitudes are scaled and the phases pass through.
    let units = read_frame(
        vec![
            UnitSpec::new("PV_MagAbove", Rate::Control, vec![c(0.0), c(0.0)], 1),
            UnitSpec::new("PV_MagNoise", Rate::Control, vec![u(0)], 1),
        ],
        vec![],
    );
    let (mut controller, mut world) = frame_engine(units, 1, &[], None);
    let got = step(&mut controller, &mut world, &test_frame(), &[], 1);
    assert_frame(&got[0], &MAG_NOISE_POLAR[0], "polar frame");
}

#[test]
fn pv_rand_comb_matches_scsynth() {
    // Buses: 0 chain, 1 wipe, 2 trig. Channel 1 carries the chain the unit passes on.
    let units = read_frame(
        vec![
            bus_in(0),
            bus_in(1),
            bus_in(2),
            UnitSpec::new("PV_RandComb", Rate::Control, vec![u(0), u(1), u(2)], 1),
            to_audio(3),
        ],
        vec![u(4)],
    );
    let (mut controller, mut world) = frame_engine(units, 2, &[], None);
    let steps: [[f32; 3]; 6] = [
        // No frame yet: the rising trig latches.
        [-1.0, 0.25, 1.0],
        // The first frame allocates and chooses, keeping the latch...
        [0.0, 0.25, 1.0],
        // ...so the next frame chooses again.
        [0.0, 0.25, 1.0],
        // The same ordering, a wider wipe.
        [0.0, 0.5, 0.0],
        // A rising trig chooses again.
        [0.0, 0.5, 1.0],
        // A full wipe clears DC and Nyquist too.
        [0.0, 1.0, 0.0],
    ];
    for (block, buses) in steps.iter().enumerate() {
        let got = step(&mut controller, &mut world, &test_frame(), buses, 2);
        assert_frame(&got[0], &RAND_COMB[block], &format!("block {block}"));
        assert_eq!(
            got[1][0].to_bits(),
            RAND_COMB_OUT[block],
            "chain out {block}"
        );
    }
}

#[test]
fn pv_rand_wipe_matches_scsynth() {
    // Buffer 1 holds B. Buses: 0 chain B, 1 wipe, 2 trig; chain A is buffer 0.
    let units = read_frame(
        vec![
            bus_in(0),
            bus_in(1),
            bus_in(2),
            UnitSpec::new(
                "PV_RandWipe",
                Rate::Control,
                vec![c(0.0), u(0), u(1), u(2)],
                1,
            ),
            to_audio(3),
        ],
        vec![u(4)],
    );
    let (mut controller, mut world) = frame_engine(units, 2, &[b_frame()], None);
    let steps: [[f32; 3]; 7] = [
        // The first frame allocates and chooses.
        [1.0, 0.25, 0.0],
        // No frame on B: the output is -1, the frame untouched, and the rising trig latches.
        [-1.0, 0.5, 1.0],
        // The latch chooses again.
        [1.0, 0.5, 1.0],
        // A full wipe copies every bin, but not DC or Nyquist.
        [1.0, 1.0, 0.0],
        // B is A itself: the rising trig still chooses again, and each bin is copied onto itself.
        [0.0, 0.5, 1.0],
        // The new ordering, from B.
        [1.0, 0.5, 0.0],
        // A negative wipe copies nothing.
        [1.0, -0.5, 0.0],
    ];
    for (block, buses) in steps.iter().enumerate() {
        let got = step(&mut controller, &mut world, &test_frame(), buses, 2);
        assert_frame(&got[0], &RAND_WIPE[block], &format!("block {block}"));
        assert_eq!(
            got[1][0].to_bits(),
            RAND_WIPE_OUT[block],
            "chain out {block}"
        );
    }
}

#[test]
fn pv_bin_scramble_matches_scsynth() {
    // Buses: 0 wipe, 1 width, 2 trig.
    let units = read_frame(
        vec![
            bus_in(0),
            bus_in(1),
            bus_in(2),
            UnitSpec::new(
                "PV_BinScramble",
                Rate::Control,
                vec![c(0.0), u(0), u(1), u(2)],
                1,
            ),
        ],
        vec![],
    );
    let (mut controller, mut world) = frame_engine(units, 1, &[], None);
    let steps: [[f32; 3]; 5] = [
        // The first frame chooses with a window of 0.2 of the spectrum.
        [0.5, 0.2, 0.0],
        // Every bin from its source.
        [1.0, 0.2, 0.0],
        // A rising trig chooses again, with a narrower window.
        [0.75, 0.05, 1.0],
        // The width is only read when choosing: the ordering stands.
        [0.3, 1.0, 0.0],
        // A rising trig chooses with the whole spectrum as the window; the wipe clips to 1.
        [2.0, 1.0, 1.0],
    ];
    for (block, buses) in steps.iter().enumerate() {
        let got = step(&mut controller, &mut world, &test_frame(), buses, 1);
        assert_frame(&got[0], &BIN_SCRAMBLE[block], &format!("block {block}"));
    }
}

#[test]
fn a_failed_table_allocation_outputs_no_frame() {
    // A unit pool too small for the first frame's table: the unit is silenced with `-1` on its
    // chain output, so nothing downstream sees a ready frame.
    for op in ["PV_RandComb", "PV_RandWipe", "PV_BinScramble"] {
        let units = vec![
            UnitSpec::new(op, Rate::Control, vec![c(0.0), c(0.0), c(1.0), c(0.0)], 1),
            to_audio(0),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ];
        let (_c, mut world) = frame_engine(units, 1, &[], Some(64));
        let mut buf = vec![0.0f32; BLOCK];
        world.fill(&mut buf, 1);
        world.fill(&mut buf, 1);
        assert!(
            buf.iter().all(|&s| s == -1.0),
            "{op}: no frame, got {buf:?}"
        );
    }
}

// Bit patterns from scsynth's `PV_UGens.cpp`, as described in the module docs.

const MAG_NOISE_COMPLEX: [[u32; 64]; 2] = [
    [
        0x3e1a74f9, 0xbe562bc6, 0x3eb710fc, 0xbe3710fc, 0xbee88b0c, 0x3e688b0c, 0xbec75801,
        0x3e475801, 0x3d3f00cc, 0xbcbf00cc, 0xbe1c2250, 0x3d9c2250, 0xbe3e60f0, 0x3dbe60f0,
        0x3e3ce534, 0xbdbce534, 0x3eeeae54, 0xbe6eae54, 0xbf1b02f5, 0x3e9b02f5, 0x3f2e8f89,
        0xbeae8f89, 0xbe13315d, 0x3d93315d, 0x3e00e294, 0xbd80e294, 0xbf4e2037, 0x3ece2037,
        0x3f21e0dc, 0xbea1e0dc, 0x3d74db98, 0xbcf4db98, 0xbe6285bd, 0x3de285bd, 0x3c235200,
        0xbba35200, 0xbf1c963c, 0x3e9c963c, 0x3f5e9ed3, 0xbede9ed3, 0x3f0b111a, 0xbe8b111a,
        0xbf7fafae, 0x3effafae, 0x3df12205, 0xbd712205, 0x3f5346b1, 0xbed346b1, 0x3f5427f0,
        0xbed427f0, 0x3f86d945, 0xbf06d945, 0xbf1aa7e3, 0x3e9aa7e3, 0x3d88ea7a, 0xbd08ea7a,
        0xbf909bff, 0x3f109bff, 0xbe9aaab7, 0x3e1aaab7, 0x3e514e85, 0xbdd14e85, 0x3f0e3a5f,
        0xbe8e3a5f,
    ],
    [
        0x3d86920a, 0x3dcd9a3f, 0xbea0ec7d, 0x3e20ec7d, 0x3e361568, 0xbdb61568, 0x3ea8bd97,
        0xbe28bd97, 0xbc27c513, 0x3ba7c513, 0x3d35fee0, 0xbcb5fee0, 0x3db3744f, 0xbd33744f,
        0xbc51f478, 0x3bd1f478, 0x3e7fe750, 0xbdffe750, 0xbead97d1, 0x3e2d97d1, 0xbeefdfc6,
        0x3e6fdfc6, 0x3d5e7c4b, 0xbcde7c4b, 0x3dc0f2c3, 0xbd40f2c3, 0xbf1d2980, 0x3e9d2980,
        0xbef3e549, 0x3e73e549, 0x3d1989f4, 0xbc9989f4, 0x3d2557f3, 0xbca557f3, 0x3b72a0a3,
        0xbaf2a0a3, 0x3f100741, 0xbe900741, 0xbefdaa3a, 0x3e7daa3a, 0xbe96afcb, 0x3e16afcb,
        0x3f465303, 0xbec65303, 0xbdceebb0, 0x3d4eebb0, 0xbf0cffd0, 0x3e8cffd0, 0x3e447061,
        0xbdc47061, 0x3e83db35, 0xbe03db35, 0xbe27941c, 0x3da7941c, 0xbcbbb122, 0x3c3bb122,
        0xbe3d7fa4, 0x3dbd7fa4, 0x3e7e6169, 0xbdfe6169, 0xbd9db71e, 0x3d1db71e, 0xbe8c3aed,
        0x3e0c3aed,
    ],
];
const MAG_NOISE_POLAR: [[u32; 64]; 1] = [[
    0x3e1a74f9, 0xbe562bc6, 0x3eccaca5, 0xbeed6338, 0xbf01fede, 0xbeed6338, 0xbededf85, 0xbeed6338,
    0x3d558c48, 0xbeed6338, 0xbe2e902b, 0xbeed6338, 0xbe54d98e, 0xbeed6338, 0x3e5330ff, 0xbeed6338,
    0x3f056d3e, 0xbeed6338, 0xbf2d4ee6, 0xbeed6338, 0x3f432a2f, 0xbeed6338, 0xbe24910c, 0xbeed6338,
    0x3e101910, 0xbeed6338, 0xbf6674a9, 0xbeed6338, 0x3f34fc4a, 0xbeed6338, 0x3d88e132, 0xbeed6338,
    0xbe7d4280, 0xbeed6338, 0x3c369900, 0xbeed6338, 0xbf2f11c6, 0xbeed6338, 0x3f78e5ae, 0xbeed6338,
    0x3f1b7b3e, 0xbeed6338, 0xbf8eeed7, 0xbeed6338, 0x3e06cc22, 0xbeed6338, 0x3f6c36c2, 0xbeed6338,
    0x3f6d3297, 0xbeed6338, 0x3f96c3f4, 0xbeed6338, 0xbf2ce913, 0xbeed6338, 0x3d9913a0, 0xbeed6338,
    0xbfa1ad9e, 0xbeed6338, 0xbeacec3d, 0xbeed6338, 0x3e6a0314, 0xbeed6338, 0x3f1f0408, 0xbeed6338,
]];
const RAND_COMB: [[u32; 64]; 6] = [
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x3f780000, 0xbef80000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f940000, 0xbf140000, 0x3f980000, 0xbf180000, 0x3f9c0000,
        0xbf1c0000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x3fb40000, 0xbf340000, 0x3fb80000,
        0xbf380000,
    ],
    [
        0x3f500000, 0xbee00000, 0x00000000, 0x00000000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x00000000, 0x00000000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x00000000, 0x00000000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x00000000, 0x00000000, 0x3f980000, 0xbf180000, 0x3f9c0000,
        0xbf1c0000, 0x00000000, 0x00000000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x00000000, 0x00000000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x3f780000, 0xbef80000, 0x3f800000,
        0xbf000000, 0x00000000, 0x00000000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f940000, 0xbf140000, 0x00000000, 0x00000000, 0x3f9c0000,
        0xbf1c0000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x3fb00000, 0xbf300000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ],
    [
        0x3f500000, 0xbee00000, 0x00000000, 0x00000000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x00000000, 0x00000000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f940000, 0xbf140000, 0x00000000, 0x00000000, 0x3f9c0000,
        0xbf1c0000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x3fb00000, 0xbf300000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x00000000,
        0x00000000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3f780000, 0xbef80000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ],
    [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ],
];
const RAND_COMB_OUT: [u32; 6] = [
    0xbf800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
];
const RAND_WIPE: [[u32; 64]; 7] = [
    [
        0x3f500000, 0xbee00000, 0x40480000, 0x404c0000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x408c0000, 0x408e0000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x40a00000, 0x40a20000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x40b80000, 0x40ba0000, 0x3f980000, 0xbf180000, 0x3f9c0000,
        0xbf1c0000, 0x40c40000, 0x40c60000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x40d80000, 0x40da0000, 0x40dc0000,
        0x40de0000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x3f780000, 0xbef80000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f940000, 0xbf140000, 0x3f980000, 0xbf180000, 0x3f9c0000,
        0xbf1c0000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x3fb40000, 0xbf340000, 0x3fb80000,
        0xbf380000,
    ],
    [
        0x3f500000, 0xbee00000, 0x40480000, 0x404c0000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x40600000, 0x40640000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x40780000, 0x407c0000, 0x40800000, 0x40820000, 0x40840000, 0x40860000, 0x40880000,
        0x408a0000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x409c0000, 0x409e0000, 0x40a00000, 0x40a20000, 0x40a40000,
        0x40a60000, 0x40a80000, 0x40aa0000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f940000, 0xbf140000, 0x40bc0000, 0x40be0000, 0x3f9c0000,
        0xbf1c0000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x40cc0000, 0x40ce0000,
        0x40d00000, 0x40d20000, 0x3fb00000, 0xbf300000, 0x40d80000, 0x40da0000, 0x40dc0000,
        0x40de0000,
    ],
    [
        0x3f500000, 0xbee00000, 0x40480000, 0x404c0000, 0x40500000, 0x40540000, 0x40580000,
        0x405c0000, 0x40600000, 0x40640000, 0x40680000, 0x406c0000, 0x40700000, 0x40740000,
        0x40780000, 0x407c0000, 0x40800000, 0x40820000, 0x40840000, 0x40860000, 0x40880000,
        0x408a0000, 0x408c0000, 0x408e0000, 0x40900000, 0x40920000, 0x40940000, 0x40960000,
        0x40980000, 0x409a0000, 0x409c0000, 0x409e0000, 0x40a00000, 0x40a20000, 0x40a40000,
        0x40a60000, 0x40a80000, 0x40aa0000, 0x40ac0000, 0x40ae0000, 0x40b00000, 0x40b20000,
        0x40b40000, 0x40b60000, 0x40b80000, 0x40ba0000, 0x40bc0000, 0x40be0000, 0x40c00000,
        0x40c20000, 0x40c40000, 0x40c60000, 0x40c80000, 0x40ca0000, 0x40cc0000, 0x40ce0000,
        0x40d00000, 0x40d20000, 0x40d40000, 0x40d60000, 0x40d80000, 0x40da0000, 0x40dc0000,
        0x40de0000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x3f780000, 0xbef80000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f940000, 0xbf140000, 0x3f980000, 0xbf180000, 0x3f9c0000,
        0xbf1c0000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x3fb40000, 0xbf340000, 0x3fb80000,
        0xbf380000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x40580000,
        0x405c0000, 0x40600000, 0x40640000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x40880000,
        0x408a0000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x40940000, 0x40960000,
        0x40980000, 0x409a0000, 0x409c0000, 0x409e0000, 0x3f780000, 0xbef80000, 0x40a40000,
        0x40a60000, 0x40a80000, 0x40aa0000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x40b80000, 0x40ba0000, 0x40bc0000, 0x40be0000, 0x40c00000,
        0x40c20000, 0x40c40000, 0x40c60000, 0x40c80000, 0x40ca0000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x40d80000, 0x40da0000, 0x40dc0000,
        0x40de0000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x3f780000, 0xbef80000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f940000, 0xbf140000, 0x3f980000, 0xbf180000, 0x3f9c0000,
        0xbf1c0000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x3fb40000, 0xbf340000, 0x3fb80000,
        0xbf380000,
    ],
];
const RAND_WIPE_OUT: [u32; 7] = [
    0x00000000, 0xbf800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
];
const BIN_SCRAMBLE: [[u32; 64]; 5] = [
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f380000, 0xbeb80000, 0x3f280000, 0xbea80000,
        0x3f500000, 0xbed00000, 0x3f300000, 0xbeb00000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f200000, 0xbea00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f840000, 0xbf040000, 0x3f600000, 0xbee00000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f680000, 0xbee80000, 0x3f8c0000, 0xbf0c0000,
        0x3fa00000, 0xbf200000, 0x3f8c0000, 0xbf0c0000, 0x3f8c0000, 0xbf0c0000, 0x3f9c0000,
        0xbf1c0000, 0x3f980000, 0xbf180000, 0x3fa40000, 0xbf240000, 0x3f9c0000, 0xbf1c0000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x3fb00000, 0xbf300000, 0x3fac0000,
        0xbf2c0000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f000000, 0xbe800000, 0x3f200000,
        0xbea00000, 0x3f080000, 0xbe880000, 0x3f380000, 0xbeb80000, 0x3f300000, 0xbeb00000,
        0x3f500000, 0xbed00000, 0x3f300000, 0xbeb00000, 0x3f300000, 0xbeb00000, 0x3f280000,
        0xbea80000, 0x3f200000, 0xbea00000, 0x3f680000, 0xbee80000, 0x3f400000, 0xbec00000,
        0x3f380000, 0xbeb80000, 0x3f840000, 0xbf040000, 0x3f600000, 0xbee00000, 0x3f580000,
        0xbed80000, 0x3f880000, 0xbf080000, 0x3f680000, 0xbee80000, 0x3f980000, 0xbf180000,
        0x3fa00000, 0xbf200000, 0x3f8c0000, 0xbf0c0000, 0x3f8c0000, 0xbf0c0000, 0x3f880000,
        0xbf080000, 0x3f980000, 0xbf180000, 0x3f980000, 0xbf180000, 0x3f9c0000, 0xbf1c0000,
        0x3f940000, 0xbf140000, 0x3fa80000, 0xbf280000, 0x3fb00000, 0xbf300000, 0x3fac0000,
        0xbf2c0000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f100000, 0xbe900000, 0x3f180000, 0xbe980000, 0x3f280000, 0xbea80000,
        0x3f280000, 0xbea80000, 0x3f300000, 0xbeb00000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f480000, 0xbec80000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000,
        0x3f680000, 0xbee80000, 0x3f680000, 0xbee80000, 0x3f780000, 0xbef80000, 0x3f780000,
        0xbef80000, 0x3f800000, 0xbf000000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f900000, 0xbf100000, 0x3f980000, 0xbf180000, 0x3f980000,
        0xbf180000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x3fb00000, 0xbf300000, 0x3fb40000,
        0xbf340000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f000000, 0xbe800000, 0x3f080000, 0xbe880000, 0x3f100000,
        0xbe900000, 0x3f180000, 0xbe980000, 0x3f200000, 0xbea00000, 0x3f280000, 0xbea80000,
        0x3f300000, 0xbeb00000, 0x3f380000, 0xbeb80000, 0x3f400000, 0xbec00000, 0x3f480000,
        0xbec80000, 0x3f500000, 0xbed00000, 0x3f580000, 0xbed80000, 0x3f600000, 0xbee00000,
        0x3f680000, 0xbee80000, 0x3f700000, 0xbef00000, 0x3f780000, 0xbef80000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f880000, 0xbf080000, 0x3f8c0000, 0xbf0c0000,
        0x3f900000, 0xbf100000, 0x3f900000, 0xbf100000, 0x3f980000, 0xbf180000, 0x3f980000,
        0xbf180000, 0x3fa00000, 0xbf200000, 0x3fa40000, 0xbf240000, 0x3fa80000, 0xbf280000,
        0x3fac0000, 0xbf2c0000, 0x3fb00000, 0xbf300000, 0x3fb00000, 0xbf300000, 0x3fb40000,
        0xbf340000,
    ],
    [
        0x3f500000, 0xbee00000, 0x3f080000, 0xbe880000, 0x3f280000, 0xbea80000, 0x3f800000,
        0xbf000000, 0x3f380000, 0xbeb80000, 0x3f680000, 0xbee80000, 0x3f780000, 0xbef80000,
        0x3f280000, 0xbea80000, 0x3f480000, 0xbec80000, 0x3f100000, 0xbe900000, 0x3f800000,
        0xbf000000, 0x3f840000, 0xbf040000, 0x3f600000, 0xbee00000, 0x3fb00000, 0xbf300000,
        0x3fa80000, 0xbf280000, 0x3f600000, 0xbee00000, 0x3f680000, 0xbee80000, 0x3f780000,
        0xbef80000, 0x3f000000, 0xbe800000, 0x3f680000, 0xbee80000, 0x3f8c0000, 0xbf0c0000,
        0x3f080000, 0xbe880000, 0x3f500000, 0xbed00000, 0x3f980000, 0xbf180000, 0x3fb00000,
        0xbf300000, 0x3f300000, 0xbeb00000, 0x3fa80000, 0xbf280000, 0x3f600000, 0xbee00000,
        0x3f840000, 0xbf040000, 0x3fa00000, 0xbf200000, 0x3f940000, 0xbf140000, 0x3f8c0000,
        0xbf0c0000,
    ],
];
