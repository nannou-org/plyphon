//! `FFTTrigger`, `PV_MagFreeze`, `PV_MagShift`, `PV_PhaseShift`, `PV_MagDiv`, `PV_BinWipe`,
//! `PV_RectComb2` and `PV_ConformalMap`.
//!
//! The spectral ops are driven over pre-filled chain buffers, with the chain signal supplied
//! directly (a constant buffer number, or a control bus where a test changes it between blocks), and
//! the frames are read straight back out of the buffers by a counter driving non-interpolating
//! `BufRd`s. One control block is exactly one frame long, so one rendered block carries each whole
//! packed frame as it stands after that block's spectral units ran.
//!
//! The expected frames are bit patterns from scsynth's own calc code (`PV_UGens.cpp`,
//! `PV_ThirdParty.cpp`) run over the same frames in a C++ harness built against scsynth's plugin
//! headers, converting with the real `ToPolarApx`/`ToComplexApx` over `SC_Complex.h`'s tables. They
//! are the same on macOS and Linux (glibc).
//!
//! Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Samples per control block, equal to [`FRAME`] so one block reads a whole frame.
const BLOCK: usize = 16;
/// Chain-buffer samples: `[dc, nyq]` and seven bins.
const FRAME: usize = 16;
/// Bins in a packed [`FRAME`] frame.
const BINS: usize = (FRAME - 2) / 2;
/// The buffer holding [`frame_x`].
const BUF_X: f32 = 0.0;
/// A buffer of a different size, for the size-mismatch cases.
const BUF_WIDE: f32 = 1.0;
/// The buffer holding [`frame_y`].
const BUF_Y: f32 = 2.0;
/// Samples in [`BUF_WIDE`].
const WIDE_FRAME: usize = 2 * FRAME;
/// The control bus carrying a chain signal where a test changes it between blocks.
const CHAIN_BUS: u32 = 0;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// A control-rate unit.
fn kr(name: &str, inputs: Vec<InputRef>) -> UnitSpec {
    UnitSpec::new(name, Rate::Control, inputs, 1)
}

/// `In.kr(CHAIN_BUS)`: the chain signal a test sets block by block.
fn chain_bus() -> UnitSpec {
    kr("In", vec![c(CHAIN_BUS as f32)])
}

/// A Cartesian packed spectrum with distinct, non-zero terms and phases that differ bin to bin.
fn frame_x() -> Vec<f32> {
    let mut data = vec![0.0f32; FRAME];
    data[0] = 0.8125;
    data[1] = -0.4375;
    for i in 0..BINS {
        data[2 + 2 * i] = 0.5 + i as f32 / 8.0;
        data[3 + 2 * i] = 0.625 - i as f32 / 4.0;
    }
    data
}

/// A second Cartesian packed spectrum of the same size, with some bins quieter than `0.5`.
fn frame_y() -> Vec<f32> {
    let mut data = vec![0.0f32; FRAME];
    data[0] = -0.3125;
    data[1] = 0.6875;
    for i in 0..BINS {
        data[2 + 2 * i] = -0.75 + i as f32 / 4.0;
        data[3 + 2 * i] = 0.375 - i as f32 / 32.0;
    }
    data
}

/// A frame's bit patterns.
fn bits(frame: &[f32]) -> Vec<u32> {
    frame.iter().map(|s| s.to_bits()).collect()
}

/// An engine holding [`frame_x`] in buffer 0, a wider frame in buffer 1 and [`frame_y`] in buffer 2,
/// running `units` as one synth over `channels` outputs.
fn frame_engine(units: Vec<UnitSpec>, channels: usize) -> (plyphon::Controller, World) {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: channels,
        ..Options::default()
    });
    let wide: Vec<f32> = (0..WIDE_FRAME).map(|i| 1.0 + i as f32).collect();
    for (index, data) in [(0, frame_x()), (1, wide), (2, frame_y())] {
        controller
            .buffer_set(index, Box::new(Buffer::from_interleaved(data, 1, SR)))
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

/// A control value carried into an audio block exactly (adding zero at audio rate, where `K2A`
/// would interpolate across a step).
fn to_audio(of: InputRef) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![of, c(0.0)],
        num_outputs: 1,
        special_index: 0,
    }
}

/// Append a frame counter driving one non-interpolating `BufRd` per buffer in `bufnums`, then an
/// `Out` of those reads followed by `extra` - after the units under test, so it reads the frames
/// they just wrote.
fn read_frames(mut units: Vec<UnitSpec>, bufnums: &[f32], extra: Vec<InputRef>) -> Vec<UnitSpec> {
    let phasor = units.len() as u32;
    units.push(UnitSpec::new(
        "Phasor",
        Rate::Audio,
        vec![c(0.0), c(1.0), c(0.0), c(FRAME as f32), c(0.0)],
        1,
    ));
    let mut out = vec![c(0.0)];
    for (k, &bufnum) in bufnums.iter().enumerate() {
        units.push(UnitSpec::new(
            "BufRd",
            Rate::Audio,
            vec![c(bufnum), u(phasor), c(1.0), c(1.0)],
            1,
        ));
        out.push(u(phasor + 1 + k as u32));
    }
    out.extend(extra);
    units.push(UnitSpec::new("Out", Rate::Audio, out, 0));
    units
}

/// The frame of buffer `bufnum` after `blocks` control blocks of `units`.
fn frame_after(units: Vec<UnitSpec>, bufnum: f32, blocks: usize) -> Vec<f32> {
    let (_c, mut world) = frame_engine(read_frames(units, &[bufnum], vec![]), 1);
    let mut out = Vec::new();
    for _ in 0..blocks {
        out = one_block(&mut world, 1);
    }
    out
}

/// `PV_MagAbove(chain, 0)`: an identity that leaves the frame in polar form.
fn polar_identity(chain: InputRef) -> UnitSpec {
    kr("PV_MagAbove", vec![chain, c(0.0)])
}

const MAG_FREEZE_HELD: [u32; 16] = [
    0x3f500000, 0xbee00000, 0x3f4ce165, 0x402b6374, 0x3f3a8eed, 0x402283fa, 0x3f42a0bd, 0x400fbeaf,
    0x3f624bdb, 0x3fc90fdb, 0x3f88b43d, 0x3f490fda, 0x3fa4bcd8, 0x3ed32776, 0x3fc352a4, 0x3e7adbb0,
];
const MAG_FREEZE_STORED: [u32; 16] = [
    0xbea00000, 0x3f300000, 0x3f56a99c, 0x402b6374, 0x3f1b54f8, 0x402283fa, 0x3eccfa64, 0x400fbeaf,
    0x3e900000, 0x3fc90fdb, 0x3eb504f3, 0x3f490fda, 0x3f0bb6c8, 0x3ed32776, 0x3f45e8b8, 0x3e7adbb0,
];

#[test]
fn pv_mag_freeze_holds_the_first_frame_magnitudes() {
    // Block 1 runs over X, block 2 over Y. With `freeze` held at 1 the first frame still stores
    // (the table must be filled before it is read), so Y comes out with X's magnitudes, DC and
    // Nyquist over its own phases.
    let run = |freeze: f32| {
        let units = vec![chain_bus(), kr("PV_MagFreeze", vec![u(0), c(freeze)])];
        let (mut controller, mut world) = frame_engine(read_frames(units, &[BUF_Y], vec![]), 1);
        controller.set_control_bus(CHAIN_BUS, BUF_X).unwrap();
        one_block(&mut world, 1);
        controller.set_control_bus(CHAIN_BUS, BUF_Y).unwrap();
        one_block(&mut world, 1)
    };
    assert_eq!(bits(&run(1.0)), MAG_FREEZE_HELD, "frozen");
    // Unfrozen, each frame stores its own magnitudes and passes through, left polar.
    assert_eq!(bits(&run(0.0)), MAG_FREEZE_STORED, "unfrozen");
}

const MAG_SHIFT_STRETCH: [u32; 16] = [
    0x3f500000, 0xbee00000, 0x3f3a8eed, 0x3f656bb1, 0x3f42a0bd, 0x3f0a461b, 0x3f624bdb, 0x3e2876aa,
    0x00000000, 0xbe1200a5, 0x3f88b43d, 0xbeb7b0ca, 0x3fa4bcd8, 0xbf01d6a4, 0x00000000, 0xbf1c6120,
];
const MAG_SHIFT_COMPRESS: [u32; 16] = [
    0x3f500000, 0xbee00000, 0x00000000, 0x3f656bb1, 0x00000000, 0x3f0a461b, 0x00000000, 0x3e2876aa,
    0x3f4ce165, 0xbe1200a5, 0x3fbe97d5, 0xbeb7b0ca, 0x3ff9da2a, 0xbf01d6a4, 0x403407be, 0xbf1c6120,
];

#[test]
fn pv_mag_shift_moves_magnitudes_and_keeps_phases() {
    let shift = |stretch: f32, shift: f32| {
        frame_after(
            vec![kr("PV_MagShift", vec![c(BUF_X), c(stretch), c(shift)])],
            BUF_X,
            1,
        )
    };
    // Bin `i` lands at `(int)(shift + i * stretch + 0.5)`: with a negative shift bin 1 truncates
    // toward zero into bin 0, and bins that land outside the spectrum are dropped.
    assert_eq!(bits(&shift(1.5, -2.25)), MAG_SHIFT_STRETCH, "stretched");
    // A compressing stretch lands pairs of bins on the same destination, where they add.
    assert_eq!(bits(&shift(0.5, 3.0)), MAG_SHIFT_COMPRESS, "compressed");
}

const PHASE_SHIFT_INTEGRATED: [u32; 16] = [
    0x3f500000, 0xbee00000, 0x3f4ce165, 0x419ce767, 0x3f3a8eed, 0x419a0e3a, 0x3f42a0bd, 0x41970cf7,
    0x3f624bdb, 0x41949808, 0x3f88b43d, 0x4192dd46, 0x3fa4bcd8, 0x4191ad54, 0x3fc352a4, 0x4190d900,
];
const PHASE_SHIFT_PLAIN: [u32; 16] = [
    0x3f500000, 0xbee00000, 0x3f4ce165, 0x412e56bb, 0x3f3a8eed, 0x4128a462, 0x3f42a0bd, 0x4122a1db,
    0x3f624bdb, 0x411db7fe, 0x3f88b43d, 0x411a427a, 0x3fa4bcd8, 0x4117e296, 0x3fc352a4, 0x411639ee,
];

#[test]
fn pv_phase_shift_integrates_its_offset() {
    // Four frames over the same buffer. Integrating, each frame adds `shift` plus the running total,
    // which wraps by `fmod` at one turn after the third frame.
    let shift = |integrate: f32| {
        frame_after(
            vec![kr("PV_PhaseShift", vec![c(BUF_X), c(2.5), c(integrate)])],
            BUF_X,
            4,
        )
    };
    assert_eq!(bits(&shift(1.0)), PHASE_SHIFT_INTEGRATED, "integrating");
    // `integrate` is truncated to an integer, so 0.9 does not integrate: each frame adds 2.5.
    assert_eq!(bits(&shift(0.9)), PHASE_SHIFT_PLAIN, "not integrating");
}

const MAG_DIV_A: [u32; 16] = [
    0x3fd00000, 0xbf22e8ba, 0x3f74558d, 0x3f656bb1, 0x3f99bb5b, 0x3f0a461b, 0x3fc2a0bd, 0x3e2876aa,
    0x3fe24bdb, 0xbe1200a5, 0x4008b43d, 0xbeb7b0ca, 0x4016ecf6, 0xbf01d6a4, 0x3ffca796, 0xbf1c6120,
];
const MAG_DIV_B: [u32; 16] = [
    0xbea00000, 0x3f300000, 0x3f56a99c, 0x402b6374, 0x3f1b54f8, 0x402283fa, 0x3eccfa64, 0x400fbeaf,
    0x3e900000, 0x3fc90fdb, 0x3eb504f3, 0x3f490fda, 0x3f0bb6c8, 0x3ed32776, 0x3f45e8b8, 0x3e7adbb0,
];
const MAG_DIV_SELF: [u32; 16] = [
    0x3f800000, 0xbf600000, 0x3f800000, 0x3f656bb1, 0x3f800000, 0x3f0a461b, 0x3f800000, 0x3e2876aa,
    0x3f800000, 0xbe1200a5, 0x3f800000, 0xbeb7b0ca, 0x3f800000, 0xbf01d6a4, 0x3f800000, 0xbf1c6120,
];

#[test]
fn pv_mag_div_divides_magnitudes_and_leaves_both_frames_polar() {
    // X / Y with divisors floored at 0.5 (Y's DC is negative and several bins are quieter). Like
    // scsynth, the op converts both frames, so Y is read back polar too.
    let units = vec![kr("PV_MagDiv", vec![c(BUF_X), c(BUF_Y), c(0.5)])];
    let (_c, mut world) = frame_engine(read_frames(units, &[BUF_X, BUF_Y], vec![]), 2);
    let out = one_block(&mut world, 2);
    assert_eq!(bits(&channel(&out, 2, 0)), MAG_DIV_A, "A");
    assert_eq!(bits(&channel(&out, 2, 1)), MAG_DIV_B, "B");
    assert_eq!(
        MAG_DIV_B, MAG_FREEZE_STORED,
        "B's end state is its own polar form"
    );

    // One buffer on both inputs divides each term by itself (floored).
    let got = frame_after(
        vec![kr("PV_MagDiv", vec![c(BUF_X), c(BUF_X), c(0.5)])],
        BUF_X,
        1,
    );
    assert_eq!(bits(&got), MAG_DIV_SELF, "self");
}

const BIN_WIPE_LOW: [u32; 16] = [
    0xbea00000, 0xbee00000, 0xbf400000, 0x3ec00000, 0xbf000000, 0x3eb00000, 0x3f400000, 0x3e000000,
    0x3f600000, 0xbe000000, 0x3f800000, 0xbec00000, 0x3f900000, 0xbf200000, 0x3fa00000, 0xbf600000,
];
const BIN_WIPE_HIGH: [u32; 16] = [
    0x3f500000, 0x3f300000, 0x3f000000, 0x3f200000, 0x3f200000, 0x3ec00000, 0x3f400000, 0x3e000000,
    0x3f600000, 0xbe000000, 0x3f800000, 0xbec00000, 0x3f000000, 0x3e600000, 0x3f400000, 0x3e400000,
];
const BIN_WIPE_ALL: [u32; 16] = [
    0xbea00000, 0x3f300000, 0xbf400000, 0x3ec00000, 0xbf000000, 0x3eb00000, 0xbe800000, 0x3ea00000,
    0x00000000, 0x3e900000, 0x3e800000, 0x3e800000, 0x3f000000, 0x3e600000, 0x3f400000, 0x3e400000,
];
const BIN_WIPE_MIXED: [u32; 16] = [
    0xbea00000, 0xbee00000, 0xbf400000, 0x3ec00000, 0xbf000000, 0x3eb00000, 0x3f42a0bd, 0x3e2876aa,
    0x3f624bdb, 0xbe1200a5, 0x3f88b43d, 0xbeb7b0ca, 0x3fa4bcd8, 0xbf01d6a4, 0x3fc352a4, 0xbf1c6120,
];

#[test]
fn pv_bin_wipe_copies_raw_bins() {
    let wipe = |w: f32| {
        frame_after(
            vec![kr("PV_BinWipe", vec![c(BUF_X), c(BUF_Y), c(w)])],
            BUF_X,
            1,
        )
    };
    // `(int)(0.3 * 7) = 2`: the lowest two bins and the DC term come from Y.
    assert_eq!(bits(&wipe(0.3)), BIN_WIPE_LOW, "low");
    // Negative: the highest two bins and the Nyquist term.
    assert_eq!(bits(&wipe(-0.3)), BIN_WIPE_HIGH, "high");
    // A full wipe either way takes all of Y, beyond it clamps.
    assert_eq!(bits(&wipe(1.0)), BIN_WIPE_ALL, "all");
    assert_eq!(bits(&wipe(-1.0)), BIN_WIPE_ALL, "all, negative");
    assert_eq!(bits(&wipe(5.0)), BIN_WIPE_ALL, "clamped");
    assert_eq!(bits(&wipe(0.0)), bits(&frame_x()), "none");

    // The copy is raw: behind a polar predecessor, A keeps its polar pairs above the wipe while the
    // wiped bins are Y's Cartesian pairs, as in scsynth.
    let got = frame_after(
        vec![
            polar_identity(c(BUF_X)),
            kr("PV_BinWipe", vec![u(0), c(BUF_Y), c(0.3)]),
        ],
        BUF_X,
        1,
    );
    assert_eq!(bits(&got), BIN_WIPE_MIXED, "mixed forms");
}

const RECT_COMB2: [u32; 16] = [
    0x3f500000, 0xbee00000, 0xbf400000, 0x3ec00000, 0x3f200000, 0x3ec00000, 0xbe800000, 0x3ea00000,
    0x3f600000, 0xbe000000, 0x3e800000, 0x3e800000, 0x3f900000, 0xbf200000, 0x3f400000, 0x3e400000,
];

#[test]
fn pv_rect_comb2_fills_the_gaps_from_b() {
    // Four teeth over eight comb steps from phase 0.25: slots alternate between A's teeth and B's
    // gaps, the DC term in a tooth and the Nyquist term in a gap.
    let got = frame_after(
        vec![kr(
            "PV_RectComb2",
            vec![c(BUF_X), c(BUF_Y), c(4.0), c(0.25), c(0.5)],
        )],
        BUF_X,
        1,
    );
    assert_eq!(bits(&got), RECT_COMB2);
}

const CONFORMAL: [u32; 16] = [
    0x3f500000, 0xbee00000, 0xbd273bf6, 0x3f444850, 0x3e384071, 0x3f25b718, 0x3ee2cabb, 0x3eefb14f,
    0x3f422a3d, 0x3e52b5c5, 0x3f91867b, 0xbe335733, 0x3fca8a54, 0xbf3a3f61, 0x400457a3, 0xbfc3291a,
];
const CONFORMAL_FROM_POLAR: [u32; 16] = [
    0x3f500000, 0xbee00000, 0xbd26aeb1, 0x3f444438, 0x3e387ecd, 0x3f25a709, 0x3ee30bda, 0x3eef4856,
    0x3f423340, 0x3e52bb3b, 0x3f918a0d, 0xbe321570, 0x3fca9520, 0xbf3a2d01, 0x40046cc0, 0xbfc31c1f,
];

#[test]
fn pv_conformal_map_maps_each_bin() {
    let map = |chain: InputRef| kr("PV_ConformalMap", vec![chain, c(0.3), c(-0.2)]);
    assert_eq!(
        bits(&frame_after(vec![map(c(BUF_X))], BUF_X, 1)),
        CONFORMAL,
        "from Cartesian"
    );
    // Behind a polar predecessor the op first converts back to Cartesian form.
    assert_eq!(
        bits(&frame_after(
            vec![polar_identity(c(BUF_X)), map(u(0))],
            BUF_X,
            1
        )),
        CONFORMAL_FROM_POLAR,
        "from polar"
    );
}

#[test]
fn two_buffer_ops_need_both_frames_and_equal_sizes() {
    let ops = [
        kr("PV_MagDiv", vec![c(BUF_X), u(0), c(0.5)]),
        kr("PV_BinWipe", vec![c(BUF_X), u(0), c(0.5)]),
        kr(
            "PV_RectComb2",
            vec![c(BUF_X), u(0), c(4.0), c(0.25), c(0.5)],
        ),
    ];
    for op in ops {
        let name = op.name.clone();
        // Channel 0 reads buffer X, channel 1 carries the chain index the op passes on.
        let units = read_frames(vec![chain_bus(), op, to_audio(u(1))], &[BUF_X], vec![u(2)]);
        let (mut controller, mut world) = frame_engine(units, 2);

        // B has no frame ready: the op outputs -1 although A does, and leaves A alone.
        controller.set_control_bus(CHAIN_BUS, -1.0).unwrap();
        let out = one_block(&mut world, 2);
        assert_eq!(channel(&out, 2, 1)[0], -1.0, "{name}: no B frame");
        assert_eq!(bits(&channel(&out, 2, 0)), bits(&frame_x()), "{name}");

        // B is a different size: A's number passes on and A is left untouched (and unconverted).
        controller.set_control_bus(CHAIN_BUS, BUF_WIDE).unwrap();
        let out = one_block(&mut world, 2);
        assert_eq!(channel(&out, 2, 1)[0], BUF_X, "{name}: size mismatch");
        assert_eq!(bits(&channel(&out, 2, 0)), bits(&frame_x()), "{name}");
    }
}

#[test]
fn per_bin_state_is_sized_from_the_first_frame() {
    for op in [
        kr("PV_MagFreeze", vec![u(0), c(0.0)]),
        kr("PV_MagShift", vec![u(0), c(1.0), c(1.0)]),
    ] {
        let name = op.name.clone();
        // Channel 0 reads the wide buffer's first `FRAME` samples, channel 1 carries the chain.
        let units = read_frames(
            vec![chain_bus(), op, to_audio(u(1))],
            &[BUF_WIDE],
            vec![u(2)],
        );
        let (mut controller, mut world) = frame_engine(units, 2);

        // No frame: -1 passes on.
        controller.set_control_bus(CHAIN_BUS, -1.0).unwrap();
        let out = one_block(&mut world, 2);
        assert_eq!(channel(&out, 2, 1)[0], -1.0, "{name}: no frame");

        // The first frame fixes the bin count; a frame of another size then passes through
        // untouched, its index still passed on.
        controller.set_control_bus(CHAIN_BUS, BUF_X).unwrap();
        one_block(&mut world, 2);
        controller.set_control_bus(CHAIN_BUS, BUF_WIDE).unwrap();
        let out = one_block(&mut world, 2);
        assert_eq!(channel(&out, 2, 1)[0], BUF_WIDE, "{name}: other size");
        let wide = channel(&out, 2, 0);
        for (i, &s) in wide.iter().enumerate() {
            assert_eq!(s, 1.0 + i as f32, "{name}: slot {i} of the other size");
        }
    }
}

#[test]
fn a_failed_first_frame_allocation_outputs_no_frame() {
    // A unit pool too small for a 1024-sample frame's per-bin memory: the op is silenced with `-1` on its chain output
    // (scsynth's `FFT_ClearUnitOutputs`), so nothing downstream sees a ready frame.
    for op in ["PV_MagFreeze", "PV_MagShift"] {
        let (mut controller, _nrt, mut world) = engine(Options {
            sample_rate: SR,
            block_size: BLOCK,
            output_channels: 1,
            unit_pool_bytes: 64,
            ..Options::default()
        });
        controller
            .buffer_set(0, Box::new(Buffer::zeroed(1024, 1, SR)))
            .unwrap();
        controller.add_synthdef(SynthDef {
            name: "t".to_string(),
            params: vec![],
            units: vec![
                kr(op, vec![c(0.0), c(1.0), c(1.0)]),
                to_audio(u(0)),
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

/// The chain signal of `FFTTrigger(buffer, hop, polar)` over `blocks` blocks of `block_size`
/// samples, with a 1024-sample buffer 3 and `locals` 16-sample `LocalBuf`s declared first.
fn trigger_outputs(
    buffer: InputRef,
    hop: f32,
    block_size: usize,
    locals: usize,
    blocks: usize,
) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(3, Box::new(Buffer::zeroed(1024, 1, SR)))
        .unwrap();
    let mut units: Vec<UnitSpec> = (0..locals)
        .map(|_| UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), c(16.0)], 1))
        .collect();
    let trig = units.len() as u32;
    units.push(kr("FFTTrigger", vec![buffer, c(hop), c(0.0)]));
    units.push(to_audio(u(trig)));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(trig + 1)],
        0,
    ));
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    (0..blocks)
        .map(|_| {
            let mut buf = vec![0.0f32; block_size];
            world.fill(&mut buf, 1);
            buf[0]
        })
        .collect()
}

#[test]
fn fft_trigger_announces_a_frame_every_hop() {
    // `(int)((1024 * 0.5) / 64) - 1 = 7` blocks of -1 between frames.
    let mut want = vec![-1.0; 7];
    want.push(3.0);
    let want: Vec<f32> = want.iter().cycle().take(20).copied().collect();
    assert_eq!(trigger_outputs(c(3.0), 0.5, 64, 0, 20), want, "hop 0.5");

    // `(int)((1024 * 0.3) / 64) - 1 = (int)4.8 - 1 = 3`.
    let want = [-1.0, -1.0, -1.0, 3.0].repeat(3);
    assert_eq!(trigger_outputs(c(3.0), 0.3, 64, 0, 12), want, "hop 0.3");

    // A buffer with no storage has no samples, so every block is a frame.
    assert_eq!(
        trigger_outputs(c(7.0), 0.5, 64, 0, 3),
        [7.0; 3],
        "missing buffer"
    );
}

#[test]
fn fft_trigger_resolves_local_buffer_numbers() {
    let capacity = Options::default().max_buffers as f32;
    // A graph-local buffer's number passes through (16 samples at a 16-sample block, hop 1: every
    // block is a frame).
    assert_eq!(
        trigger_outputs(u(0), 1.0, 16, 1, 2),
        [capacity; 2],
        "local buffer"
    );
    // A local number past the synth's local buffers falls back to world buffer 0, whose number the
    // unit then outputs.
    assert_eq!(
        trigger_outputs(c(capacity + 5.0), 0.5, 64, 1, 2),
        [0.0; 2],
        "out-of-range local buffer"
    );
}

const TRIGGER_POLAR: [u32; 16] = [
    0x3f500000, 0xbee00000, 0x3ecfb4f4, 0x3e95a554, 0x3f14ec37, 0x3e69fe08, 0x3f3e8512, 0x3dbe5c6e,
    0x3f5e45ea, 0xbdde1680, 0x3f6e46be, 0xbebb31a0, 0x3f69ab92, 0xbf2859fe, 0x3f4d4620, 0xbf757c05,
];

#[test]
fn fft_trigger_tags_the_frame_coordinates() {
    // `PV_ConformalMap(chain, 0, 0)` is the identity on Cartesian bins, but first converts a polar
    // frame back to Cartesian form - so it shows which form `FFTTrigger` told the chain the frame
    // is in. With 16-sample frames and blocks and hop 1, every block is a frame.
    let run = |polar: f32| {
        frame_after(
            vec![
                kr("FFTTrigger", vec![c(BUF_X), c(1.0), c(polar)]),
                kr("PV_ConformalMap", vec![u(0), c(0.0), c(0.0)]),
            ],
            BUF_X,
            1,
        )
    };
    let x = frame_x();
    assert_eq!(bits(&run(0.0)), bits(&x), "Cartesian: untouched");

    // Tagged polar, X's pairs are read as (magnitude, phase) and converted with `ToComplexApx`.
    assert_eq!(bits(&run(1.0)), TRIGGER_POLAR, "polar: converted");
    // Only exactly 1 means polar.
    assert_eq!(bits(&run(0.5)), bits(&x), "0.5 is Cartesian");
}
