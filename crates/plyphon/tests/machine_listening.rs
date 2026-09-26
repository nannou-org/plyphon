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
    let mut inputs = vec![u(0, 0)];
    inputs.extend(args.iter().map(|&a| c(a)));
    let mut units = vec![
        UnitSpec::new("In", Rate::Control, vec![c(CHAIN_BUS as f32)], 1),
        UnitSpec::new(name, Rate::Control, inputs, outputs),
    ];
    // Adding zero at audio rate carries each control output into its block exactly.
    for o in 0..outputs {
        units.push(UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![u(1, o as u32), c(0.0)],
            num_outputs: 1,
            special_index: 0,
        });
    }
    let mut out = vec![c(0.0)];
    out.extend((0..outputs).map(|o| u(2 + o as u32, 0)));
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
