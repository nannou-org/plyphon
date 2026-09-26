//! `GrainTap` against scsynth's own `DelayUGens.cpp`, compiled from source and run on a fresh
//! stream 0 (`RGen::init(0)`) at 48 kHz with 64-sample blocks, over a fixed buffer (a saw plus a
//! parabola, arithmetic only, so the harness computes the same samples). Each case pins the FNV-1a
//! hash of every output sample's bit pattern over 64 blocks, plus ten samples' bit patterns.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Sample `n` of the buffer.
fn bufsig(n: usize) -> f32 {
    let x = n as f64 / 37.3;
    let t = x - x.floor();
    let y = n as f64 / 211.7;
    let u = y - y.floor();
    (0.5 * (2.0 * t - 1.0) + 0.3 * (1.0 - 8.0 * u * (1.0 - u))) as f32
}

/// Render `blocks` blocks of `GrainTap.ar(0, params...)` over a `frames` x `channels` buffer 0.
fn render(frames: usize, channels: usize, params: [f32; 5], blocks: usize) -> Vec<f32> {
    let samples = (0..frames * channels).map(bufsig).collect();
    let mut inputs = vec![c(0.0)];
    inputs.extend(params.iter().map(|&v| c(v)));
    let units = vec![
        UnitSpec::new("GrainTap", Rate::Audio, inputs, 1),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![c(0.0), InputRef::Unit { unit: 0, output: 0 }],
            0,
        ),
    ];
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(samples, channels, SR)))
        .expect("buffer_set");
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; BLOCK * blocks];
    world.fill(&mut out, 1);
    out
}

/// FNV-1a over each sample's bit pattern, as little-endian bytes.
fn fnv(samples: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for s in samples {
        for b in s.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

const PICKS: [usize; 10] = [0, 1, 63, 64, 100, 500, 1000, 2000, 3000, 4095];

fn check(frames: usize, channels: usize, params: [f32; 5], bits: [u32; 10], hash: u64) {
    let out = render(frames, channels, params, 64);
    let got: Vec<u32> = PICKS.iter().map(|&i| out[i].to_bits()).collect();
    assert_eq!(got, bits, "{params:?}: picked samples differ from scsynth");
    assert_eq!(
        fnv(&out),
        hash,
        "{params:?}: the render differs from scsynth"
    );
}

#[test]
fn grain_tap_matches_scsynth() {
    // (grainDur, pchRatio, pchDispersion, timeDispersion, overlap)
    // No dispersion: every grain still draws its two random values.
    check(
        8192,
        1,
        [0.02, 1.0, 0.0, 0.0, 2.0],
        [
            0x00000000, 0x3a143740, 0xbd11dcfd, 0xbce89e65, 0x3e1b9f36, 0x3ebd37b8, 0xbef0d137,
            0xbea02724, 0x3e541710, 0xbee78afa,
        ],
        0x25e910168f359c89,
    );
    // Pitched up, with pitch and time dispersion.
    check(
        8192,
        1,
        [0.03, 1.5, 0.3, 0.01, 4.0],
        [
            0x00000000, 0xb9995768, 0x3ba57e5a, 0x3c1650c8, 0x3bba5142, 0xbf13076e, 0x3eed557f,
            0x3de1ad9e, 0xbf8d57f9, 0xbf07d0fc,
        ],
        0xf8471306ee5f6cf9,
    );
    // Pitched down, with dispersion wide enough for some grains to play backwards.
    check(
        16384,
        1,
        [0.05, 0.5, 0.8, 0.05, 8.0],
        [
            0x00000000, 0xba4842c1, 0x3c8d7792, 0x3cea6d64, 0x3caa5b8d, 0x3eabf648, 0x3d9eb518,
            0xbdc986d7, 0xbfc0285c, 0x3e919ee7,
        ],
        0x1d0b1b0ac3923d35,
    );
    // An overlap of 40 wants more than the 32 grains: a grain start with none free draws nothing.
    // The buffer is stereo, which `GrainTap` reads as one interleaved line of 8192 samples.
    check(
        4096,
        2,
        [0.05, 1.2, 0.5, 0.02, 40.0],
        [
            0x00000000, 0x383345d9, 0xbd638adf, 0xbd58e829, 0x3da392d4, 0xbf9652c7, 0x3ea46578,
            0xc08dcb63, 0xbfaec577, 0xc008d564,
        ],
        0x5dad9e569b255cd9,
    );
}

#[test]
fn grain_tap_is_silent_on_a_buffer_that_is_not_a_power_of_two() {
    // `GrainTap_Ctor` rejects a buffer whose sample count is not a power of two for good.
    let out = render(5000, 1, [0.02, 1.0, 0.0, 0.0, 2.0], 4);
    assert!(out.iter().all(|&s| s == 0.0));
}
