//! `Pitch` against scsynth's own `DelayUGens.cpp`, compiled from source and run at 48 kHz with
//! 64-sample blocks. The input is a test signal played from a buffer by `PlayBuf` (every sample for
//! an audio-rate input, one sample per block for a control-rate one): two periodic sections, a saw
//! plus a parabola, separated by a near-silent gap below the amplitude threshold. The signal uses
//! only arithmetic, so the harness computes the same samples. Each case pins the FNV-1a hash of
//! both outputs' bit patterns over the render, and the bit patterns of eleven blocks.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
}

/// Sample `n` of the test signal: period `p1` at amplitude 0.5 until `gap0`, then amplitude 0.001
/// until `gap1`, then period `p2` at amplitude 0.8.
fn sig(n: usize, p1: f64, p2: f64, gap0: usize, gap1: usize) -> f32 {
    let (p, amp) = if n < gap0 {
        (p1, 0.5)
    } else if n < gap1 {
        (p1, 0.001)
    } else {
        (p2, 0.8)
    };
    let x = n as f64 / p;
    let t = x - x.floor();
    (amp * (0.6 * (2.0 * t - 1.0) + 0.4 * (1.0 - 8.0 * t * (1.0 - t)))) as f32
}

/// The signal's shape: its two periods (in samples of the input's rate) and the gap's bounds.
struct Signal {
    p1: f64,
    p2: f64,
    gap0: usize,
    gap1: usize,
}

/// Render `blocks` blocks of `Pitch.kr(PlayBuf, params...)` and return its two outputs, one value
/// per block. `audio` plays the signal at audio rate, otherwise at control rate.
fn render(audio: bool, params: [f32; 10], blocks: usize, s: &Signal) -> (Vec<f32>, Vec<f32>) {
    let (frames, buf_sr, rate) = if audio {
        (blocks * BLOCK, SR, Rate::Audio)
    } else {
        // At control rate `PlayBuf` advances by `rate * bufSampleRate * controlDur` frames a block,
        // so a buffer at the control rate plays one frame per block.
        (blocks, SR / BLOCK as f64, Rate::Control)
    };
    let samples = (0..frames)
        .map(|n| sig(n, s.p1, s.p2, s.gap0, s.gap1))
        .collect();
    let mut pitch_inputs = vec![u(0, 0)];
    pitch_inputs.extend(params.iter().map(|&v| c(v)));
    let units = vec![
        // PlayBuf(bufnum 0, rate 1, trigger 0, startPos 0, loop 0).
        UnitSpec::new(
            "PlayBuf",
            rate,
            vec![c(0.0), c(1.0), c(0.0), c(0.0), c(0.0)],
            1,
        ),
        UnitSpec::new("Pitch", Rate::Control, pitch_inputs, 2),
        // Hold each block's two values across the block.
        UnitSpec::new("Select", Rate::Audio, vec![c(0.0), u(1, 0)], 1),
        UnitSpec::new("Select", Rate::Audio, vec![c(0.0), u(1, 1)], 1),
        UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2, 0), u(3, 0)], 0),
    ];
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 2,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(samples, 1, buf_sr)))
        .expect("buffer_set");
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; BLOCK * blocks * 2];
    world.fill(&mut out, 2);
    let freq = out.iter().step_by(BLOCK * 2).copied().collect();
    let has = out.iter().skip(1).step_by(BLOCK * 2).copied().collect();
    (freq, has)
}

/// FNV-1a over each value's bit pattern, as little-endian bytes.
fn fnv(values: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Blocks `k * blocks / 10 + 3` for `k` in `0..10`, then the last block.
fn picks(blocks: usize) -> Vec<usize> {
    let mut picks: Vec<usize> = (0..10).map(|k| k * blocks / 10 + 3).collect();
    picks.push(blocks - 1);
    picks
}

/// Expected `(hash, picked bits)` for one output.
type Expected = (u64, [u32; 11]);

fn check(
    audio: bool,
    params: [f32; 10],
    blocks: usize,
    s: Signal,
    freq_expected: Expected,
    has_expected: Expected,
) {
    let (freq, has) = render(audio, params, blocks, &s);
    for (name, out, (hash, bits)) in [
        ("freq", &freq, freq_expected),
        ("hasFreq", &has, has_expected),
    ] {
        let got: Vec<u32> = picks(blocks).iter().map(|&i| out[i].to_bits()).collect();
        assert_eq!(
            got, bits,
            "{params:?} {name}: picked blocks differ from scsynth"
        );
        assert_eq!(fnv(out), hash, "{params:?} {name}: differs from scsynth");
    }
}

#[test]
fn pitch_audio_rate_defaults_match_scsynth() {
    // The language defaults: initFreq 440, minFreq 60, maxFreq 4000, execFreq 100, maxBins 16,
    // median 1, ampThreshold 0.01, peakThreshold 0.5, downSample 1, clar 0.
    check(
        true,
        [440.0, 60.0, 4000.0, 100.0, 16.0, 1.0, 0.01, 0.5, 1.0, 0.0],
        300,
        Signal {
            p1: 97.3,
            p2: 61.7,
            gap0: 6000,
            gap1: 7500,
        },
        (
            0x52134b6e6a175576,
            [
                0x43dc0000, 0x43f6cb47, 0x43f6f64d, 0x43f6f77a, 0x43f52cb6, 0x444227fc, 0x44422c49,
                0x44422d42, 0x44424065, 0x44423fd9, 0x444241be,
            ],
        ),
        (
            0x1d097a3d7e4ec908,
            [
                0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x00000000, 0x3f800000, 0x3f800000,
                0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
            ],
        ),
    );
}

#[test]
fn pitch_audio_rate_median_clarity_downsampling_match_scsynth() {
    // A 7-point median, the clarity output, and a downsampling of 3, which does not divide the
    // block, so the read position carries over between blocks.
    check(
        true,
        [220.0, 80.0, 2000.0, 200.0, 4.0, 7.0, 0.02, 0.3, 3.0, 1.0],
        300,
        Signal {
            p1: 97.3,
            p2: 161.9,
            gap0: 6000,
            gap1: 7500,
        },
        (
            0x8a1e52742fb8c9ea,
            [
                0x435c0000, 0x43f6541d, 0x43f7e125, 0x43f65b4e, 0x43f72cfc, 0x43943054, 0x43941bf4,
                0x439429e1, 0x43942ac6, 0x4394335a, 0x43941593,
            ],
        ),
        (
            0x90fd70777487e2e5,
            [
                0x00000000, 0x3f70d5cf, 0x3f717235, 0x3f6dab97, 0x00000000, 0x3f7caff2, 0x3f7e17c3,
                0x3f7cd665, 0x3f7c98db, 0x3f77a1ca, 0x3f7eba65,
            ],
        ),
    );
}

#[test]
fn pitch_audio_rate_coarse_bins_match_scsynth() {
    // `maxBinsPerOctave` 3 steps through long lags so coarsely that the peak search lands off the
    // peak and slides to it, and sometimes locks onto the wrong octave.
    check(
        true,
        [440.0, 30.0, 1000.0, 70.0, 3.0, 3.0, 0.01, 0.3, 2.0, 1.0],
        300,
        Signal {
            p1: 311.1,
            p2: 187.4,
            gap0: 6000,
            gap1: 7500,
        },
        (
            0xf9ee6342cbef655f,
            [
                0x43dc0000, 0x43dc0000, 0x431a8e1c, 0x41f6d070, 0x41f9de29, 0x431a8eb7, 0x43801659,
                0x437fb614, 0x437fbca4, 0x438000a2, 0x438006ea,
            ],
        ),
        (
            0x19c324a0fd7cc05d,
            [
                0x00000000, 0x00000000, 0x3f739609, 0x3f76f7ea, 0x3f73e599, 0x3f71d257, 0x3f74b52d,
                0x3f78915f, 0x3f742a45, 0x3f730508, 0x3f75ed2b,
            ],
        ),
    );
}

#[test]
fn pitch_control_rate_matches_scsynth() {
    // A control-rate input takes `Pitch_next_k`: one sample a block, at 750 Hz.
    check(
        false,
        [100.0, 20.0, 300.0, 50.0, 16.0, 3.0, 0.01, 0.5, 1.0, 1.0],
        1200,
        Signal {
            p1: 7.3,
            p2: 4.1,
            gap0: 500,
            gap1: 600,
        },
        (
            0x7b5fcd6c861d5d17,
            [
                0x42c80000, 0x42c80000, 0x42d0dd8c, 0x42d1b005, 0x42ce72dd, 0x42ce72dd, 0x4338f4f7,
                0x43392094, 0x4338d0bb, 0x433919bf, 0x433919bf,
            ],
        ),
        (
            0x098bcb4a3a6c624c,
            [
                0x00000000, 0x3f55a869, 0x3f5fb417, 0x3f5419ba, 0x3f535a72, 0x00000000, 0x3f68f720,
                0x3f68fdb8, 0x3f69107e, 0x3f690fd1, 0x3f691f10,
            ],
        ),
    );
    // Downsampled by 2: one sample every other block.
    check(
        false,
        [100.0, 10.0, 200.0, 30.0, 8.0, 1.0, 0.01, 0.4, 2.0, 0.0],
        1200,
        Signal {
            p1: 5.7,
            p2: 3.3,
            gap0: 500,
            gap1: 600,
        },
        (
            0x02e50f34c0f8d2ee,
            [
                0x42c80000, 0x42c80000, 0x42fecae8, 0x42ff0cf5, 0x42fecc6c, 0x42fefe2d, 0x42fefe2d,
                0x42957606, 0x4295bdf4, 0x42952219, 0x42955298,
            ],
        ),
        (
            0x416c2840b88d5cc8,
            [
                0x00000000, 0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x00000000,
                0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
            ],
        ),
    );
}
