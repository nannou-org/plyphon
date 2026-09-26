//! `Duty` and `TDuty` read their `reset` input by its rate, as scsynth's `Duty_Ctor`/`TDuty_Ctor`
//! select `_next_dk`/`_da`/`_dd` (`DemandUGens.cpp`): a control-rate reset fires on its block value,
//! an audio-rate reset on a rising edge at any sample, and a demand-rate reset is a stream of
//! durations between resets.
//!
//! The expected renders come from scsynth's own `DemandUGens.cpp` driven by a minimal graph at
//! 48 kHz with 64-sample blocks: `dur = Dseq([0.0005, 0.0007, 0.0011], inf)`,
//! `level = Dseries(inf, 0, 1)`, and a reset of `Impulse.ar(375, 0.25)` (a rising edge at sample
//! 96 and every 128 samples after, mid-block), `Dseq([0.003, 0.0021], inf)` or `0`. Each case pins
//! the FNV-1a hash of every output sample's bits, plus the first change points so a mismatch shows
//! where it starts.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

#[derive(Clone, Copy)]
enum Reset {
    /// `Impulse.ar(375, 0.25)`.
    Audio,
    /// `Dseq([0.003, 0.0021], inf)`.
    Demand,
    /// The constant `0`.
    Zero,
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

fn dseq(items: &[f32]) -> UnitSpec {
    let mut inputs = vec![c(f32::INFINITY)];
    inputs.extend(items.iter().map(|&v| c(v)));
    UnitSpec::new("Dseq", Rate::Demand, inputs, 1)
}

/// Render `blocks` blocks of `name` (`Duty` or `TDuty`, the latter with `gap_first`) at `rate`: every
/// sample at audio rate, one value per block at control rate. Channel 1 carries the reset signal
/// when it is audio-rate, so the test can check its edges.
fn render(name: &str, rate: Rate, reset: Reset, gap_first: Option<f32>, blocks: usize) -> Vec<f32> {
    let mut units = vec![
        dseq(&[0.0005, 0.0007, 0.0011]),
        UnitSpec::new(
            "Dseries",
            Rate::Demand,
            vec![c(f32::INFINITY), c(0.0), c(1.0)],
            1,
        ),
    ];
    let reset_in = match reset {
        Reset::Audio => {
            units.push(UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![c(375.0), c(0.25)],
                1,
            ));
            u(2)
        }
        Reset::Demand => {
            units.push(dseq(&[0.003, 0.0021]));
            u(2)
        }
        Reset::Zero => c(0.0),
    };
    let mut inputs = vec![u(0), reset_in, c(0.0), u(1)];
    inputs.extend(gap_first.map(c));
    let duty = units.len() as u32;
    units.push(UnitSpec::new(name, rate, inputs, 1));
    let mut signal = duty;
    if rate == Rate::Control {
        signal = units.len() as u32;
        units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(duty)], 1));
    }
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(signal)],
        0,
    ));
    if let Reset::Audio = reset {
        units.push(UnitSpec::new("Out", Rate::Audio, vec![c(1.0), u(2)], 0));
    }

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 2,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "duty".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("duty", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; blocks * BLOCK * 2];
    world.fill(&mut buf, 2);

    if let Reset::Audio = reset {
        // The reset really is a mid-block rising edge at 96 and every 128 samples after.
        for (n, frame) in buf.chunks(2).enumerate() {
            let edge = n >= 96 && (n - 96) % 128 == 0;
            assert_eq!(frame[1], if edge { 1.0 } else { 0.0 }, "reset at {n}");
        }
    }
    let out: Vec<f32> = buf.chunks(2).map(|f| f[0]).collect();
    if rate == Rate::Control {
        // `K2A` ramps from the previous block's value, so block `k`'s value is exactly the first
        // sample of block `k + 1`.
        out.iter()
            .step_by(BLOCK)
            .skip(1)
            .take(blocks - 1)
            .copied()
            .collect()
    } else {
        out
    }
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

/// The first `n` points where the output changes, as `(index, value)`.
fn changes(out: &[f32], n: usize) -> Vec<(usize, f32)> {
    let mut v = Vec::new();
    for (i, &x) in out.iter().enumerate() {
        if i == 0 || x.to_bits() != out[i - 1].to_bits() {
            v.push((i, x));
            if v.len() == n {
                break;
            }
        }
    }
    v
}

fn check(out: &[f32], first: &[(usize, f32)], hash: u64) {
    assert_eq!(
        changes(out, first.len()),
        first,
        "change points differ from scsynth"
    );
    assert_eq!(fnv(out), hash, "the render differs from scsynth");
}

#[test]
fn duty_audio_rate_reset_fires_mid_block() {
    // scsynth's `Duty_next_da`: the edge at sample 96 restarts the level stream at 0 there, not at
    // the next block boundary (128).
    let out = render("Duty", Rate::Audio, Reset::Audio, None, 16);
    check(
        &out,
        &[
            (0, 0.0),
            (25, 1.0),
            (58, 2.0),
            (96, 0.0),
            (121, 1.0),
            (154, 2.0),
            (207, 3.0),
            (224, 0.0),
            (249, 1.0),
            (282, 2.0),
            (335, 3.0),
            (352, 0.0),
        ],
        0x0995_0aea_26e9_0118,
    );
}

#[test]
fn duty_demand_rate_reset_counts_its_own_durations() {
    // scsynth's `Duty_next_dd`: resets every 0.003 s then 0.0021 s (144 then ~101 samples).
    let out = render("Duty", Rate::Audio, Reset::Demand, None, 16);
    check(
        &out,
        &[
            (0, 0.0),
            (25, 1.0),
            (58, 2.0),
            (111, 3.0),
            (135, 4.0),
            (144, 0.0),
            (169, 1.0),
            (202, 2.0),
            (246, 0.0),
            (271, 1.0),
            (304, 2.0),
            (357, 3.0),
        ],
        0x5b75_db15_760f_2595,
    );
}

#[test]
fn duty_control_rate_demand_reset() {
    // The same streams at control rate, counting in control blocks.
    let out = render("Duty", Rate::Control, Reset::Demand, None, 401);
    check(
        &out,
        &[
            (0, 0.0),
            (1, 1.0),
            (2, 2.0),
            (3, 0.0),
            (4, 1.0),
            (5, 0.0),
            (6, 1.0),
            (7, 2.0),
            (8, 3.0),
            (9, 0.0),
            (10, 1.0),
            (11, 0.0),
        ],
        0x7128_d312_0c1e_9218,
    );
}

#[test]
fn duty_counts_in_single_precision() {
    // A second of fractional durations with no reset: the count accumulates in `float`, as
    // scsynth's `m_count` does, so the boundaries land where scsynth's do.
    let out = render("Duty", Rate::Audio, Reset::Zero, None, 750);
    check(
        &out,
        &[
            (0, 0.0),
            (25, 1.0),
            (58, 2.0),
            (111, 3.0),
            (135, 4.0),
            (168, 5.0),
            (221, 6.0),
            (245, 7.0),
            (279, 8.0),
            (332, 9.0),
            (356, 10.0),
            (389, 11.0),
        ],
        0x0e69_3947_e5cf_d68c,
    );
}

#[test]
fn tduty_audio_rate_reset_fires_mid_block() {
    // scsynth's `TDuty_next_da`: the edge at 96 restarts the level stream, so the impulse there
    // carries level 0.
    let out = render("TDuty", Rate::Audio, Reset::Audio, Some(0.0), 16);
    check(
        &out,
        &[
            (0, 0.0),
            (25, 1.0),
            (26, 0.0),
            (58, 2.0),
            (59, 0.0),
            (121, 1.0),
            (122, 0.0),
            (154, 2.0),
            (155, 0.0),
            (207, 3.0),
            (208, 0.0),
            (249, 1.0),
        ],
        0x399b_bcfe_a651_91c8,
    );
}

#[test]
fn tduty_demand_rate_reset_with_gap_first() {
    // scsynth's `TDuty_next_dd`, with `gapFirst`: the constructor pulls the first reset duration,
    // then one `dur`.
    let out = render("TDuty", Rate::Audio, Reset::Demand, Some(1.0), 16);
    check(
        &out,
        &[
            (0, 0.0),
            (58, 1.0),
            (59, 0.0),
            (111, 2.0),
            (112, 0.0),
            (135, 3.0),
            (136, 0.0),
            (169, 1.0),
            (170, 0.0),
            (202, 2.0),
            (203, 0.0),
            (271, 1.0),
        ],
        0x5ce1_8049_c432_56b8,
    );
}
