//! The constructor-only randoms (`IRand`, `LinRand`, `NRand`), `TWindex` and `CoinGate`, drawing
//! from the synth's random stream. Each expected value is what scsynth's own `RGen` gives after
//! `RGen::init(0)` (stream 0 of a fresh World), following the `NoiseUGens.cpp` and `OscUGens.cpp`
//! code for the same inputs and trigger streams.

use plyphon::{
    AddAction, BuildError, InputRef, Options, ROOT_GROUP_ID, Rate, RateInfo, SynthDef,
    UnitRegistry, UnitSpec, engine,
};

const SR: f64 = 48_000.0;
/// Samples per control block at the default engine options.
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// Play `units`, tapping each unit in `taps` onto its own output channel, and render `blocks`
/// control blocks. A scalar tap goes through `K2A`, which holds it; a control-rate tap goes through
/// `Select.ar(0, [tap])`, which holds each block's value across the block. Returns one buffer per tap.
fn render(mut units: Vec<UnitSpec>, taps: &[u32], blocks: usize) -> Vec<Vec<f32>> {
    let mut outs = vec![c(0.0)];
    for &tap in taps {
        let src = match units[tap as usize].rate {
            Rate::Audio => tap,
            Rate::Control => {
                units.push(UnitSpec::new(
                    "Select",
                    Rate::Audio,
                    vec![c(0.0), u(tap)],
                    1,
                ));
                units.len() as u32 - 1
            }
            _ => {
                units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(tap)], 1));
                units.len() as u32 - 1
            }
        };
        outs.push(u(src));
    }
    units.push(UnitSpec::new("Out", Rate::Audio, outs, 0));
    let channels = taps.len();
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * blocks * channels];
    world.fill(&mut buf, channels);
    (0..channels)
        .map(|ch| buf.iter().skip(ch).step_by(channels).copied().collect())
        .collect()
}

/// Compile a one-unit def, returning the build error if any.
fn try_compile(unit: UnitSpec) -> Result<(), BuildError> {
    let rate = RateInfo::new(SR, BLOCK);
    SynthDef {
        name: "t".to_string(),
        params: vec![],
        units: vec![unit],
    }
    .compile(
        &UnitRegistry::with_builtins(),
        &rate,
        &rate,
        64,
        32,
        None,
        1,
    )
    .map(|_| ())
}

/// The value a scalar-rate unit writes in its constructor: the first sample of its held output.
fn scalar(name: &str, inputs: &[f32]) -> f32 {
    let unit = UnitSpec::new(
        name,
        Rate::Scalar,
        inputs.iter().map(|&v| c(v)).collect(),
        1,
    );
    let out = &render(vec![unit], &[0], 1)[0];
    assert!(out.iter().all(|s| s.to_bits() == out[0].to_bits()), "held");
    out[0]
}

#[test]
fn irand_draws_one_integer_in_its_constructor() {
    // `IRand_Ctor`: `(float)(rgen.irand(hi - lo + 1) + lo)` with both bounds truncated to `int`.
    let cases: [(f32, f32, u32); 4] = [
        (-5.0, 10.0, 0x4100_0000),       // 8
        (10.0, 5.0, 0x40c0_0000),        // lo above hi: irand(-4) + 10 = 6
        (-2.7, 3.9, 0x4040_0000),        // truncates to [-2, 3]: 3
        (0.0, 1_000_000.0, 0x4951_5d00), // 857552
    ];
    for (lo, hi, expected) in cases {
        assert_eq!(
            scalar("IRand", &[lo, hi]).to_bits(),
            expected,
            "IRand({lo}, {hi})"
        );
    }
}

#[test]
fn linrand_keeps_the_smaller_or_larger_of_two_draws() {
    // `LinRand_Ctor`: two `frand`s; the smaller when `(int)minmax <= 0`, else the larger.
    let low = 0x400d_ed0b; // 2.21759
    let high = 0x4092_532f; // 4.57265
    for (minmax, expected) in [
        (0.0, low),
        (-3.0, low),
        (0.9, low),
        (1.0, high),
        (2.0, high),
    ] {
        assert_eq!(
            scalar("LinRand", &[2.0, 5.0, minmax]).to_bits(),
            expected,
            "LinRand(2, 5, {minmax})"
        );
    }
}

#[test]
fn nrand_averages_n_draws() {
    // `NRand_Ctor`: `(sum / n) * (hi - lo) + lo` over `n = (int)in` draws.
    for (n, expected) in [
        (1.0, 0x401b_887e),
        (3.0, 0x3ee4_02a4),
        (2.9, 0x3f5c_33c4),
        (12.0, 0x3f72_160a),
        (-2.0, 0xbf80_0000), // no draws: -0 * range + lo = lo
    ] {
        assert_eq!(
            scalar("NRand", &[-1.0, 3.0, n]).to_bits(),
            expected,
            "NRand(-1, 3, {n})"
        );
    }
    // `n = 0` divides a zero sum by zero.
    assert!(scalar("NRand", &[-1.0, 3.0, 0.0]).is_nan());
}

#[test]
fn constructor_draws_run_in_synthdef_order() {
    // IRand takes draw 1, LinRand draws 2 and 3, NRand(n = 3) draws 4 to 6, Rand draw 7.
    let units = vec![
        UnitSpec::new("IRand", Rate::Scalar, vec![c(0.0), c(100.0)], 1),
        UnitSpec::new("LinRand", Rate::Scalar, vec![c(0.0), c(1.0), c(1.0)], 1),
        UnitSpec::new("NRand", Rate::Scalar, vec![c(0.0), c(1.0), c(3.0)], 1),
        UnitSpec::new("Rand", Rate::Scalar, vec![c(0.0), c(1.0)], 1),
    ];
    let taps = render(units, &[0, 1, 2, 3], 1);
    let firsts: Vec<u32> = taps.iter().map(|t| t[0].to_bits()).collect();
    assert_eq!(firsts, [0x42ac_0000, 0x3e1d_9c70, 0x3ed9_b57c, 0x3f22_5838]);

    // A non-positive `n` draws nothing, so the `Rand` after it takes the stream's first value.
    let units = vec![
        UnitSpec::new("NRand", Rate::Scalar, vec![c(0.5), c(1.0), c(-2.0)], 1),
        UnitSpec::new("Rand", Rate::Scalar, vec![c(0.0), c(1.0)], 1),
    ];
    let taps = render(units, &[0, 1], 1);
    assert_eq!(taps[0][0].to_bits(), 0x3f00_0000);
    assert_eq!(taps[1][0].to_bits(), 0x3f5b_887e);
}

#[test]
fn constructor_only_randoms_exist_only_at_scalar_rate() {
    // scsynth's constructors set no calc function, so no calc rate is defined for them.
    for (name, inputs) in [("IRand", 2), ("LinRand", 3), ("NRand", 3)] {
        for rate in [Rate::Control, Rate::Audio] {
            let unit = UnitSpec::new(name, rate, vec![c(0.0); inputs], 1);
            assert_eq!(
                try_compile(unit),
                Err(BuildError::UnsupportedRate(rate)),
                "{name} at {rate:?}"
            );
        }
        let unit = UnitSpec::new(name, Rate::Scalar, vec![c(0.0); inputs - 1], 1);
        assert_eq!(
            try_compile(unit),
            Err(BuildError::WrongInputCount),
            "{name}"
        );
    }
}

/// `Impulse` at `rate` with frequency `freq` and phase `phase`.
fn impulse(rate: Rate, freq: f32, phase: f32) -> UnitSpec {
    UnitSpec::new("Impulse", rate, vec![c(freq), c(phase)], 1)
}

/// Every `period`th sample of `signal`, from sample `first`.
fn every(signal: &[f32], first: usize, period: usize) -> Vec<f32> {
    signal.iter().skip(first).step_by(period).copied().collect()
}

/// The value of a held control-rate tap in each block.
fn per_block(signal: &[f32]) -> Vec<f32> {
    every(signal, 0, BLOCK)
}

/// `TWindex` at `rate` over `weights`.
fn twindex(rate: Rate, trig: InputRef, normalize: f32, weights: &[f32]) -> UnitSpec {
    let mut inputs = vec![trig, c(normalize)];
    inputs.extend(weights.iter().map(|&w| c(w)));
    UnitSpec::new("TWindex", rate, inputs, 1)
}

/// Render `TWindex.ar(Impulse.ar(6000), weights, normalize)` for two blocks, check each draw is held
/// until the next impulse, and return the index held from each impulse.
fn twindex_ar(normalize: f32, weights: &[f32]) -> Vec<f32> {
    let units = vec![
        impulse(Rate::Audio, 6000.0, 0.0),
        twindex(Rate::Audio, u(0), normalize, weights),
    ];
    let taps = render(units, &[0, 1], 2);
    for (i, &t) in taps[0].iter().enumerate() {
        assert_eq!(t, if i % 8 == 0 { 1.0 } else { 0.0 }, "trigger at {i}");
    }
    let held = every(&taps[1], 0, 8);
    for (i, &o) in taps[1].iter().enumerate() {
        assert_eq!(o, held[i / 8], "held at {i}");
    }
    held
}

#[test]
fn audio_rate_twindex_draws_a_weighted_index_per_trigger() {
    // The constructor's draw (index 2) holds through the impulse at sample 0: `TWindex_Ctor` sets
    // `m_trig = 1`, so it is no edge. Each later impulse draws again.
    assert_eq!(
        twindex_ar(0.0, &[0.2, 0.5, 0.3]),
        [
            2., 0., 0., 1., 1., 1., 1., 2., 0., 2., 1., 1., 0., 2., 1., 1.
        ],
    );
}

#[test]
fn twindex_scales_by_the_weight_sum_only_when_normalize_is_one() {
    assert_eq!(
        twindex_ar(1.0, &[3.0, 1.0, 1.0]),
        [
            2., 0., 0., 0., 0., 0., 1., 2., 0., 2., 0., 0., 0., 2., 0., 0.
        ],
    );
    // Unnormalized, the draw is scaled by 1, which the first weight always reaches.
    assert_eq!(twindex_ar(0.0, &[3.0, 1.0, 1.0]), vec![0.0; 16]);
}

#[test]
fn twindex_falls_back_to_its_input_count() {
    // Weights summing below 1 without `normalize`: a draw past their sum yields the unit's input
    // count (4), as `TWindex_chooseNewIndex` starts its index at `mNumInputs`.
    assert_eq!(
        twindex_ar(0.0, &[0.1, 0.2]),
        [
            4., 0., 1., 4., 4., 4., 4., 4., 0., 4., 4., 4., 0., 4., 4., 4.
        ],
    );
    // A `NaN` weight poisons the running sum, so only the first weight can be picked; every other
    // draw falls back to the input count (5).
    assert_eq!(
        twindex_ar(0.0, &[0.5, f32::NAN, 0.5]),
        [
            5., 0., 0., 5., 0., 0., 5., 5., 0., 5., 0., 5., 0., 5., 5., 0.
        ],
    );
    // No weights at all: always the input count (2).
    assert_eq!(twindex_ar(0.0, &[]), vec![2.0; 16]);
}

#[test]
fn control_rate_twindex_reads_the_first_trigger_sample_of_each_block() {
    // A control-rate trigger high every other block, and an audio-rate one with an impulse at the
    // first sample of every other block: both draw on blocks 2, 4 and 6.
    let expected = [2., 2., 0., 0., 0., 0., 1., 1.];
    let units = vec![
        impulse(Rate::Control, 375.0, 0.0),
        twindex(Rate::Control, u(0), 0.0, &[0.2, 0.5, 0.3]),
    ];
    assert_eq!(per_block(&render(units, &[1], 8)[0]), expected);
    let units = vec![
        impulse(Rate::Audio, 375.0, 0.0),
        twindex(Rate::Control, u(0), 0.0, &[0.2, 0.5, 0.3]),
    ];
    let taps = render(units, &[0, 1], 8);
    for (i, &t) in taps[0].iter().enumerate() {
        assert_eq!(t, if i % 128 == 0 { 1.0 } else { 0.0 }, "trigger at {i}");
    }
    assert_eq!(per_block(&taps[1]), expected);
}

#[test]
fn scalar_twindex_draws_only_in_its_constructor() {
    let units = vec![twindex(Rate::Scalar, c(1.0), 0.0, &[0.2, 0.5, 0.3])];
    let out = &render(units, &[0], 2)[0];
    assert!(out.iter().all(|&o| o == 2.0));
}

#[test]
fn twindex_needs_its_trigger_and_normalize_inputs() {
    let unit = UnitSpec::new("TWindex", Rate::Control, vec![c(1.0)], 1);
    assert_eq!(try_compile(unit), Err(BuildError::WrongInputCount));
}

/// `a * b` at `rate`.
fn mul(rate: Rate, a: InputRef, b: InputRef) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate,
        inputs: vec![a, b],
        num_outputs: 1,
        special_index: 2,
    }
}

#[test]
fn audio_rate_coingate_passes_each_trigger_edge_by_chance() {
    // `Impulse.ar(6000) * 0.75`: a 0.75 impulse every 8 samples from sample 0, and 0.75 in its
    // constructor too, which `CoinGate_Ctor` latches - so the impulse at sample 0 is no edge.
    let units = vec![
        impulse(Rate::Audio, 6000.0, 0.0),
        mul(Rate::Audio, u(0), c(0.75)),
        UnitSpec::new("CoinGate", Rate::Audio, vec![c(0.5), u(1)], 1),
    ];
    let taps = render(units, &[1, 2], 4);
    let (trig, out) = (&taps[0], &taps[1]);
    for (i, &t) in trig.iter().enumerate() {
        assert_eq!(t, if i % 8 == 0 { 0.75 } else { 0.0 }, "trigger at {i}");
    }
    for (i, &o) in out.iter().enumerate() {
        if i % 8 != 0 {
            assert_eq!(o, 0.0, "no output between triggers at {i}");
        }
    }
    let passed: Vec<u8> = every(out, 0, 8)
        .iter()
        .map(|&o| u8::from(o == 0.75))
        .collect();
    assert_eq!(
        passed,
        [
            0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
            1, 0, 0
        ]
    );

    // With phase 0.5 the constructor sample is low, so the first impulse (sample 4) is an edge.
    let units = vec![
        impulse(Rate::Audio, 6000.0, 0.5),
        UnitSpec::new("CoinGate", Rate::Audio, vec![c(0.5), u(0)], 1),
    ];
    let taps = render(units, &[0, 1], 2);
    assert_eq!(every(&taps[0], 4, 8), vec![1.0; 16], "impulses at 4 + 8k");
    assert_eq!(
        every(&taps[1], 4, 8),
        [
            0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1.
        ],
    );
}

#[test]
fn control_rate_coingate_checks_one_edge_a_block() {
    // `Impulse.kr(375)` is high every other block from block 0 (and in its constructor).
    let units = vec![
        impulse(Rate::Control, 375.0, 0.0),
        UnitSpec::new("CoinGate", Rate::Control, vec![c(0.5), u(0)], 1),
    ];
    let taps = render(units, &[0, 1], 16);
    let trig = per_block(&taps[0]);
    for (b, &t) in trig.iter().enumerate() {
        assert_eq!(
            t,
            if b % 2 == 0 { 1.0 } else { 0.0 },
            "trigger in block {b}"
        );
    }
    assert_eq!(
        per_block(&taps[1]),
        [
            0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.
        ],
    );
}

#[test]
fn coingate_draws_on_every_edge_even_when_it_cannot_pass() {
    // `rgen.frand() < NaN` never passes, but the draw still happens, shifting the `TWindex` after
    // it: alone, the `TWindex` would hold 2, 2, 0, 0, 0, 0, 1, 1.
    let units = vec![
        impulse(Rate::Control, 375.0, 0.0),
        UnitSpec::new("CoinGate", Rate::Control, vec![c(f32::NAN), u(0)], 1),
        twindex(Rate::Control, u(0), 0.0, &[0.2, 0.5, 0.3]),
    ];
    let taps = render(units, &[1, 2], 8);
    assert_eq!(per_block(&taps[0]), vec![0.0; 8]);
    assert_eq!(per_block(&taps[1]), [2., 2., 0., 0., 1., 1., 1., 1.]);
}

#[test]
fn coingate_needs_its_probability_and_trigger_inputs() {
    let unit = UnitSpec::new("CoinGate", Rate::Control, vec![c(0.5)], 1);
    assert_eq!(try_compile(unit), Err(BuildError::WrongInputCount));
}
