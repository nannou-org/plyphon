//! The constructor-only randoms (`IRand`, `LinRand`, `NRand`), drawing from the synth's random
//! stream. Each expected value is what scsynth's own `RGen` gives after `RGen::init(0)` (stream 0 of
//! a fresh World), following the `NoiseUGens.cpp` constructors for the same inputs.

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
