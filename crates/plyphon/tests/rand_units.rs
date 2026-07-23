//! The init- and trigger-time randoms (`Rand`, `ExpRand`, `TRand`, `TIRand`, `RandSeed`,
//! `RandID`) and the RNG-driven operators (unary `asInt`, binary `rrand`/`exprand`): range,
//! hold, redraw-on-trigger, and shared-stream re-seeding behaviour.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Samples per control block at the default engine options.
const BLOCK: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

/// `Out.ar(0, Unit{src})`.
fn out(src: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: src,
                output: 0,
            },
        ],
        0,
    )
}

/// Build a world playing `units` as the def `t`, and render `blocks` control blocks.
fn render(units: Vec<UnitSpec>, blocks: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * blocks];
    world.fill(&mut buf, 1);
    buf
}

/// `Impulse.ar(freq, 0)` - a single-sample `1.0` every `SR / freq` samples, starting at sample 0.
fn impulse_ar(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "Impulse",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

#[test]
fn rand_holds_one_draw_in_range() {
    // Rand(100, 200) -> K2A -> Out: one init-time draw, held for the synth's life.
    let buf = render(
        vec![
            UnitSpec::new(
                "Rand",
                Rate::Scalar,
                vec![InputRef::Constant(100.0), InputRef::Constant(200.0)],
                1,
            ),
            UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
        4,
    );
    let value = buf[0];
    assert!((100.0..200.0).contains(&value), "in range: {value}");
    assert!(buf.iter().all(|&s| s == value), "held constant");
}

#[test]
fn exprand_holds_one_draw_in_range() {
    let buf = render(
        vec![
            UnitSpec::new(
                "ExpRand",
                Rate::Scalar,
                vec![InputRef::Constant(1.0), InputRef::Constant(1000.0)],
                1,
            ),
            UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
        4,
    );
    let value = buf[0];
    assert!((1.0..1000.0).contains(&value), "in range: {value}");
    assert!(buf.iter().all(|&s| s == value), "held constant");
}

#[test]
fn trand_redraws_on_trigger_and_holds_between() {
    // A one-sample impulse at the start of every 64-sample block drives TRand: the first block
    // keeps the init draw (the trigger level is latched at spawn), every later block re-draws on
    // the rising edge at its first sample and holds the value across the block.
    let buf = render(
        vec![
            impulse_ar(SR as f32 / BLOCK as f32),
            UnitSpec::new(
                "TRand",
                Rate::Audio,
                vec![
                    InputRef::Constant(10.0),
                    InputRef::Constant(20.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                1,
            ),
            out(1),
        ],
        4,
    );
    let block_values: Vec<f32> = (0..4).map(|b| buf[b * BLOCK]).collect();
    for (b, &value) in block_values.iter().enumerate() {
        assert!((10.0..20.0).contains(&value), "block {b} in range: {value}");
        assert!(
            buf[b * BLOCK..(b + 1) * BLOCK].iter().all(|&s| s == value),
            "block {b} held"
        );
    }
    assert!(
        block_values.windows(2).any(|w| w[0] != w[1]),
        "re-draws across blocks: {block_values:?}"
    );
}

#[test]
fn tirand_draws_integers_in_range() {
    let buf = render(
        vec![
            impulse_ar(SR as f32 / BLOCK as f32),
            UnitSpec::new(
                "TIRand",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(3.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                1,
            ),
            out(1),
        ],
        32,
    );
    let draws: Vec<f32> = (0..32).map(|b| buf[b * BLOCK]).collect();
    for &value in &draws {
        assert_eq!(value.fract(), 0.0, "integer draw: {value}");
        assert!((0.0..=3.0).contains(&value), "in range: {value}");
    }
    assert!(draws.windows(2).any(|w| w[0] != w[1]), "varies: {draws:?}");
}

#[test]
fn randseed_restarts_the_shared_stream() {
    // RandSeed fires every second control block (Impulse.kr at half the control rate) and re-keys
    // the synth's stream to a constant, so the TRand draw sequence repeats with period 2:
    // blocks 0/2/4 draw the fresh-stream value, blocks 1/3/5 the one after it.
    let control_rate = SR as f32 / BLOCK as f32;
    let units = vec![
        UnitSpec::new(
            "Impulse",
            Rate::Control,
            vec![
                InputRef::Constant(control_rate / 2.0),
                InputRef::Constant(0.0),
            ],
            1,
        ),
        UnitSpec::new(
            "RandSeed",
            Rate::Control,
            vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(12345.0),
            ],
            1,
        ),
        impulse_ar(control_rate),
        UnitSpec::new(
            "TRand",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Constant(1.0),
                InputRef::Unit { unit: 2, output: 0 },
            ],
            1,
        ),
        out(3),
    ];
    let buf = render(units.clone(), 6);
    let draws: Vec<f32> = (0..6).map(|b| buf[b * BLOCK]).collect();
    assert_eq!(draws[0], draws[2], "seeded stream repeats: {draws:?}");
    assert_eq!(draws[2], draws[4], "seeded stream repeats: {draws:?}");
    assert_eq!(draws[1], draws[3], "second draw repeats: {draws:?}");
    assert_ne!(draws[0], draws[1], "distinct draws within a period");

    // The sequence is a pure function of the seed constant: a fresh engine reproduces it exactly.
    let again = render(units, 6);
    assert_eq!(buf, again, "seeded sequence reproduces across engines");
}

#[test]
fn rand_id_consumes_input_and_outputs_zero() {
    let buf = render(
        vec![
            UnitSpec::new("RandID", Rate::Control, vec![InputRef::Constant(3.0)], 1),
            UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
        1,
    );
    assert!(buf.iter().all(|&s| s == 0.0));
}

#[test]
fn unary_as_int_truncates_toward_zero_and_as_float_is_identity() {
    for (index, input, expected) in [
        (7, 2.7f32, 2.0f32),
        (7, -2.7, -2.0),
        (7, 0.9, 0.0),
        (6, 2.7, 2.7),
        (6, -0.25, -0.25),
    ] {
        let buf = render(
            vec![
                UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(input)], 1),
                UnitSpec {
                    name: "UnaryOpUGen".to_string(),
                    rate: Rate::Audio,
                    inputs: vec![InputRef::Unit { unit: 0, output: 0 }],
                    num_outputs: 1,
                    special_index: index,
                },
                out(1),
            ],
            1,
        );
        assert_eq!(buf[0], expected, "op {index} on {input}");
    }
}

#[test]
fn binary_rand_ops_draw_between_inputs_per_sample() {
    // Audio-rate ranges follow scsynth's calc variants: `rrand_aa` draws with the *bipolar*
    // `frand2`, so `rrand(100, 200).ar` is uniform over [0, 200) (twice the requested width,
    // extending below lo - shipped scsynth behaviour); `exprand_aa` stays within [100, 200).
    for (index, range, below_lo) in [(47i16, 0.0f32..200.0f32, true), (48, 100.0..200.0, false)] {
        let buf = render(
            vec![
                UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(100.0)], 1),
                UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(200.0)], 1),
                UnitSpec {
                    name: "BinaryOpUGen".to_string(),
                    rate: Rate::Audio,
                    inputs: vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Unit { unit: 1, output: 0 },
                    ],
                    num_outputs: 1,
                    special_index: index,
                },
                out(2),
            ],
            4,
        );
        assert!(
            buf.iter().all(|s| range.contains(s)),
            "op {index} stays in {range:?}"
        );
        assert!(
            buf.windows(2).any(|w| w[0] != w[1]),
            "op {index} draws fresh values per sample"
        );
        assert_eq!(
            buf.iter().any(|&s| s < 100.0),
            below_lo,
            "op {index}: bipolar draws land below lo iff rrand at audio rate"
        );
    }
}

#[test]
fn two_instances_of_one_def_decorrelate() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Rand",
                Rate::Scalar,
                vec![InputRef::Constant(0.0), InputRef::Constant(1.0)],
                1,
            ),
            UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("first synth");
    let first = first_sample(&mut world);
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("second synth");
    // Both instances sum into bus 0: the sum only doubles if the second instance drew the exact
    // same value, which distinct per-instance seeds make (astronomically) improbable.
    let summed = first_sample(&mut world);
    assert!(
        (summed - 2.0 * first).abs() > 1e-7,
        "draws decorrelate: {first} then {summed}"
    );
}

/// The first output sample of the world's next block.
fn first_sample(world: &mut World) -> f32 {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf[0]
}

#[test]
fn a_spawn_does_not_replay_the_previous_spawns_first_unit_stream() {
    // The graph's shared-stream seed must stay off the per-unit reseed ladder of *neighbouring*
    // spawns too: with the old `base - SEED_STEP` seed it equalled the previous spawn's unit-0
    // value, so a WhiteNoise there and this graph's rrand draws replayed one underlying stream.
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 2,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "noise".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller.add_synthdef(SynthDef {
        name: "stream".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.0)], 1),
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                num_outputs: 1,
                special_index: 47, // rrand
            },
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(1.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("noise", ROOT_GROUP_ID, AddAction::Tail)
        .expect("noise synth");
    controller
        .synth_new("stream", ROOT_GROUP_ID, AddAction::Tail)
        .expect("stream synth");
    let mut buf = vec![0.0f32; BLOCK * 2];
    world.fill(&mut buf, 2);
    let noise: Vec<f32> = buf.iter().step_by(2).copied().collect();
    let stream: Vec<f32> = buf.iter().skip(1).step_by(2).copied().collect();
    // WhiteNoise emits the bipolar `frand2` map of its stream. A replayed stream would make the
    // rrand(0, 1) output match under the unipolar map (`(noise + 1) / 2`) or the bipolar one
    // (`noise` itself), depending on the rrand draw convention.
    let close = |a: f32, b: f32| (a - b).abs() < 1e-7;
    assert!(
        !noise
            .iter()
            .zip(&stream)
            .all(|(n, s)| close((n + 1.0) * 0.5, *s)),
        "graph stream replays the previous spawn's unit-0 stream (unipolar map)"
    );
    assert!(
        !noise.iter().zip(&stream).all(|(n, s)| close(*n, *s)),
        "graph stream replays the previous spawn's unit-0 stream (bipolar map)"
    );
}
