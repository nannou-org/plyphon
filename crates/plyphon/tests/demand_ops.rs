//! Demand-rate `BinaryOpUGen`/`UnaryOpUGen`: the math operators compiled as demand *sources*, which
//! pull their operands and yield the operator applied to them.
//!
//! Each def hangs an audio-rate `Demand` consumer off the operator and triggers it from a control
//! bus, so the test drives one pull per rising edge and reads the value the operator produced
//! straight out of the rendered block. A second `Demand` output carries an independent counter, so a
//! held (exhausted) value can be told apart from a pull that did not happen.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Samples per control block at the options these tests use.
const BLOCK: usize = 64;
/// The control bus carrying the consumer's trigger.
const TRIG_BUS: u32 = 0;
/// The control bus carrying the consumer's reset.
const RESET_BUS: u32 = 1;

/// SuperCollider's `opAdd`.
const OP_ADD: i16 = 0;
/// SuperCollider's `opMul`.
const OP_MUL: i16 = 2;
/// SuperCollider's `opGT`.
const OP_GT: i16 = 9;
/// SuperCollider's `opMax`.
const OP_MAX: i16 = 13;
/// SuperCollider's `opNeg`.
const OP_NEG: i16 = 0;
/// SuperCollider's `opSin`.
const OP_SIN: i16 = 28;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// `Dseq(repeats, items...)` - a sequence source.
fn dseq(items: &[f32], repeats: f32) -> UnitSpec {
    let mut inputs = vec![c(repeats)];
    inputs.extend(items.iter().map(|&v| c(v)));
    UnitSpec::new("Dseq", Rate::Demand, inputs, 1)
}

/// `BinaryOpUGen.dr(a, b)` with operator `op`.
fn binary(op: i16, a: InputRef, b: InputRef) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Demand,
        inputs: vec![a, b],
        num_outputs: 1,
        special_index: op,
    }
}

/// `UnaryOpUGen.dr(a)` with operator `op`.
fn unary(op: i16, a: InputRef) -> UnitSpec {
    UnitSpec {
        name: "UnaryOpUGen".to_string(),
        rate: Rate::Demand,
        inputs: vec![a],
        num_outputs: 1,
        special_index: op,
    }
}

/// A driver for the demand sources in `units`: an audio-rate `Demand` triggered from [`TRIG_BUS`]
/// and reset from [`RESET_BUS`], with one output per source in `sources`, each on its own channel.
fn driver(mut units: Vec<UnitSpec>, sources: Vec<InputRef>) -> (Vec<UnitSpec>, usize) {
    let channels = sources.len();
    let bus_trig = units.len() as u32;
    units.push(UnitSpec::new(
        "In",
        Rate::Control,
        vec![c(TRIG_BUS as f32)],
        1,
    ));
    units.push(UnitSpec::new(
        "In",
        Rate::Control,
        vec![c(RESET_BUS as f32)],
        1,
    ));
    let mut inputs = vec![u(bus_trig), u(bus_trig + 1)];
    inputs.extend(sources);
    let demand = bus_trig + 2;
    units.push(UnitSpec::new("Demand", Rate::Audio, inputs, channels));
    let mut out = vec![c(0.0)];
    out.extend((0..channels).map(|k| InputRef::Unit {
        unit: demand,
        output: k as u32,
    }));
    units.push(UnitSpec::new("Out", Rate::Audio, out, 0));
    (units, channels)
}

/// A running engine over `units`, with the trigger and reset buses cleared.
fn start(units: Vec<UnitSpec>, channels: usize) -> (plyphon::Controller, World) {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: channels,
        ..Options::default()
    });
    controller.set_control_bus(TRIG_BUS, 0.0).expect("set bus");
    controller.set_control_bus(RESET_BUS, 0.0).expect("set bus");
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

/// Render one block, returning each channel's held value.
fn held(world: &mut World, channels: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; BLOCK * channels];
    world.fill(&mut buf, channels);
    (0..channels).map(|ch| buf[ch]).collect()
}

/// Raise the trigger for one block (one pull), then lower it, returning the values the pull held.
fn pull(controller: &mut plyphon::Controller, world: &mut World, channels: usize) -> Vec<f32> {
    controller.set_control_bus(TRIG_BUS, 1.0).expect("set bus");
    let values = held(world, channels);
    controller.set_control_bus(TRIG_BUS, 0.0).expect("set bus");
    held(world, channels);
    values
}

/// Raise the reset for one block, then lower it. The consumer resets before it triggers within a
/// block, so the trigger stays low here and the reset is the block's only effect.
fn reset(controller: &mut plyphon::Controller, world: &mut World, channels: usize) {
    controller.set_control_bus(RESET_BUS, 1.0).expect("set bus");
    held(world, channels);
    controller.set_control_bus(RESET_BUS, 0.0).expect("set bus");
    held(world, channels);
}

#[test]
fn demand_binary_ops_match_reference_semantics() {
    // `a` carries a NaN in the middle of its sequence and `b` counts on regardless, so the third
    // pull tells whether `b` advanced during the NaN pull: no short-circuit means it did, and the
    // sum lands on `3 + 20` rather than `3 + 10`.
    //
    // The `max` operands avoid a pair of equal-comparing zeroes: `f32::max` may return either of
    // them, so the sign of a `±0.0` result is unspecified, where the reference's `max` always
    // returns the second.
    let a = [1.0f32, f32::NAN, 3.0, 4.0];
    let b = [0.0f32, 10.0, 20.0, 30.0];
    for (op, expect) in [
        (OP_ADD, [1.0f32, f32::NAN, 23.0, 34.0]),
        (OP_MUL, [0.0, f32::NAN, 60.0, 120.0]),
        (OP_GT, [1.0, f32::NAN, 0.0, 0.0]),
        (OP_MAX, [1.0, f32::NAN, 20.0, 30.0]),
    ] {
        let units = vec![
            dseq(&a, f32::INFINITY),
            dseq(&b, f32::INFINITY),
            binary(op, u(0), u(1)),
        ];
        let (units, channels) = driver(units, vec![u(2)]);
        let (mut controller, mut world) = start(units, channels);

        // The consumer starts holding 0, and the binary operator does not prime, so nothing has been
        // pulled before the first trigger.
        assert_eq!(
            held(&mut world, channels)[0],
            0.0,
            "op {op}: the binary constructor must not prime"
        );

        let mut previous = 0.0f32;
        for (k, want) in expect.into_iter().enumerate() {
            let got = pull(&mut controller, &mut world, channels)[0];
            if want.is_nan() {
                // The consumer holds its previous value on an exhausted pull, so a NaN result shows
                // up as a value that did not move.
                assert_eq!(got, previous, "op {op} pull {k}: a NaN operand yields NaN");
            } else {
                // Pure arithmetic and comparison: the result is exact.
                assert_eq!(got, want, "op {op} pull {k}");
                previous = got;
            }
        }
    }
}

#[test]
fn demand_binary_ops_reset_and_exhaust() {
    // `a` runs out after two items while `b` runs forever. Channel 1 carries an independent counter
    // pulled by the same consumer, so a frozen channel 0 is visibly an exhausted operator rather
    // than a pull that never happened.
    let units = vec![
        dseq(&[1.0, 2.0], 1.0),
        dseq(&[10.0, 20.0, 30.0, 40.0], f32::INFINITY),
        binary(OP_ADD, u(0), u(1)),
        UnitSpec::new(
            "Dseries",
            Rate::Demand,
            vec![c(f32::INFINITY), c(1.0), c(1.0)],
            1,
        ),
    ];
    let (units, channels) = driver(units, vec![u(2), u(3)]);
    let (mut controller, mut world) = start(units, channels);

    for (k, want) in [11.0f32, 22.0].into_iter().enumerate() {
        let got = pull(&mut controller, &mut world, channels);
        assert_eq!(got[0], want, "pull {k}");
        assert_eq!(got[1], (k + 1) as f32, "the counter advances on pull {k}");
    }

    // Exhausted: the operator yields NaN, which the consumer shows by holding, while the counter
    // proves the pulls are still happening.
    for k in 2..4 {
        let got = pull(&mut controller, &mut world, channels);
        assert_eq!(got[0], 22.0, "pull {k} is exhausted, so the value holds");
        assert_eq!(got[1], (k + 1) as f32, "the counter advances on pull {k}");
    }

    // A reset reaches every demand child of the operator - both operands, not just the exhausted
    // one - so the sequence resumes exactly as it started rather than from where `b` was left.
    reset(&mut controller, &mut world, channels);
    for (k, want) in [11.0f32, 22.0].into_iter().enumerate() {
        let got = pull(&mut controller, &mut world, channels);
        assert_eq!(got[0], want, "pull {k} after a reset");
    }
}

#[test]
fn demand_unary_op_protocol_and_ctor_prime() {
    // The reference's unary constructor runs the unit's calc function once, consuming one element of
    // the operand before any consumer pulls. A negation over a counting sequence makes that visible:
    // the first value a consumer sees is the sequence's *second* item.
    let items = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let units = vec![dseq(&items, f32::INFINITY), unary(OP_NEG, u(0))];
    let (units, channels) = driver(units, vec![u(1)]);
    let (mut controller, mut world) = start(units, channels);
    for (k, want) in [-2.0f32, -3.0, -4.0].into_iter().enumerate() {
        let got = pull(&mut controller, &mut world, channels)[0];
        assert_eq!(got, want, "pull {k} after the constructor's prime");
    }

    // A reset reaches the operand but does not re-arm the prime: the constructor runs once per
    // instantiation, so the sequence resumes at its first item rather than its second.
    reset(&mut controller, &mut world, channels);
    assert_eq!(
        pull(&mut controller, &mut world, channels)[0],
        -1.0,
        "a reset restarts the operand without repeating the constructor's pull"
    );

    // A reset that lands before the first produce also spends the prime: in the reference the
    // construction-time pull happens first and the reset then rewinds the operand over it, so the
    // first produced value is the sequence's *first* item, not its second.
    let units = vec![dseq(&items, f32::INFINITY), unary(OP_NEG, u(0))];
    let (units, channels) = driver(units, vec![u(1)]);
    let (mut controller, mut world) = start(units, channels);
    reset(&mut controller, &mut world, channels);
    assert_eq!(
        pull(&mut controller, &mut world, channels)[0],
        -1.0,
        "a reset before the first produce leaves the operand at its first item"
    );

    // An exhausted operand yields NaN, which the consumer shows by holding.
    let units = vec![dseq(&[7.0, 8.0], 1.0), unary(OP_NEG, u(0))];
    let (units, channels) = driver(units, vec![u(1)]);
    let (mut controller, mut world) = start(units, channels);
    // The prime consumes item 0, so the first pull yields item 1 and the operand is then spent.
    assert_eq!(pull(&mut controller, &mut world, channels)[0], -8.0);
    assert_eq!(
        pull(&mut controller, &mut world, channels)[0],
        -8.0,
        "an exhausted operand yields NaN, so the held value does not move"
    );

    // The special index selects the operator: 28 is sine, which the engine evaluates through its own
    // float math, so the sequence's own values are the oracle for *which* operator ran.
    let angles = [0.0f32, 0.5, 1.0, 1.5];
    let units = vec![dseq(&angles, f32::INFINITY), unary(OP_SIN, u(0))];
    let (units, channels) = driver(units, vec![u(1)]);
    let (mut controller, mut world) = start(units, channels);
    for (k, &angle) in angles.iter().enumerate().skip(1) {
        let got = pull(&mut controller, &mut world, channels)[0];
        assert!(
            (got - angle.sin()).abs() < 1e-6,
            "pull {k}: sine of {angle} is {got}"
        );
    }
}

#[test]
fn demand_op_unlisted_special_falls_through_like_the_reference() {
    // The reference's demand switches fall through to `add_d` (binary) and `thru_d` (unary) for an
    // operator they do not list: `opUnsignedShift` (28) and an out-of-range index here.
    for op in [28i16, 999] {
        let (units, channels) = driver(vec![binary(op, c(3.0), c(4.0))], vec![u(0)]);
        let (mut controller, mut world) = start(units, channels);
        assert_eq!(
            pull(&mut controller, &mut world, channels),
            [7.0],
            "binary {op} adds"
        );
    }
    // `opIsNil` (2) and an out-of-range index pass their operand through.
    for op in [2i16, 999] {
        let (units, channels) = driver(vec![unary(op, c(3.0))], vec![u(0)]);
        let (mut controller, mut world) = start(units, channels);
        assert_eq!(
            pull(&mut controller, &mut world, channels),
            [3.0],
            "unary {op} passes through"
        );
    }
}
