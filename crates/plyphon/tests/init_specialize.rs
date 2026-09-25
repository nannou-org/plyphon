//! `SynthDef::specialize_init`: allocation-sizing inputs computed from rates, parameters and scalar
//! math are proven to constants the way scsynth's constructors read them, and everything else is
//! refused with a cause.

use plyphon::{
    AddAction, BuildError, InitCause, InputRef, Options, Param, ROOT_GROUP_ID, Rate, RateInfo,
    SynthDef, UnitRegistry, UnitSpec, engine,
};
use plyphon_unit::unit::select::select_index;

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

/// SuperCollider's `opAdd` / `opMul` binary selectors.
const ADD: i16 = 0;
const MUL: i16 = 2;
/// SuperCollider's `opRecip` unary selector.
const RECIP: i16 = 16;
/// A unary selector the operator table does not resolve (`opIsNil`).
const UNMAPPED: i16 = 3;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

fn p(index: u32) -> InputRef {
    InputRef::Param(index)
}

fn op(name: &str, inputs: Vec<InputRef>, special_index: i16) -> UnitSpec {
    UnitSpec {
        special_index,
        ..UnitSpec::new(name, Rate::Control, inputs, 1)
    }
}

fn def(params: Vec<Param>, units: Vec<UnitSpec>) -> SynthDef {
    SynthDef {
        name: "def".to_string(),
        params,
        units,
    }
}

fn rate() -> RateInfo {
    RateInfo::new(SR, BLOCK)
}

fn compile(def: &SynthDef) -> Result<(), BuildError> {
    let control = RateInfo::new(SR / BLOCK as f64, 1);
    let registry = UnitRegistry::with_builtins();
    def.compile(&registry, &rate(), &control, 1024, 128, None, 1)
        .map(|_| ())
}

fn constant_at(def: &SynthDef, unit: usize, input: usize) -> f32 {
    match def.units[unit].inputs[input] {
        InputRef::Constant(v) => v,
        other => panic!("expected a constant at unit {unit} input {input}, got {other:?}"),
    }
}

/// `DC.ar(0) -> DelayC.ar(dc, maxdelaytime)` with `maxdelaytime` computed by `units`, which start
/// at index 1; the delay is the last unit.
fn delay_sized_by(params: Vec<Param>, mut units: Vec<UnitSpec>, max_delay: InputRef) -> SynthDef {
    units.insert(0, UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1));
    units.push(UnitSpec::new(
        "DelayC",
        Rate::Audio,
        vec![u(0), max_delay, c(0.01)],
        1,
    ));
    def(params, units)
}

#[test]
fn rate_expression_is_proven() {
    // `LocalBuf(1, SampleRate.ir * 0.1)`.
    let d = def(
        vec![],
        vec![
            UnitSpec::new("SampleRate", Rate::Scalar, vec![], 1),
            op("BinaryOpUGen", vec![u(0), c(0.1)], MUL),
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), u(1)], 1),
        ],
    );
    assert!(matches!(
        compile(&d),
        Err(BuildError::AuxRequiresConstant { input: 1 })
    ));

    let s = d.specialize_init(&rate(), &[]).unwrap();
    assert_eq!(s.rewritten, vec![(2, 1)]);
    assert_eq!(constant_at(&s.def, 2, 1), SR as f32 * 0.1);
    assert!(s.dependencies.is_empty());
    compile(&s.def).unwrap();
}

#[test]
fn param_is_proven_and_its_live_use_is_kept() {
    // One param sizes the delay and also drives an oscillator; only the sizing input is rewritten.
    let d = delay_sized_by(
        vec![Param::control("size", 2.0)],
        vec![UnitSpec::new("SinOsc", Rate::Audio, vec![p(0), c(0.0)], 1)],
        p(0),
    );
    let s = d.specialize_init(&rate(), &[2.0]).unwrap();
    assert_eq!(s.rewritten, vec![(2, 1)]);
    assert_eq!(constant_at(&s.def, 2, 1), 2.0);
    assert!(matches!(s.def.units[1].inputs[0], InputRef::Param(0)));
    assert_eq!(s.dependencies, vec![0]);
    compile(&s.def).unwrap();
}

#[test]
fn trigger_param_proves_to_zero() {
    // scsynth's `TrigControl` constructor clears its output, so constructors downstream read 0.
    let d = delay_sized_by(vec![Param::trig("t_size", 5.0)], vec![], p(0));
    let s = d.specialize_init(&rate(), &[5.0]).unwrap();
    assert_eq!(constant_at(&s.def, 1, 1), 0.0);
    assert!(s.dependencies.is_empty());
}

#[test]
fn select_proves_only_the_selected_branch() {
    // `Select(sel, [K2A(x), x_ar])` choosing the scalar branch: the audio branch is not traversed.
    let d = def(
        vec![
            Param::control("sel", 0.0),
            Param::control("x", 0.03),
            Param::audio("x_ar", 0.5),
        ],
        vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
            UnitSpec::new("K2A", Rate::Audio, vec![p(1)], 1),
            UnitSpec::new("Select", Rate::Audio, vec![p(0), u(1), p(2)], 1),
            UnitSpec::new(
                "PitchShift",
                Rate::Audio,
                vec![u(0), u(2), c(1.0), c(0.0), c(0.0)],
                1,
            ),
        ],
    );
    let s = d.specialize_init(&rate(), &[0.0, 0.03, 0.5]).unwrap();
    assert_eq!(s.rewritten, vec![(3, 1)]);
    assert_eq!(constant_at(&s.def, 3, 1), 0.03);
    assert_eq!(s.dependencies, vec![0, 1]);
    compile(&s.def).unwrap();
}

#[test]
fn select_picks_the_same_branch_as_the_running_unit() {
    const BRANCHES: [f32; 3] = [11.0, 22.0, 33.0];
    let sized = def(
        vec![Param::control("sel", 0.0)],
        vec![
            UnitSpec::new(
                "Select",
                Rate::Control,
                vec![p(0), c(BRANCHES[0]), c(BRANCHES[1]), c(BRANCHES[2])],
                1,
            ),
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), u(0)], 1),
        ],
    );

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "picked".to_string(),
        ..def(
            vec![Param::control("sel", 0.0)],
            vec![
                UnitSpec::new("DC", Rate::Audio, vec![c(BRANCHES[0])], 1),
                UnitSpec::new("DC", Rate::Audio, vec![c(BRANCHES[1])], 1),
                UnitSpec::new("DC", Rate::Audio, vec![c(BRANCHES[2])], 1),
                UnitSpec::new("Select", Rate::Audio, vec![p(0), u(0), u(1), u(2)], 1),
                UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(3)], 0),
            ],
        )
    });
    let node = controller
        .synth_new("picked", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // Out-of-range and NaN selectors pick the first input, as scsynth's wrapping `(int32)in + 1`
    // does.
    assert_eq!(select_index(3e9, 4), 1);
    assert_eq!(select_index(-3e9, 4), 1);
    assert_eq!(select_index(f32::NAN, 4), 1);

    for selector in [-3e9, -0.5, 0.0, 0.9, 1.5, 2.0, 5.0, 3e9, f32::NAN] {
        let expected = BRANCHES[select_index(selector, 4) - 1];
        let s = sized.specialize_init(&rate(), &[selector]).unwrap();
        assert_eq!(constant_at(&s.def, 1, 1), expected, "selector {selector}");

        controller.set_control(node, 0, selector).unwrap();
        let mut block = [0.0f32; BLOCK];
        world.fill(&mut block, 1);
        assert_eq!(block[0], expected, "selector {selector}");
    }
}

#[test]
fn fft_window_size_is_proven() {
    // `FFT(LocalBuf(2048), in, winsize: n)` with `n` a parameter.
    let d = def(
        vec![Param::control("n", 1024.0)],
        vec![
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), c(2048.0)], 1),
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
            UnitSpec::new(
                "FFT",
                Rate::Control,
                vec![u(0), u(1), c(0.5), c(0.0), c(1.0), p(0)],
                1,
            ),
        ],
    );
    let s = d.specialize_init(&rate(), &[1024.0]).unwrap();
    assert_eq!(s.rewritten, vec![(2, 5)]);
    assert_eq!(constant_at(&s.def, 2, 5), 1024.0);
    compile(&s.def).unwrap();
}

#[test]
fn unprovable_inputs_report_a_cause() {
    let cases = [
        (
            "bus read",
            vec![UnitSpec::new("In", Rate::Control, vec![c(0.0)], 1)],
            InitCause::Signal,
        ),
        (
            "init-time random",
            vec![UnitSpec::new("Rand", Rate::Scalar, vec![c(0.1), c(0.2)], 1)],
            InitCause::Random,
        ),
        (
            "buffer metadata",
            vec![UnitSpec::new("BufDur", Rate::Scalar, vec![p(0)], 1)],
            InitCause::BufferMetadataUnavailable,
        ),
        (
            "unresolved operator",
            vec![op("UnaryOpUGen", vec![c(0.05)], UNMAPPED)],
            InitCause::Unsupported,
        ),
        (
            "non-finite result",
            vec![op("UnaryOpUGen", vec![p(0)], RECIP)],
            InitCause::NonFinite,
        ),
    ];
    for (what, units, cause) in cases {
        let d = delay_sized_by(vec![Param::control("x", 0.0)], units, u(1));
        let error = d.specialize_init(&rate(), &[0.0]).unwrap_err();
        assert_eq!(
            (error.unit, error.input, error.cause),
            (2, 1, cause),
            "{what}"
        );
    }

    // A failure reports the params it traversed, so the host can retry when they change.
    let d = delay_sized_by(
        vec![Param::control("x", 0.0)],
        vec![UnitSpec::new("BufDur", Rate::Scalar, vec![p(0)], 1)],
        u(1),
    );
    assert_eq!(
        d.specialize_init(&rate(), &[0.0]).unwrap_err().dependencies,
        vec![0]
    );

    // A param reference past the supplied values is refused, never read out of bounds.
    let d = delay_sized_by(vec![Param::control("x", 0.0)], vec![], p(0));
    let error = d.specialize_init(&rate(), &[]).unwrap_err();
    assert_eq!(error.cause, InitCause::Unsupported);
}

#[test]
fn constant_inputs_are_left_alone() {
    let d = delay_sized_by(vec![Param::control("freq", 440.0)], vec![], c(0.25));
    let s = d.specialize_init(&rate(), &[440.0]).unwrap();
    assert!(s.rewritten.is_empty());
    assert!(s.dependencies.is_empty());
    assert_eq!(format!("{:?}", s.def), format!("{d:?}"));
}

#[test]
fn shared_subexpressions_evaluate_in_linear_time() {
    // sclang's `n.do { a = a + a }` builds a diamond DAG that costs `2^depth` without memoization.
    const DEPTH: u32 = 64;
    let mut units = vec![UnitSpec::new("SampleRate", Rate::Scalar, vec![], 1)];
    for level in 0..DEPTH {
        units.push(op("BinaryOpUGen", vec![u(level), u(level)], ADD));
    }
    units.push(op("BinaryOpUGen", vec![u(DEPTH), c(0.0)], MUL));
    units.push(op("BinaryOpUGen", vec![u(DEPTH + 1), c(64.0)], ADD));
    units.push(UnitSpec::new(
        "LocalBuf",
        Rate::Scalar,
        vec![c(1.0), u(DEPTH + 2)],
        1,
    ));
    let s = def(vec![], units).specialize_init(&rate(), &[]).unwrap();
    assert_eq!(constant_at(&s.def, DEPTH as usize + 3, 1), 64.0);
}
