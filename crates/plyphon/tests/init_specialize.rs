//! Initialization specialization (`SynthDef::specialize_init`): proving allocation-only unit inputs
//! to constants under scsynth's constructor-time first-sample semantics, the deterministic
//! classification of every expression that cannot be proven, and the per-site allocation bound.

use plyphon::{
    AddAction, AuxDynamicCause, BuildError, InitEnvironment, InputRef, Options, Param,
    ROOT_GROUP_ID, Rate, RateInfo, SynthDef, UnitRegistry, UnitSpec, World, engine, graph_rates,
};
use plyphon_unit::unit::init_only::{MAX_AUX_ELEMS, init_only_inputs};
use plyphon_unit::unit::select::select_index;

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

/// SuperCollider's `opMul` binary selector.
const MUL: i16 = 2;
/// SuperCollider's `opRecip` unary selector.
const RECIP: i16 = 16;
/// SuperCollider's `opThru` unary selector - an identity the shipped table resolves.
const THRU: i16 = 47;
/// A unary selector the shipped table does not resolve (SC's `opIsNil` region).
const UNMAPPED: i16 = 3;

/// A constant input.
fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Output 0 of unit `unit`.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// The value of parameter `index`.
fn p(index: u32) -> InputRef {
    InputRef::Param(index)
}

/// A `UnitSpec` carrying an explicit operator selector (`UnaryOpUGen`/`BinaryOpUGen`).
fn op(name: &str, rate: Rate, inputs: Vec<InputRef>, special_index: i16) -> UnitSpec {
    UnitSpec {
        name: name.to_string(),
        rate,
        inputs,
        num_outputs: 1,
        special_index,
    }
}

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    }
}

/// The graph rate pair these tests specialize and compile against, derived through the same helper
/// `compile` uses so the evaluator and the built units can never disagree.
fn rates() -> (RateInfo, RateInfo) {
    let audio = RateInfo::new(SR, BLOCK);
    let control = RateInfo::new(SR / BLOCK as f64, 1);
    graph_rates(&audio, &control, None, 1).expect("nominal graph rates")
}

/// An initialization environment at the nominal rates with `params` as the authored bases.
fn env(params: &[f32]) -> InitEnvironment {
    InitEnvironment {
        audio: rates().0,
        params: params.to_vec(),
    }
}

/// Compile `def` with the built-in registry at the nominal rates, returning the result so a test can
/// assert the build error.
fn try_compile(def: &SynthDef) -> Result<(), BuildError> {
    let (audio, control) = rates();
    def.compile(
        &UnitRegistry::with_builtins(),
        &audio,
        &control,
        1024,
        128,
        None,
        1,
    )
    .map(|_| ())
}

/// Render one control block of mono audio.
fn one_block(world: &mut World) -> Vec<f32> {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf
}

/// Zero crossings in `block` - a frequency-sensitive statistic, so a faster oscillator scores higher.
fn sign_changes(block: &[f32]) -> usize {
    block
        .windows(2)
        .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
        .count()
}

/// The constant `def`'s unit `unit` carries at input `input`, or a panic naming what is there.
fn constant_at(def: &SynthDef, unit: usize, input: usize, what: &str) -> f32 {
    match def.units[unit].inputs[input] {
        InputRef::Constant(v) => v,
        other => panic!("{what}: expected a baked constant, got {other:?}"),
    }
}

/// The `AuxRequiresConstant` cause `def` fails specialization with, or a panic naming the outcome.
fn dynamic_cause(def: &SynthDef, environment: &InitEnvironment, what: &str) -> AuxDynamicCause {
    match def.specialize_init(environment) {
        Err(error) => match error.error {
            BuildError::AuxRequiresConstant { cause, .. } => cause,
            other => panic!("{what}: expected AuxRequiresConstant, got {other:?}"),
        },
        Ok(_) => panic!("{what}: expected a dynamic declared input, but the definition proved"),
    }
}

#[test]
fn pure_scalar_aux_expression_is_proven() {
    // The rates-only class: `LocalBuf` frames sized from `SampleRate.ir * k`. No parameter is
    // involved, so the proof depends on the rate environment alone.
    const SECONDS: f32 = 0.1;
    let def = SynthDef {
        name: "scalar".to_string(),
        params: vec![],
        units: vec![
            // 0: SampleRate.ir - the compile-context invariant the frames chain roots in.
            UnitSpec::new("SampleRate", Rate::Control, vec![], 1),
            // 1: SampleRate * SECONDS - resolved through the shipped binary-op table.
            op("BinaryOpUGen", Rate::Control, vec![u(0), c(SECONDS)], MUL),
            // 2: LocalBuf(1, frames) - the declared allocation site.
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), u(1)], 1),
        ],
    };

    let specialized = def
        .specialize_init(&env(&[]))
        .expect("a pure scalar chain proves");
    assert_eq!(
        specialized.rewritten,
        vec![(2, 1)],
        "only the declared `frames` input is rewritten"
    );
    assert_eq!(
        constant_at(&specialized.def, 2, 1, "LocalBuf frames"),
        SR as f32 * SECONDS,
        "the baked frame count is the shipped multiply of the graph rate"
    );
    assert!(
        specialized.dependencies.params.is_empty(),
        "no parameter influenced the proof, got {:?}",
        specialized.dependencies.params
    );

    // The unspecialized definition is exactly the failure specialization exists to close.
    assert_eq!(
        try_compile(&def),
        Err(BuildError::AuxRequiresConstant {
            input: 1,
            cause: AuxDynamicCause::Unsupported,
        }),
        "a bare compile still rejects the wired frames input"
    );
    try_compile(&specialized.def).expect("the specialized definition compiles");
}

#[test]
fn select_gated_param_aux_specializes_from_initial_values() {
    // The imported-corpus shape: every value parameter reaches its consumer through
    // `Select(x_sel, K2A(x), x_ar)`, with the hidden selector choosing the scalar branch. The
    // audio branch carries a *different* default, so proving the wrong branch is visible.
    const WINDOW: f32 = 0.03;
    const AUDIO_BRANCH: f32 = 0.5;
    let def = SynthDef {
        name: "gated".to_string(),
        params: vec![
            Param::control("sel", 0.0),
            Param::control("x", WINDOW),
            Param::audio("x_ar", AUDIO_BRANCH),
        ],
        units: vec![
            // 0: SinOsc.ar - the signal PitchShift transposes.
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
            // 1: K2A.ar(x) - the scalar branch, rate-lifted.
            UnitSpec::new("K2A", Rate::Audio, vec![p(1)], 1),
            // 2: Select.ar(sel, K2A(x), x_ar).
            UnitSpec::new("Select", Rate::Audio, vec![p(0), u(1), p(2)], 1),
            // 3: PitchShift.ar(in, windowSize, 1, 0, 0) - the declared allocation site.
            UnitSpec::new(
                "PitchShift",
                Rate::Audio,
                vec![u(0), u(2), c(1.0), c(0.0), c(0.0)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(3)], 0),
        ],
    };

    let specialized = def
        .specialize_init(&env(&[0.0, WINDOW, AUDIO_BRANCH]))
        .expect("a Select-gated parameter proves from its authored bases");
    assert_eq!(
        specialized.rewritten,
        vec![(3, 1)],
        "only PitchShift's windowSize is rewritten"
    );
    assert_eq!(
        constant_at(&specialized.def, 3, 1, "PitchShift windowSize"),
        WINDOW,
        "the selected scalar branch supplies the size, not the audio branch"
    );
    assert_eq!(
        specialized.dependencies.params,
        vec![0, 1],
        "the selector is traversed alongside the branch it selected, and the unselected \
         audio branch is not"
    );

    assert!(
        matches!(
            try_compile(&def),
            Err(BuildError::AuxRequiresConstant { input: 1, .. })
        ),
        "the unspecialized definition still fails to compile"
    );
    try_compile(&specialized.def).expect("the specialized definition compiles");
}

#[test]
fn select_index_conversion_matches_runtime_unit() {
    // The evaluator and the compiled `Select` share one index conversion, so a non-integral or
    // out-of-range selector can never pick different branches on the two sides.
    const BRANCHES: [f32; 3] = [11.0, 22.0, 33.0];
    const SELECTORS: [f32; 7] = [-0.5, 0.0, 0.9, 1.0, 1.5, 2.0, 5.0];

    // The specialization side: three constant branches feeding `LocalBuf` frames.
    let sized = SynthDef {
        name: "sized".to_string(),
        params: vec![Param::control("sel", 0.0)],
        units: vec![
            UnitSpec::new(
                "Select",
                Rate::Control,
                vec![p(0), c(BRANCHES[0]), c(BRANCHES[1]), c(BRANCHES[2])],
                1,
            ),
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), u(0)], 1),
        ],
    };

    // The runtime side: the same three branches as constant signals through a compiled `Select`.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "picked".to_string(),
        params: vec![Param::control("sel", 0.0)],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(BRANCHES[0])], 1),
            UnitSpec::new("DC", Rate::Audio, vec![c(BRANCHES[1])], 1),
            UnitSpec::new("DC", Rate::Audio, vec![c(BRANCHES[2])], 1),
            UnitSpec::new("Select", Rate::Audio, vec![p(0), u(0), u(1), u(2)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(3)], 0),
        ],
    });
    let node = controller
        .synth_new("picked", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    let mut seen = Vec::new();
    for selector in SELECTORS {
        // The shipped conversion truncates toward zero, then clamps into the branch range.
        let expected = BRANCHES[select_index(selector, 1 + BRANCHES.len()) - 1];
        seen.push(expected);

        let specialized = sized
            .specialize_init(&env(&[selector]))
            .expect("a proven selector proves the branch it picks");
        assert_eq!(
            constant_at(&specialized.def, 1, 1, "LocalBuf frames"),
            expected,
            "selector {selector} must bake the branch `select_index` picks"
        );

        controller.set_control(node, 0, selector).expect("set sel");
        let rendered = one_block(&mut world);
        assert_eq!(
            rendered[0], expected,
            "selector {selector} must render the branch `select_index` picks"
        );
    }
    assert!(
        seen.contains(&BRANCHES[0]) && seen.contains(&BRANCHES[1]) && seen.contains(&BRANCHES[2]),
        "the sweep must exercise every branch, got {seen:?}"
    );
}

#[test]
fn param_init_value_specializes_aux_without_replacing_signal_use() {
    // One parameter feeds two inputs: `DelayC`'s init-only `maxdelaytime` (declared, so it bakes)
    // and `SinOsc`'s frequency (not declared, so it stays a live control).
    const BASE: f32 = 2.0;
    const SLOW: f32 = 100.0;
    const FAST: f32 = 6_000.0;
    let def = SynthDef {
        name: "dual".to_string(),
        params: vec![Param::control("size", BASE)],
        units: vec![
            // 0: SinOsc.ar(size) - a live signal use of the same parameter.
            UnitSpec::new("SinOsc", Rate::Audio, vec![p(0), c(0.0)], 1),
            // 1: DelayC.ar(in, size, 0.01) - the declared allocation site.
            UnitSpec::new("DelayC", Rate::Audio, vec![u(0), p(0), c(0.01)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    };

    let specialized = def
        .specialize_init(&env(&[BASE]))
        .expect("a parameter base proves the declared input");
    assert_eq!(
        specialized.rewritten,
        vec![(1, 1)],
        "only the delay's maxdelaytime is rewritten"
    );
    assert_eq!(
        constant_at(&specialized.def, 1, 1, "DelayC maxdelaytime"),
        BASE,
        "the delay line is sized from the authored base"
    );
    assert!(
        matches!(specialized.def.units[0].inputs[0], InputRef::Param(0)),
        "the oscillator's frequency must stay a parameter reference, got {:?}",
        specialized.def.units[0].inputs[0]
    );
    assert_eq!(specialized.dependencies.params, vec![0]);
    assert!(
        matches!(
            try_compile(&def),
            Err(BuildError::AuxRequiresConstant { input: 1, .. })
        ),
        "the unspecialized definition still fails to compile"
    );

    // The surviving parameter reference must still track `set_control` on the audio thread.
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(specialized.def.clone());
    let node = controller
        .synth_new("dual", ROOT_GROUP_ID, AddAction::Tail)
        .expect("the specialized definition spawns");
    controller.set_control(node, 0, SLOW).expect("set size");
    let slow = one_block(&mut world);
    controller.set_control(node, 0, FAST).expect("set size");
    let fast = one_block(&mut world);
    assert!(
        sign_changes(&fast) > sign_changes(&slow),
        "a live update must change the rendered frequency: {} crossings at {SLOW} Hz, {} at {FAST} Hz",
        sign_changes(&slow),
        sign_changes(&fast),
    );
}

#[test]
fn buf_rooted_aux_reports_buffer_cause() {
    // Buffer-metadata-backed proving is deferred, so any proof rooted in a `Buf*` info unit
    // classifies as buffer-gated - unconditionally, with no registry plumbing behind it.
    let def = SynthDef {
        name: "buffered".to_string(),
        params: vec![
            Param::control("bufnum", 0.0),
            Param::control("stretch", 4.0),
        ],
        units: vec![
            // 0: BufSampleRate.kr(bufnum) - the metadata leaf.
            UnitSpec::new("BufSampleRate", Rate::Control, vec![p(0)], 1),
            // 1: BufSampleRate * stretch.
            op("BinaryOpUGen", Rate::Control, vec![u(0), p(1)], MUL),
            // 2: LocalBuf(1, frames).
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), u(1)], 1),
        ],
    };

    let error = def
        .specialize_init(&env(&[0.0, 4.0]))
        .expect_err("a buffer-rooted size cannot be proven");
    assert_eq!(
        error.error,
        BuildError::AuxRequiresConstant {
            input: 1,
            cause: AuxDynamicCause::BufferMetadataUnavailable,
        },
    );
    // The `Buf*` leaf traverses its buffer input for the attempted-dependency record (its value
    // is unused - no metadata channel exists here), so both the buffer-selecting param and the
    // sibling operand are reported: a corrective edit to either re-triggers the region.
    assert_eq!(
        error.dependencies.params,
        vec![0, 1],
        "both the buffer-selecting param and the traversed sibling operand are reported"
    );
}

#[test]
fn random_initializer_reports_random_cause() {
    // scsynth folds a scalar `Rand` at construction; plyphon deliberately does not, since baking
    // one draw would freeze a seed into the definition's identity.
    let def = SynthDef {
        name: "random".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
            // 1: Rand.ir(0.1, 0.2) - an init-time draw.
            UnitSpec::new("Rand", Rate::Control, vec![c(0.1), c(0.2)], 1),
            UnitSpec::new(
                "PitchShift",
                Rate::Audio,
                vec![u(0), u(1), c(1.0), c(0.0), c(0.0)],
                1,
            ),
        ],
    };

    let error = def
        .specialize_init(&env(&[]))
        .expect_err("an init-time random size stays unsupported");
    assert_eq!(
        error.error,
        BuildError::AuxRequiresConstant {
            input: 1,
            cause: AuxDynamicCause::Random,
        },
    );
    assert!(
        error.dependencies.params.is_empty(),
        "no parameter was traversed, got {:?}",
        error.dependencies.params
    );
}

#[test]
fn signal_driven_aux_reports_signal_cause() {
    // A bus read is a genuinely signal-driven size, not merely an expression the evaluator does
    // not recognise: coverage classification depends on the two being distinguishable.
    for (name, reader) in [
        ("in", UnitSpec::new("In", Rate::Control, vec![c(0.0)], 1)),
        ("localin", UnitSpec::new("LocalIn", Rate::Audio, vec![], 1)),
    ] {
        let def = SynthDef {
            name: name.to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
                reader,
                // 2: DelayC.ar(in, maxdelaytime = the bus read, 0.01).
                UnitSpec::new("DelayC", Rate::Audio, vec![u(0), u(1), c(0.01)], 1),
            ],
        };

        let cause = dynamic_cause(&def, &env(&[]), name);
        assert_eq!(
            cause,
            AuxDynamicCause::Signal,
            "{name}: a bus read is a signal"
        );
        assert_ne!(
            cause,
            AuxDynamicCause::Unsupported,
            "{name}: a bus read must not be lumped in with unrecognised expressions"
        );

        // The builder's own raise site - reached by callers that never specialize - keeps the
        // undifferentiated cause.
        assert_eq!(
            try_compile(&def),
            Err(BuildError::AuxRequiresConstant {
                input: 1,
                cause: AuxDynamicCause::Unsupported,
            }),
            "{name}: a bare compile raises the builder's cause"
        );
    }
}

#[test]
fn unsupported_op_index_stays_dynamic() {
    // Operators resolve through the shipped tables; an index they do not resolve is `Dynamic`
    // rather than a guessed kernel.
    let def = |special_index| SynthDef {
        name: "operator".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
            op("UnaryOpUGen", Rate::Control, vec![c(0.05)], special_index),
            UnitSpec::new("DelayC", Rate::Audio, vec![u(0), u(1), c(0.01)], 1),
        ],
    };

    assert_eq!(
        dynamic_cause(&def(UNMAPPED), &env(&[]), "unmapped operator"),
        AuxDynamicCause::Unsupported,
    );

    // The same chain through an operator the table *does* resolve proves, so the failure is the
    // selector and not the shape.
    let specialized = def(THRU)
        .specialize_init(&env(&[]))
        .expect("a resolved operator proves");
    assert_eq!(
        constant_at(&specialized.def, 2, 1, "DelayC maxdelaytime"),
        0.05,
    );
}

#[test]
fn param_index_past_env_end_is_dynamic() {
    // A host transform may append parameters after the authored ones, so the environment can be
    // shorter than the definition's parameter list: a reference past its end is `Dynamic`, never a
    // panic or a read of a neighbouring lane.
    let def = SynthDef {
        name: "appended".to_string(),
        params: vec![
            Param::control("size", 0.05),
            Param::control("appended", 1.0),
        ],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
            UnitSpec::new("DelayC", Rate::Audio, vec![u(0), p(1), c(0.01)], 1),
        ],
    };

    let error = def
        .specialize_init(&env(&[0.05]))
        .expect_err("a reference past the environment cannot be proven");
    assert_eq!(
        error.error,
        BuildError::AuxRequiresConstant {
            input: 1,
            cause: AuxDynamicCause::Unsupported,
        },
    );
    assert!(
        error.dependencies.params.is_empty(),
        "an unreadable index records no dependency, got {:?}",
        error.dependencies.params
    );

    // A total environment proves the very same definition, so the failure is the short vector.
    let specialized = def
        .specialize_init(&env(&[0.05, 1.0]))
        .expect("a total environment proves");
    assert_eq!(
        constant_at(&specialized.def, 1, 1, "DelayC maxdelaytime"),
        1.0
    );
    assert_eq!(specialized.dependencies.params, vec![1]);
}

#[test]
fn non_finite_aux_expression_is_rejected() {
    // A proof that reduces to a non-finite size fails at the allocation site rather than sizing
    // from a nonsense number. A syntactic constant would have folded at authoring time, so the
    // infinity has to come from a parameter base: `1 / 0`.
    let def = SynthDef {
        name: "infinite".to_string(),
        params: vec![Param::control("size", 0.0)],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
            op("UnaryOpUGen", Rate::Control, vec![p(0)], RECIP),
            UnitSpec::new("DelayC", Rate::Audio, vec![u(0), u(1), c(0.01)], 1),
        ],
    };

    let error = def
        .specialize_init(&env(&[0.0]))
        .expect_err("a non-finite proven size is rejected");
    assert_eq!(
        error.error,
        BuildError::AuxNonFinite {
            unit: "DelayC".to_string(),
            input: 1,
        },
    );
    assert_eq!(
        error.dependencies.params,
        vec![0],
        "the traversed base is reported so a corrective edit re-triggers"
    );

    // A finite base proves the same chain, so the failure is the value and not the expression.
    let specialized = def
        .specialize_init(&env(&[4.0]))
        .expect("a finite reciprocal proves");
    assert_eq!(
        constant_at(&specialized.def, 2, 1, "DelayC maxdelaytime"),
        0.25
    );
}

/// The element count `line_len` computes for a delay line of `secs` at the test rates - the
/// builder's own saturating accumulation, so a test can name the exact bound crossing.
fn delay_line_elems(secs: f32) -> u64 {
    ((secs.max(0.0) as f64 * SR + 1.0).ceil() as u64)
        .saturating_add(BLOCK as u64)
        .max(1)
}

/// `DC.ar(0) -> DelayC.ar(in, maxdelaytime, 0.01)`: one delay line and nothing else.
fn delay_def(name: &str, params: Vec<Param>, max_delay: InputRef) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params,
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
            UnitSpec::new("DelayC", Rate::Audio, vec![u(0), max_delay, c(0.01)], 1),
        ],
    }
}

/// Compile `def` expecting the allocation bound to reject it, returning `(unit, input, elements)`.
fn expect_out_of_range(def: &SynthDef, what: &str) -> (String, usize, u64) {
    match try_compile(def) {
        Err(BuildError::AuxSizeOutOfRange {
            unit,
            input,
            elements,
            limit,
        }) => {
            assert_eq!(limit, MAX_AUX_ELEMS, "{what}: the reported bound");
            (unit, input, elements)
        }
        other => panic!("{what}: expected AuxSizeOutOfRange, got {other:?}"),
    }
}

#[test]
fn aux_size_out_of_range_is_rejected() {
    // Regime 1 - one step over. The comparison is exact, so the largest fitting `f32` max-delay and
    // the next `f32` above it land on opposite sides of the bound.
    let mut fits = ((MAX_AUX_ELEMS - BLOCK as u64) as f64 / SR) as f32;
    while delay_line_elems(fits) > MAX_AUX_ELEMS {
        fits = f32::from_bits(fits.to_bits() - 1);
    }
    let stepped = f32::from_bits(fits.to_bits() + 1);
    let (unit, input, elements) = expect_out_of_range(
        &delay_def("step", vec![], c(stepped)),
        "a delay one step over the bound",
    );
    assert_eq!((unit.as_str(), input), ("Delay", 1));
    assert_eq!(elements, delay_line_elems(stepped));
    assert!(
        elements > MAX_AUX_ELEMS && elements <= MAX_AUX_ELEMS + 8,
        "one step over must be rejected by a hair, not by a rounded comparison: {elements}"
    );
    // The check does not fire for ordinary sizes.
    try_compile(&delay_def("modest", vec![], c(0.05)))
        .expect("an ordinary delay line still compiles");

    // Regime 2 - truncating: a size past `u32` would have wrapped to a short line whose mask then
    // indexes an empty slice on the audio thread. Driven through `specialize_init` + a compile of
    // the specialized definition, so the bound is proven on the specialization path too.
    let proven = delay_def("truncating", vec![Param::control("size", 1e5)], p(0))
        .specialize_init(&env(&[1e5]))
        .expect("a large finite base proves");
    assert_eq!(constant_at(&proven.def, 1, 1, "DelayC maxdelaytime"), 1e5);
    let (unit, input, elements) =
        expect_out_of_range(&proven.def, "a specialized delay in the truncating regime");
    assert_eq!((unit.as_str(), input), ("Delay", 1));
    assert!(
        elements >= 1 << 31,
        "the count must be compared before any narrowing cast, got {elements}"
    );

    let (unit, input, elements) = expect_out_of_range(
        &SynthDef {
            name: "shifted".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
                UnitSpec::new(
                    "PitchShift",
                    Rate::Audio,
                    vec![u(0), c(1e5), c(1.0), c(0.0), c(0.0)],
                    1,
                ),
            ],
        },
        "a PitchShift window in the truncating regime",
    );
    assert_eq!((unit.as_str(), input), ("PitchShift", 1));
    assert!(
        elements >= 1 << 31,
        "compared before narrowing, got {elements}"
    );

    // Regime 3 - saturating: every accumulation saturates end to end, so an absurd size arrives at
    // the comparison as a huge count rather than wrapping into a small one.
    let (unit, input, elements) = expect_out_of_range(
        &delay_def("saturating", vec![], c(1e30)),
        "a delay in the saturating regime",
    );
    assert_eq!((unit.as_str(), input), ("Delay", 1));
    assert_eq!(elements, u64::MAX, "the float -> integer cast saturates");

    let (unit, input, elements) = expect_out_of_range(
        &SynthDef {
            name: "wide".to_string(),
            params: vec![],
            units: vec![UnitSpec::new(
                "LocalBuf",
                Rate::Scalar,
                vec![c(4_294_967_296.0), c(4_294_967_296.0)],
                1,
            )],
        },
        "a 2^32 x 2^32 LocalBuf",
    );
    assert_eq!((unit.as_str(), input), ("LocalBuf", 0));
    assert_eq!(
        elements,
        1 << 32,
        "each factor is bound-checked before the product, so the first one fails"
    );

    // The product is checked in its own right: two in-range factors whose product is not.
    let (unit, input, elements) = expect_out_of_range(
        &SynthDef {
            name: "product".to_string(),
            params: vec![],
            units: vec![UnitSpec::new(
                "LocalBuf",
                Rate::Scalar,
                vec![c(4096.0), c(8192.0)],
                1,
            )],
        },
        "a LocalBuf whose product exceeds the bound",
    );
    assert_eq!((unit.as_str(), input), ("LocalBuf", 1));
    assert_eq!(elements, 4096 * 8192);
    try_compile(&SynthDef {
        name: "small".to_string(),
        params: vec![],
        units: vec![UnitSpec::new(
            "LocalBuf",
            Rate::Scalar,
            vec![c(2.0), c(64.0)],
            1,
        )],
    })
    .expect("an ordinary LocalBuf still compiles");

    let (unit, input, elements) = expect_out_of_range(
        &SynthDef {
            name: "verb".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("Impulse", Rate::Audio, vec![c(2.0), c(0.0)], 1),
                UnitSpec::new(
                    "GVerb",
                    Rate::Audio,
                    vec![
                        u(0),    // in
                        c(10.0), // roomsize
                        c(3.0),  // revtime
                        c(0.5),  // damping
                        c(0.5),  // inputbw
                        c(15.0), // spread
                        c(0.5),  // drylevel
                        c(0.7),  // earlyreflevel
                        c(0.5),  // taillevel
                        c(1e30), // maxroomsize
                    ],
                    2,
                ),
            ],
        },
        "a GVerb maxroomsize in the saturating regime",
    );
    assert_eq!((unit.as_str(), input), ("GVerb", 9));
    assert_eq!(elements, u64::MAX, "the room-size chain saturates");

    let (unit, input, elements) = expect_out_of_range(
        &SynthDef {
            name: "normalized".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("SinOsc", Rate::Audio, vec![c(220.0), c(0.0)], 1),
                UnitSpec::new("Normalizer", Rate::Audio, vec![u(0), c(0.5), c(1e30)], 1),
            ],
        },
        "a Normalizer duration in the saturating regime",
    );
    assert_eq!((unit.as_str(), input), ("LookAhead", 2));
    assert_eq!(elements, u64::MAX, "the three-window product saturates");
}

#[test]
fn specialize_init_is_deterministic_and_identity_when_unneeded() {
    // Determinism: one definition and one environment always yield the same result, bit for bit.
    const SECONDS: f32 = 0.07;
    let proving = SynthDef {
        name: "proving".to_string(),
        params: vec![Param::control("size", SECONDS)],
        units: vec![
            UnitSpec::new("SampleRate", Rate::Control, vec![], 1),
            op("BinaryOpUGen", Rate::Control, vec![u(0), p(0)], MUL),
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), u(1)], 1),
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
            UnitSpec::new("DelayC", Rate::Audio, vec![u(3), p(0), c(0.01)], 1),
        ],
    };
    let first = proving
        .specialize_init(&env(&[SECONDS]))
        .expect("the chain proves");
    let second = proving
        .specialize_init(&env(&[SECONDS]))
        .expect("the chain proves again");
    assert_eq!(
        format!("{first:?}"),
        format!("{second:?}"),
        "the whole result is reproducible"
    );
    assert_eq!(first.dependencies, second.dependencies);
    assert_eq!(first.rewritten, second.rewritten);
    for (index, (a, b)) in first.def.units.iter().zip(&second.def.units).enumerate() {
        assert_eq!(
            format!("{:?}", a.inputs),
            format!("{:?}", b.inputs),
            "unit {index}'s inputs are reproducible"
        );
    }
    assert!(
        !first.rewritten.is_empty(),
        "the fixture must actually rewrite something for the identity leg below to mean anything"
    );

    // Identity: a definition whose declared inputs are already syntactic constants comes back
    // structurally unchanged, so every definition that compiles today keeps its identity material.
    let unneeded = SynthDef {
        name: "unneeded".to_string(),
        params: vec![Param::control("freq", 440.0)],
        units: vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![p(0), c(0.0)], 1),
            UnitSpec::new("DelayC", Rate::Audio, vec![u(0), c(0.25), c(0.01)], 1),
            UnitSpec::new("LocalBuf", Rate::Scalar, vec![c(1.0), c(512.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
    };
    let identity = unneeded
        .specialize_init(&env(&[440.0]))
        .expect("nothing needed proving");
    assert!(
        identity.rewritten.is_empty(),
        "nothing was rewritten, got {:?}",
        identity.rewritten
    );
    assert!(
        identity.dependencies.params.is_empty(),
        "nothing was traversed, got {:?}",
        identity.dependencies.params
    );
    assert_eq!(
        format!("{:?}", identity.def),
        format!("{unneeded:?}"),
        "the returned definition is structurally identical to the input"
    );
}

#[test]
fn init_only_input_table_is_exactly_the_enumerated_aux_sites() {
    // The table is the evaluator's entire rewrite scope, so it is pinned to the exact enumeration
    // of allocation-sizing inputs: a new site upstream fails here until it is declared or
    // deliberately excluded.
    for name in [
        "DelayN", "DelayL", "DelayC", "CombN", "CombL", "CombC", "AllpassN", "AllpassL", "AllpassC",
    ] {
        assert_eq!(
            init_only_inputs(name),
            [1usize],
            "{name} declares its maxdelaytime"
        );
    }
    assert_eq!(init_only_inputs("Pluck"), [2usize], "Pluck's maxdelaytime");
    assert_eq!(
        init_only_inputs("PitchShift"),
        [1usize],
        "PitchShift's windowSize"
    );
    assert_eq!(
        init_only_inputs("LocalBuf"),
        [0usize, 1],
        "LocalBuf's channels and frames"
    );
    assert_eq!(init_only_inputs("Gendy1"), [8usize], "Gendy1's initCPs");
    for name in ["Limiter", "Normalizer"] {
        assert_eq!(
            init_only_inputs(name),
            [2usize],
            "{name}'s look-ahead duration"
        );
    }
    assert_eq!(init_only_inputs("Median"), [0usize], "Median's length");
    assert_eq!(
        init_only_inputs("GVerb"),
        [1usize, 5, 9],
        "GVerb's roomsize, spread and maxroomsize"
    );

    // The deliberate exclusions, and an ordinary unit, declare nothing.
    for name in ["FFT", "IFFT", "SendReply", "Poll", "SinOsc"] {
        assert!(
            init_only_inputs(name).is_empty(),
            "{name} declares no init-only input, got {:?}",
            init_only_inputs(name)
        );
    }
}
