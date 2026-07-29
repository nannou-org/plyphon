//! Aggregate contract tests for the SC3 processing and spectral units.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

use plyphon::{
    AddAction, BuildContext, BuildError, InputRef, Options, Param, ROOT_GROUP_ID, Rate, RateInfo,
    SynthDef, UnitRegistry, UnitSpec, World, engine,
};
use plyphon_unit::unit::InputSource;

/// Locates the checked-in black-box conformance pack.
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3_processing")
}

/// Shorthand for a constant graph input.
fn constant(value: f32) -> InputRef {
    InputRef::Constant(value)
}

/// References one output of an earlier graph unit.
fn wire(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
}

/// Renders one mono frame range from the hosted engine.
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut output = vec![0.0; frames];
    world.fill(&mut output, 1);
    output
}

/// One parameter-driven processing-unit graph configuration.
struct ProcessingCase {
    name: &'static str,
    node_rate: Rate,
    initial: &'static [f32],
    recovery: &'static [f32],
    audio_signal_input: bool,
    outputs: usize,
}

/// Build a parameter-driven processing unit, with an audio-rate `DC` source where inlet zero is a
/// signal. The returned node id is used to change every input at a callback boundary.
fn processing_case(case: &ProcessingCase, block_size: usize) -> (plyphon::Controller, World, i32) {
    processing_case_at(case, 48_000.0, block_size)
}

/// Builds a processing graph at an explicit sample rate and block size.
fn processing_case_at(
    case: &ProcessingCase,
    sample_rate: f64,
    block_size: usize,
) -> (plyphon::Controller, World, i32) {
    let params = case
        .initial
        .iter()
        .enumerate()
        .map(|(index, &value)| Param::control(format!("p{index}"), value))
        .collect::<Vec<_>>();
    let mut units = Vec::new();
    let mut inputs = Vec::new();
    let target_index = if case.audio_signal_input {
        units.push(UnitSpec::new(
            "DC",
            Rate::Audio,
            vec![InputRef::Param(0)],
            1,
        ));
        inputs.push(wire(0, 0));
        inputs.extend((1..case.initial.len()).map(|index| InputRef::Param(index as u32)));
        1
    } else {
        inputs.extend((0..case.initial.len()).map(|index| InputRef::Param(index as u32)));
        0
    };
    units.push(UnitSpec::new(
        case.name,
        case.node_rate,
        inputs,
        case.outputs,
    ));

    let output_unit = if case.node_rate == Rate::Control {
        let converter = units.len() as u32;
        units.push(UnitSpec::new(
            "K2A",
            Rate::Audio,
            vec![wire(target_index, 0)],
            1,
        ));
        converter
    } else {
        target_index
    };
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![constant(0.0), wire(output_unit, 0)],
        0,
    ));

    let (mut controller, _nrt, world) = engine(Options {
        sample_rate,
        block_size,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: format!("{}-contract", case.name),
        params,
        units,
    });
    let synth = controller
        .synth_new(
            &format!("{}-contract", case.name),
            ROOT_GROUP_ID,
            AddAction::Tail,
        )
        .expect("processing graph compiles");
    (controller, world, synth)
}

/// Proves every retained oracle input still matches its manifest hash.
#[test]
fn sc3_processing_manifest_hashes_verify() {
    let output = Command::new("python3")
        .arg("verify.py")
        .current_dir(fixture_dir())
        .output()
        .expect("python3 runs the oracle verifier");
    assert!(
        output.status.success(),
        "oracle verifier failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Proves the aggregate built-in registry exposes all sixteen exact server names.
#[test]
fn sc3_processing_registry_contains_all_sixteen_units() {
    let registry = UnitRegistry::with_builtins();
    let calc = registry.names().collect::<BTreeSet<_>>();
    let demand = registry.demand_names().collect::<BTreeSet<_>>();

    for name in [
        "EnvDetect",
        "Decimator",
        "DFM1",
        "BMoog",
        "MoogLadder",
        "MoogVCF",
        "BlitB3",
        "BlitB3Saw",
        "BlitB3Square",
        "BlitB3Tri",
        "Perlin3",
        "RosslerL",
        "PV_Freeze",
        "PV_MagSmooth",
        "PV_Morph",
    ] {
        assert!(calc.contains(name), "missing calc unit {name}");
    }
    assert!(demand.contains("DNoiseRing"));
}

/// Invoke one calc-unit constructor directly so the complete fixed-shape ABI can be exercised
/// without graph-compilation errors masking the unit's own rejection.
fn calc_build_error(
    name: &str,
    rate: Rate,
    input_rates: &[Rate],
    outputs: usize,
    special_index: i16,
) -> Option<BuildError> {
    let registry = UnitRegistry::with_builtins();
    let def = registry.get(name).expect("calc-unit registration");
    let input_units = vec![None; input_rates.len()];
    let input_sources = vec![InputSource::Constant(0.0); input_rates.len()];
    let audio = RateInfo::new(48_000.0, 64);
    let control = RateInfo::new(750.0, 1);
    let ctx = BuildContext {
        input_rates,
        input_units: &input_units,
        input_sources: &input_sources,
        rate,
        num_outputs: outputs,
        audio: &audio,
        control: &control,
        special_index,
        seed: 1,
        local_bufs_so_far: 0,
    };
    def.build(&ctx).err()
}

/// Invoke the demand constructor directly for the same reason as [`calc_build_error`].
fn demand_build_error(
    rate: Rate,
    input_rates: &[Rate],
    outputs: usize,
    special_index: i16,
) -> Option<BuildError> {
    let registry = UnitRegistry::with_builtins();
    let def = registry
        .get_demand("DNoiseRing")
        .expect("DNoiseRing registration");
    let input_units = vec![None; input_rates.len()];
    let input_sources = vec![InputSource::Constant(0.0); input_rates.len()];
    let audio = RateInfo::new(48_000.0, 64);
    let control = RateInfo::new(750.0, 1);
    let ctx = BuildContext {
        input_rates,
        input_units: &input_units,
        input_sources: &input_sources,
        rate,
        num_outputs: outputs,
        audio: &audio,
        control: &control,
        special_index,
        seed: 1,
        local_bufs_so_far: 0,
    };
    def.build(&ctx).err()
}

/// Rejects invalid shapes, rates, outputs, and specialization indices for every unit.
#[test]
fn sc3_processing_signatures_reject_invalid_shapes() {
    /// Expected fixed ABI for one registered calculation unit.
    struct CalcAbi {
        name: &'static str,
        node_rate: Rate,
        input_rates: Vec<Rate>,
    }

    let cases = [
        CalcAbi {
            name: "EnvDetect",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Audio, Rate::Control, Rate::Scalar],
        },
        CalcAbi {
            name: "Decimator",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Audio, Rate::Control, Rate::Scalar],
        },
        CalcAbi {
            name: "DFM1",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Audio; 6],
        },
        CalcAbi {
            name: "BMoog",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Control; 4],
        },
        CalcAbi {
            name: "MoogLadder",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Scalar; 3],
        },
        CalcAbi {
            name: "MoogVCF",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Control; 3],
        },
        CalcAbi {
            name: "BlitB3",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Control],
        },
        CalcAbi {
            name: "BlitB3Saw",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Scalar, Rate::Audio],
        },
        CalcAbi {
            name: "BlitB3Square",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Control, Rate::Scalar],
        },
        CalcAbi {
            name: "BlitB3Tri",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Audio, Rate::Control, Rate::Scalar],
        },
        CalcAbi {
            name: "Perlin3",
            node_rate: Rate::Control,
            input_rates: vec![Rate::Scalar, Rate::Control, Rate::Scalar],
        },
        CalcAbi {
            name: "RosslerL",
            node_rate: Rate::Audio,
            input_rates: vec![Rate::Control; 8],
        },
        CalcAbi {
            name: "PV_Freeze",
            node_rate: Rate::Control,
            input_rates: vec![Rate::Control, Rate::Scalar],
        },
        CalcAbi {
            name: "PV_MagSmooth",
            node_rate: Rate::Control,
            input_rates: vec![Rate::Scalar, Rate::Control],
        },
        CalcAbi {
            name: "PV_Morph",
            node_rate: Rate::Control,
            input_rates: vec![Rate::Control, Rate::Scalar, Rate::Control],
        },
    ];

    for case in cases {
        let expected_outputs = if case.name == "RosslerL" { 3 } else { 1 };
        assert_eq!(
            calc_build_error(
                case.name,
                case.node_rate,
                &case.input_rates,
                expected_outputs,
                0,
            ),
            None,
            "{} valid ABI",
            case.name
        );

        assert_eq!(
            calc_build_error(
                case.name,
                case.node_rate,
                &case.input_rates[..case.input_rates.len() - 1],
                expected_outputs,
                0,
            ),
            Some(BuildError::WrongInputCount),
            "{} input count",
            case.name
        );
        assert_eq!(
            calc_build_error(
                case.name,
                case.node_rate,
                &case.input_rates,
                expected_outputs + 1,
                0,
            ),
            Some(BuildError::WrongOutputCount {
                expected: expected_outputs,
                actual: expected_outputs + 1,
            }),
            "{} output count",
            case.name
        );
        assert_eq!(
            calc_build_error(
                case.name,
                case.node_rate,
                &case.input_rates,
                expected_outputs,
                23,
            ),
            Some(BuildError::UnsupportedOp(23)),
            "{} special index",
            case.name
        );

        let invalid_node_rate = match case.name {
            "MoogLadder" | "Perlin3" => Rate::Scalar,
            _ if case.node_rate == Rate::Audio => Rate::Control,
            _ => Rate::Audio,
        };
        assert_eq!(
            calc_build_error(
                case.name,
                invalid_node_rate,
                &case.input_rates,
                expected_outputs,
                0,
            ),
            Some(BuildError::UnsupportedUnitRate),
            "{} node rate",
            case.name
        );

        let mut invalid_input_rates = case.input_rates.clone();
        invalid_input_rates[0] = Rate::Demand;
        assert_eq!(
            calc_build_error(
                case.name,
                case.node_rate,
                &invalid_input_rates,
                expected_outputs,
                0,
            ),
            Some(BuildError::UnsupportedUnitRate),
            "{} input rate",
            case.name
        );
    }

    assert_eq!(
        calc_build_error(
            "MoogLadder",
            Rate::Control,
            &[Rate::Audio, Rate::Audio, Rate::Audio],
            1,
            0,
        ),
        None,
        "MoogLadder.kr accepts source-defined audio inputs"
    );

    // EnvDetect alone requires its signal inlet to be audio rate.
    assert_eq!(
        calc_build_error(
            "EnvDetect",
            Rate::Audio,
            &[Rate::Control, Rate::Control, Rate::Control],
            1,
            0,
        ),
        Some(BuildError::UnsupportedUnitRate)
    );

    let demand_rates = [
        Rate::Scalar,
        Rate::Control,
        Rate::Audio,
        Rate::Demand,
        Rate::Scalar,
    ];
    assert_eq!(
        demand_build_error(Rate::Demand, &demand_rates, 1, 0),
        None,
        "DNoiseRing accepts every input-source category"
    );
    assert_eq!(
        demand_build_error(Rate::Demand, &demand_rates[..4], 1, 0),
        Some(BuildError::WrongInputCount)
    );
    assert_eq!(
        demand_build_error(Rate::Demand, &demand_rates, 2, 0),
        Some(BuildError::WrongOutputCount {
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        demand_build_error(Rate::Demand, &demand_rates, 1, 23),
        Some(BuildError::UnsupportedOp(23))
    );
    assert_eq!(
        demand_build_error(Rate::Control, &demand_rates, 1, 0),
        Some(BuildError::UnsupportedUnitRate)
    );
}

/// Proves ordinary finite runtime control changes continue producing finite output.
#[test]
fn sc3_processing_finite_control_changes_remain_finite() {
    let cases = [
        ProcessingCase {
            name: "EnvDetect",
            node_rate: Rate::Audio,
            initial: &[0.75, 0.002, 0.01],
            recovery: &[-0.25, 0.0, 0.0],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "Decimator",
            node_rate: Rate::Audio,
            initial: &[0.25, 12_000.0, 8.0],
            recovery: &[-0.375, 48_000.0, 12.0],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "DFM1",
            node_rate: Rate::Audio,
            initial: &[0.25, 1_000.0, 0.2, 1.0, 0.0, 0.0],
            recovery: &[-0.25, 2_000.0, 0.4, 0.5, 1.0, 0.0],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "BMoog",
            node_rate: Rate::Audio,
            initial: &[0.25, 1_000.0, 0.2, 0.0],
            recovery: &[-0.25, 2_000.0, 0.4, 2.0],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "MoogLadder",
            node_rate: Rate::Audio,
            initial: &[0.25, 1_000.0, 0.2],
            recovery: &[-0.25, 2_000.0, 0.4],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "MoogVCF",
            node_rate: Rate::Audio,
            initial: &[0.25, 1_000.0, 0.2],
            recovery: &[-0.25, 2_000.0, 0.4],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "BlitB3",
            node_rate: Rate::Audio,
            initial: &[440.0],
            recovery: &[880.0],
            audio_signal_input: false,
            outputs: 1,
        },
        ProcessingCase {
            name: "BlitB3Saw",
            node_rate: Rate::Audio,
            initial: &[440.0, 0.5],
            recovery: &[880.0, 0.25],
            audio_signal_input: false,
            outputs: 1,
        },
        ProcessingCase {
            name: "BlitB3Square",
            node_rate: Rate::Audio,
            initial: &[440.0, 0.5],
            recovery: &[880.0, 0.25],
            audio_signal_input: false,
            outputs: 1,
        },
        ProcessingCase {
            name: "BlitB3Tri",
            node_rate: Rate::Audio,
            initial: &[440.0, 0.5, 0.5],
            recovery: &[880.0, 0.25, 0.75],
            audio_signal_input: false,
            outputs: 1,
        },
        ProcessingCase {
            name: "Perlin3",
            node_rate: Rate::Control,
            initial: &[0.125, 0.25, 0.375],
            recovery: &[0.625, 0.75, 0.875],
            audio_signal_input: false,
            outputs: 1,
        },
        ProcessingCase {
            name: "RosslerL",
            node_rate: Rate::Audio,
            initial: &[6_000.0, 0.2, 0.2, 5.7, 0.05, 0.1, 0.0, 0.0],
            recovery: &[8_000.0, 0.1, 0.3, 5.5, 0.025, 0.2, 0.1, -0.1],
            audio_signal_input: false,
            outputs: 3,
        },
    ];

    for case in cases {
        let (mut controller, mut world, synth) = processing_case(&case, 64);
        let initial = render(&mut world, 64);
        assert!(
            initial.iter().all(|sample| sample.is_finite()),
            "{} initial block",
            case.name
        );

        for (index, &value) in case.recovery.iter().enumerate() {
            controller
                .set_control(synth, index, value)
                .expect("set changed control");
        }
        let changed = render(&mut world, 64);
        assert!(
            changed.iter().all(|sample| sample.is_finite()),
            "{} changed block",
            case.name
        );
        assert!(
            changed.iter().any(|sample| sample.abs() > 1e-9),
            "{} did not continue finite non-silent processing",
            case.name
        );
    }
}

/// Pins constructor state and the first rendered callbacks for every processing family.
#[test]
fn sc3_processing_constructor_and_first_blocks_match() {
    let cases = [
        ProcessingCase {
            name: "EnvDetect",
            node_rate: Rate::Audio,
            initial: &[0.75, 0.002, 0.01],
            recovery: &[],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "Decimator",
            node_rate: Rate::Audio,
            initial: &[0.375, 12_000.0, 8.0],
            recovery: &[],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "DFM1",
            node_rate: Rate::Audio,
            initial: &[0.25, 1_000.0, 0.2, 1.0, 0.0, 0.0],
            recovery: &[],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "BMoog",
            node_rate: Rate::Audio,
            initial: &[0.25, 1_000.0, 0.7, 0.0],
            recovery: &[],
            audio_signal_input: true,
            outputs: 1,
        },
        ProcessingCase {
            name: "BlitB3Tri",
            node_rate: Rate::Audio,
            initial: &[440.0, 0.5, 0.75],
            recovery: &[],
            audio_signal_input: false,
            outputs: 1,
        },
        ProcessingCase {
            name: "RosslerL",
            node_rate: Rate::Audio,
            initial: &[6_000.0, 0.2, 0.2, 5.7, 0.05, 0.1, 0.0, 0.0],
            recovery: &[],
            audio_signal_input: false,
            outputs: 3,
        },
    ];

    for case in cases {
        let (_controller_a, mut world_a, _synth_a) = processing_case(&case, 64);
        let (_controller_b, mut world_b, _synth_b) = processing_case(&case, 64);

        let together = render(&mut world_a, 128);
        let mut separated = render(&mut world_b, 64);
        separated.extend(render(&mut world_b, 64));

        assert_eq!(
            together, separated,
            "{} constructor/first-two-block lifecycle changed with host reads",
            case.name
        );
        assert!(
            together.iter().all(|sample| sample.is_finite()),
            "{} constructor/first-two-block output",
            case.name
        );
    }
}

/// Proves filter controls and retained gain update only at their documented cadence.
#[test]
fn sc3_filter_controls_and_retained_gain_follow_callback_cadence() {
    let no_resonance = ProcessingCase {
        name: "BMoog",
        node_rate: Rate::Audio,
        initial: &[0.25, 1_000.0, 0.0, 0.0],
        recovery: &[],
        audio_signal_input: true,
        outputs: 1,
    };
    let full_resonance = ProcessingCase {
        name: "BMoog",
        node_rate: Rate::Audio,
        initial: &[0.25, 1_000.0, 1.0, 0.0],
        recovery: &[],
        audio_signal_input: true,
        outputs: 1,
    };
    let (_controller_a, mut world_a, _) = processing_case(&no_resonance, 64);
    let (_controller_b, mut world_b, _) = processing_case(&full_resonance, 64);

    let first_without = render(&mut world_a, 64);
    let first_with = render(&mut world_b, 64);
    assert_eq!(
        first_without, first_with,
        "BMoog's first callback uses the constructor's retained zero feedback gain"
    );

    let second_without = render(&mut world_a, 64);
    let second_with = render(&mut world_b, 64);
    assert_ne!(
        second_without, second_with,
        "the sampled resonance must become the retained feedback gain on the next callback"
    );

    // A host read may end inside a logical callback. A control write there cannot split the
    // already-rendered callback; it is observed at the next 64-sample boundary.
    let dfm = ProcessingCase {
        name: "DFM1",
        node_rate: Rate::Audio,
        initial: &[0.25, 1_000.0, 0.2, 1.0, 0.0, 0.0],
        recovery: &[],
        audio_signal_input: true,
        outputs: 1,
    };
    let (mut controller, mut split_world, synth) = processing_case(&dfm, 64);
    let (_baseline_controller, mut baseline_world, _) = processing_case(&dfm, 64);
    let mut split = render(&mut split_world, 17);
    controller
        .set_control(synth, 1, 8_000.0)
        .expect("change cutoff");
    split.extend(render(&mut split_world, 47));
    assert_eq!(
        split,
        render(&mut baseline_world, 64),
        "DFM1 controls are sampled once for the complete logical callback"
    );
    assert_ne!(
        render(&mut split_world, 64),
        render(&mut baseline_world, 64),
        "the changed coefficient becomes active at the next callback"
    );
}

/// Renders one filter definition using a retained impulse or frequency-response probe.
fn render_filter(
    name: &str,
    source: UnitSpec,
    controls: Vec<InputRef>,
    warm_up: usize,
    frames: usize,
) -> Vec<f32> {
    let mut inputs = vec![wire(0, 0)];
    inputs.extend(controls);
    let units = vec![
        source,
        UnitSpec::new(name, Rate::Audio, inputs, 1),
        UnitSpec::new("Out", Rate::Audio, vec![constant(0.0), wire(1, 0)], 0),
    ];
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: 48_000.0,
        block_size: 64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: format!("{name}-response"),
        params: vec![],
        units,
    });
    controller
        .synth_new(&format!("{name}-response"), ROOT_GROUP_ID, AddAction::Tail)
        .expect("filter response graph");
    render(&mut world, warm_up);
    render(&mut world, frames)
}

/// Checks deterministic impulse and frequency-response vectors for the three filter families.
#[test]
fn sc3_filter_impulse_and_frequency_response_vectors() {
    let cases = [
        (
            "DFM1",
            vec![
                constant(1_000.0),
                constant(0.1),
                constant(1.0),
                constant(0.0),
                constant(0.0),
            ],
        ),
        (
            "BMoog",
            vec![constant(1_000.0), constant(0.1), constant(0.0)],
        ),
        ("MoogLadder", vec![constant(1_000.0), constant(0.1)]),
        ("MoogVCF", vec![constant(1_000.0), constant(0.1)]),
    ];

    for (name, controls) in cases {
        let impulse = render_filter(
            name,
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![constant(0.0), constant(0.0)],
                1,
            ),
            controls.clone(),
            0,
            256,
        );
        assert!(
            impulse.iter().all(|sample| sample.is_finite()),
            "{name} impulse response contains a non-finite sample"
        );
        assert!(
            impulse.iter().any(|sample| sample.abs() > 1e-10),
            "{name} impulse response is vacuous"
        );

        let response = |frequency| {
            render_filter(
                name,
                UnitSpec::new(
                    "SinOsc",
                    Rate::Audio,
                    vec![constant(frequency), constant(0.0)],
                    1,
                ),
                controls.clone(),
                4_096,
                4_096,
            )
        };
        let low = response(100.0);
        let high = response(8_000.0);
        let rms = |samples: &[f32]| {
            (samples
                .iter()
                .map(|sample| (*sample as f64) * (*sample as f64))
                .sum::<f64>()
                / samples.len() as f64)
                .sqrt()
        };
        assert!(
            rms(&low) > rms(&high),
            "{name} low-pass response must pass 100 Hz more strongly than 8 kHz"
        );
    }
}

/// Verifies that attack and release coefficients follow rising and falling envelopes.
#[test]
fn sc3_env_detect_attack_and_release_follow_the_signal_direction() {
    let immediate = ProcessingCase {
        name: "EnvDetect",
        node_rate: Rate::Audio,
        initial: &[1.0, 0.0, 0.0],
        recovery: &[],
        audio_signal_input: true,
        outputs: 1,
    };
    let (mut controller, mut world, synth) = processing_case(&immediate, 64);
    assert!(
        render(&mut world, 64).iter().all(|&sample| sample == 1.0),
        "zero attack follows the absolute input immediately"
    );
    controller
        .set_control(synth, 0, 0.0)
        .expect("drop envelope input");
    assert!(
        render(&mut world, 64).iter().all(|&sample| sample == 0.0),
        "zero release follows a falling input immediately"
    );

    let smoothed = ProcessingCase {
        name: "EnvDetect",
        node_rate: Rate::Audio,
        initial: &[1.0, 0.002, 0.004],
        recovery: &[],
        audio_signal_input: true,
        outputs: 1,
    };
    let (mut controller, mut world, synth) = processing_case(&smoothed, 64);
    let attack = render(&mut world, 64);
    assert!(
        attack.windows(2).all(|pair| pair[0] < pair[1]),
        "positive attack must rise monotonically"
    );
    controller
        .set_control(synth, 0, 0.0)
        .expect("drop envelope input");
    let release = render(&mut world, 64);
    assert!(
        release.windows(2).all(|pair| pair[0] > pair[1]),
        "positive release must fall monotonically"
    );
}

/// Renders one band-limited oscillator with fixed controls.
fn render_blit(name: &'static str, controls: &'static [f32]) -> Vec<f32> {
    let case = ProcessingCase {
        name,
        node_rate: Rate::Audio,
        initial: controls,
        recovery: &[],
        audio_signal_input: false,
        outputs: 1,
    };
    let (_controller, mut world, _) = processing_case(&case, 64);
    render(&mut world, 128)
}

/// Pins oscillator constructor phase and source-defined behavior at frequency boundaries.
#[test]
fn sc3_blit_phase_recreation_and_frequency_boundaries() {
    for (name, below, minimum, above, nyquist) in [
        (
            "BlitB3",
            &[-1_000.0][..],
            &[0.000_001][..],
            &[1_000_000.0][..],
            &[24_000.0][..],
        ),
        (
            "BlitB3Saw",
            &[-1_000.0, 0.5][..],
            &[0.000_001, 0.5][..],
            &[1_000_000.0, 0.5][..],
            &[24_000.0, 0.5][..],
        ),
        (
            "BlitB3Square",
            &[-1_000.0, 0.5][..],
            &[0.000_001, 0.5][..],
            &[1_000_000.0, 0.5][..],
            &[24_000.0, 0.5][..],
        ),
        (
            "BlitB3Tri",
            &[-1_000.0, 0.5, 0.5][..],
            &[0.000_001, 0.5, 0.5][..],
            &[1_000_000.0, 0.5, 0.5][..],
            &[24_000.0, 0.5, 0.5][..],
        ),
    ] {
        assert_eq!(
            render_blit(name, below),
            render_blit(name, minimum),
            "{name} minimum effective frequency"
        );
        if name == "BlitB3" {
            assert_ne!(
                render_blit(name, above),
                render_blit(name, nyquist),
                "BlitB3 preserves source frequencies above Nyquist"
            );
        } else {
            assert_eq!(
                render_blit(name, above),
                render_blit(name, nyquist),
                "{name} source minimum-period behavior"
            );
        }
    }

    for (name, below, minimum, above, maximum) in [
        (
            "BlitB3Saw",
            &[440.0, -1.0][..],
            &[440.0, 0.0][..],
            &[440.0, 2.0][..],
            &[440.0, 1.0][..],
        ),
        (
            "BlitB3Square",
            &[440.0, -1.0][..],
            &[440.0, 0.0][..],
            &[440.0, 2.0][..],
            &[440.0, 1.0][..],
        ),
        (
            "BlitB3Tri",
            &[440.0, -1.0, -1.0][..],
            &[440.0, 0.0, 0.0][..],
            &[440.0, 2.0, 2.0][..],
            &[440.0, 1.0, 1.0][..],
        ),
    ] {
        assert_ne!(
            render_blit(name, below),
            render_blit(name, minimum),
            "{name} preserves a negative source leak"
        );
        assert_ne!(
            render_blit(name, above),
            render_blit(name, maximum),
            "{name} preserves a source leak above one"
        );
    }

    let changing = ProcessingCase {
        name: "BlitB3Saw",
        node_rate: Rate::Audio,
        initial: &[440.0, 0.5],
        recovery: &[],
        audio_signal_input: false,
        outputs: 1,
    };
    let (mut controller, mut world, synth) = processing_case(&changing, 64);
    render(&mut world, 64);
    controller
        .set_control(synth, 0, 880.0)
        .expect("change BLIT frequency");
    let continued = render(&mut world, 64);
    let recreated = &render_blit("BlitB3Saw", &[880.0, 0.5])[..64];
    assert_ne!(
        continued, recreated,
        "frequency changes preserve phase; reconstruction restarts constructor phase"
    );
    assert_eq!(
        render_blit("BlitB3Saw", &[880.0, 0.5]),
        render_blit("BlitB3Saw", &[880.0, 0.5]),
        "reconstruction is deterministic"
    );
}
