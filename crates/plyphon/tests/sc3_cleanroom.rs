//! Black-box conformance for the five oracle-only Spec 100 units.

use plyphon::{
    AddAction, Buffer, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
const APX_PHASE_ABS_FLOOR: f32 = 0.0016;

fn c(value: f32) -> InputRef {
    InputRef::Constant(value)
}

fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

fn output(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
}

fn binary(rate: Rate, left: InputRef, right: InputRef, special_index: i16) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate,
        inputs: vec![left, right],
        num_outputs: 1,
        special_index,
    }
}

fn fixture(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect()
}

fn render_block(world: &mut World, channels: usize) -> Vec<f32> {
    let mut block = vec![0.0; BLOCK * channels];
    world.fill(&mut block, channels);
    block
}

fn assert_oracle(label: &str, actual: &[f32], expected: &[f32]) {
    assert!(
        actual.len() <= expected.len(),
        "{label} rendered {} values for an {}-value oracle",
        actual.len(),
        expected.len()
    );
    assert!(
        expected.iter().filter(|&&value| value != 0.0).count() > 16,
        "{label} reference must be non-vacuous"
    );
    assert!(
        actual.iter().all(|value| value.is_finite()),
        "{label} emitted a non-finite value"
    );

    let mut first_failure = None;
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1.0e-4 + 1.0e-3 * expected.abs();
        let error = (actual - expected).abs();
        if error > worst.3 {
            worst = (index, actual, expected, error);
        }
        if error > tolerance && first_failure.is_none() {
            first_failure = Some((index, actual, expected, tolerance));
        }
    }
    assert!(
        first_failure.is_none(),
        "{label} first mismatch {:?}; worst mismatch at {}: expected {}, got {}, abs error {}",
        first_failure,
        worst.0,
        worst.2,
        worst.1,
        worst.3
    );
}

/// Express `actual` on the phase branch nearest `expected` without changing angular distance.
fn nearest_phase(actual: f32, expected: f32) -> f32 {
    let difference = actual - expected;
    let difference = if difference > core::f32::consts::PI {
        difference - core::f32::consts::TAU
    } else if difference < -core::f32::consts::PI {
        difference + core::f32::consts::TAU
    } else {
        difference
    };
    expected + difference
}

/// Build a Plyphon `Phasor` that emits the same first value as scsynth's zero-triggered form.
///
/// Plyphon's pre-existing `Phasor` starts its accumulator at zero, while scsynth starts a
/// zero-triggered phasor at `start`. A one-shot initial reset keeps the oracle input stream
/// byte-for-byte equivalent without making this Spec 100 test depend on an unrelated timing-unit
/// correction.
fn sc_started_phasor(rate: Rate, step: f32, start: f32, end: f32) -> UnitSpec {
    UnitSpec::new(
        "Phasor",
        rate,
        vec![c(1.0), c(step), c(start), c(end), c(start - step)],
        1,
    )
}

#[test]
fn decimator_matches_sc314() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 10,
        ..Options::default()
    });
    let mut units = vec![
        sc_started_phasor(Rate::Audio, 0.03125, -1.0, 1.0),
        binary(Rate::Audio, u(0), c(-1.0), 2),
        UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
    ];
    for inputs in [
        vec![u(0), InputRef::Param(0), InputRef::Param(1)],
        vec![u(0), c(0.0), c(8.0)],
        vec![u(0), c(24_000.0), c(8.5)],
        vec![u(0), c(48_000.0), c(1.0)],
        vec![u(0), c(12_000.0), c(0.999)],
        vec![u(0), c(12_000.0), c(1.001)],
        vec![u(0), c(12_000.0), c(30.999)],
        vec![u(0), c(12_000.0), c(31.0)],
        vec![u(1), c(16_000.0), c(4.0)],
        vec![u(2), c(48_000.0), c(12.0)],
    ] {
        units.push(UnitSpec::new("Decimator", Rate::Audio, inputs, 1));
    }
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        core::iter::once(c(0.0)).chain((3..13).map(u)).collect(),
        0,
    ));
    controller.add_synthdef(SynthDef {
        name: "cleanroom-decimator".to_string(),
        params: vec![
            Param::control("changingRate", 44_100.0),
            Param::control("changingBits", 24.0),
        ],
        units,
    });
    let synth = controller
        .synth_new("cleanroom-decimator", ROOT_GROUP_ID, AddAction::Tail)
        .expect("spawn Decimator oracle graph");

    let mut actual = Vec::with_capacity(448 * 10);
    for block in 0..7 {
        if block == 3 {
            controller.set_control(synth, 0, 12_000.0).unwrap();
            controller.set_control(synth, 1, 8.5).unwrap();
        } else if block == 5 {
            controller.set_control(synth, 0, 48_000.0).unwrap();
            controller.set_control(synth, 1, 31.0).unwrap();
        }
        actual.extend(render_block(&mut world, 10));
    }
    assert_oracle(
        "Decimator",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/decimator.f32")),
    );
}

#[test]
fn bmoog_matches_sc314() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 4,
        ..Options::default()
    });
    let mut units = vec![sc_started_phasor(Rate::Audio, 0.021875, -0.8, 0.8)];
    for mode in 0..4 {
        units.push(UnitSpec::new(
            "BMoog",
            Rate::Audio,
            vec![u(0), InputRef::Param(0), InputRef::Param(1), c(mode as f32)],
            1,
        ));
    }
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(1), u(2), u(3), u(4)],
        0,
    ));
    controller.add_synthdef(SynthDef {
        name: "cleanroom-bmoog".to_string(),
        params: vec![Param::control("cutoff", 440.0), Param::control("q", 0.2)],
        units,
    });
    let synth = controller
        .synth_new("cleanroom-bmoog", ROOT_GROUP_ID, AddAction::Tail)
        .expect("spawn BMoog oracle graph");

    let mut actual = Vec::with_capacity(448 * 4);
    for block in 0..7 {
        let controls = match block {
            1 => Some((1_200.0, 0.65)),
            3 => Some((6_000.0, 0.9)),
            5 => Some((220.0, 0.1)),
            _ => None,
        };
        if let Some((cutoff, q)) = controls {
            controller.set_control(synth, 0, cutoff).unwrap();
            controller.set_control(synth, 1, q).unwrap();
        }
        actual.extend(render_block(&mut world, 4));
    }
    assert_oracle(
        "BMoog",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/bmoog.f32")),
    );
}

fn render_perlin(rate: Rate) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 3,
        ..Options::default()
    });
    let (x_step, y_step, z_step) = if rate == Rate::Audio {
        (0.0078125, 0.01171875, 0.015625)
    } else {
        (0.125, 0.1875, 0.25)
    };
    let mut units = vec![
        sc_started_phasor(rate, x_step, -1.25, 1.25),
        sc_started_phasor(rate, y_step, -0.75, 1.75),
        sc_started_phasor(rate, z_step, -1.5, 1.5),
        UnitSpec::new("MulAdd", rate, vec![u(0), c(1.0), c(0.25)], 1),
        UnitSpec::new("MulAdd", rate, vec![u(1), c(1.0), c(-0.5)], 1),
        UnitSpec::new("MulAdd", rate, vec![u(2), c(1.0), c(0.75)], 1),
        UnitSpec::new("MulAdd", rate, vec![u(0), c(-1.0), c(0.0)], 1),
        UnitSpec::new("MulAdd", rate, vec![u(1), c(0.5), c(0.0)], 1),
        UnitSpec::new("MulAdd", rate, vec![u(2), c(1.25), c(0.0)], 1),
        UnitSpec::new("Perlin3", rate, vec![u(0), u(1), u(2)], 1),
        UnitSpec::new("Perlin3", rate, vec![u(3), u(4), u(5)], 1),
        UnitSpec::new("Perlin3", rate, vec![u(6), u(7), u(8)], 1),
    ];
    let outputs = if rate == Rate::Audio {
        vec![u(9), u(10), u(11)]
    } else {
        for source in 9..12 {
            units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(source)], 1));
        }
        vec![u(12), u(13), u(14)]
    };
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        core::iter::once(c(0.0)).chain(outputs).collect(),
        0,
    ));
    controller.add_synthdef(SynthDef {
        name: "cleanroom-perlin".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("cleanroom-perlin", ROOT_GROUP_ID, AddAction::Tail)
        .expect("spawn Perlin3 oracle graph");

    let mut actual = Vec::with_capacity(448 * 3);
    for _ in 0..7 {
        actual.extend(render_block(&mut world, 3));
    }
    actual
}

#[test]
fn perlin3_ar_and_kr_match_sc314() {
    assert_oracle(
        "Perlin3.ar",
        &render_perlin(Rate::Audio),
        &fixture(include_bytes!("fixtures/sc3_processing/perlin3_ar.f32")),
    );
    assert_oracle(
        "Perlin3.kr",
        &render_perlin(Rate::Control),
        &fixture(include_bytes!("fixtures/sc3_processing/perlin3_kr.f32")),
    );
}

#[test]
fn rossler_l_matches_sc314() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 3,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "cleanroom-rossler".to_string(),
        params: vec![
            Param::control("freq", 6_000.0),
            Param::control("a", 0.2),
            Param::control("b", 0.2),
            Param::control("c", 5.7),
            Param::control("h", 0.05),
            Param::control("xi", 0.1),
            Param::control("yi", 0.0),
            Param::control("zi", 0.0),
        ],
        units: vec![
            UnitSpec::new(
                "RosslerL",
                Rate::Audio,
                (0..8).map(InputRef::Param).collect(),
                3,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![c(0.0), output(0, 0), output(0, 1), output(0, 2)],
                0,
            ),
        ],
    });
    let synth = controller
        .synth_new("cleanroom-rossler", ROOT_GROUP_ID, AddAction::Tail)
        .expect("spawn RosslerL oracle graph");

    let mut actual = Vec::with_capacity(448 * 3);
    for block in 0..7 {
        if block == 2 {
            for (index, value) in [(5, 0.35), (6, -0.2), (7, 0.15)] {
                controller.set_control(synth, index, value).unwrap();
            }
        } else if block == 4 {
            for (index, value) in [(0, 12_000.0), (1, 0.36), (2, 0.35), (3, 4.5), (4, 0.03)] {
                controller.set_control(synth, index, value).unwrap();
            }
        } else if block == 6 {
            controller.set_control(synth, 0, 48_000.0).unwrap();
        }
        actual.extend(render_block(&mut world, 3));
    }
    assert_oracle(
        "RosslerL",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/rossler_l.f32")),
    );
}

fn spectrum_reader(buffer_unit: u32, slot: usize) -> UnitSpec {
    UnitSpec::new(
        "BufRd",
        Rate::Audio,
        vec![u(buffer_unit), c(slot as f32), c(1.0), c(1.0)],
        1,
    )
}

/// Append the read/latch/ramp chain used by `Demand.kr(..., UnpackFFT(...)).asArray`.
fn decoded_spectrum_value(
    units: &mut Vec<UnitSpec>,
    buffer_unit: u32,
    ready_unit: u32,
    slot: usize,
) {
    let reader = units.len() as u32;
    units.push(spectrum_reader(buffer_unit, slot));
    let sampled = units.len() as u32;
    units.push(UnitSpec::new("A2K", Rate::Control, vec![u(reader)], 1));
    let held = units.len() as u32;
    units.push(UnitSpec::new(
        "Latch",
        Rate::Control,
        vec![u(sampled), u(ready_unit)],
        1,
    ));
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(held)], 1));
}

#[test]
fn pv_freeze_matches_sc314_decoded_spectrum_and_resynthesis() {
    const CHANNELS: usize = 134;
    const FRAMES: usize = 1_536;
    const FFT_SIZE: usize = 128;
    const READY_EVENTS: [usize; 11] = [65, 193, 321, 449, 577, 705, 833, 961, 1_089, 1_217, 1_345];
    const DECODED_FRAMES: [usize; 11] = [
        128, 256, 384, 512, 640, 768, 896, 1_024, 1_152, 1_280, 1_408,
    ];

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: CHANNELS,
        ..Options::default()
    });
    controller
        .buffer_set(
            10,
            Box::new(Buffer::from_interleaved(
                fixture(include_bytes!("fixtures/sc3_processing/pv_source_a.f32")),
                1,
                SR,
            )),
        )
        .expect("install deterministic PV oracle source");

    let mut units = vec![
        sc_started_phasor(Rate::Audio, 1.0, 0.0, FRAMES as f32),
        UnitSpec::new("BufRd", Rate::Audio, vec![c(10.0), u(0), c(0.0), c(1.0)], 1),
        UnitSpec::new(
            "LocalBuf",
            Rate::Scalar,
            vec![c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        UnitSpec::new(
            "FFT",
            Rate::Control,
            vec![u(2), u(1), c(1.0), c(0.0), c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        UnitSpec::new(
            "PV_Freeze",
            Rate::Control,
            vec![u(3), InputRef::Param(0)],
            1,
        ),
        binary(Rate::Control, u(4), c(-1.0), 9),
    ];

    let raw_token = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(4)], 1));
    let ready = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(5)], 1));
    let freeze = units.len() as u32;
    units.push(UnitSpec::new(
        "K2A",
        Rate::Audio,
        vec![InputRef::Param(0)],
        1,
    ));

    let mut decoded = Vec::with_capacity(FFT_SIZE + 2);
    decoded_spectrum_value(&mut units, 2, 5, 0);
    decoded.push(u((units.len() - 1) as u32));
    decoded.push(c(0.0));
    for slot in 2..FFT_SIZE {
        decoded_spectrum_value(&mut units, 2, 5, slot);
        decoded.push(u((units.len() - 1) as u32));
    }
    decoded_spectrum_value(&mut units, 2, 5, 1);
    decoded.push(u((units.len() - 1) as u32));
    decoded.push(c(0.0));
    let ifft = units.len() as u32;
    units.push(UnitSpec::new(
        "IFFT",
        Rate::Audio,
        vec![u(4), c(0.0), c(FFT_SIZE as f32)],
        1,
    ));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        core::iter::once(c(0.0))
            .chain(core::iter::once(u(ifft)))
            .chain([u(raw_token), u(ready), u(freeze)])
            .chain(decoded)
            .collect(),
        0,
    ));

    controller.add_synthdef(SynthDef {
        name: "cleanroom-pv-freeze".to_string(),
        params: vec![Param::control("freeze", 0.0)],
        units,
    });
    let synth = controller
        .synth_new("cleanroom-pv-freeze", ROOT_GROUP_ID, AddAction::Tail)
        .expect("spawn PV_Freeze oracle graph");

    let mut actual = Vec::with_capacity(FRAMES * CHANNELS);
    for block in 0..(FRAMES / BLOCK) {
        let freeze = match block {
            0..=6 => 0.0,
            7..=10 => 1.0,
            11..=14 => 0.0,
            _ => 1.0,
        };
        if block > 0 {
            controller.set_control(synth, 0, freeze).unwrap();
        }
        actual.extend(render_block(&mut world, CHANNELS));
    }
    let expected = fixture(include_bytes!("fixtures/sc3_processing/pv_freeze.f32"));

    let mut comparable_actual = Vec::new();
    let mut comparable_expected = Vec::new();
    for frame in READY_EVENTS {
        for window_frame in frame..frame + BLOCK {
            comparable_actual.push(actual[window_frame * CHANNELS]);
            comparable_expected.push(expected[window_frame * 135]);
        }
    }
    for frame in DECODED_FRAMES {
        comparable_actual.extend_from_slice(&actual[frame * CHANNELS + 1..frame * CHANNELS + 4]);
        comparable_expected.extend_from_slice(&expected[frame * 135 + 1..frame * 135 + 4]);
        for decoded in 0..130 {
            let actual_value = actual[frame * CHANNELS + 4 + decoded];
            let expected_value = expected[frame * 135 + 5 + decoded];
            if decoded % 2 == 1 && decoded != 1 && decoded != 129 {
                let adjusted = nearest_phase(actual_value, expected_value);
                let error = (adjusted - expected_value).abs();
                let tolerance = APX_PHASE_ABS_FLOOR.max(1.0e-4 + 1.0e-3 * expected_value.abs());
                assert!(
                    error <= tolerance,
                    "PV_Freeze frame {frame} decoded phase {decoded}: expected {expected_value}, got {actual_value}, circular error {error}, tolerance {tolerance}"
                );
            } else {
                comparable_actual.push(actual_value);
                comparable_expected.push(expected_value);
            }
        }
    }
    assert_oracle("PV_Freeze", &comparable_actual, &comparable_expected);
}

/// Proves the pinned moving-impulse fixture carries non-vacuous learned phase history.
#[test]
fn pv_freeze_phase_history_matches_sc314_across_moving_impulses() {
    const CHANNELS: usize = 135;
    const ORDINARY_BINS: core::ops::Range<usize> = 1..64;
    const PRE_FREEZE_FRAMES: [usize; 3] = [128, 256, 384];
    const FROZEN_RUNS: [(usize, &[usize]); 2] =
        [(384, &[512, 640]), (896, &[1_024, 1_152, 1_280, 1_408])];

    pv_freeze_matches_sc314_decoded_spectrum_and_resynthesis();
    let expected = fixture(include_bytes!("fixtures/sc3_processing/pv_freeze.f32"));
    let phase = |frame: usize, bin: usize| expected[frame * CHANNELS + 6 + 2 * bin];
    let circular_distance = |actual: f32, wanted: f32| {
        let difference = actual - wanted;
        difference.sin().atan2(difference.cos()).abs()
    };
    let wrap_once = |value: f32| {
        if value > core::f32::consts::PI {
            value - core::f32::consts::TAU
        } else if value < -core::f32::consts::PI {
            value + core::f32::consts::TAU
        } else {
            value
        }
    };

    let mut pre_freeze_phase_bits = Vec::new();
    for frame in PRE_FREEZE_FRAMES {
        for bin in ORDINARY_BINS {
            pre_freeze_phase_bits.push(phase(frame, bin).to_bits());
        }
    }
    pre_freeze_phase_bits.sort_unstable();
    pre_freeze_phase_bits.dedup();
    assert!(
        pre_freeze_phase_bits.len() >= 3,
        "the reference must expose at least three distinct ordinary-bin phases before freeze"
    );

    let mut learned_difference_bits = Vec::new();
    for (base_frame, frozen_frames) in FROZEN_RUNS {
        for bin in ORDINARY_BINS {
            let learned = phase(frozen_frames[0], bin) - phase(base_frame, bin);
            assert!(
                circular_distance(learned, 0.0) > APX_PHASE_ABS_FLOOR,
                "frame {}, bin {bin}: learned phase difference is vacuous",
                frozen_frames[0]
            );
            learned_difference_bits.push(learned.to_bits());

            let mut previous = phase(base_frame, bin);
            for &frame in frozen_frames {
                let wanted = wrap_once(previous + learned);
                let actual = phase(frame, bin);
                let error = circular_distance(actual, wanted);
                assert!(
                    error <= APX_PHASE_ABS_FLOOR * 2.0,
                    "frame {frame}, bin {bin}: expected wrapped phase {wanted}, got {actual}, circular error {error}"
                );
                previous = actual;
            }
        }
    }
    learned_difference_bits.sort_unstable();
    learned_difference_bits.dedup();
    assert!(
        learned_difference_bits.len() >= 4,
        "the reference must retain distinct non-zero learned phase differences"
    );
}
