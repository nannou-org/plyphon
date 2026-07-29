//! Differential SC 3.14.1 coverage for the compatible-source processing units.

use plyphon::{
    AddAction, Buffer, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
const DSP_FRAMES: usize = 640;
const APX_PHASE_ABS_FLOOR: f32 = 0.0016;

/// Shorthand for a constant graph input.
fn c(value: f32) -> InputRef {
    InputRef::Constant(value)
}

/// Shorthand for one SynthDef parameter input.
fn p(index: u32) -> InputRef {
    InputRef::Param(index)
}

/// Shorthand for output zero of an earlier unit.
fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

/// References one selected output of an earlier unit.
fn uo(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
}

/// Decodes a little-endian `f32` fixture.
fn fixture(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0, "fixture contains whole f32 values");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect()
}

/// Builds a `MulAdd` graph unit around one source.
fn mul_add(rate: Rate, source: u32, mul: f32, add: f32) -> UnitSpec {
    UnitSpec::new("MulAdd", rate, vec![u(source), c(mul), c(add)], 1)
}

/// Builds a three-input sum graph unit.
fn sum3(rate: Rate, a: u32, b: u32, c_unit: u32) -> UnitSpec {
    UnitSpec::new("Sum3", rate, vec![u(a), u(b), u(c_unit)], 1)
}

/// Builds an addition unit from two prior unit outputs.
fn add(rate: Rate, left: u32, right: u32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate,
        inputs: vec![u(left), u(right)],
        num_outputs: 1,
        special_index: 0,
    }
}

/// Builds a binary-operator unit with an explicit specialization index.
fn binary(rate: Rate, left: InputRef, right: InputRef, special_index: i16) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate,
        inputs: vec![left, right],
        num_outputs: 1,
        special_index,
    }
}

/// Plyphon's existing zero-triggered `Phasor` starts at zero, while scsynth starts at `start`.
/// An initial one-shot reset supplies the same source stream without changing the timing unit.
fn sc_started_phasor(step: f32, start: f32, end: f32) -> UnitSpec {
    UnitSpec::new(
        "Phasor",
        Rate::Audio,
        vec![c(1.0), c(step), c(start), c(end), c(start - step)],
        1,
    )
}

/// Appends one hardware-output unit for the selected unit outputs.
fn append_output(units: &mut Vec<UnitSpec>, channels: &[(u32, u32)]) {
    let mut inputs = vec![c(0.0)];
    inputs.extend(channels.iter().map(|&(unit, output)| uo(unit, output)));
    units.push(UnitSpec::new("Out", Rate::Audio, inputs, 0));
}

/// Renders one interleaved engine segment.
fn render_segment(world: &mut World, frames: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; frames * channels];
    world.fill(&mut output, channels);
    output
}

/// Render a definition on the capture file's audible timeline.
///
/// The NRT server writes the block following time zero first, so score events appear one block
/// earlier in the retained raw vector, and the score's final `n_free` produces a silent last block.
fn render_scheduled(
    def: SynthDef,
    channels: usize,
    frames: usize,
    events: &[(usize, &[(usize, f32)])],
) -> Vec<f32> {
    render_scheduled_with_buffers(def, channels, frames, &[], events)
}

/// Renders a scheduled graph after installing explicit global buffers.
fn render_scheduled_with_buffers(
    def: SynthDef,
    channels: usize,
    frames: usize,
    sources: &[(usize, &[u8])],
    events: &[(usize, &[(usize, f32)])],
) -> Vec<f32> {
    let name = def.name.clone();
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: channels,
        input_channels: 0,
        ..Options::default()
    });
    for &(index, bytes) in sources {
        controller
            .buffer_set(
                index,
                Box::new(Buffer::from_interleaved(fixture(bytes), 1, SR)),
            )
            .expect("install deterministic oracle source");
    }
    controller.add_synthdef(def);
    let synth = controller
        .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail)
        .expect("oracle comparison graph");

    let mut output = Vec::with_capacity(frames * channels);
    let mut cursor = 0;
    for &(frame, writes) in events {
        assert!(frame >= cursor && frame <= frames);
        output.extend(render_segment(&mut world, frame - cursor, channels));
        for &(index, value) in writes {
            controller
                .set_control(synth, index, value)
                .expect("scheduled control");
        }
        cursor = frame;
    }
    let live_end = frames - BLOCK;
    output.extend(render_segment(&mut world, live_end - cursor, channels));
    controller.free(synth).expect("free oracle graph");
    output.extend(render_segment(&mut world, BLOCK, channels));
    output
}

/// Compares a finite, non-vacuous render with a retained DSP oracle.
fn assert_dsp_oracle(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} vector length differs"
    );
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-4 + 1e-3 * expected.abs();
        let ratio = (actual - expected).abs() / tolerance;
        if !actual.is_finite() || !expected.is_finite() || ratio > worst.3 {
            worst = (index, actual, expected, ratio);
        }
    }
    assert!(
        worst.1.is_finite() && worst.2.is_finite() && worst.3 <= 1.0,
        "{label} worst sample {}: expected {}, got {}, tolerance ratio {}",
        worst.0,
        worst.2,
        worst.1,
        worst.3
    );
}

/// Compares decoded control values after reproducing the retained `K2A` interpolation.
fn assert_k2a_decoded_block_oracle(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    channels: usize,
    block_ends: &[usize],
) {
    let mut actual_controls = Vec::with_capacity(block_ends.len() * channels);
    let mut expected_controls = Vec::with_capacity(block_ends.len() * channels);
    for &end in block_ends {
        let start = end + 1 - BLOCK;
        for channel in 0..channels {
            actual_controls.push(actual[end * channels + channel]);
            let block_start = expected[start * channels + channel];
            let block_end = expected[end * channels + channel];
            expected_controls.push(block_start + (block_end - block_start) * (64.0 / 63.0));
        }
    }
    assert_dsp_oracle(label, &actual_controls, &expected_controls);
}

/// Matches the retained envelope-follower capture.
#[test]
fn env_detect_matches_sc314_compatible_source_vector() {
    let mut units = vec![
        UnitSpec::new("SinOsc", Rate::Audio, vec![c(337.0), c(0.3)], 1),
        mul_add(Rate::Audio, 0, 0.55, 0.0),
        UnitSpec::new("LFSaw", Rate::Audio, vec![c(73.0), c(0.1)], 1),
        mul_add(Rate::Audio, 2, 0.25, 0.0),
        sc_started_phasor(0.004_375, -0.2, 0.2),
        sum3(Rate::Audio, 1, 3, 4),
        UnitSpec::new("EnvDetect", Rate::Audio, vec![u(5), p(0), p(1)], 1),
    ];
    append_output(&mut units, &[(6, 0)]);
    let actual = render_scheduled(
        SynthDef {
            name: "oracle-env-detect".to_string(),
            params: vec![
                Param::control("attack", 100.0),
                Param::control("release", 0.0),
            ],
            units,
        },
        1,
        DSP_FRAMES,
        &[
            (64, &[(0, 8.0), (1, 2.0)]),
            (192, &[(0, 0.0), (1, 12.0)]),
            (320, &[(0, 250.0), (1, 50.0)]),
            (448, &[(0, 32.0), (1, 4.0)]),
        ],
    );
    assert_dsp_oracle(
        "EnvDetect",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/env_detect.f32")),
    );
}

/// Matches the retained nonlinear-filter capture, including deterministic noise.
#[test]
fn dfm1_outputs_match_sc314_compatible_source_vector() {
    let mut units = vec![
        UnitSpec::new(
            "Phasor",
            Rate::Audio,
            vec![c(0.0), c(1.0), c(0.0), c(DSP_FRAMES as f32), c(0.0)],
            1,
        ),
        UnitSpec::new("BufRd", Rate::Audio, vec![c(10.0), u(0), c(0.0), c(1.0)], 1),
        UnitSpec::new(
            "DFM1",
            Rate::Audio,
            vec![u(1), p(0), p(1), p(2), c(0.0), c(0.0)],
            1,
        ),
        UnitSpec::new(
            "DFM1",
            Rate::Audio,
            vec![u(1), p(0), p(1), p(2), c(1.0), c(0.0)],
            1,
        ),
    ];
    append_output(&mut units, &[(2, 0), (3, 0)]);
    let actual = render_scheduled_with_buffers(
        SynthDef {
            name: "oracle-dfm1".to_string(),
            params: vec![
                Param::control("freq", 1_000.0),
                Param::control("res", 0.05),
                Param::control("inputGain", 1.5),
            ],
            units,
        },
        2,
        DSP_FRAMES,
        &[(
            10,
            include_bytes!("fixtures/sc3_processing/dfm1_source.f32"),
        )],
        &[
            (64, &[(0, 1_800.0), (1, 0.1), (2, 1.75)]),
            (192, &[(0, 3_200.0), (1, 0.15), (2, 1.25)]),
            (320, &[(0, 700.0), (1, 0.08), (2, 2.0)]),
            (448, &[(0, 4_400.0), (1, 0.2), (2, 1.0)]),
        ],
    );
    assert_dsp_oracle(
        "DFM1",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/dfm1.f32")),
    );
}

/// Returns the shared callback-boundary control schedule for Moog filter captures.
fn moog_events() -> [(usize, &'static [(usize, f32)]); 4] {
    [
        (64, &[(0, 1_600.0), (1, 0.55)]),
        (192, &[(0, 7_200.0), (1, 0.85)]),
        (320, &[(0, 220.0), (1, 0.1)]),
        (448, &[(0, 3_500.0), (1, 0.7)]),
    ]
}

/// Matches the audio-rate Moog ladder capture.
#[test]
fn moog_ladder_ar_matches_sc314_compatible_source_vector() {
    let mut units = vec![
        UnitSpec::new("LFSaw", Rate::Audio, vec![c(131.0), c(0.2)], 1),
        mul_add(Rate::Audio, 0, 0.3, 0.0),
        UnitSpec::new("SinOsc", Rate::Audio, vec![c(389.0), c(0.4)], 1),
        mul_add(Rate::Audio, 2, 0.2, 0.0),
        add(Rate::Audio, 1, 3),
        UnitSpec::new("SinOsc", Rate::Audio, vec![c(17.0), c(0.1)], 1),
        mul_add(Rate::Audio, 5, 3_125.0, 3_375.0),
        UnitSpec::new("LFTri", Rate::Audio, vec![c(11.0), c(0.3)], 1),
        mul_add(Rate::Audio, 7, 0.4, 0.5),
        UnitSpec::new("MoogLadder", Rate::Audio, vec![u(4), u(6), u(8)], 1),
        UnitSpec::new("MoogLadder", Rate::Audio, vec![u(4), p(0), p(1)], 1),
    ];
    append_output(&mut units, &[(9, 0), (10, 0)]);
    let actual = render_scheduled(
        SynthDef {
            name: "oracle-moog-ladder-ar".to_string(),
            params: vec![Param::control("cutoff", 440.0), Param::control("res", 0.2)],
            units,
        },
        2,
        DSP_FRAMES,
        &moog_events(),
    );
    assert_dsp_oracle(
        "MoogLadder.ar",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/moog_ladder_ar.f32")),
    );
}

/// Matches the control-rate Moog ladder capture.
#[test]
fn moog_ladder_kr_matches_sc314_compatible_source_vector() {
    let mut units = vec![
        UnitSpec::new("MoogLadder", Rate::Control, vec![p(0), p(1), p(2)], 1),
        UnitSpec::new("MoogLadder", Rate::Control, vec![p(3), p(4), p(5)], 1),
        UnitSpec::new("DC", Rate::Audio, vec![u(0)], 1),
        UnitSpec::new("DC", Rate::Audio, vec![u(1)], 1),
    ];
    append_output(&mut units, &[(2, 0), (3, 0)]);
    let actual = render_scheduled(
        SynthDef {
            name: "oracle-moog-ladder-kr".to_string(),
            params: vec![
                Param::control("source_a", 0.45),
                Param::control("cutoff_a", 120.0),
                Param::control("res_a", 0.1),
                Param::control("source_b", 0.3),
                Param::control("cutoff_b", 180.0),
                Param::control("res_b", 0.2),
            ],
            units,
        },
        2,
        DSP_FRAMES,
        &[
            (
                64,
                &[
                    (0, 0.48),
                    (1, 160.0),
                    (2, 0.15),
                    (3, 0.27),
                    (4, 220.0),
                    (5, 0.24),
                ],
            ),
            (
                192,
                &[
                    (0, 0.44),
                    (1, 240.0),
                    (2, 0.2),
                    (3, 0.33),
                    (4, 300.0),
                    (5, 0.28),
                ],
            ),
            (
                320,
                &[
                    (0, 0.5),
                    (1, 80.0),
                    (2, 0.12),
                    (3, 0.29),
                    (4, 140.0),
                    (5, 0.14),
                ],
            ),
            (
                448,
                &[
                    (0, 0.46),
                    (1, 280.0),
                    (2, 0.25),
                    (3, 0.35),
                    (4, 340.0),
                    (5, 0.32),
                ],
            ),
        ],
    );
    let expected = fixture(include_bytes!("fixtures/sc3_processing/moog_ladder_kr.f32"));
    assert_k2a_decoded_block_oracle(
        "MoogLadder.kr",
        &actual,
        &expected,
        2,
        &[63, 127, 191, 255, 319, 383, 447, 511, 575],
    );
}

/// Matches every retained MoogVCF input-rate specialization.
#[test]
fn moog_vcf_all_input_rate_specializations_match_sc314() {
    let mut units = vec![
        UnitSpec::new("LFSaw", Rate::Audio, vec![c(127.0), c(0.2)], 1),
        mul_add(Rate::Audio, 0, 0.32, 0.0),
        UnitSpec::new("SinOsc", Rate::Audio, vec![c(383.0), c(0.4)], 1),
        mul_add(Rate::Audio, 2, 0.18, 0.0),
        add(Rate::Audio, 1, 3),
        UnitSpec::new("SinOsc", Rate::Audio, vec![c(19.0), c(0.1)], 1),
        mul_add(Rate::Audio, 5, 2_990.0, 3_210.0),
        UnitSpec::new("LFTri", Rate::Audio, vec![c(13.0), c(0.3)], 1),
        mul_add(Rate::Audio, 7, 0.4, 0.5),
        UnitSpec::new("MoogVCF", Rate::Audio, vec![u(4), p(0), p(1)], 1),
        UnitSpec::new("MoogVCF", Rate::Audio, vec![u(4), u(6), p(1)], 1),
        UnitSpec::new("MoogVCF", Rate::Audio, vec![u(4), p(0), u(8)], 1),
        UnitSpec::new("MoogVCF", Rate::Audio, vec![u(4), u(6), u(8)], 1),
    ];
    append_output(&mut units, &[(9, 0), (10, 0), (11, 0), (12, 0)]);
    let actual = render_scheduled(
        SynthDef {
            name: "oracle-moog-vcf".to_string(),
            params: vec![Param::control("cutoff", 440.0), Param::control("res", 0.2)],
            units,
        },
        4,
        DSP_FRAMES,
        &[
            (64, &[(0, 1_400.0), (1, 0.5)]),
            (192, &[(0, 6_800.0), (1, 0.85)]),
            (320, &[(0, 180.0), (1, 0.1)]),
            (448, &[(0, 3_200.0), (1, 0.7)]),
        ],
    );
    assert_dsp_oracle(
        "MoogVCF",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/moog_vcf.f32")),
    );
}

/// Matches all band-limited oscillator captures across frequency boundaries.
#[test]
fn all_blit_b3_outputs_match_sc314_across_boundaries() {
    let mut units = vec![
        UnitSpec::new("BlitB3", Rate::Audio, vec![p(0)], 1),
        UnitSpec::new("BlitB3Saw", Rate::Audio, vec![p(0), p(1)], 1),
        UnitSpec::new("BlitB3Square", Rate::Audio, vec![p(0), p(1)], 1),
        UnitSpec::new("BlitB3Tri", Rate::Audio, vec![p(0), p(1), p(2)], 1),
    ];
    append_output(&mut units, &[(0, 0), (1, 0), (2, 0), (3, 0)]);
    let actual = render_scheduled(
        SynthDef {
            name: "oracle-blit-b3".to_string(),
            params: vec![
                Param::control("freq", 440.0),
                Param::control("leak", 0.99),
                Param::control("leak2", 0.97),
            ],
            units,
        },
        4,
        DSP_FRAMES,
        &[
            (64, &[(0, 0.0)]),
            (192, &[(0, -220.0)]),
            (320, &[(0, 24_000.0)]),
            (448, &[(0, 880.0), (1, 0.93), (2, 0.89)]),
        ],
    );
    assert_dsp_oracle(
        "BlitB3 family",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/blit_b3.f32")),
    );
}

/// Builds a one-repeat demand sequence from constant values.
fn dseq(values: &[f32]) -> UnitSpec {
    let mut inputs = vec![c(f32::INFINITY)];
    inputs.extend(values.iter().copied().map(c));
    UnitSpec::new("Dseq", Rate::Demand, inputs, 1)
}

/// Matches deterministic demand-ring lanes and shared random draw order exactly.
#[test]
fn dnoise_ring_deterministic_lanes_match_sc314_exactly() {
    let mut units = vec![
        UnitSpec::new("Impulse", Rate::Audio, vec![c(750.0), c(0.0)], 1),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![c(0.0), c(0.5), c(1.0), c(4.0), c(11.0)],
            1,
        ),
        UnitSpec::new("Demand", Rate::Audio, vec![u(0), c(0.0), u(1)], 1),
        dseq(&[0.0, 1.0]),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![c(1.0), u(3), c(1.0), c(4.0), c(11.0)],
            1,
        ),
        UnitSpec::new("Demand", Rate::Audio, vec![u(0), c(0.0), u(4)], 1),
        dseq(&[0.0, 1.0, 1.0, 0.0]),
        dseq(&[1.0, 0.0, 1.0, 0.0]),
        dseq(&[1.0, 2.0, 3.0, 0.0]),
        UnitSpec::new(
            "DNoiseRing",
            Rate::Demand,
            vec![u(6), u(7), u(8), c(4.0), c(13.0)],
            1,
        ),
        UnitSpec::new("Demand", Rate::Audio, vec![u(0), c(0.0), u(9)], 1),
    ];
    append_output(&mut units, &[(2, 0), (5, 0), (10, 0)]);

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 3,
        input_channels: 0,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "oracle-dnoise-deterministic".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new(
            "oracle-dnoise-deterministic",
            ROOT_GROUP_ID,
            AddAction::Tail,
        )
        .expect("DNoiseRing oracle graph");
    let actual = render_segment(&mut world, 1_024, 3);
    let expected = fixture(include_bytes!("fixtures/sc3_processing/dnoise_ring.f32"));
    for frame in 0..1_024 {
        for (actual_channel, fixture_channel) in [(0, 3), (1, 5), (2, 7)] {
            assert_eq!(
                actual[frame * 3 + actual_channel].to_bits(),
                expected[frame * 10 + fixture_channel].to_bits(),
                "DNoiseRing deterministic lane {fixture_channel}, frame {frame}"
            );
        }
    }
}

/// Builds a no-interpolation reader for one packed-spectrum slot.
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
) -> InputRef {
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
    let ramped = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(held)], 1));
    u(ramped)
}

/// Appends decoded packed-spectrum channels to one graph.
fn append_decoded_spectrum(
    units: &mut Vec<UnitSpec>,
    buffer_unit: u32,
    ready_unit: u32,
    fft_size: usize,
) -> Vec<InputRef> {
    let mut decoded = Vec::with_capacity(fft_size + 2);
    decoded.push(decoded_spectrum_value(units, buffer_unit, ready_unit, 0));
    decoded.push(c(0.0));
    for slot in 2..fft_size {
        decoded.push(decoded_spectrum_value(units, buffer_unit, ready_unit, slot));
    }
    decoded.push(decoded_spectrum_value(units, buffer_unit, ready_unit, 1));
    decoded.push(c(0.0));
    decoded
}

/// Renders a phase-vocoder graph across an explicit callback control schedule.
fn render_pv_oracle(
    name: &str,
    units: Vec<UnitSpec>,
    sources: &[(usize, &[u8])],
    schedule: &[f32],
) -> Vec<f32> {
    const CHANNELS: usize = 135;
    const FRAMES: usize = 1_536;

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: CHANNELS,
        input_channels: 0,
        ..Options::default()
    });
    for &(index, bytes) in sources {
        controller
            .buffer_set(
                index,
                Box::new(Buffer::from_interleaved(fixture(bytes), 1, SR)),
            )
            .expect("install deterministic PV source");
    }
    controller.add_synthdef(SynthDef {
        name: name.to_string(),
        params: vec![Param::control("amount", schedule[0])],
        units,
    });
    let synth = controller
        .synth_new(name, ROOT_GROUP_ID, AddAction::Tail)
        .expect("PV oracle graph");

    let mut output = Vec::with_capacity(FRAMES * CHANNELS);
    for block in 0..FRAMES / BLOCK {
        if block % 2 == 1 {
            let callback = ((block - 1) / 2).min(schedule.len() - 1);
            controller
                .set_control(synth, 0, schedule[callback])
                .expect("PV callback control");
        }
        output.extend(render_segment(&mut world, BLOCK, CHANNELS));
    }
    output
}

/// Expresses an actual phase on the branch nearest the expected phase.
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

/// Compares resynthesis, tokens, controls, and decoded bins with a retained PV oracle.
fn assert_pv_oracle(label: &str, actual: &[f32], expected: &[f32]) {
    const CHANNELS: usize = 135;
    const READY_EVENTS: [usize; 11] = [65, 193, 321, 449, 577, 705, 833, 961, 1_089, 1_217, 1_345];
    const DECODED_FRAMES: [usize; 11] = [
        128, 256, 384, 512, 640, 768, 896, 1_024, 1_152, 1_280, 1_408,
    ];

    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} vector length differs"
    );
    let mut comparable_actual = Vec::new();
    let mut comparable_expected = Vec::new();

    // Each ready callback contributes one complete, non-vacuous IFFT hop.
    for frame in READY_EVENTS {
        for window_frame in frame..frame + BLOCK {
            comparable_actual.push(actual[window_frame * CHANNELS]);
            comparable_expected.push(expected[window_frame * CHANNELS]);
        }
    }

    // Compare the stable control-rate token, ready state, amount, and every decoded bin.
    for frame in DECODED_FRAMES {
        comparable_actual.extend_from_slice(&actual[frame * CHANNELS + 1..frame * CHANNELS + 4]);
        comparable_expected
            .extend_from_slice(&expected[frame * CHANNELS + 1..frame * CHANNELS + 4]);
        for decoded in 0..130 {
            let actual_value = actual[frame * CHANNELS + 5 + decoded];
            let expected_value = expected[frame * CHANNELS + 5 + decoded];
            if decoded % 2 == 1 && decoded != 1 && decoded != 129 {
                let adjusted = nearest_phase(actual_value, expected_value);
                let error = (adjusted - expected_value).abs();
                let tolerance = APX_PHASE_ABS_FLOOR.max(1.0e-4 + 1.0e-3 * expected_value.abs());
                assert!(
                    error <= tolerance,
                    "{label} frame {frame} decoded phase {decoded}: expected {expected_value}, got {actual_value}, circular error {error}, tolerance {tolerance}"
                );
            } else {
                comparable_actual.push(actual_value);
                comparable_expected.push(expected_value);
            }
        }
    }

    assert!(
        comparable_expected
            .iter()
            .filter(|&&value| value != 0.0)
            .count()
            > 128,
        "{label} fixture must exercise non-zero spectral state"
    );
    assert_dsp_oracle(label, &comparable_actual, &comparable_expected);
}

/// Builds the deterministic magnitude-smoothing oracle graph.
fn pv_mag_smooth_graph() -> Vec<UnitSpec> {
    const FRAMES: usize = 1_536;
    const FFT_SIZE: usize = 128;

    let mut units = vec![
        sc_started_phasor(1.0, 0.0, FRAMES as f32),
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
        UnitSpec::new("PV_MagSmooth", Rate::Control, vec![u(3), p(0)], 1),
        binary(Rate::Control, u(4), c(-1.0), 9),
    ];

    let token = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(4)], 1));
    let ready = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(5)], 1));
    let amount = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![p(0)], 1));
    let decoded = append_decoded_spectrum(&mut units, 2, 5, FFT_SIZE);
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
            .chain([u(ifft), u(token), u(ready), u(amount), c(0.0)])
            .chain(decoded)
            .collect(),
        0,
    ));
    units
}

/// Matches decoded spectra and resynthesis for magnitude smoothing.
#[test]
fn pv_mag_smooth_matches_sc314_decoded_spectrum_and_resynthesis() {
    let actual = render_pv_oracle(
        "oracle-pv-mag-smooth",
        pv_mag_smooth_graph(),
        &[(
            10,
            include_bytes!("fixtures/sc3_processing/pv_source_a.f32"),
        )],
        &[0.1, 0.1, 0.25, 0.75, 1.0, 0.0, 0.5, 0.2],
    );
    assert_pv_oracle(
        "PV_MagSmooth",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/pv_mag_smooth.f32")),
    );
}

/// Builds the deterministic two-source spectral-morph oracle graph.
fn pv_morph_graph() -> Vec<UnitSpec> {
    const FRAMES: usize = 1_536;
    const FFT_SIZE: usize = 128;

    let mut units = vec![
        sc_started_phasor(1.0, 0.0, FRAMES as f32),
        UnitSpec::new("BufRd", Rate::Audio, vec![c(10.0), u(0), c(0.0), c(1.0)], 1),
        UnitSpec::new("BufRd", Rate::Audio, vec![c(11.0), u(0), c(0.0), c(1.0)], 1),
        UnitSpec::new(
            "LocalBuf",
            Rate::Scalar,
            vec![c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        UnitSpec::new(
            "FFT",
            Rate::Control,
            vec![u(3), u(1), c(1.0), c(0.0), c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        UnitSpec::new(
            "LocalBuf",
            Rate::Scalar,
            vec![c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        UnitSpec::new(
            "FFT",
            Rate::Control,
            vec![u(5), u(2), c(1.0), c(0.0), c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        UnitSpec::new("PV_Morph", Rate::Control, vec![u(4), u(6), p(0)], 1),
        binary(Rate::Control, u(7), c(-1.0), 9),
    ];

    let token = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(7)], 1));
    let ready = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(8)], 1));
    let amount = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![p(0)], 1));
    let decoded = append_decoded_spectrum(&mut units, 3, 8, FFT_SIZE);
    let ifft = units.len() as u32;
    units.push(UnitSpec::new(
        "IFFT",
        Rate::Audio,
        vec![u(7), c(0.0), c(FFT_SIZE as f32)],
        1,
    ));

    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        core::iter::once(c(0.0))
            .chain([u(ifft), u(token), u(ready), u(amount), c(0.0)])
            .chain(decoded)
            .collect(),
        0,
    ));
    units
}

/// Matches decoded spectra and resynthesis for spectral morphing.
#[test]
fn pv_morph_matches_sc314_decoded_spectrum_and_resynthesis() {
    let actual = render_pv_oracle(
        "oracle-pv-morph",
        pv_morph_graph(),
        &[
            (
                10,
                include_bytes!("fixtures/sc3_processing/pv_source_a.f32"),
            ),
            (
                11,
                include_bytes!("fixtures/sc3_processing/pv_source_b.f32"),
            ),
        ],
        &[0.0, 0.25, 0.5, 0.75, 1.0, 0.6, 0.2, 0.9],
    );
    assert_pv_oracle(
        "PV_Morph",
        &actual,
        &fixture(include_bytes!("fixtures/sc3_processing/pv_morph.f32")),
    );
}

/// Proves raw source-phase interpolation from pinned, distinct moving-impulse spectra.
#[test]
fn pv_morph_raw_phase_interpolation_matches_sc314_for_distinct_sources() {
    const MORPH_CHANNELS: usize = 135;
    const SOURCE_PHASE_CHANNELS: usize = 126;
    const INSPECTED: [(usize, f32); 3] = [(128, 0.0), (384, 0.5), (640, 1.0)];

    pv_morph_matches_sc314_decoded_spectrum_and_resynthesis();
    let morphed = fixture(include_bytes!("fixtures/sc3_processing/pv_morph.f32"));
    let source_phases = fixture(include_bytes!(
        "fixtures/sc3_processing/pv_morph_source_phases.f32"
    ));
    let circular_distance = |actual: f32, wanted: f32| {
        let difference = actual - wanted;
        difference.sin().atan2(difference.cos()).abs()
    };

    for (frame, morph) in INSPECTED {
        for bin in 1..64 {
            let source_a = source_phases[frame * SOURCE_PHASE_CHANNELS + bin - 1];
            let source_b = source_phases[frame * SOURCE_PHASE_CHANNELS + 63 + bin - 1];
            let source_distance = circular_distance(source_a, source_b);
            assert!(
                source_distance > APX_PHASE_ABS_FLOOR,
                "frame {frame}, bin {bin}: source A/B phases are not distinct"
            );

            let wanted = (1.0 - morph) * source_a + morph * source_b;
            let actual = morphed[frame * MORPH_CHANNELS + 6 + 2 * bin];
            let error = circular_distance(actual, wanted);
            assert!(
                error <= APX_PHASE_ABS_FLOOR,
                "frame {frame}, bin {bin}, morph {morph}: expected raw phase interpolation {wanted}, got {actual}, circular error {error}"
            );
        }
    }
}
