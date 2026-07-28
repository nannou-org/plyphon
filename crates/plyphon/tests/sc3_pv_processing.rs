//! SC3 PV ABI, decoded-bin behavior, ownership, staged state, and invalid-frame recovery.

use plyphon::{
    AddAction, Buffer, BuildContext, BuildError, InputRef, Options, Param, ROOT_GROUP_ID, Rate,
    RateInfo, SynthDef, UnitRegistry, UnitSpec, World, engine,
};
use plyphon_dsp::buffer::SpectrumCoord;
use plyphon_unit::unit::{InputSource, pv};

const SR: f64 = 48_000.0;
const FFT_SIZE: usize = 64;

/// A mono packed-spectrum buffer with the requested coordinate tag.
fn spectrum_buffer(data: Vec<f32>, coord: SpectrumCoord) -> Buffer {
    spectrum_buffer_size(data, coord, FFT_SIZE)
}

/// Builds a packed-spectrum buffer with an explicit logical FFT size.
fn spectrum_buffer_size(mut data: Vec<f32>, coord: SpectrumCoord, fft_size: usize) -> Buffer {
    data.resize(fft_size, 0.0);
    let mut buffer = Buffer::from_interleaved(data, 1, SR);
    buffer.set_coord(coord);
    buffer
}

/// A no-interpolation `BufRd` that exposes one packed-spectrum slot as audio.
fn read_slot(buffer: f32, slot: f32) -> UnitSpec {
    UnitSpec::new(
        "BufRd",
        Rate::Audio,
        vec![
            InputRef::Constant(buffer),
            InputRef::Constant(slot),
            InputRef::Constant(1.0),
            InputRef::Constant(1.0),
        ],
        1,
    )
}

/// Render one fixed control block into `channels` interleaved channels.
fn render_block(world: &mut World, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; 64 * channels];
    world.fill(&mut output, channels);
    output
}

/// Return channel `channel` from the first frame of an interleaved render.
fn channel(output: &[f32], channel: usize) -> f32 {
    output[channel]
}

/// Invoke a registered PV constructor directly with a synthetic ABI.
fn pv_build_error(
    name: &str,
    rate: Rate,
    input_rates: &[Rate],
    outputs: usize,
    special_index: i16,
) -> Option<BuildError> {
    let registry = UnitRegistry::with_builtins();
    let def = registry.get(name).expect("PV registration");
    let input_units = vec![None; input_rates.len()];
    let input_sources = vec![InputSource::Constant(0.0); input_rates.len()];
    let audio = RateInfo::new(SR, 64);
    let control = RateInfo::new(SR / 64.0, 1);
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

/// Rejects invalid PV shapes, rates, output counts, and specialization indices.
#[test]
fn sc3_pv_units_reject_invalid_shapes_rates_and_special_index() {
    for (name, inputs) in [("PV_Freeze", 2), ("PV_MagSmooth", 2), ("PV_Morph", 3)] {
        let valid = vec![Rate::Control; inputs];
        assert_eq!(
            pv_build_error(name, Rate::Control, &valid[..inputs - 1], 1, 0),
            Some(BuildError::WrongInputCount),
            "{name} input count"
        );
        assert_eq!(
            pv_build_error(name, Rate::Control, &valid, 2, 0),
            Some(BuildError::WrongOutputCount {
                expected: 1,
                actual: 2,
            }),
            "{name} output count"
        );
        assert_eq!(
            pv_build_error(name, Rate::Control, &valid, 1, 9),
            Some(BuildError::UnsupportedOp(9)),
            "{name} special index"
        );
        assert_eq!(
            pv_build_error(name, Rate::Audio, &valid, 1, 0),
            Some(BuildError::UnsupportedUnitRate),
            "{name} node rate"
        );
        let mut audio_input = valid.clone();
        audio_input[0] = Rate::Audio;
        assert_eq!(
            pv_build_error(name, Rate::Control, &audio_input, 1, 0),
            Some(BuildError::UnsupportedUnitRate),
            "{name} buffer-token input rate"
        );
        let mut audio_modulation = valid.clone();
        *audio_modulation.last_mut().expect("modulation input") = Rate::Audio;
        assert_eq!(
            pv_build_error(name, Rate::Control, &audio_modulation, 1, 0),
            None,
            "{name} audio-rate modulation input"
        );
        assert_eq!(
            pv_build_error(name, Rate::Control, &valid, 1, 0),
            None,
            "{name} valid ABI"
        );
    }
}

/// Pins SuperCollider's approximate-polar grid while proving the existing exact path is unchanged.
#[test]
fn sc3_pv_approximate_polar_conversion_matches_pinned_core_and_is_isolated() {
    const COMPLEX_BINS: [(f32, f32); 4] = [(3.0, 4.1), (-2.7, 5.3), (-6.2, -1.9), (4.4, -7.7)];
    const APPROXIMATE_BITS: [(u32, u32); 4] = [
        (0x40a2_8d11, 0x3f70_746a),
        (0x40be_5d73, 0x4002_b3d4),
        (0x40cf_761d, 0x405c_0c16),
        (0x410d_f269, 0x40a7_6de7),
    ];
    const EXACT_BITS: [(u32, u32); 4] = [
        (0x40a2_9243, 0x3f70_693b),
        (0x40be_56e9, 0x4002_af84),
        (0x40cf_81d1, 0xc036_07d3),
        (0x410d_e54f, 0xbf86_9c79),
    ];

    let packed = core::iter::once(0.0)
        .chain(core::iter::once(0.0))
        .chain(
            COMPLEX_BINS
                .into_iter()
                .flat_map(|(real, imag)| [real, imag]),
        )
        .collect::<Vec<_>>();
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 8,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(spectrum_buffer(packed.clone(), SpectrumCoord::Complex)),
        )
        .expect("approximate-path spectrum");
    let mut units = vec![UnitSpec::new(
        "PV_MagSmooth",
        Rate::Control,
        vec![InputRef::Constant(0.0), InputRef::Constant(0.0)],
        1,
    )];
    units.extend((2..10).map(|slot| read_slot(0.0, slot as f32)));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        core::iter::once(InputRef::Constant(0.0))
            .chain((1..9).map(|unit| InputRef::Unit { unit, output: 0 }))
            .collect(),
        0,
    ));
    controller.add_synthdef(SynthDef {
        name: "pv-approximate-polar-grid".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("pv-approximate-polar-grid", ROOT_GROUP_ID, AddAction::Tail)
        .expect("PV_MagSmooth approximate-polar graph");
    let approximate = render_block(&mut world, 8);
    let approximate_bits = (0..8)
        .map(|index| channel(&approximate, index).to_bits())
        .collect::<Vec<_>>();
    let expected_approximate_bits = APPROXIMATE_BITS
        .into_iter()
        .flat_map(|(magnitude, phase)| [magnitude, phase])
        .collect::<Vec<_>>();
    assert_eq!(
        approximate_bits, expected_approximate_bits,
        "PV_MagSmooth must use the pinned SC 3.14.1 approximate-polar table construction"
    );

    let mut exact = spectrum_buffer(packed, SpectrumCoord::Complex);
    pv::to_polar(&mut exact.view_mut()).expect("existing exact polar view");
    let exact_bits = exact.data()[2..10]
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let expected_exact_bits = EXACT_BITS
        .into_iter()
        .flat_map(|(magnitude, phase)| [magnitude, phase])
        .collect::<Vec<_>>();
    assert_eq!(
        exact_bits, expected_exact_bits,
        "the pre-existing exact polar conversion bits changed"
    );
    assert!(
        approximate_bits
            .iter()
            .zip(&exact_bits)
            .any(|(approximate, exact)| approximate != exact),
        "the off-grid quadrants must distinguish approximate and exact conversion"
    );
}

/// Covers freeze warm-up stages, freeze toggles, and FFT-size reinitialization.
#[test]
fn pv_freeze_three_stage_warmup_freeze_unfreeze_and_size_reset() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 4,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(spectrum_buffer(
                vec![1.0, 2.0, 3.0, 0.1],
                SpectrumCoord::Polar,
            )),
        )
        .expect("buffer");
    controller.add_synthdef(SynthDef {
        name: "freeze-stages".to_string(),
        params: vec![Param::control("freeze", 0.0)],
        units: vec![
            UnitSpec::new(
                "PV_Freeze",
                Rate::Control,
                vec![InputRef::Constant(0.0), InputRef::Param(0)],
                1,
            ),
            read_slot(0.0, 0.0),
            read_slot(0.0, 1.0),
            read_slot(0.0, 2.0),
            read_slot(0.0, 3.0),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Unit { unit: 4, output: 0 },
                ],
                0,
            ),
        ],
    });
    let synth = controller
        .synth_new("freeze-stages", ROOT_GROUP_ID, AddAction::Tail)
        .expect("PV_Freeze graph");

    let first = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&first, 0),
            channel(&first, 1),
            channel(&first, 2),
            channel(&first, 3),
        ),
        (1.0, 2.0, 3.0, 0.1),
        "stage zero establishes the frame size without mutating it"
    );

    for (slot, value) in [5.0, 6.0, 7.0, 0.2].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("second frame");
    }
    let second = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&second, 0),
            channel(&second, 1),
            channel(&second, 2),
            channel(&second, 3),
        ),
        (5.0, 6.0, 7.0, 0.2),
        "stage one stores the first complete magnitude/phase frame"
    );

    controller
        .set_control(synth, 0, 1.0)
        .expect("engage freeze");
    for (slot, value) in [9.0, 10.0, 11.0, 0.4].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("third frame");
    }
    let third = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&third, 0),
            channel(&third, 1),
            channel(&third, 2),
            channel(&third, 3),
        ),
        (5.0, 6.0, 7.0, 0.4),
        "stage two establishes phase differences while freezing stored magnitudes"
    );

    for (slot, value) in [13.0, 14.0, 15.0, 0.8].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("frozen frame");
    }
    let frozen = render_block(&mut world, 4);
    assert_eq!(channel(&frozen, 0), 5.0, "frozen DC");
    assert_eq!(channel(&frozen, 1), 6.0, "frozen Nyquist");
    assert_eq!(channel(&frozen, 2), 7.0, "frozen magnitude");
    assert!(
        (channel(&frozen, 3) - 0.6).abs() < 1e-6,
        "phase advances by the retained difference"
    );

    controller
        .set_control(synth, 0, f32::NAN)
        .expect("non-finite freeze control");
    for (slot, value) in [17.0, 18.0, 19.0, 1.0].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("invalid-control frame");
    }
    let retained = render_block(&mut world, 4);
    assert_eq!(
        (channel(&retained, 0), channel(&retained, 2)),
        (5.0, 7.0),
        "a non-finite control retains the previous freeze state"
    );
    assert!(
        (channel(&retained, 3) - 0.8).abs() < 1e-6,
        "retained freeze continues coherent phase"
    );

    controller
        .set_control(synth, 0, 0.0)
        .expect("release freeze");
    for (slot, value) in [21.0, 22.0, 23.0, 1.2].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("unfrozen frame");
    }
    let unfrozen = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&unfrozen, 0),
            channel(&unfrozen, 1),
            channel(&unfrozen, 2),
            channel(&unfrozen, 3),
        ),
        (21.0, 22.0, 23.0, 1.2),
        "unfreezing stores and passes the incoming frame"
    );

    controller
        .buffer_set(
            0,
            Box::new(spectrum_buffer_size(
                vec![31.0, 32.0, 33.0, 1.4],
                SpectrumCoord::Polar,
                1_024,
            )),
        )
        .expect("replace with a different supported FFT size");
    let resized = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&resized, 0),
            channel(&resized, 1),
            channel(&resized, 2),
            channel(&resized, 3),
        ),
        (31.0, 32.0, 33.0, 1.4),
        "a size change restarts stage zero and does not reuse stale state"
    );
}

/// Proves invalid tokens and malformed frames are atomic and followed by exact recovery.
#[test]
fn pv_invalid_tokens_and_frames_are_atomic_then_recover() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 4,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(spectrum_buffer(
                vec![1.0, 2.0, 3.0, 0.25],
                SpectrumCoord::Polar,
            )),
        )
        .expect("buffer");
    controller.add_synthdef(SynthDef {
        name: "smooth-invalid-recovery".to_string(),
        params: vec![Param::control("token", 0.0)],
        units: vec![
            UnitSpec::new(
                "PV_MagSmooth",
                Rate::Control,
                vec![InputRef::Param(0), InputRef::Constant(0.5)],
                1,
            ),
            read_slot(0.0, 0.0),
            read_slot(0.0, 1.0),
            read_slot(0.0, 2.0),
            read_slot(0.0, 3.0),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Unit { unit: 4, output: 0 },
                ],
                0,
            ),
        ],
    });
    let synth = controller
        .synth_new("smooth-invalid-recovery", ROOT_GROUP_ID, AddAction::Tail)
        .expect("PV_MagSmooth graph");
    render_block(&mut world, 4);

    for (slot, value) in [5.0, 6.0, 7.0, 0.75].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("new spectrum");
    }
    controller
        .set_control(synth, 0, 0.5)
        .expect("fractional token");
    let fractional = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&fractional, 0),
            channel(&fractional, 1),
            channel(&fractional, 2),
            channel(&fractional, 3),
        ),
        (5.0, 6.0, 7.0, 0.75),
        "a fractional token is a byte-preserving no-op"
    );

    controller.set_control(synth, 0, 0.0).expect("valid token");
    let recovered_token = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&recovered_token, 0),
            channel(&recovered_token, 1),
            channel(&recovered_token, 2),
            channel(&recovered_token, 3),
        ),
        (3.0, 4.0, 5.0, 0.75),
        "the valid retry uses memory from before the rejected token"
    );

    controller
        .buffer_set_sample(0, 2, f32::NAN)
        .expect("non-finite bin");
    let rejected_frame = render_block(&mut world, 4);
    assert!(
        channel(&rejected_frame, 2).is_nan(),
        "a rejected spectrum is not partially repaired or converted"
    );

    for (slot, value) in [9.0, 10.0, 11.0, 1.25].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("finite retry");
    }
    let recovered_frame = render_block(&mut world, 4);
    assert_eq!(
        (
            channel(&recovered_frame, 0),
            channel(&recovered_frame, 1),
            channel(&recovered_frame, 2),
            channel(&recovered_frame, 3),
        ),
        (6.0, 7.0, 8.0, 1.25),
        "the finite retry uses state from before the rejected frame"
    );
}

/// Verifies magnitude smoothing while preserving incoming decoded phases.
#[test]
fn pv_mag_smooth_retains_and_updates_decoded_components() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 4,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(spectrum_buffer(
                vec![1.0, 2.0, 3.0, 0.25],
                SpectrumCoord::Polar,
            )),
        )
        .expect("buffer");
    let units = vec![
        UnitSpec::new(
            "PV_MagSmooth",
            Rate::Control,
            vec![InputRef::Constant(0.0), InputRef::Param(0)],
            1,
        ),
        read_slot(0.0, 0.0),
        read_slot(0.0, 1.0),
        read_slot(0.0, 2.0),
        read_slot(0.0, 3.0),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Unit { unit: 4, output: 0 },
            ],
            0,
        ),
    ];
    controller.add_synthdef(SynthDef {
        name: "smooth".to_string(),
        params: vec![Param::control("factor", 0.5)],
        units,
    });
    let synth = controller
        .synth_new("smooth", ROOT_GROUP_ID, AddAction::Tail)
        .expect("PV_MagSmooth graph");

    let first = render_block(&mut world, 4);
    assert_eq!(
        (channel(&first, 0), channel(&first, 1), channel(&first, 2)),
        (1.0, 2.0, 3.0),
        "the first ready frame initializes memory unchanged"
    );

    for (slot, value) in [5.0, 6.0, 7.0, 0.75].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("change spectrum");
    }
    let second = render_block(&mut world, 4);
    assert_eq!(channel(&second, 0), 3.0, "smoothed DC");
    assert_eq!(channel(&second, 1), 4.0, "smoothed Nyquist");
    assert_eq!(channel(&second, 2), 5.0, "smoothed magnitude");
    assert_eq!(channel(&second, 3), 0.75, "phase remains incoming");

    controller
        .set_control(synth, 0, 2.0)
        .expect("factor above one");
    for (slot, value) in [9.0, 10.0, 11.0, 1.25].into_iter().enumerate() {
        controller
            .buffer_set_sample(0, slot, value)
            .expect("change spectrum");
    }
    let clamped = render_block(&mut world, 4);
    assert_eq!(channel(&clamped, 0), 3.0, "factor clamps to one");
    assert_eq!(channel(&clamped, 1), 4.0, "factor clamps to one");
    assert_eq!(channel(&clamped, 2), 5.0, "factor clamps to one");
    assert_eq!(channel(&clamped, 3), 1.25, "phase is never smoothed");
}

/// Verifies that morphing reads the B spectrum without changing its bytes or coordinate tag.
#[test]
fn pv_morph_decodes_b_without_mutating_it() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 6,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(spectrum_buffer(
                vec![1.0, 2.0, 1.0, 0.0],
                SpectrumCoord::Polar,
            )),
        )
        .expect("buffer A");
    controller
        .buffer_set(
            1,
            Box::new(spectrum_buffer(
                vec![5.0, 6.0, 3.0, 4.0],
                SpectrumCoord::Complex,
            )),
        )
        .expect("buffer B");
    let units = vec![
        UnitSpec::new(
            "PV_Morph",
            Rate::Control,
            vec![
                InputRef::Constant(0.0),
                InputRef::Constant(1.0),
                InputRef::Constant(0.5),
            ],
            1,
        ),
        read_slot(0.0, 0.0),
        read_slot(0.0, 1.0),
        read_slot(0.0, 2.0),
        read_slot(0.0, 3.0),
        read_slot(1.0, 2.0),
        read_slot(1.0, 3.0),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Unit { unit: 4, output: 0 },
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Unit { unit: 6, output: 0 },
            ],
            0,
        ),
    ];
    controller.add_synthdef(SynthDef {
        name: "morph".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("morph", ROOT_GROUP_ID, AddAction::Tail)
        .expect("PV_Morph graph");
    let output = render_block(&mut world, 6);

    assert_eq!(channel(&output, 0), 5.0, "DC copied from B");
    assert_eq!(channel(&output, 1), 6.0, "Nyquist copied from B");
    assert!((channel(&output, 2) - 3.0).abs() < 1e-6, "magnitude");
    let expected_phase = 0.5 * 4.0f32.atan2(3.0);
    assert!(
        (channel(&output, 3) - expected_phase).abs() < 1e-6,
        "raw phase interpolation"
    );
    assert_eq!(channel(&output, 4), 3.0, "B real component unchanged");
    assert_eq!(channel(&output, 5), 4.0, "B imaginary component unchanged");
}

/// Rejects non-finite morph results atomically and recovers on the next valid frame.
#[test]
fn pv_morph_rejects_overflowing_complex_bins_then_recovers() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 2,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(spectrum_buffer(
                vec![1.0, 2.0, f32::MAX, f32::MAX],
                SpectrumCoord::Complex,
            )),
        )
        .expect("buffer A");
    controller
        .buffer_set(
            1,
            Box::new(spectrum_buffer(
                vec![5.0, 6.0, 0.0, 1.0],
                SpectrumCoord::Complex,
            )),
        )
        .expect("buffer B");
    controller.add_synthdef(SynthDef {
        name: "recover".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "PV_Morph",
                Rate::Control,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            read_slot(0.0, 2.0),
            read_slot(0.0, 3.0),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("recover", ROOT_GROUP_ID, AddAction::Tail)
        .expect("PV_Morph recovery graph");

    let rejected = render_block(&mut world, 2);
    assert_eq!(channel(&rejected, 0), f32::MAX);
    assert_eq!(channel(&rejected, 1), f32::MAX);

    controller
        .buffer_set_sample(0, 2, 3.0)
        .expect("finite real");
    controller
        .buffer_set_sample(0, 3, 4.0)
        .expect("finite imaginary");
    let recovered = render_block(&mut world, 2);
    assert_eq!(channel(&recovered, 0), 1.0, "B magnitude at morph one");
    assert!(
        (channel(&recovered, 1) - core::f32::consts::FRAC_PI_2).abs() < 1e-6,
        "B phase at morph one"
    );
}

/// Builds a low-amplitude sine source suitable for FFT resynthesis checks.
fn scaled_sine(frequency: f32) -> [UnitSpec; 2] {
    [
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(frequency), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.5),
            ],
            num_outputs: 1,
            special_index: 2,
        },
    ]
}

/// Builds one FFT unit for the selected buffer and source unit.
fn fft(buffer: f32, source: u32) -> UnitSpec {
    UnitSpec::new(
        "FFT",
        Rate::Control,
        vec![
            InputRef::Constant(buffer),
            InputRef::Unit {
                unit: source,
                output: 0,
            },
            InputRef::Constant(0.5),
            InputRef::Constant(0.0),
            InputRef::Constant(1.0),
            InputRef::Constant(1_024.0),
        ],
        1,
    )
}

/// Builds one IFFT unit for the selected spectral chain.
fn ifft(chain: u32) -> UnitSpec {
    UnitSpec::new(
        "IFFT",
        Rate::Audio,
        vec![
            InputRef::Unit {
                unit: chain,
                output: 0,
            },
            InputRef::Constant(0.0),
            InputRef::Constant(1_024.0),
        ],
        1,
    )
}

/// Renders one phase-vocoder operator inside a non-vacuous FFT/IFFT graph.
fn pv_resynthesis(name: &str) -> Vec<f32> {
    const SIZE: usize = 1_024;
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(vec![0.0; SIZE], 1, SR)),
        )
        .expect("FFT buffer A");

    let units = if name == "PV_Morph" {
        controller
            .buffer_set(
                1,
                Box::new(Buffer::from_interleaved(vec![0.0; SIZE], 1, SR)),
            )
            .expect("FFT buffer B");
        let mut units = scaled_sine(234.375).into_iter().collect::<Vec<_>>();
        units.push(fft(0.0, 1));
        units.push(UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(468.75), InputRef::Constant(0.0)],
            1,
        ));
        units.push(UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![
                InputRef::Unit { unit: 3, output: 0 },
                InputRef::Constant(0.5),
            ],
            num_outputs: 1,
            special_index: 2,
        });
        units.push(fft(1.0, 4));
        units.push(UnitSpec::new(
            "PV_Morph",
            Rate::Control,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Constant(0.5),
            ],
            1,
        ));
        units.push(ifft(6));
        units.push(UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 7, output: 0 },
            ],
            0,
        ));
        units
    } else {
        let mut units = scaled_sine(234.375).into_iter().collect::<Vec<_>>();
        units.push(fft(0.0, 1));
        let extra = if name == "PV_Freeze" { 0.0 } else { 0.25 };
        units.push(UnitSpec::new(
            name,
            Rate::Control,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Constant(extra),
            ],
            1,
        ));
        units.push(ifft(3));
        units.push(UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 4, output: 0 },
            ],
            0,
        ));
        units
    };

    controller.add_synthdef(SynthDef {
        name: format!("{name}-resynthesis"),
        params: vec![],
        units,
    });
    controller
        .synth_new(
            &format!("{name}-resynthesis"),
            ROOT_GROUP_ID,
            AddAction::Tail,
        )
        .expect("FFT/PV/IFFT graph");
    let mut output = vec![0.0; 16_384];
    world.fill(&mut output, 1);
    output
}

/// Proves every phase-vocoder operator resynthesizes finite non-silent audio.
#[test]
fn pv_units_resynthesize_non_vacuous_fft_chains() {
    for name in ["PV_Freeze", "PV_MagSmooth", "PV_Morph"] {
        let output = pv_resynthesis(name);
        let tail = &output[8_192..];
        assert!(
            tail.iter().all(|sample| sample.is_finite()),
            "{name} FFT/PV/IFFT chain emitted a non-finite sample"
        );
        let rms =
            (tail.iter().map(|sample| sample * sample).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(
            rms > 0.01,
            "{name} FFT/PV/IFFT chain is vacuous (RMS {rms})"
        );
        assert!(
            tail.chunks_exact(1_024)
                .take(4)
                .all(|window| window.iter().any(|sample| sample.abs() > 1e-4)),
            "{name} must produce non-zero audio in multiple ready-frame windows"
        );
    }
}
