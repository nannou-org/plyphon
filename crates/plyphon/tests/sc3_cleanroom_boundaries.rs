//! Semantic regression checks for the retained SC3 clean-room boundary extension.

const RENDERED: usize = 448;

/// Proves every retained boundary-oracle asset still matches its manifest hash.
#[test]
fn sc3_processing_boundary_manifest_hashes_verify() {
    let output = std::process::Command::new("python3")
        .arg("verify_boundaries.py")
        .current_dir(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sc3_processing"),
        )
        .output()
        .expect("python3 runs the boundary-oracle verifier");
    assert!(
        output.status.success(),
        "boundary verifier failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn fixture(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32")))
        .collect()
}

fn value(values: &[f32], channels: usize, frame: usize, channel: usize) -> f32 {
    values[frame * channels + channel]
}

fn exact_columns(
    values: &[f32],
    channels: usize,
    left: usize,
    right: usize,
    mut frames: core::ops::Range<usize>,
) -> bool {
    frames.all(|frame| {
        value(values, channels, frame, left).to_bits()
            == value(values, channels, frame, right).to_bits()
    })
}

#[test]
fn decimator_boundary_oracle_pins_cadence_and_non_finite_controls() {
    let boundary = fixture(include_bytes!(
        "fixtures/sc3_processing/decimator_boundaries.f32"
    ));
    for frame in 0..RENDERED {
        assert_eq!(value(&boundary, 5, frame, 1), 0.0, "negative rate");
        assert_eq!(value(&boundary, 5, frame, 2), 0.0, "zero rate");
    }
    assert!(exact_columns(&boundary, 5, 3, 4, 0..RENDERED));
    for frame in 1..RENDERED {
        assert_ne!(
            value(&boundary, 5, frame - 1, 3).to_bits(),
            value(&boundary, 5, frame, 3).to_bits(),
            "sample-rate lane must update every sample"
        );
    }

    let controls = fixture(include_bytes!(
        "fixtures/sc3_processing/decimator_controls.f32"
    ));
    assert!(exact_columns(&controls, 9, 0, 2, 0..RENDERED));
    assert!(exact_columns(&controls, 9, 1, 7, 0..RENDERED));
    assert!(exact_columns(&controls, 9, 4, 5, 0..RENDERED));
    assert!(exact_columns(&controls, 9, 4, 8, 0..RENDERED));
    let nan_frames: Vec<_> = (0..RENDERED)
        .filter(|&frame| value(&controls, 9, frame, 3).is_nan())
        .collect();
    assert_eq!(nan_frames, (66..194).collect::<Vec<_>>());
    assert!(exact_columns(&controls, 9, 3, 8, 194..RENDERED));
}

#[test]
fn bmoog_boundary_oracle_pins_finite_unclamped_and_non_finite_branches() {
    let cases = [
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_0.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_1.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_2.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_3.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_4.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_5.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_6.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_7.f32"
        )),
        fixture(include_bytes!(
            "fixtures/sc3_processing/bmoog_boundaries_8.f32"
        )),
    ];
    for case in &cases {
        assert!(case[..RENDERED * 4].iter().all(|sample| sample.is_finite()));
        assert!(exact_columns(case, 4, 0, 3, 0..RENDERED));
    }
    for (actual, boundary) in [(0, 5), (1, 5), (2, 6), (3, 7), (4, 8)] {
        assert!(
            (0..RENDERED).any(|frame| {
                value(&cases[actual], 4, frame, 0).to_bits()
                    != value(&cases[boundary], 4, frame, 0).to_bits()
            }),
            "out-of-range case must not equal its finite boundary"
        );
    }

    let controls = fixture(include_bytes!("fixtures/sc3_processing/bmoog_controls.f32"));
    for channel in 0..3 {
        assert!((0..129).all(|frame| value(&controls, 7, frame, channel).is_finite()));
        assert!((129..RENDERED).all(|frame| value(&controls, 7, frame, channel).is_nan()));
    }
    for channel in 3..6 {
        assert!(exact_columns(&controls, 7, channel, 6, 0..RENDERED));
    }
}

#[test]
fn perlin3_boundary_oracle_pins_negative_cells_and_256_periodicity() {
    let values = fixture(include_bytes!(
        "fixtures/sc3_processing/perlin3_boundaries.f32"
    ));
    for group in [
        [0, 3, 6].as_slice(),
        &[1, 4, 7],
        &[2, 5, 8],
        &[9, 10, 11, 12, 13],
    ] {
        let expected = value(&values, 14, 0, group[0]).to_bits();
        assert!(
            group
                .iter()
                .all(|&channel| value(&values, 14, 0, channel).to_bits() == expected)
        );
    }
    assert_ne!(
        value(&values, 14, 0, 0).to_bits(),
        value(&values, 14, 0, 1).to_bits()
    );
    assert_ne!(
        value(&values, 14, 0, 1).to_bits(),
        value(&values, 14, 0, 2).to_bits()
    );
}

#[test]
fn rossler_l_boundary_oracle_pins_floor_destabilization_and_recovery() {
    let boundary = fixture(include_bytes!(
        "fixtures/sc3_processing/rossler_l_boundaries.f32"
    ));
    for (left, right) in [(0, 3), (0, 6), (1, 4), (1, 7), (2, 5), (2, 8)] {
        assert!(exact_columns(&boundary, 12, left, right, 0..RENDERED));
    }
    for frame in 3..RENDERED {
        assert!((9..12).all(|channel| value(&boundary, 12, frame, channel).is_nan()));
    }

    let controls = fixture(include_bytes!(
        "fixtures/sc3_processing/rossler_l_controls.f32"
    ));
    for (kind, reference) in [(0, 72), (1, 72), (2, 75)] {
        let lane = kind * 8 * 3;
        for coordinate in 0..3 {
            assert!(exact_columns(
                &controls,
                78,
                lane + coordinate,
                reference + coordinate,
                0..RENDERED
            ));
        }
    }
    for kind in 0..3 {
        for control in 5..8 {
            let lane = (kind * 8 + control) * 3;
            let invalid: Vec<_> = (0..RENDERED)
                .filter(|&frame| {
                    (0..3).any(|coordinate| {
                        !value(&controls, 78, frame, lane + coordinate).is_finite()
                    })
                })
                .collect();
            assert_eq!(invalid, (64..199).collect::<Vec<_>>());
        }
    }
}

#[test]
fn pv_freeze_boundary_oracle_pins_early_and_non_finite_freeze() {
    let early = fixture(include_bytes!(
        "fixtures/sc3_processing/pv_freeze_early.f32"
    ));
    let pulses: Vec<_> = (0..768)
        .filter(|&frame| value(&early, 135, frame, 4) >= 0.5)
        .collect();
    assert_eq!(pulses, [0, 65, 193, 321, 449, 577]);
    for frame in [384, 512, 640] {
        for bin in 0..65 {
            let channel = 5 + 2 * bin;
            assert!(
                (value(&early, 135, frame, channel) - value(&early, 135, 256, channel)).abs()
                    <= 1.0e-6
            );
        }
    }

    let controls = fixture(include_bytes!(
        "fixtures/sc3_processing/pv_freeze_controls.f32"
    ));
    assert!(exact_columns(&controls, 25, 0, 15, 0..1472));
    assert!(exact_columns(&controls, 25, 10, 15, 0..1472));
    assert!(exact_columns(&controls, 25, 5, 20, 0..1472));
    assert!(value(&controls, 25, 384, 3).is_nan());
    assert_eq!(value(&controls, 25, 384, 8), f32::INFINITY);
    assert_eq!(value(&controls, 25, 384, 13), f32::NEG_INFINITY);
}

#[test]
fn oracle_only_signed_zero_inputs_match_positive_zero() {
    let values = fixture(include_bytes!(
        "fixtures/sc3_processing/signed_zero_boundaries.f32"
    ));
    for frame in 0..RENDERED {
        assert_eq!(value(&values, 82, frame, 0), f32::INFINITY);
        assert_eq!(value(&values, 82, frame, 1), f32::NEG_INFINITY);
    }

    let mut pairs = vec![
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, 17),
        (18, 19),
        (20, 21),
        (22, 23),
        (24, 25),
    ];
    for control in 0..8 {
        let first = 26 + control * 6;
        pairs.extend(
            (0..3)
                .filter(|&coordinate| control != 5 || coordinate != 0)
                .map(|coordinate| (first + coordinate, first + 3 + coordinate)),
        );
    }
    pairs.extend([(74, 78), (76, 80), (77, 81)]);
    for (positive, negative) in pairs {
        assert!(
            exact_columns(&values, 82, positive, negative, 0..RENDERED),
            "signed-zero oracle lanes {positive}/{negative} differ"
        );
    }
    assert_eq!(value(&values, 82, 0, 56).to_bits(), 0.0f32.to_bits());
    assert_eq!(value(&values, 82, 0, 59).to_bits(), (-0.0f32).to_bits());
    assert!(exact_columns(&values, 82, 56, 59, 1..RENDERED));
    assert!([128, 256, 384].into_iter().all(|frame| {
        value(&values, 82, frame, 75) >= 0.0 && value(&values, 82, frame, 79) >= 0.0
    }));
}
