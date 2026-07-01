//! The selection/indexing units: `Select` (pick one of several signal inputs) and the buffer-lookup
//! family `Index` (clip), `IndexL` (linear interp), `WrapIndex` and `FoldIndex`.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
    to_wavetable,
};

const SR: f64 = 48_000.0;

fn out_unit(src: u32) -> UnitSpec {
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

/// Render one block of a graph, optionally with a lookup table set at buffer 0; returns a mid sample.
fn run(units: Vec<UnitSpec>, table: Option<&[f32]>) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    if let Some(t) = table {
        controller
            .buffer_set(0, Box::new(Buffer::from_interleaved(t.to_vec(), 1, SR)))
            .expect("buffer_set");
    }
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; 64];
    world.fill(&mut buf, 1);
    buf[32]
}

/// `Select.ar(which, [DC(0.1), DC(0.2), DC(0.3)]) -> Out`, `which` a constant.
fn select(which: f32) -> f32 {
    let units = vec![
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.1)], 1),
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.2)], 1),
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.3)], 1),
        UnitSpec::new(
            "Select",
            Rate::Audio,
            vec![
                InputRef::Constant(which),
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Unit { unit: 1, output: 0 },
                InputRef::Unit { unit: 2, output: 0 },
            ],
            1,
        ),
        out_unit(3),
    ];
    run(units, None)
}

#[test]
fn select_picks_the_indexed_input() {
    assert!((select(0.0) - 0.1).abs() < 1e-6, "which 0 -> item 0");
    assert!((select(1.0) - 0.2).abs() < 1e-6, "which 1 -> item 1");
    assert!((select(2.0) - 0.3).abs() < 1e-6, "which 2 -> item 2");
    // Out-of-range indices clip to the ends.
    assert!(
        (select(5.0) - 0.3).abs() < 1e-6,
        "over-range clips to the last"
    );
    assert!(
        (select(-1.0) - 0.1).abs() < 1e-6,
        "under-range clips to the first"
    );
}

/// `<name>.ar(bufnum=0, DC(index)) -> Out` against the given table.
fn index(name: &str, table: &[f32], index_val: f32) -> f32 {
    let units = vec![
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(index_val)], 1),
        UnitSpec::new(
            name,
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 0, output: 0 },
            ],
            1,
        ),
        out_unit(1),
    ];
    run(units, Some(table))
}

#[test]
fn index_clips_to_the_nearest_slot() {
    let t = [10.0, 20.0, 30.0, 40.0];
    assert!((index("Index", &t, 0.0) - 10.0).abs() < 1e-4);
    assert!((index("Index", &t, 2.0) - 30.0).abs() < 1e-4);
    assert!(
        (index("Index", &t, 2.9) - 30.0).abs() < 1e-4,
        "truncates toward the low slot"
    );
    assert!(
        (index("Index", &t, 5.0) - 40.0).abs() < 1e-4,
        "over-range clips high"
    );
    assert!(
        (index("Index", &t, -3.0) - 10.0).abs() < 1e-4,
        "under-range clips low"
    );
}

#[test]
fn index_l_interpolates_between_slots() {
    let t = [10.0, 20.0, 30.0, 40.0];
    assert!(
        (index("IndexL", &t, 1.5) - 25.0).abs() < 1e-4,
        "halfway between 20 and 30"
    );
    assert!(
        (index("IndexL", &t, 0.25) - 12.5).abs() < 1e-4,
        "quarter between 10 and 20"
    );
    assert!(
        (index("IndexL", &t, 3.0) - 40.0).abs() < 1e-4,
        "top slot, no slot above"
    );
}

#[test]
fn wrap_index_wraps_the_index() {
    let t = [10.0, 20.0, 30.0, 40.0]; // indices 0..=3
    assert!(
        (index("WrapIndex", &t, 4.0) - 10.0).abs() < 1e-4,
        "4 wraps to 0"
    );
    assert!(
        (index("WrapIndex", &t, 5.0) - 20.0).abs() < 1e-4,
        "5 wraps to 1"
    );
    assert!(
        (index("WrapIndex", &t, -1.0) - 40.0).abs() < 1e-4,
        "-1 wraps to 3"
    );
}

#[test]
fn fold_index_folds_the_index() {
    let t = [10.0, 20.0, 30.0, 40.0]; // indices 0..=3
    assert!(
        (index("FoldIndex", &t, 4.0) - 30.0).abs() < 1e-4,
        "4 folds to 2"
    );
    assert!(
        (index("FoldIndex", &t, 5.0) - 20.0).abs() < 1e-4,
        "5 folds to 1"
    );
    assert!(
        (index("FoldIndex", &t, -1.0) - 20.0).abs() < 1e-4,
        "-1 folds to 1"
    );
}

/// `Shaper.ar(bufnum=0, DC(x)) -> Out` against a wavetable-format transfer `table`.
fn shape(table: &[f32], x: f32) -> f32 {
    let units = vec![
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(x)], 1),
        UnitSpec::new(
            "Shaper",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 0, output: 0 },
            ],
            1,
        ),
        out_unit(1),
    ];
    run(units, Some(table))
}

#[test]
fn shaper_identity_transfer_is_transparent() {
    // A ramp -1..1 is the identity transfer f(u) = u, so shaping x returns ~x.
    let n = 64;
    let ramp: Vec<f32> = (0..n).map(|i| -1.0 + 2.0 * i as f32 / n as f32).collect();
    let wt = to_wavetable(&ramp);
    for x in [-0.7f32, -0.2, 0.0, 0.3, 0.6] {
        assert!(
            (shape(&wt, x) - x).abs() < 1e-2,
            "identity shape({x}) = {}",
            shape(&wt, x)
        );
    }
}

#[test]
fn shaper_applies_a_nonlinear_transfer() {
    // A T2 Chebyshev transfer f(u) = 2u^2 - 1: shaping x returns ~2x^2 - 1.
    let n = 128;
    let t2: Vec<f32> = (0..n)
        .map(|i| {
            let u = -1.0 + 2.0 * i as f32 / n as f32;
            2.0 * u * u - 1.0
        })
        .collect();
    let wt = to_wavetable(&t2);
    for x in [-0.6f32, -0.25, 0.0, 0.4, 0.7] {
        let expected = 2.0 * x * x - 1.0;
        assert!(
            (shape(&wt, x) - expected).abs() < 2e-2,
            "T2 shape({x}) = {}, expected {expected}",
            shape(&wt, x)
        );
    }
}

/// `DegreeToKey.ar(bufnum=0, DC(degree), octave) -> Out` against a scale `table`.
fn degree_to_key(table: &[f32], degree: f32, octave: f32) -> f32 {
    let units = vec![
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(degree)], 1),
        UnitSpec::new(
            "DegreeToKey",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(octave),
            ],
            1,
        ),
        out_unit(1),
    ];
    run(units, Some(table))
}

#[test]
fn degree_to_key_maps_scale_degrees_with_octave_wrap() {
    // A C-major scale (7 degrees, semitone offsets), octave = 12.
    let scale = [0.0f32, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0];
    let key = |d: f32| degree_to_key(&scale, d, 12.0);
    assert!((key(0.0) - 0.0).abs() < 1e-4, "degree 0 -> key 0");
    assert!((key(3.0) - 5.0).abs() < 1e-4, "degree 3 -> key 5");
    assert!(
        (key(7.0) - 12.0).abs() < 1e-4,
        "degree 7 -> octave up, degree 0"
    );
    assert!(
        (key(8.0) - 14.0).abs() < 1e-4,
        "degree 8 -> octave up, degree 1"
    );
    // Negative degrees wrap down (Euclidean modulo, unlike scsynth's C `%` at exact multiples).
    assert!(
        (key(-1.0) - -1.0).abs() < 1e-4,
        "degree -1 -> top degree an octave down"
    );
    assert!(
        (key(-7.0) - -12.0).abs() < 1e-4,
        "degree -7 -> degree 0 an octave down"
    );
}

#[test]
fn index_of_missing_buffer_is_silent() {
    // No table set at buffer 0: the lookup yields 0.
    let units = vec![
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
        UnitSpec::new(
            "Index",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 0, output: 0 },
            ],
            1,
        ),
        out_unit(1),
    ];
    assert!(
        run(units, None).abs() < 1e-6,
        "missing table should be silent"
    );
}
