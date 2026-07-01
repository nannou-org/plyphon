//! Exercise the range-shaping units (`Clip`, `Wrap`, `Fold`, `ModDif`, `InRange`, `InRect`,
//! `LinExp`, `Unwrap`) and `XLine`, driven mostly by a full-scale `SinOsc`.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn run(units: Vec<UnitSpec>, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut i = 0;
    while out.len() < frames {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

fn sin(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

fn out(unit: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![InputRef::Constant(0.0), InputRef::Unit { unit, output: 0 }],
        0,
    )
}

/// `shaper.ar(SinOsc.ar(200), ..params) -> Out`, with the sine as unit 0 and the shaper as unit 1.
fn on_sine(shaper: &str, params: Vec<InputRef>) -> Vec<f32> {
    let mut inputs = vec![InputRef::Unit { unit: 0, output: 0 }];
    inputs.extend(params);
    run(
        vec![
            sin(200.0),
            UnitSpec::new(shaper, Rate::Audio, inputs, 1),
            out(1),
        ],
        SR as usize / 20,
    )
}

#[test]
fn clip_clamps_to_bounds() {
    let o = on_sine(
        "Clip",
        vec![InputRef::Constant(-0.3), InputRef::Constant(0.3)],
    );
    assert!(
        o.iter().all(|&x| (-0.3001..=0.3001).contains(&x)),
        "Clip exceeded its bounds"
    );
    // A full-scale sine spends time beyond ±0.3, so the clamp is exercised.
    assert!(
        o.iter().any(|&x| x > 0.29) && o.iter().any(|&x| x < -0.29),
        "Clip should reach both bounds"
    );
}

#[test]
fn wrap_stays_within_range() {
    let o = on_sine(
        "Wrap",
        vec![InputRef::Constant(-0.5), InputRef::Constant(0.5)],
    );
    assert!(
        o.iter().all(|&x| (-0.5001..0.5001).contains(&x)),
        "Wrap escaped [-0.5, 0.5)"
    );
    // Wrapping a ±1 sine produces discontinuities, so it visits the full range.
    assert!(
        o.iter().any(|&x| x > 0.4) && o.iter().any(|&x| x < -0.4),
        "Wrap should span its range"
    );
}

#[test]
fn fold_stays_within_range() {
    let o = on_sine(
        "Fold",
        vec![InputRef::Constant(-0.5), InputRef::Constant(0.5)],
    );
    assert!(
        o.iter().all(|&x| (-0.5001..=0.5001).contains(&x)),
        "Fold escaped [-0.5, 0.5]"
    );
    assert!(
        o.iter().any(|&x| x > 0.4) && o.iter().any(|&x| x < -0.4),
        "Fold should span its range"
    );
}

#[test]
fn moddif_gives_ring_distance() {
    // ModDif(sine, 0, 1): the distance from 0 on a mod-1 ring is in [0, 0.5].
    let o = on_sine(
        "ModDif",
        vec![InputRef::Constant(0.0), InputRef::Constant(1.0)],
    );
    assert!(
        o.iter().all(|&x| (-0.0001..=0.5001).contains(&x)),
        "ModDif out of [0, 0.5]"
    );
    assert!(
        o.iter().any(|&x| x > 0.4),
        "ModDif should reach ~0.5 where |sine| ~ 0.5"
    );
}

#[test]
fn in_range_flags_membership() {
    // InRange(sine, -0.5, 0.5): a full-scale sine is inside [-0.5, 0.5] about a third of the time.
    let o = on_sine(
        "InRange",
        vec![InputRef::Constant(-0.5), InputRef::Constant(0.5)],
    );
    assert!(
        o.iter().all(|&x| x == 0.0 || x == 1.0),
        "InRange must be 0 or 1"
    );
    let inside = o.iter().filter(|&&x| x == 1.0).count() as f32 / o.len() as f32;
    assert!(
        (inside - 0.33).abs() < 0.08,
        "InRange should be ~1/3 inside, got {inside}"
    );
}

#[test]
fn in_rect_flags_membership() {
    // InRect(SinOsc(200), SinOsc(150), -0.5, -0.5, 0.5, 0.5): 1 when both coords are in [-0.5, 0.5].
    let o = run(
        vec![
            sin(200.0),
            sin(150.0),
            UnitSpec::new(
                "InRect",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(-0.5),
                    InputRef::Constant(-0.5),
                    InputRef::Constant(0.5),
                    InputRef::Constant(0.5),
                ],
                1,
            ),
            out(2),
        ],
        SR as usize / 20,
    );
    assert!(
        o.iter().all(|&x| x == 0.0 || x == 1.0),
        "InRect must be 0 or 1"
    );
    assert!(
        o.contains(&1.0) && o.contains(&0.0),
        "InRect should see both states"
    );
}

#[test]
fn lin_exp_maps_onto_an_exponential_range() {
    // LinExp(sine in [-1, 1], -1, 1, 100, 1000): output stays within [100, 1000], the endpoints
    // reached at the sine's peaks.
    let o = on_sine(
        "LinExp",
        vec![
            InputRef::Constant(-1.0),
            InputRef::Constant(1.0),
            InputRef::Constant(100.0),
            InputRef::Constant(1000.0),
        ],
    );
    assert!(
        o.iter().all(|&x| (99.0..=1001.0).contains(&x)),
        "LinExp escaped [100, 1000]"
    );
    assert!(
        o.iter().any(|&x| x > 900.0) && o.iter().any(|&x| x < 120.0),
        "LinExp should span its range"
    );
}

#[test]
fn unwrap_makes_a_wrapped_ramp_continuous() {
    // A Phasor sawtooth (0..1, wrapping) unwrapped in [0, 1] becomes a continuously rising ramp.
    let o = run(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.0)], 1), // no reset trigger
            UnitSpec::new(
                "Phasor",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.002),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Unwrap",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            out(2),
        ],
        SR as usize / 10,
    );
    assert!(o.iter().all(|s| s.is_finite()));
    // Rises well past the [0, 1) wrap range, and does so essentially monotonically.
    assert!(
        *o.last().unwrap() > 5.0,
        "Unwrap should accumulate past 5, got {}",
        o.last().unwrap()
    );
    let backsteps = o.windows(2).filter(|w| w[1] < w[0] - 1e-3).count();
    assert!(
        backsteps < 5,
        "Unwrap should be monotonic, but stepped back {backsteps} times"
    );
}

#[test]
fn xline_ramps_exponentially() {
    // XLine(100, 1000, 0.25): rises from 100 to 1000, monotonically.
    let o = run(
        vec![
            UnitSpec::new(
                "XLine",
                Rate::Audio,
                vec![
                    InputRef::Constant(100.0),
                    InputRef::Constant(1000.0),
                    InputRef::Constant(0.25),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out(0),
        ],
        SR as usize / 2,
    );
    assert!(
        (o[0] - 100.0).abs() < 1.0,
        "XLine should start at 100, got {}",
        o[0]
    );
    assert!(
        (*o.last().unwrap() - 1000.0).abs() < 1.0,
        "XLine should end at 1000, got {}",
        o.last().unwrap()
    );
    assert!(
        o.windows(2).all(|w| w[1] >= w[0] - 1e-3),
        "XLine should rise monotonically"
    );
}
