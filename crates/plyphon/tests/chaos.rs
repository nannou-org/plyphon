//! Exercise the chaotic map generators (`CuspN`, `QuadN`, `LinCongN`, `GbmanN`, `StandardN`,
//! `LatoocarfianN`): each should produce a bounded, non-constant, sample-and-held signal.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

/// Render `name(consts) -> Out` for `frames` samples.
fn chaos(name: &str, consts: &[f32], frames: usize) -> Vec<f32> {
    let inputs = consts.iter().map(|&c| InputRef::Constant(c)).collect();
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "c".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(name, Rate::Audio, inputs, 1),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("c", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}

fn transitions(o: &[f32]) -> usize {
    o.windows(2).filter(|w| w[0] != w[1]).count()
}

/// Assert `o` is finite, within `±bound`, clearly non-constant, and held (a `freq`=1000 Hz map over
/// 4800 samples iterates ~100 times, far fewer than the sample count).
fn assert_chaotic(o: &[f32], bound: f32) {
    assert!(o.iter().all(|s| s.is_finite()), "chaos output not finite");
    assert!(
        o.iter().all(|&x| x.abs() <= bound),
        "chaos output exceeded ±{bound}"
    );
    let (min, max) = o
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    assert!(max - min > 0.1, "chaos output barely varies ({min}..{max})");
    let t = transitions(o);
    assert!(
        (20..o.len() / 8).contains(&t),
        "unexpected sample-and-hold rate: {t} transitions"
    );
}

const N: usize = 4800; // 0.1 s

#[test]
fn cusp_n_is_bounded_chaos() {
    // CuspN(1000, a=1, b=1.9, xi=0).
    assert_chaotic(&chaos("CuspN", &[1000.0, 1.0, 1.9, 0.0], N), 10.0);
}

#[test]
fn quad_n_is_bounded_chaos() {
    // QuadN(1000, a=1, b=-1, c=-0.75, xi=0) - the logistic-family quadratic.
    assert_chaotic(&chaos("QuadN", &[1000.0, 1.0, -1.0, -0.75, 0.0], N), 4.0);
}

#[test]
fn lin_cong_n_stays_in_unit_range() {
    // LinCongN(1000, a=1.1, c=0.13, m=1, xi=0) - output scaled to [-1, 1).
    let o = chaos("LinCongN", &[1000.0, 1.1, 0.13, 1.0, 0.0], N);
    assert_chaotic(&o, 1.001);
}

#[test]
fn gbman_n_is_bounded_chaos() {
    // GbmanN(1000, xi=1.2, yi=2.1) - the Gingerbreadman map.
    assert_chaotic(&chaos("GbmanN", &[1000.0, 1.2, 2.1], N), 20.0);
}

#[test]
fn standard_n_stays_in_unit_range() {
    // StandardN(1000, k=1, xi=0.5, yi=0) - output scaled to [-1, 1).
    assert_chaotic(&chaos("StandardN", &[1000.0, 1.0, 0.5, 0.0], N), 1.001);
}

#[test]
fn latoocarfian_n_is_bounded_chaos() {
    // LatoocarfianN(1000, a=1, b=3, c=0.5, d=0.5, xi=0.5, yi=0.5).
    assert_chaotic(
        &chaos("LatoocarfianN", &[1000.0, 1.0, 3.0, 0.5, 0.5, 0.5, 0.5], N),
        3.0,
    );
}
