//! Exercise the panning / spatial units (`LinPan2`, `Balance2`, `XFade2`, `LinXFade2`, `Rotate2`)
//! with steady `DC` sources, checking the per-channel gains scsynth produces.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn dc(v: f32) -> UnitSpec {
    UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(v)], 1)
}

/// Render `channels` interleaved outputs of `units` (last unit an `Out`), returning the steady value
/// per channel (sampled well after the start).
fn steady(units: Vec<UnitSpec>, channels: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "p".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("p", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; 256 * channels];
    world.fill(&mut buf, channels);
    // The last frame, per channel.
    (0..channels)
        .map(|c| buf[(256 - 1) * channels + c])
        .collect()
}

fn out2(unit: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit, output: 0 },
            InputRef::Unit { unit, output: 1 },
        ],
        0,
    )
}

fn out1(unit: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![InputRef::Constant(0.0), InputRef::Unit { unit, output: 0 }],
        0,
    )
}

fn close(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-3, "expected {b}, got {a}");
}

#[test]
fn lin_pan2_splits_linearly() {
    let mk = |pos: f32| {
        steady(
            vec![
                dc(1.0),
                UnitSpec::new(
                    "LinPan2",
                    Rate::Audio,
                    vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Constant(pos),
                        InputRef::Constant(1.0),
                    ],
                    2,
                ),
                out2(1),
            ],
            2,
        )
    };
    let l = mk(-1.0);
    close(l[0], 1.0);
    close(l[1], 0.0);
    let c = mk(0.0);
    close(c[0], 0.5);
    close(c[1], 0.5);
    let r = mk(1.0);
    close(r[0], 0.0);
    close(r[1], 1.0);
}

#[test]
fn balance2_is_equal_power() {
    let mk = |pos: f32| {
        steady(
            vec![
                dc(1.0),
                dc(1.0),
                UnitSpec::new(
                    "Balance2",
                    Rate::Audio,
                    vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Unit { unit: 1, output: 0 },
                        InputRef::Constant(pos),
                        InputRef::Constant(1.0),
                    ],
                    2,
                ),
                out2(2),
            ],
            2,
        )
    };
    // Centre: both channels at the equal-power 0.707.
    let c = mk(0.0);
    close(c[0], std::f32::consts::FRAC_1_SQRT_2);
    close(c[1], std::f32::consts::FRAC_1_SQRT_2);
    let l = mk(-1.0);
    close(l[0], 1.0);
    close(l[1], 0.0);
    let r = mk(1.0);
    close(r[0], 0.0);
    close(r[1], 1.0);
}

#[test]
fn xfade2_equal_power_crossfade() {
    // XFade2(DC(1), DC(-1), pos): -1 -> +1 (all A), +1 -> -1 (all B), 0 -> ~0.
    let mk = |pos: f32| {
        steady(
            vec![
                dc(1.0),
                dc(-1.0),
                UnitSpec::new(
                    "XFade2",
                    Rate::Audio,
                    vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Unit { unit: 1, output: 0 },
                        InputRef::Constant(pos),
                        InputRef::Constant(1.0),
                    ],
                    1,
                ),
                out1(2),
            ],
            1,
        )[0]
    };
    close(mk(-1.0), 1.0);
    close(mk(1.0), -1.0);
    close(mk(0.0), 0.0);
}

#[test]
fn lin_xfade2_linear_crossfade() {
    // LinXFade2(DC(0), DC(1), pos): amp = pos*0.5+0.5; out = amp.
    let mk = |pos: f32| {
        steady(
            vec![
                dc(0.0),
                dc(1.0),
                UnitSpec::new(
                    "LinXFade2",
                    Rate::Audio,
                    vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Unit { unit: 1, output: 0 },
                        InputRef::Constant(pos),
                    ],
                    1,
                ),
                out1(2),
            ],
            1,
        )[0]
    };
    close(mk(-1.0), 0.0);
    close(mk(0.0), 0.5);
    close(mk(1.0), 1.0);
}

#[test]
fn rotate2_rotates_the_field() {
    // Rotate2(DC(1), DC(0), pos): out_x = cos(pi*pos), out_y = -sin(pi*pos).
    let mk = |pos: f32| {
        steady(
            vec![
                dc(1.0),
                dc(0.0),
                UnitSpec::new(
                    "Rotate2",
                    Rate::Audio,
                    vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Unit { unit: 1, output: 0 },
                        InputRef::Constant(pos),
                    ],
                    2,
                ),
                out2(2),
            ],
            2,
        )
    };
    let z = mk(0.0);
    close(z[0], 1.0);
    close(z[1], 0.0);
    // pos = 0.5 -> quarter turn: x -> 0, y -> -1.
    let q = mk(0.5);
    close(q[0], 0.0);
    close(q[1], -1.0);
}
