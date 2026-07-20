//! The linear-ramp smoothers `Ramp` (resample-and-interpolate) and `VarLag` (ramp from `start` to the
//! input).

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR).floor();
    let w = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in samples {
        let s = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
}

fn render(units: Vec<UnitSpec>, frames: usize) -> Vec<f32> {
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
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = Vec::with_capacity(frames + 256);
    let mut buf = vec![0.0f32; 256];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

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

#[test]
fn ramp_resamples_and_smooths() {
    // Ramp re-samples its input every `lagTime` (here 100 Hz); a fast tone aliases away, a slow one
    // passes through the piecewise-linear reconstruction.
    let sine = |f: f32| {
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(f), InputRef::Constant(0.0)],
            1,
        )
    };
    let ramp = |f: f32| {
        render(
            vec![
                sine(f),
                UnitSpec::new(
                    "Ramp",
                    Rate::Audio,
                    vec![
                        InputRef::Unit { unit: 0, output: 0 },
                        InputRef::Constant(0.01),
                    ],
                    1,
                ),
                out_unit(1),
            ],
            SR as usize / 4,
        )
    };
    let fast = goertzel(&ramp(650.0), 650.0);
    let slow = goertzel(&ramp(20.0), 20.0);
    assert!(
        slow > 5.0 * fast,
        "Ramp should pass slow changes and smooth fast ones (slow={slow}, fast={fast})"
    );
}

/// `VarLag.ar(in, time, start) -> Out`, rendered for `frames`.
fn render_varlag(in_val: f32, time: f32, start: f32, frames: usize) -> Vec<f32> {
    render(
        vec![
            UnitSpec::new(
                "VarLag",
                Rate::Audio,
                vec![
                    InputRef::Constant(in_val),
                    InputRef::Constant(time),
                    InputRef::Constant(start),
                ],
                1,
            ),
            out_unit(0),
        ],
        frames,
    )
}

#[test]
fn varlag_ramps_from_start_to_input() {
    // From start = 0 toward in = 1 over 50 ms, then holds at 1.
    let out = render_varlag(1.0, 0.05, 0.0, (0.15 * SR) as usize);
    assert!(out.iter().all(|s| s.is_finite()), "VarLag must stay finite");
    assert!(out[0].abs() < 0.05, "starts at `start` (0), got {}", out[0]);
    let mid = (0.025 * SR) as usize; // halfway through the 50 ms ramp
    assert!(
        (0.3..0.7).contains(&out[mid]),
        "about halfway up at 25 ms, got {}",
        out[mid]
    );
    // Monotonic non-decreasing over the ramp.
    let end = (0.05 * SR) as usize;
    assert!(
        out[..end].windows(2).all(|w| w[1] >= w[0] - 1e-6),
        "the ramp is monotonic"
    );
    let late = (0.1 * SR) as usize;
    assert!(
        (out[late] - 1.0).abs() < 0.01,
        "holds at the input (1) after the ramp, got {}",
        out[late]
    );
}
