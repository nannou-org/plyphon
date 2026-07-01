//! The exponential-lag cascades: `Lag2`/`Lag3` (progressively smoother low-passes) and the asymmetric
//! `LagUD`/`Lag2UD`/`Lag3UD` (separate rise/fall smoothing).

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

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

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + 256);
    let mut buf = vec![0.0f32; 256];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `<name>.ar(SinOsc.ar(sig_freq), coefs...) -> Out`.
fn render_filter(name: &str, sig_freq: f32, coefs: &[f32]) -> Vec<f32> {
    let mut inputs = vec![InputRef::Unit { unit: 0, output: 0 }];
    inputs.extend(coefs.iter().map(|&c| InputRef::Constant(c)));
    let units = vec![
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(sig_freq), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec::new(name, Rate::Audio, inputs, 1),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 1, output: 0 },
            ],
            0,
        ),
    ];
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
    render(&mut world, SR as usize / 4)
}

fn mean(samples: &[f32]) -> f32 {
    samples.iter().sum::<f32>() / samples.len() as f32
}

#[test]
fn lag_cascades_low_pass_progressively() {
    // A 2000 Hz sine through Lag / Lag2 / Lag3 (lagTime 0.005 s): each extra one-pole stage attenuates
    // the high tone more.
    let e1 = goertzel(&render_filter("Lag", 2000.0, &[0.005]), 2000.0);
    let e2 = goertzel(&render_filter("Lag2", 2000.0, &[0.005]), 2000.0);
    let e3 = goertzel(&render_filter("Lag3", 2000.0, &[0.005]), 2000.0);
    assert!(e2 < e1, "Lag2 attenuates more than Lag ({e2} < {e1})");
    assert!(e3 < e2, "Lag3 attenuates more than Lag2 ({e3} < {e2})");
    // A low sine passes through Lag3 far more than the high one is rejected.
    let low = goertzel(&render_filter("Lag3", 50.0, &[0.005]), 50.0);
    assert!(low > 20.0 * e3, "Lag3 passes 50 Hz far more than 2000 Hz");
}

#[test]
fn lag_ud_variants_track_asymmetrically() {
    // lagU = 0 (instant rise), lagD = 0.05 (slow fall): a peak follower whose mean sits above 0.
    // lagU = 0.05, lagD = 0 (instant fall): a trough follower whose mean sits below 0.
    for name in ["LagUD", "Lag2UD", "Lag3UD"] {
        let up = render_filter(name, 200.0, &[0.0, 0.05]);
        let down = render_filter(name, 200.0, &[0.05, 0.0]);
        assert!(up.iter().all(|s| s.is_finite()), "{name} must stay finite");
        assert!(
            mean(&up) > 0.1,
            "{name} fast-up should track the peaks (mean {})",
            mean(&up)
        );
        assert!(
            mean(&down) < -0.1,
            "{name} fast-down should track the troughs (mean {})",
            mean(&down)
        );
    }
}
