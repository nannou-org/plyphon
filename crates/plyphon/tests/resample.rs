//! Intra-graph resampling (scsynth's `Resample(n)`): a def whose graph runs at `n`x the World sample
//! rate, the interior oversampled to anti-alias nonlinear units, the boundary `In` zero-order-holding
//! the World-rate bus up and `Out` decimating the oversampled interior back down (naive decimation,
//! as scsynth does). For a band-limited linear chain this is transparent - the decimated even samples
//! of the 2x sine are exactly the World-rate sine - which checks the ZOH/decimate mechanism end to
//! end. Invalid factors are rejected at compile.

use plyphon::{
    AddAction, BuildError, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, SynthNewError,
    UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + 64);
    let mut buf = vec![0.0f32; 64];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR as f32).floor();
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

/// `SinOsc.ar(220) -> Out.ar(0)` - a band-limited linear chain.
fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(220.0), InputRef::Constant(0.0)],
                1,
            ),
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
    }
}

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

#[test]
fn resample_is_transparent_for_a_band_limited_chain() {
    // A 2x-oversampled band-limited sine, decimated back to the World rate, equals the non-resampled
    // sine sample for sample: the even samples of the 2x sine are the World-rate sine.
    let plain = {
        let (mut c, _nrt, mut world) = engine(opts());
        c.add_synthdef(sine_def());
        c.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail).unwrap();
        render(&mut world, 4096)
    };
    let resampled = {
        let (mut c, _nrt, mut world) = engine(opts());
        c.add_synthdef_resampled(sine_def(), 2);
        c.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail).unwrap();
        render(&mut world, 4096)
    };

    assert!(
        plain.iter().any(|s| s.abs() > 0.1),
        "the reference sine was silent"
    );
    let max_diff = plain
        .iter()
        .zip(&resampled)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Not bit-exact: the oversampled oscillator accumulates phase in twice as many (finer) steps, so
    // it drifts from the World-rate one by a tiny float amount (here ~2e-4, about -75 dB, audibly
    // identical - and scsynth's 2x oscillator drifts the same way). What matters is that the ZOH /
    // decimate boundary preserves the band-limited sine, not noise.
    assert!(
        max_diff < 1e-3,
        "oversampling a band-limited chain must be transparent (max sample diff {max_diff})"
    );
}

#[test]
fn invalid_resample_factors_are_rejected() {
    // Not a power of two, and zero: both must fail at compile (surfaced by the first `synth_new`).
    for bad in [3usize, 0] {
        let (mut c, _nrt, _world) = engine(opts());
        c.add_synthdef_resampled(sine_def(), bad);
        let err = c
            .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
            .expect_err("an invalid resample factor must be rejected");
        assert!(
            matches!(
                err,
                SynthNewError::Build(BuildError::InvalidResample { .. })
            ),
            "expected InvalidResample for factor {bad}, got {err:?}"
        );
    }
}

#[test]
fn a_resampled_synth_runs_and_sounds_at_pitch() {
    // A 4x-oversampled synth produces finite, audible output dominated by its fundamental - the
    // sub-block loop, ZOH and decimate carry the signal end to end.
    let (mut c, _nrt, mut world) = engine(opts());
    c.add_synthdef_resampled(sine_def(), 4);
    c.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail).unwrap();
    let out = render(&mut world, 8192);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "resampled output was non-finite"
    );
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "resampled synth was silent"
    );
    assert!(
        goertzel(&out, 220.0) > 5.0 * goertzel(&out, 660.0),
        "the 220 Hz fundamental should dominate"
    );
}
