//! The low-frequency and dynamic noise generators: `LFNoise0/1/2`, `LFClipNoise`, `LFDNoise0/1/3`
//! and `LFDClipNoise`. Each emits a new random value at an average `freq`; the tests check the
//! output stays bounded, changes at roughly the right rate, and has the right between-value shape
//! (held step, linear ramp, or smooth curve).

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

/// Render a one-unit graph (`name.ar(inputs) -> Out`) for one second after a short settle, across
/// varied block sizes to shake out block-boundary bugs.
fn render(units: Vec<UnitSpec>, frames: usize, settle: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "n".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("n", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = Vec::with_capacity(frames + settle + 512);
    let mut buf = Vec::new();
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut i = 0;
    while out.len() < frames + settle {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.split_off(settle).into_iter().take(frames).collect()
}

/// `name.ar(freq) -> Out`, one second after a 0.02 s settle.
fn noise_out(name: &str, freq: f32) -> Vec<f32> {
    let units = vec![
        UnitSpec::new(name, Rate::Audio, vec![InputRef::Constant(freq)], 1),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 0, output: 0 },
            ],
            0,
        ),
    ];
    render(units, SR as usize, SR as usize / 50)
}

/// Count positions where the value changes from the previous sample (segment boundaries for a
/// piecewise-constant signal).
fn changes(out: &[f32]) -> usize {
    out.windows(2).filter(|w| w[0] != w[1]).count()
}

/// The largest jump between consecutive samples.
fn max_jump(out: &[f32]) -> f32 {
    out.windows(2)
        .fold(0.0f32, |m, w| m.max((w[1] - w[0]).abs()))
}

#[test]
fn lf_noise0_holds_random_steps() {
    // LFNoise0.ar(1000) at 48 kHz holds each value for ~48 samples, so ~1000 steps per second.
    let out = noise_out("LFNoise0", 1000.0);
    assert!(
        out.iter().all(|&x| (-1.0..1.0).contains(&x)),
        "LFNoise0 out of range"
    );
    let n = changes(&out);
    assert!(
        (700..1300).contains(&n),
        "LFNoise0(1000) should step ~1000 times/s, got {n}"
    );
}

#[test]
fn lf_clip_noise_is_plus_or_minus_one() {
    let out = noise_out("LFClipNoise", 1000.0);
    assert!(
        out.iter().all(|&x| x == 1.0 || x == -1.0),
        "LFClipNoise must be exactly ±1"
    );
    let n = changes(&out);
    assert!(
        (300..1300).contains(&n),
        "LFClipNoise(1000) should switch on the order of 1000 times/s, got {n}"
    );
}

#[test]
fn lf_noise1_ramps_without_jumps() {
    // LFNoise1.ar(500): straight-line ramps between random targets, so no large sample-to-sample
    // jumps and the signal is anything but constant.
    let out = noise_out("LFNoise1", 500.0);
    assert!(out.iter().all(|s| s.is_finite()), "LFNoise1 not finite");
    assert!(out.iter().all(|&x| x.abs() < 1.1), "LFNoise1 out of range");
    assert!(max_jump(&out) < 0.1, "LFNoise1 should ramp, not jump");
    let (lo, hi) = out
        .iter()
        .fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(hi - lo > 0.3, "LFNoise1 should vary, span {}", hi - lo);
}

#[test]
fn lf_noise2_is_smooth_and_bounded() {
    // LFNoise2.ar(500): quadratic interpolation - smooth (small steps) and roughly bounded (it can
    // overshoot ±1 a little, as in scsynth).
    let out = noise_out("LFNoise2", 500.0);
    assert!(out.iter().all(|s| s.is_finite()), "LFNoise2 not finite");
    assert!(
        out.iter().all(|&x| x.abs() < 2.0),
        "LFNoise2 wildly out of range"
    );
    assert!(max_jump(&out) < 0.1, "LFNoise2 should be smooth");
    assert!(
        changes(&out) > out.len() / 2,
        "LFNoise2 should vary continuously"
    );
}

#[test]
fn lf_dnoise0_holds_random_steps() {
    // The dynamic step generator: a floating phase, so transitions land off the sample grid, but the
    // average rate still tracks freq.
    let out = noise_out("LFDNoise0", 1000.0);
    assert!(out.iter().all(|&x| x.abs() < 1.0), "LFDNoise0 out of range");
    let n = changes(&out);
    assert!(
        (700..1300).contains(&n),
        "LFDNoise0(1000) should step ~1000 times/s, got {n}"
    );
}

#[test]
fn lf_dclip_noise_is_plus_or_minus_one() {
    let out = noise_out("LFDClipNoise", 1000.0);
    assert!(
        out.iter().all(|&x| x == 1.0 || x == -1.0),
        "LFDClipNoise must be exactly ±1"
    );
}

#[test]
fn lf_dnoise1_ramps_without_jumps() {
    let out = noise_out("LFDNoise1", 500.0);
    assert!(out.iter().all(|s| s.is_finite()), "LFDNoise1 not finite");
    assert!(out.iter().all(|&x| x.abs() < 1.1), "LFDNoise1 out of range");
    assert!(max_jump(&out) < 0.1, "LFDNoise1 should ramp, not jump");
}

#[test]
fn lf_dnoise3_is_smooth_and_capped() {
    // LFDNoise3 scales each random value by 0.8 so the cubic overshoot cannot exceed 1.
    let out = noise_out("LFDNoise3", 500.0);
    assert!(out.iter().all(|s| s.is_finite()), "LFDNoise3 not finite");
    assert!(
        out.iter().all(|&x| x.abs() < 1.05),
        "LFDNoise3 should stay within ±1"
    );
    assert!(max_jump(&out) < 0.1, "LFDNoise3 should be smooth");
}

#[test]
fn lf_dnoise0_accepts_audio_rate_frequency() {
    // The whole point of the LFD* family: freq may be modulated at audio rate. Feed an audio-rate
    // DC(1000) and confirm it still steps at ~1000/s (exercising the per-sample freq read).
    let units = vec![
        UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1000.0)], 1),
        UnitSpec::new(
            "LFDNoise0",
            Rate::Audio,
            vec![InputRef::Unit { unit: 0, output: 0 }],
            1,
        ),
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
    let out = render(units, SR as usize, SR as usize / 50);
    let n = changes(&out);
    assert!(
        (700..1300).contains(&n),
        "audio-rate freq LFDNoise0 should step ~1000/s, got {n}"
    );
}
