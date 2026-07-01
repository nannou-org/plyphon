//! The wavetable oscillators `Osc` (interpolating) and `OscN` (truncating), reading a buffer filled
//! in scsynth's `(a, b)` wavetable format (via `to_wavetable`).

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
    to_wavetable,
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

/// One cycle of `f` sampled at `n` points, packed into scsynth's `(a, b)` wavetable format.
fn wavetable(n: usize, f: impl Fn(f32) -> f32) -> Vec<f32> {
    let samples: Vec<f32> = (0..n)
        .map(|i| f(std::f32::consts::TAU * i as f32 / n as f32))
        .collect();
    to_wavetable(&samples)
}

/// Render `frames` of a one-synth graph (its `out_src` unit routed to `Out`) with an optional buffer
/// installed at index 0.
fn render(
    mut units: Vec<UnitSpec>,
    out_src: u32,
    buffer: Option<Vec<f32>>,
    frames: usize,
) -> Vec<f32> {
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: out_src,
                output: 0,
            },
        ],
        0,
    ));
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    if let Some(data) = buffer {
        controller
            .buffer_set(0, Box::new(Buffer::from_interleaved(data, 1, SR as f64)))
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
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}

/// An oscillator (`Osc`/`OscN`) reading buffer 0 at `freq` with no phase offset.
fn osc(name: &str, freq: f32) -> UnitSpec {
    UnitSpec::new(
        name,
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Constant(freq),
            InputRef::Constant(0.0),
        ],
        1,
    )
}

#[test]
fn osc_plays_a_sine_wavetable() {
    let table = wavetable(1024, f32::sin);
    let out = render(vec![osc("Osc", 440.0)], 0, Some(table), SR as usize / 4);
    assert!(out.iter().all(|s| s.is_finite()), "Osc must stay finite");
    assert!(
        out.iter().all(|&s| s.abs() < 2.0),
        "Osc should stay bounded"
    );
    let at = goertzel(&out, 440.0);
    let off = goertzel(&out, 660.0);
    assert!(
        at > 20.0 * off,
        "should be a clean 440 Hz sine (440={at}, 660={off})"
    );
}

#[test]
fn osc_sums_wavetable_partials() {
    // A wavetable holding a fundamental plus a half-amplitude third harmonic.
    let table = wavetable(1024, |p| p.sin() + 0.5 * (3.0 * p).sin());
    let out = render(vec![osc("Osc", 300.0)], 0, Some(table), SR as usize / 4);
    let f1 = goertzel(&out, 300.0);
    let f3 = goertzel(&out, 900.0);
    let mid = goertzel(&out, 600.0);
    assert!(f1 > 10.0 * mid, "energy at the fundamental (300)");
    assert!(f3 > 10.0 * mid, "energy at the third harmonic (900)");
    assert!(f1 > f3, "the fundamental (amp 1 vs 0.5) dominates");
}

#[test]
fn osc_missing_buffer_is_silent() {
    let out = render(vec![osc("Osc", 440.0)], 0, None, 256);
    assert!(out.iter().all(|&s| s == 0.0), "a missing buffer is silent");
}

#[test]
fn osc_non_power_of_two_wavetable_is_silent() {
    // 1000 logical samples -> 2000 frames, not a power of two: scsynth (and plyphon) zero the output.
    let table = wavetable(1000, f32::sin);
    assert_eq!(table.len(), 2000);
    let out = render(vec![osc("Osc", 440.0)], 0, Some(table), 256);
    assert!(
        out.iter().all(|&s| s == 0.0),
        "a non-power-of-two wavetable is silent"
    );
}

#[test]
fn oscn_plays_a_plain_buffer() {
    // OscN reads a plain (non-wavetable-format) buffer, so pass raw samples, not `to_wavetable`.
    let plain: Vec<f32> = (0..1024)
        .map(|i| (std::f32::consts::TAU * i as f32 / 1024.0).sin())
        .collect();
    let out = render(vec![osc("OscN", 220.0)], 0, Some(plain), SR as usize / 4);
    assert!(out.iter().all(|s| s.is_finite()), "OscN must stay finite");
    assert!(
        out.iter().all(|&s| s.abs() < 2.0),
        "OscN should stay bounded"
    );
    let at = goertzel(&out, 220.0);
    let off = goertzel(&out, 330.0);
    assert!(
        at > 8.0 * off,
        "should carry a 220 Hz tone (220={at}, 330={off})"
    );
}
