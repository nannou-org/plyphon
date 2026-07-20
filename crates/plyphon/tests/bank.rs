//! The additive/modal resonator banks: `Klang` (a fixed sum of sine partials) and `Klank` (a bank of
//! decaying resonators driven by an excitation input).

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

/// Render `frames` of a one-synth graph (its last unit routed to `Out`) after a short settle.
fn render(mut units: Vec<UnitSpec>, out_src: u32, frames: usize) -> Vec<f32> {
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
    controller.add_synthdef(SynthDef {
        name: "b".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("b", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = vec![0.0f32; frames];
    world.fill(&mut out, 1);
    out
}

/// A `Klang` unit from `(freqscale, freqoffset)` and interleaved `(freq, amp, phase)` triples.
fn klang(freqscale: f32, freqoffset: f32, partials: &[(f32, f32, f32)]) -> UnitSpec {
    let mut inputs = vec![
        InputRef::Constant(freqscale),
        InputRef::Constant(freqoffset),
    ];
    for &(f, a, p) in partials {
        inputs.push(InputRef::Constant(f));
        inputs.push(InputRef::Constant(a));
        inputs.push(InputRef::Constant(p));
    }
    UnitSpec::new("Klang", Rate::Audio, inputs, 1)
}

#[test]
fn klang_single_partial_is_a_sine() {
    let out = render(
        vec![klang(1.0, 0.0, &[(440.0, 1.0, 0.0)])],
        0,
        SR as usize / 4,
    );
    assert!(out.iter().all(|s| s.is_finite()), "Klang must stay finite");
    assert!(
        out.iter().all(|&s| s.abs() < 2.0),
        "Klang should stay bounded"
    );
    let at = goertzel(&out, 440.0);
    let off = goertzel(&out, 660.0);
    assert!(
        at > 20.0 * off,
        "should be a clean 440 Hz sine (440={at}, 660={off})"
    );
}

#[test]
fn klang_sums_partials_with_freqscale() {
    // Two partials at 440 (amp 1) and 880 (amp 0.5), with freqscale 2 so the written freqs halve.
    let out = render(
        vec![klang(2.0, 0.0, &[(220.0, 1.0, 0.0), (440.0, 0.5, 0.0)])],
        0,
        SR as usize / 4,
    );
    let f1 = goertzel(&out, 440.0);
    let f2 = goertzel(&out, 880.0);
    let mid = goertzel(&out, 600.0);
    assert!(f1 > 10.0 * mid, "energy at the first partial (440)");
    assert!(f2 > 10.0 * mid, "energy at the second partial (880)");
    assert!(f1 > f2, "the louder partial (amp 1 vs 0.5) dominates");
}

/// A `Klank` unit: `input` unit index, `(freqscale, freqoffset, decayscale)`, and `(freq, amp, ring)`
/// triples.
fn klank(input: u32, scales: (f32, f32, f32), partials: &[(f32, f32, f32)]) -> UnitSpec {
    let mut inputs = vec![
        InputRef::Unit {
            unit: input,
            output: 0,
        },
        InputRef::Constant(scales.0),
        InputRef::Constant(scales.1),
        InputRef::Constant(scales.2),
    ];
    for &(f, a, r) in partials {
        inputs.push(InputRef::Constant(f));
        inputs.push(InputRef::Constant(a));
        inputs.push(InputRef::Constant(r));
    }
    UnitSpec::new("Klank", Rate::Audio, inputs, 1)
}

#[test]
fn klank_rings_at_its_mode() {
    // An Impulse clock excites a single 440 Hz mode; the bank rings there.
    let units = vec![
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(4.0), InputRef::Constant(0.0)],
            1,
        ),
        klank(0, (1.0, 0.0, 1.0), &[(440.0, 1.0, 0.6)]),
    ];
    let out = render(units, 1, SR as usize / 2);
    assert!(out.iter().all(|s| s.is_finite()), "Klank must stay finite");
    assert!(
        out.iter().all(|&s| s.abs() < 8.0),
        "Klank should stay bounded"
    );
    let at = goertzel(&out, 440.0);
    let off = goertzel(&out, 660.0);
    assert!(at > 8.0 * off, "should ring at 440 (440={at}, 660={off})");
}

#[test]
fn klank_rings_multiple_modes() {
    let units = vec![
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(3.0), InputRef::Constant(0.0)],
            1,
        ),
        klank(0, (1.0, 0.0, 1.0), &[(300.0, 1.0, 0.5), (700.0, 0.8, 0.5)]),
    ];
    let out = render(units, 1, SR as usize / 2);
    let m1 = goertzel(&out, 300.0);
    let m2 = goertzel(&out, 700.0);
    let off = goertzel(&out, 500.0);
    assert!(m1 > 5.0 * off, "should ring at 300 (300={m1}, 500={off})");
    assert!(m2 > 5.0 * off, "should ring at 700 (700={m2}, 500={off})");
}
