//! `GVerb`: a large Griesinger-style FDN reverb - four recirculating delay lines mixed by a Hadamard
//! matrix, with early-reflection taps and a diffused stereo tail.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

fn out_ch(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
}

fn rms(s: &[f32]) -> f32 {
    if s.is_empty() {
        return 0.0;
    }
    (s.iter().map(|&x| x * x).sum::<f32>() / s.len() as f32).sqrt()
}

/// `Impulse(2) -> GVerb(...) -> Out.ar(0, [L, R])`, rendered as stereo.
fn render(dry: f32, revtime: f32, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 2,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "g".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("Impulse", Rate::Audio, vec![c(2.0), c(0.0)], 1),
            UnitSpec::new(
                "GVerb",
                Rate::Audio,
                vec![
                    u(0),       // in
                    c(10.0),    // roomsize (const)
                    c(revtime), // revtime
                    c(0.5),     // damping
                    c(0.5),     // inputbw
                    c(15.0),    // spread (const)
                    c(dry),     // drylevel
                    c(0.7),     // earlyreflevel
                    c(0.5),     // taillevel
                    c(300.0),   // maxroomsize (const)
                ],
                2,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![c(0.0), out_ch(1, 0), out_ch(1, 1)],
                0,
            ),
        ],
    });
    controller
        .synth_new("g", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; frames * 2];
    world.fill(&mut out, 2);
    out
}

#[test]
fn gverb_produces_a_decaying_stereo_tail() {
    let out = render(0.0, 0.8, SR as usize);
    assert!(out.iter().all(|s| s.is_finite()), "GVerb must stay finite");
    assert!(out.iter().all(|&s| s.abs() < 8.0), "GVerb stays bounded");

    let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
    let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
    assert!(
        rms(&ch0) > 1e-4 && rms(&ch1) > 1e-4,
        "both reverb channels should sound (l={}, r={})",
        rms(&ch0),
        rms(&ch1)
    );
    // A diffuse stereo field: the two channels are not identical.
    assert!(
        ch0.iter().zip(&ch1).any(|(a, b)| (a - b).abs() > 1e-4),
        "the two reverb channels should differ (stereo)"
    );
    // The tail decays: the last eighth is quieter than the loud early region.
    let n = ch0.len();
    let early = rms(&ch0[n / 8..n / 4]);
    let late = rms(&ch0[7 * n / 8..]);
    assert!(
        early > 2.0 * late,
        "the reverb tail should decay (early={early}, late={late})"
    );
}

#[test]
fn gverb_dry_level_passes_input() {
    // With a high dry level and (relatively) quiet tail, the impulses in the input are clearly present
    // in the output. The dry path adds `x * drylevel` to both channels.
    let out = render(1.0, 0.5, SR as usize / 4);
    assert!(out.iter().all(|s| s.is_finite()), "GVerb must stay finite");
    // The dry impulses make the output peak near the impulse amplitude (>= drylevel * 1.0 region).
    let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    assert!(
        peak > 0.5,
        "the dry impulses should come through (peak={peak})"
    );
}
