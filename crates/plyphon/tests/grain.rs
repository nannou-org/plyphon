//! Granular synthesis: `GrainSin` spawns windowed sine grains on a trigger and pans them across the
//! output channels.

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

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Render `units` (interleaved) into `channels` output channels.
fn render(units: Vec<UnitSpec>, channels: usize, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "g".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("g", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; frames * channels];
    world.fill(&mut out, channels);
    out
}

fn impulse(freq: f32) -> UnitSpec {
    UnitSpec::new("Impulse", Rate::Audio, vec![c(freq), c(0.0)], 1)
}

/// `GrainSin(numChannels, Impulse(unit 0), dur, freq, pan, envbufnum=-1, maxGrains=32)`.
fn grain_sin(num_ch: usize, freq: f32, pan: f32) -> UnitSpec {
    UnitSpec::new(
        "GrainSin",
        Rate::Audio,
        vec![u(0), c(0.03), c(freq), c(pan), c(-1.0), c(32.0)],
        num_ch,
    )
}

#[test]
fn grain_sin_rings_at_the_grain_frequency() {
    // A 20 Hz impulse train triggers 440 Hz windowed sine grains.
    let out = render(
        vec![
            impulse(20.0),
            grain_sin(1, 440.0, 0.0),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 4,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "GrainSin must stay finite"
    );
    assert!(
        out.iter().all(|&s| s.abs() < 3.0),
        "windowed grains stay bounded"
    );
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 660.0),
        "grains should ring at 440 (440={at})"
    );
    // The grains actually sound (they are not all silent).
    assert!(at > 0.02, "grains should be audible (440={at})");
}

/// Per-channel energy at 440 Hz for a stereo `GrainSin` panned by `pan`.
fn stereo_energy(pan: f32) -> (f32, f32) {
    let out = render(
        vec![
            impulse(20.0),
            grain_sin(2, 440.0, pan),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    c(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 1, output: 1 },
                ],
                0,
            ),
        ],
        2,
        SR as usize / 4,
    );
    let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
    let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
    (goertzel(&ch0, 440.0), goertzel(&ch1, 440.0))
}

#[test]
fn grain_sin_pans_across_channels() {
    // pan = -1 puts the grains in channel 0; pan = +1 in channel 1.
    let (l0, l1) = stereo_energy(-1.0);
    let (r0, r1) = stereo_energy(1.0);
    assert!(
        l0 > 5.0 * l1,
        "pan -1 favours channel 0 (ch0={l0}, ch1={l1})"
    );
    assert!(
        r1 > 5.0 * r0,
        "pan +1 favours channel 1 (ch0={r0}, ch1={r1})"
    );
}

#[test]
fn grain_fm_has_carrier_and_sidebands() {
    // A long FM grain (carrier 440, modulator 110, index 2) resolves the full FM sideband series at
    // 440 +/- k*110; at index 2 the first sidebands (330, 550) exceed the carrier (Bessel J1 > J0).
    let out = render(
        vec![
            impulse(1.0), // a single, long grain over the render
            UnitSpec::new(
                "GrainFM",
                Rate::Audio,
                vec![
                    u(0),
                    c(0.2),
                    c(440.0),
                    c(110.0),
                    c(2.0),
                    c(0.0),
                    c(-1.0),
                    c(32.0),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 4,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "GrainFM must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 3.0), "GrainFM stays bounded");
    // 495 Hz sits between the 440 and 550 components (an ~empty bin).
    let off = goertzel(&out, 495.0);
    let carrier = goertzel(&out, 440.0);
    let side = goertzel(&out, 550.0);
    assert!(
        carrier > 10.0 * off,
        "the carrier should sound (440={carrier}, 495={off})"
    );
    assert!(
        side > 10.0 * off,
        "the 550 Hz FM sideband should appear (550={side}, 495={off})"
    );
    assert!(
        side > carrier,
        "at index 2 the first sideband exceeds the carrier (550={side}, 440={carrier})"
    );
}

#[test]
fn grain_in_windows_the_input() {
    // GrainIn windows a live 440 Hz sine, so its grains ring at 440.
    let out = render(
        vec![
            impulse(20.0),
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            UnitSpec::new(
                "GrainIn",
                Rate::Audio,
                vec![u(0), c(0.03), u(1), c(0.0), c(-1.0), c(32.0)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
        ],
        1,
        SR as usize / 4,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "GrainIn must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 3.0), "GrainIn stays bounded");
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 660.0),
        "windows the 440 input (440={at})"
    );
    assert!(at > 0.02, "the windowed grains should sound");
}
