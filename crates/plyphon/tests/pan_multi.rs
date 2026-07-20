//! The multichannel / ambisonic panners: `Pan4`, `PanAz`, `PanB`, `PanB2`, `BiPanB2`, `DecodeB2`.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// Render `units` into `channels` output channels, returning per-channel RMS.
fn channel_rms(units: Vec<UnitSpec>, channels: usize) -> Vec<f32> {
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
        .synth_new("p", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let frames = 512;
    let mut out = vec![0.0f32; frames * channels];
    world.fill(&mut out, channels);
    (0..channels)
        .map(|ch| {
            let s: f32 = out.iter().skip(ch).step_by(channels).map(|&x| x * x).sum();
            (s / frames as f32).sqrt()
        })
        .collect()
}

/// An `Out.ar(0, [pan.0, pan.1, ...])` routing the panner's `n` outputs to the bus.
fn out_n(pan: u32, n: usize) -> UnitSpec {
    let mut inputs = vec![c(0.0)];
    inputs.extend((0..n).map(|k| InputRef::Unit {
        unit: pan,
        output: k as u32,
    }));
    UnitSpec::new("Out", Rate::Audio, inputs, 0)
}

#[test]
fn pan4_places_the_corner() {
    // A DC signal at the front-left corner (xpos -1, ypos +1) lands entirely in channel 0 (LF).
    let rms = channel_rms(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("Pan4", Rate::Audio, vec![u(0), c(-1.0), c(1.0), c(1.0)], 4),
            out_n(1, 4),
        ],
        4,
    );
    assert!(
        (rms[0] - 1.0).abs() < 1e-4,
        "front-left corner in ch0: {rms:?}"
    );
    assert!(
        rms[1] < 1e-4 && rms[2] < 1e-4 && rms[3] < 1e-4,
        "no energy elsewhere: {rms:?}"
    );
}

#[test]
fn panb_encodes_azimuth() {
    // At azimuth 0 the source is straight ahead: W (omni) and X (front-back) sound, Y (left-right) is
    // silent. At azimuth 0.5 (a quarter turn) it swings to the side: Y sounds, X is silent.
    let ahead = channel_rms(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("PanB", Rate::Audio, vec![u(0), c(0.0), c(0.0), c(1.0)], 4),
            out_n(1, 4),
        ],
        4,
    );
    assert!(ahead[0] > 0.5, "W is always present: {ahead:?}");
    assert!(ahead[1] > 0.9, "X (front) full at azimuth 0: {ahead:?}");
    assert!(ahead[2] < 1e-4, "Y silent straight ahead: {ahead:?}");

    let side = channel_rms(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("PanB", Rate::Audio, vec![u(0), c(0.5), c(0.0), c(1.0)], 4),
            out_n(1, 4),
        ],
        4,
    );
    assert!(side[2] > 0.9, "Y full at a quarter turn: {side:?}");
    assert!(side[1] < 1e-4, "X silent at a quarter turn: {side:?}");
}

#[test]
fn panb2_and_bipanb2_encode_planar() {
    // PanB2: azimuth 0 -> W + X, no Y.
    let b2 = channel_rms(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("PanB2", Rate::Audio, vec![u(0), c(0.0), c(1.0)], 3),
            out_n(1, 3),
        ],
        3,
    );
    assert!(
        b2[0] > 0.5 && b2[1] > 0.9 && b2[2] < 1e-4,
        "PanB2 ahead: {b2:?}"
    );

    // BiPanB2: inA = 1, inB = 0 -> W = (a+b)*0.707, X = (a-b)*cos, Y = (a-b)*sin. At azimuth 0, Y = 0.
    let bi = channel_rms(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1),
            UnitSpec::new("BiPanB2", Rate::Audio, vec![u(0), u(1), c(0.0), c(1.0)], 3),
            out_n(2, 3),
        ],
        3,
    );
    assert!(
        bi[0] > 0.5 && bi[1] > 0.9 && bi[2] < 1e-4,
        "BiPanB2 ahead: {bi:?}"
    );
}

#[test]
fn decodeb2_spreads_omni_evenly() {
    // Decoding a pure-W (omnidirectional) B-format signal to 4 speakers gives equal level in every
    // speaker (W has no directional bias); X and Y are silent.
    let rms = channel_rms(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1), // W
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1), // X
            UnitSpec::new("DC", Rate::Audio, vec![c(0.0)], 1), // Y
            UnitSpec::new("DecodeB2", Rate::Audio, vec![u(0), u(1), u(2), c(0.5)], 4),
            out_n(3, 4),
        ],
        4,
    );
    let first = rms[0];
    assert!(first > 0.1, "the omni source should sound: {rms:?}");
    assert!(
        rms.iter().all(|&r| (r - first).abs() < 1e-4),
        "an omni source decodes evenly to every speaker: {rms:?}"
    );
}

#[test]
fn panaz_localises_around_the_ring() {
    // Panning a DC signal to one position on a 4-speaker ring: the energy is localized (not equal in
    // all speakers) and the whole thing stays finite and bounded.
    let rms = channel_rms(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            // PanAz(numChans=4): in, pos, level, width, orientation.
            UnitSpec::new(
                "PanAz",
                Rate::Audio,
                vec![u(0), c(0.0), c(1.0), c(2.0), c(0.5)],
                4,
            ),
            out_n(1, 4),
        ],
        4,
    );
    assert!(
        rms.iter().all(|r| r.is_finite()),
        "PanAz stays finite: {rms:?}"
    );
    let total: f32 = rms.iter().sum();
    assert!(total > 0.1, "the panned source should sound: {rms:?}");
    let max = rms.iter().cloned().fold(0.0f32, f32::max);
    let min = rms.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        max > min + 0.1,
        "PanAz should localise (not equal in all speakers): {rms:?}"
    );
}
