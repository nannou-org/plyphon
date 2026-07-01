//! `MoogFF` (the Moog-ladder resonant low-pass) and `Median` (running-median spike rejection).

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

fn out_unit(src: u32) -> UnitSpec {
    UnitSpec::new("Out", Rate::Audio, vec![InputRef::Constant(0.0), u(src)], 0)
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
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail)
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

/// Energy at `sig_freq` after `MoogFF.ar(SinOsc.ar(sig_freq), freq, gain, 0)`.
fn moog_energy(sig_freq: f32, freq: f32, gain: f32) -> f32 {
    let out = render(
        vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(sig_freq), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "MoogFF",
                Rate::Audio,
                vec![
                    u(0),
                    InputRef::Constant(freq),
                    InputRef::Constant(gain),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out_unit(1),
        ],
        SR as usize / 4,
    );
    goertzel(&out, sig_freq)
}

#[test]
fn moogff_low_passes() {
    // 4-pole low-pass at 500 Hz, no resonance: passes 200, rejects 3000.
    let low = moog_energy(200.0, 500.0, 0.0);
    let high = moog_energy(3000.0, 500.0, 0.0);
    assert!(
        low > 10.0 * high,
        "MoogFF should low-pass (low={low}, high={high})"
    );
}

#[test]
fn moogff_resonates_with_gain() {
    // High feedback gain boosts the band around the cutoff.
    let flat = moog_energy(500.0, 500.0, 0.0);
    let reso = moog_energy(500.0, 500.0, 3.5);
    assert!(
        reso > 1.5 * flat,
        "resonance should boost the cutoff band (flat={flat}, reso={reso})"
    );
}

fn mul(src: u32, k: f32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![u(src), InputRef::Constant(k)],
        num_outputs: 1,
        special_index: 2,
    }
}

fn add_const(src: u32, k: f32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![u(src), InputRef::Constant(k)],
        num_outputs: 1,
        special_index: 0,
    }
}

#[test]
fn median_rejects_impulsive_spikes() {
    // A 0.3 DC baseline with sparse one-sample spikes to 1.0 (an Impulse * 0.7 + 0.3).
    let spiky = vec![
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(50.0), InputRef::Constant(0.0)],
            1,
        ),
        mul(0, 0.7),
        add_const(1, 0.3),
    ];

    // The raw signal has the 1.0 spikes.
    let mut raw_units = spiky.clone();
    raw_units.push(out_unit(2));
    let raw = render(raw_units, SR as usize / 4);
    assert!(
        raw.iter().cloned().fold(0.0f32, f32::max) > 0.9,
        "the input should carry 1.0 spikes"
    );

    // Median(5) rejects the isolated spikes, leaving the 0.3 baseline.
    let mut med_units = spiky;
    med_units.push(UnitSpec::new(
        "Median",
        Rate::Audio,
        vec![InputRef::Constant(5.0), u(2)],
        1,
    ));
    med_units.push(out_unit(3));
    let med = render(med_units, SR as usize / 4);
    // Skip the startup transient (the window seeds from the first sample, itself a spike).
    assert!(
        med[64..].iter().all(|&s| (0.25..0.35).contains(&s)),
        "Median should flatten to the 0.3 baseline, spikes removed"
    );
}
