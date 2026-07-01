//! The resonant EQ/formant filters `MidEQ` (parametric peak/notch) and `Formlet` (formant resonator).

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

fn sine(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

/// The gain `<name>` applies to a `sig_freq` sine, measured as filtered energy / raw sine energy.
fn filter_gain(name: &str, coefs: &[f32], sig_freq: f32) -> f32 {
    let mut inputs = vec![u(0)];
    inputs.extend(coefs.iter().map(|&c| InputRef::Constant(c)));
    let filtered = render(
        vec![
            sine(sig_freq),
            UnitSpec::new(name, Rate::Audio, inputs, 1),
            out_unit(1),
        ],
        SR as usize / 4,
    );
    let raw = render(vec![sine(sig_freq), out_unit(0)], SR as usize / 4);
    goertzel(&filtered, sig_freq) / goertzel(&raw, sig_freq)
}

#[test]
fn mideq_is_transparent_at_zero_db() {
    let g = filter_gain("MidEQ", &[1000.0, 1.0, 0.0], 1000.0);
    assert!(
        (g - 1.0).abs() < 0.02,
        "0 dB MidEQ passes the signal (gain {g})"
    );
}

#[test]
fn mideq_boosts_and_cuts_its_band() {
    // +12 dB at 1000 Hz (~4x) boosts the band centre; a distant 200 Hz tone is left near unity.
    let boost_at = filter_gain("MidEQ", &[1000.0, 1.0, 12.0], 1000.0);
    let boost_off = filter_gain("MidEQ", &[1000.0, 1.0, 12.0], 200.0);
    assert!(
        boost_at > 2.5,
        "+12 dB should boost the centre (gain {boost_at})"
    );
    assert!(
        (boost_off - 1.0).abs() < 0.3,
        "outside the band is ~unchanged (gain {boost_off})"
    );
    // -12 dB (~0.25x) cuts the centre.
    let cut_at = filter_gain("MidEQ", &[1000.0, 1.0, -12.0], 1000.0);
    assert!(cut_at < 0.5, "-12 dB should cut the centre (gain {cut_at})");
}

#[test]
fn formlet_resonates_at_its_frequency() {
    // An impulse train through a Formlet at 800 Hz rings there.
    let units = vec![
        UnitSpec::new(
            "Impulse",
            Rate::Audio,
            vec![InputRef::Constant(8.0), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec::new(
            "Formlet",
            Rate::Audio,
            vec![
                u(0),
                InputRef::Constant(800.0),
                InputRef::Constant(0.005), // attack
                InputRef::Constant(0.08),  // decay
            ],
            1,
        ),
        out_unit(1),
    ];
    let out = render(units, SR as usize / 2);
    assert!(
        out.iter().all(|s| s.is_finite()),
        "Formlet must stay finite"
    );
    assert!(
        out.iter().all(|&s| s.abs() < 4.0),
        "Formlet should stay bounded"
    );
    let at = goertzel(&out, 800.0);
    assert!(at > 8.0 * goertzel(&out, 400.0), "rings at 800, not 400");
    assert!(at > 8.0 * goertzel(&out, 1600.0), "rings at 800, not 1600");
}
