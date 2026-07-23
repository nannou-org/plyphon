//! The BEQSuite RBJ biquads `BLowPass`, `BHiPass`, `BBandPass`, `BPeakEQ`, `BLowShelf` and
//! `BHiShelf`: exact-unity identity settings, passband/stopband shapes, and boost/cut magnitudes.

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
fn peak_eq_is_transparent_at_zero_db() {
    // With db = 0 the numerator equals the denominator, so BPeakEQ is an exact passthrough.
    for sig in [200.0, 1000.0] {
        let g = filter_gain("BPeakEQ", &[1000.0, 1.0, 0.0], sig);
        assert!(
            (g - 1.0).abs() < 0.02,
            "0 dB BPeakEQ passes {sig} Hz (gain {g})"
        );
    }
}

#[test]
fn shelves_are_transparent_at_zero_db() {
    for sig in [100.0, 8000.0] {
        let lo = filter_gain("BLowShelf", &[1000.0, 1.0, 0.0], sig);
        assert!(
            (lo - 1.0).abs() < 0.02,
            "0 dB BLowShelf passes {sig} Hz (gain {lo})"
        );
        let hi = filter_gain("BHiShelf", &[1000.0, 1.0, 0.0], sig);
        assert!(
            (hi - 1.0).abs() < 0.02,
            "0 dB BHiShelf passes {sig} Hz (gain {hi})"
        );
    }
}

#[test]
fn low_pass_passes_lows_and_cuts_highs() {
    let low = filter_gain("BLowPass", &[1000.0, 1.0], 100.0);
    assert!(
        (low - 1.0).abs() < 0.06,
        "100 Hz through a 1 kHz BLowPass is near unity (gain {low})"
    );
    let high = filter_gain("BLowPass", &[1000.0, 1.0], 10_000.0);
    assert!(
        high < 0.02,
        "10 kHz through a 1 kHz BLowPass is strongly attenuated (gain {high})"
    );
}

#[test]
fn hi_pass_passes_highs_and_cuts_lows() {
    let high = filter_gain("BHiPass", &[1000.0, 1.0], 10_000.0);
    assert!(
        (high - 1.0).abs() < 0.06,
        "10 kHz through a 1 kHz BHiPass is near unity (gain {high})"
    );
    let low = filter_gain("BHiPass", &[1000.0, 1.0], 100.0);
    assert!(
        low < 0.02,
        "100 Hz through a 1 kHz BHiPass is strongly attenuated (gain {low})"
    );
}

#[test]
fn peak_eq_boosts_and_cuts_by_the_requested_db() {
    // +12 dB at the centre is a 10^(12/20) ~ 3.98x amplitude boost; -12 dB the reciprocal cut.
    let want = 10f32.powf(12.0 / 20.0);
    let boost = filter_gain("BPeakEQ", &[1000.0, 1.0, 12.0], 1000.0);
    assert!(
        (boost / want - 1.0).abs() < 0.05,
        "+12 dB BPeakEQ boosts its centre by ~{want} (gain {boost})"
    );
    let cut = filter_gain("BPeakEQ", &[1000.0, 1.0, -12.0], 1000.0);
    assert!(
        (cut * want - 1.0).abs() < 0.05,
        "-12 dB BPeakEQ cuts its centre to ~{} (gain {cut})",
        1.0 / want
    );
}

#[test]
fn band_pass_is_unity_at_centre_and_attenuates_off_centre() {
    let centre = filter_gain("BBandPass", &[1000.0, 1.0], 1000.0);
    assert!(
        (centre - 1.0).abs() < 0.03,
        "BBandPass is unity at its centre (gain {centre})"
    );
    // A one-octave band centred on 1 kHz: tones >= 2 octaves off-centre fall well outside it.
    let below = filter_gain("BBandPass", &[1000.0, 1.0], 200.0);
    assert!(
        below < 0.25,
        "200 Hz falls outside a one-octave 1 kHz band (gain {below})"
    );
    let above = filter_gain("BBandPass", &[1000.0, 1.0], 4000.0);
    assert!(
        above < 0.25,
        "4 kHz falls outside a one-octave 1 kHz band (gain {above})"
    );
}

#[test]
fn shelves_boost_their_side_by_the_requested_db() {
    // Far below (above) the corner a +12 dB low (high) shelf reaches its full 10^(12/20) ~ 3.98x
    // plateau, while the opposite side stays near unity.
    let want = 10f32.powf(12.0 / 20.0);
    let lo = filter_gain("BLowShelf", &[1000.0, 1.0, 12.0], 100.0);
    assert!(
        (lo / want - 1.0).abs() < 0.06,
        "+12 dB BLowShelf boosts 100 Hz by ~{want} (gain {lo})"
    );
    let lo_far = filter_gain("BLowShelf", &[1000.0, 1.0, 12.0], 8000.0);
    assert!(
        (lo_far - 1.0).abs() < 0.06,
        "+12 dB BLowShelf leaves 8 kHz near unity (gain {lo_far})"
    );
    let hi = filter_gain("BHiShelf", &[1000.0, 1.0, 12.0], 8000.0);
    assert!(
        (hi / want - 1.0).abs() < 0.06,
        "+12 dB BHiShelf boosts 8 kHz by ~{want} (gain {hi})"
    );
    let hi_far = filter_gain("BHiShelf", &[1000.0, 1.0, 12.0], 100.0);
    assert!(
        (hi_far - 1.0).abs() < 0.06,
        "+12 dB BHiShelf leaves 100 Hz near unity (gain {hi_far})"
    );
}
