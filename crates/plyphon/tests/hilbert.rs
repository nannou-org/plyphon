//! `Hilbert` (analytic-pair phase splitter) and `FreqShift` (single-sideband frequency shifter).

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

fn u_out(unit: u32, output: u32) -> InputRef {
    InputRef::Unit { unit, output }
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
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail, &[])
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

/// One of `Hilbert.ar(SinOsc.ar(sig_freq))`'s two outputs, routed to `Out`.
fn hilbert_out(sig_freq: f32, output: u32) -> Vec<f32> {
    render(
        vec![
            sine(sig_freq),
            UnitSpec::new("Hilbert", Rate::Audio, vec![u_out(0, 0)], 2),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![InputRef::Constant(0.0), u_out(1, output)],
                0,
            ),
        ],
        SR as usize / 4,
    )
}

#[test]
fn hilbert_outputs_a_quadrature_pair() {
    // The two outputs are ~90 degrees apart, so for a sine their squares sum to a nearly constant
    // analytic magnitude (unlike a raw sine, whose square swings between 0 and 1).
    let real = hilbert_out(1000.0, 0);
    let imag = hilbert_out(1000.0, 1);
    let n = real.len();
    let mag: Vec<f32> = (2000..n)
        .map(|i| real[i] * real[i] + imag[i] * imag[i])
        .collect();
    let max = mag.iter().cloned().fold(0.0f32, f32::max);
    let min = mag.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        min > 0.15,
        "the analytic magnitude should be nonzero (min={min})"
    );
    assert!(
        min > 0.4 * max,
        "the analytic magnitude should be roughly constant (min={min}, max={max})"
    );
}

/// `FreqShift.ar(SinOsc.ar(sig_freq), shift, 0) -> Out`.
fn render_freqshift(sig_freq: f32, shift: f32) -> Vec<f32> {
    render(
        vec![
            sine(sig_freq),
            UnitSpec::new(
                "FreqShift",
                Rate::Audio,
                vec![
                    u_out(0, 0),
                    InputRef::Constant(shift),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![InputRef::Constant(0.0), u_out(1, 0)],
                0,
            ),
        ],
        SR as usize / 4,
    )
}

#[test]
fn freqshift_is_single_sideband() {
    // A 1000 Hz sine shifted by 200 Hz lands on one sideband (800 or 1200), suppressing the other and
    // the carrier.
    let out = render_freqshift(1000.0, 200.0);
    let up = goertzel(&out, 1200.0);
    let down = goertzel(&out, 800.0);
    let carrier = goertzel(&out, 1000.0);
    assert!(
        up.max(down) > 8.0 * up.min(down),
        "one sideband dominates (1200={up}, 800={down})"
    );
    assert!(
        up.max(down) > 8.0 * carrier,
        "the carrier at 1000 is shifted away (carrier={carrier})"
    );

    // The opposite shift moves the energy to the mirror sideband.
    let out_neg = render_freqshift(1000.0, -200.0);
    let up_neg = goertzel(&out_neg, 1200.0);
    let down_neg = goertzel(&out_neg, 800.0);
    assert!(
        (up > down) != (up_neg > down_neg),
        "a negative shift moves the opposite way"
    );
}
