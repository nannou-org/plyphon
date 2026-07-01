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

/// Render `frames` of a one-synth graph with a bank of `(index, wavetable)` buffers installed.
fn render_bank(
    mut units: Vec<UnitSpec>,
    out_src: u32,
    banks: &[(usize, Vec<f32>)],
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
    for (index, data) in banks {
        controller
            .buffer_set(
                *index,
                Box::new(Buffer::from_interleaved(data.clone(), 1, SR as f64)),
            )
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

#[test]
fn cosc_beats_around_the_carrier() {
    // Two copies detuned by 4 Hz (438 and 442) beat against each other around a 440 Hz carrier.
    let table = wavetable(1024, f32::sin);
    let units = vec![UnitSpec::new(
        "COsc",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Constant(440.0),
            InputRef::Constant(4.0),
        ],
        1,
    )];
    let out = render(units, 0, Some(table), SR as usize / 2);
    assert!(out.iter().all(|s| s.is_finite()), "COsc must stay finite");
    assert!(
        out.iter().all(|&s| s.abs() < 4.0),
        "COsc should stay bounded"
    );
    // The two copies sit at 438 and 442 Hz (440 +/- beats/2); energy stays around the carrier.
    let at = goertzel(&out, 438.0) + goertzel(&out, 442.0);
    let off = goertzel(&out, 660.0);
    assert!(
        at > 20.0 * off,
        "energy stays around the carrier (438/442={at}, 660={off})"
    );
    // The two detuned copies beat: the windowed peak swells and fades across the buffer.
    let win = SR as usize / 40;
    let peaks: Vec<f32> = out
        .chunks(win)
        .map(|c| c.iter().fold(0.0f32, |m, &s| m.max(s.abs())))
        .collect();
    let loud = peaks.iter().cloned().fold(0.0f32, f32::max);
    let quiet = peaks.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        loud > 2.0 * quiet.max(1e-4),
        "the detuned copies should beat (loud={loud}, quiet={quiet})"
    );
}

/// A `VOsc`/`VOsc3` unit's input list: `bufpos` then the frequencies.
fn vosc(name: &str, bufpos: f32, freqs: &[f32]) -> UnitSpec {
    let mut inputs = vec![InputRef::Constant(bufpos)];
    inputs.extend(freqs.iter().map(|&f| InputRef::Constant(f)));
    if name == "VOsc" {
        inputs.push(InputRef::Constant(0.0)); // phase offset
    }
    UnitSpec::new(name, Rate::Audio, inputs, 1)
}

#[test]
fn vosc_crossfades_between_wavetables() {
    // A two-member bank: slot 0 is a fundamental table, slot 1 is a second-harmonic table (read at
    // `freq` it sounds an octave up).
    let fundamental = wavetable(1024, f32::sin);
    let octave = wavetable(1024, |p| (2.0 * p).sin());
    let banks = [(0, fundamental), (1, octave)];

    // bufpos 0.0 reads slot 0 only: a clean fundamental.
    let low = render_bank(
        vec![vosc("VOsc", 0.0, &[300.0])],
        0,
        &banks,
        SR as usize / 4,
    );
    let f1 = goertzel(&low, 300.0);
    let f2 = goertzel(&low, 600.0);
    assert!(
        f1 > 20.0 * f2,
        "bufpos 0 is the fundamental (300={f1}, 600={f2})"
    );

    // bufpos 0.5 crossfades halfway, so both the fundamental and its octave are present.
    let mixed = render_bank(
        vec![vosc("VOsc", 0.5, &[300.0])],
        0,
        &banks,
        SR as usize / 4,
    );
    let m1 = goertzel(&mixed, 300.0);
    let m2 = goertzel(&mixed, 600.0);
    let mid = goertzel(&mixed, 450.0);
    assert!(
        m1 > 8.0 * mid,
        "the fundamental survives the crossfade (300={m1})"
    );
    assert!(
        m2 > 8.0 * mid,
        "the octave enters at the crossfade (600={m2})"
    );
}

#[test]
fn vosc_missing_bank_member_is_silent() {
    // Only slots 0 and 1 exist; bufpos 1.0 needs slots 1 and 2, so it silences (like scsynth).
    let banks = [
        (0, wavetable(1024, f32::sin)),
        (1, wavetable(1024, f32::sin)),
    ];
    let out = render_bank(vec![vosc("VOsc", 1.0, &[300.0])], 0, &banks, 256);
    assert!(
        out.iter().all(|&s| s == 0.0),
        "a bank position missing its upper neighbour is silent"
    );
}

#[test]
fn vosc3_sums_three_voices() {
    // Both bank members are the same fundamental table, so bufpos 0 reads a plain sine; the three
    // voices at 200/300/400 Hz should each show up.
    let banks = [
        (0, wavetable(1024, f32::sin)),
        (1, wavetable(1024, f32::sin)),
    ];
    let out = render_bank(
        vec![vosc("VOsc3", 0.0, &[200.0, 300.0, 400.0])],
        0,
        &banks,
        SR as usize / 4,
    );
    assert!(out.iter().all(|s| s.is_finite()), "VOsc3 must stay finite");
    let mid = goertzel(&out, 250.0);
    for f in [200.0, 300.0, 400.0] {
        assert!(
            goertzel(&out, f) > 8.0 * mid,
            "voice at {f} Hz should be present"
        );
    }
}
