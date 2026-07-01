//! `DelTapWr`/`DelTapRd`: a shared delay line held in a mono buffer. One writer advances a wrapping
//! head (carried through the audio wire as bit-cast integers); one or more readers tap the buffer at
//! independent delays behind that head. Exercises the writer/reader split, the cross-block line, the
//! interpolated read, and several taps on one line.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

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

/// Allocate a mono buffer of `buf_frames` at bufnum 0, run `units` for `frames_out` mono samples.
fn render(units: Vec<UnitSpec>, frames_out: usize, buf_frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    });
    controller
        .buffer_alloc(0, buf_frames, 1, SR as f64)
        .expect("buffer_alloc");
    controller.add_synthdef(SynthDef {
        name: "d".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; frames_out];
    world.fill(&mut out, 1);
    out
}

#[test]
fn deltap_delays_a_dc_step_across_blocks() {
    // DelTapWr writes a constant 1.0 into the (zeroed) buffer; DelTapRd taps `k` samples behind the
    // head, so it reads the zeroed line for the first `k` samples, then the written constant - a step
    // at `k`, which spans more than one control block.
    let k = 100usize;
    let out = render(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("DelTapWr", Rate::Audio, vec![c(0.0), u(0)], 1),
            UnitSpec::new(
                "DelTapRd",
                Rate::Audio,
                vec![c(0.0), u(1), c(k as f32 / SR), c(1.0)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
        ],
        3 * BLOCK,
        1024,
    );

    assert!(k > BLOCK, "delay must span >1 block (k = {k})");
    // The step lands at the tap `k`; `delTime` is an f32 seconds value, so its sample position can
    // round to k+/-1 (scsynth's `(int)(phaseIn - delTime)` truncates the same way). Assert silence
    // strictly before the tap and the constant strictly after, leaving the one-sample transition free.
    for (i, &s) in out.iter().enumerate().take(k) {
        assert!(s.abs() < 1e-6, "pre-delay silence at {i}: {s}");
    }
    for (i, &s) in out.iter().enumerate().skip(k + 2) {
        assert!((s - 1.0).abs() < 1e-6, "delayed constant at {i}: {s}");
    }
}

#[test]
fn deltap_reads_a_delayed_sine() {
    // A 440 Hz sine written and tapped back at a fixed delay is still a 440 Hz sine (delay is a pure
    // time shift), finite and bounded.
    let out = render(
        vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            UnitSpec::new("DelTapWr", Rate::Audio, vec![c(0.0), u(0)], 1),
            UnitSpec::new(
                "DelTapRd",
                Rate::Audio,
                vec![c(0.0), u(1), c(0.005), c(2.0)], // 5 ms delay, linear interp
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
        ],
        SR as usize / 4,
        2048,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "DelTapRd must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 4.0), "DelTapRd stays bounded");
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 880.0),
        "the delayed sine rings at 440 (440={at})"
    );
    assert!(at > 0.1, "the delayed sine should sound (440={at})");
}

#[test]
fn one_line_feeds_several_taps() {
    // Two DelTapRd tap one DelTapWr at different delays: each is its own step from the shared line, at
    // its own delay. Summed here; each step contributes independently.
    let (k1, k2) = (40usize, 150usize);
    let out = render(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("DelTapWr", Rate::Audio, vec![c(0.0), u(0)], 1),
            UnitSpec::new(
                "DelTapRd",
                Rate::Audio,
                vec![c(0.0), u(1), c(k1 as f32 / SR), c(1.0)],
                1,
            ),
            UnitSpec::new(
                "DelTapRd",
                Rate::Audio,
                vec![c(0.0), u(1), c(k2 as f32 / SR), c(1.0)],
                1,
            ),
            // sum the two taps
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![u(2), u(3)],
                num_outputs: 1,
                special_index: 0, // add
            },
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(4)], 0),
        ],
        3 * BLOCK,
        1024,
    );

    // Before either tap opens: silence. Between k1 and k2: exactly one tap (1.0). After k2: both (2.0).
    assert!(
        out[..k1].iter().all(|&s| s.abs() < 1e-6),
        "silent before k1"
    );
    assert!(
        out[k1 + 1..k2].iter().all(|&s| (s - 1.0).abs() < 1e-6),
        "one tap open between k1 and k2"
    );
    assert!(
        out[k2 + 1..].iter().all(|&s| (s - 2.0).abs() < 1e-6),
        "both taps open after k2"
    );
}
