//! `BufRd`: read a buffer wherever a `phase` input points (unlike `PlayBuf`, whose head advances
//! internally). Driven by a `Phasor` it reads the buffer sequentially, like a playback.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f32 = 48_000.0;

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

fn sine_buffer(frames: usize, cycles: usize) -> Buffer {
    let samples: Vec<f32> = (0..frames)
        .map(|i| (std::f32::consts::TAU * cycles as f32 * i as f32 / frames as f32).sin())
        .collect();
    Buffer::from_interleaved(samples, 1, SR as f64)
}

fn render(units: Vec<UnitSpec>, frames: usize, buf: Buffer) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.buffer_set(0, Box::new(buf)).expect("buffer_set");
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

#[test]
fn buf_rd_plays_back_via_a_phasor() {
    // A Phasor sweeping 0..4800 (rate 1, looping) drives BufRd through a seamless 440 Hz buffer, so
    // the output is a 440 Hz tone.
    let out = render(
        vec![
            // Phasor.ar(trig=0, rate=1, start=0, end=4800, resetPos=0).
            UnitSpec::new(
                "Phasor",
                Rate::Audio,
                vec![c(0.0), c(1.0), c(0.0), c(4800.0), c(0.0)],
                1,
            ),
            // BufRd.ar(numChannels=1, bufnum=0, phase, loop=1, interpolation=2).
            UnitSpec::new("BufRd", Rate::Audio, vec![c(0.0), u(0), c(1.0), c(2.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        SR as usize / 4,
        sine_buffer(4800, 44),
    );
    assert!(out.iter().all(|s| s.is_finite()), "BufRd must stay finite");
    assert!(out.iter().all(|&s| s.abs() < 3.0), "BufRd stays bounded");
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 880.0),
        "BufRd plays the 440 buffer (440={at})"
    );
    assert!(at > 0.1, "the buffer should sound (440={at})");
}

#[test]
fn buf_rd_reads_a_static_frame() {
    // A constant phase reads one fixed frame of the buffer every sample - a DC output equal to that
    // sample. Frame 12 of a 16-sample ramp buffer holds 12/16.
    let ramp: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
    let out = render(
        vec![
            UnitSpec::new(
                "BufRd",
                Rate::Audio,
                vec![c(0.0), c(12.0), c(1.0), c(1.0)],
                1,
            ), // interp=1 (none)
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
        128,
        Buffer::from_interleaved(ramp, 1, SR as f64),
    );
    assert!(
        out.iter().all(|&s| (s - 12.0 / 16.0).abs() < 1e-6),
        "a static phase reads frame 12 = 0.75: got {}",
        out[0]
    );
}
