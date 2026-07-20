//! `VDiskIn`: play a disk-streamed tone at a variable rate. Cue a stream, feed a 440 Hz tone from
//! "off the audio thread", and confirm VDiskIn transposes it by `rate` (1 -> 440, 2 -> 880, 0.5 -> 220)
//! via the transport's cubic resampling.

use std::f32::consts::TAU;

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, StreamProducer, SynthDef, UnitSpec, engine,
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

/// Stands in for an off-RT feeder: tops the queue up with a continuous 440 Hz mono tone, phase-continuous.
struct ToneFeeder {
    next_frame: usize,
}

impl ToneFeeder {
    fn fill(&mut self, producer: &mut StreamProducer) {
        while let Some(mut chunk) = producer.take_empty() {
            let frames = chunk.capacity();
            let samples = chunk.samples_mut();
            for (f, slot) in samples.iter_mut().enumerate() {
                let t = (self.next_frame + f) as f32;
                *slot = (TAU * 440.0 * t / SR).sin() * 0.6;
            }
            chunk.set_frames(frames);
            match producer.push(chunk) {
                Ok(()) => self.next_frame += frames,
                Err(chunk) => {
                    producer.return_empty(chunk);
                    break;
                }
            }
        }
    }
}

/// Render VDiskIn at `rate` over ~0.25 s, keeping the queue fed ahead of playback.
fn render(rate: f32) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    // A generous queue so a fast rate never underruns: 8 chunks of 1024 frames.
    let mut producer = controller.buffer_cue(0, 1, SR as f64, 1024, 8).unwrap();
    let mut feeder = ToneFeeder { next_frame: 0 };
    feeder.fill(&mut producer);

    controller.add_synthdef(SynthDef {
        name: "vd".to_string(),
        params: vec![],
        units: vec![
            // VDiskIn.ar(1, bufnum=0, rate, loop=0, sendID=0).
            UnitSpec::new(
                "VDiskIn",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(rate),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("vd", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    let mut out = Vec::new();
    while out.len() < SR as usize / 4 {
        feeder.fill(&mut producer);
        let mut buf = vec![0.0f32; 512];
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out
}

#[test]
fn vdisk_in_rate_one_plays_native() {
    let out = render(1.0);
    assert!(out.iter().all(|s| s.is_finite()), "VDiskIn stays finite");
    assert!(out.iter().any(|s| s.abs() > 0.1), "the stream should sound");
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "rate 1 plays the native 440 Hz"
    );
}

#[test]
fn vdisk_in_rate_two_transposes_up_an_octave() {
    let out = render(2.0);
    assert!(out.iter().all(|s| s.is_finite()), "VDiskIn stays finite");
    let up = goertzel(&out, 880.0);
    assert!(
        up > 5.0 * goertzel(&out, 440.0),
        "rate 2 transposes 440 up to 880 (880={up})"
    );
}

#[test]
fn vdisk_in_rate_half_transposes_down_an_octave() {
    let out = render(0.5);
    let down = goertzel(&out, 220.0);
    assert!(
        down > 5.0 * goertzel(&out, 440.0),
        "rate 0.5 transposes 440 down to 220 (220={down})"
    );
}
