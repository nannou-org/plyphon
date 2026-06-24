//! Disk-streaming playback: cue a stream, feed it chunks of a continuous tone from "off the audio
//! thread" (the test stands in for an async feeder), and confirm `DiskIn` plays the streamed tone.
//! Also checks that freeing a cued stream routes it to the trash ring for off-RT dropping.

use std::f32::consts::TAU;

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, StreamProducer, SynthDef, UnitSpec, World,
    engine,
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

/// Stands in for an off-RT feeder: tops the queue up with the next frames of a continuous 440 Hz
/// mono tone, keeping phase across chunks so the stream is seamless.
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
                // Queue full: return the chunk unused (phase not advanced) and stop.
                Err(chunk) => {
                    producer.return_empty(chunk);
                    break;
                }
            }
        }
    }
}

/// `DiskIn.ar(1, bufnum = 0) -> Out.ar(0)`.
fn disk_in_def() -> SynthDef {
    SynthDef {
        name: "stream".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DiskIn", Rate::Audio, vec![InputRef::Constant(0.0)], 1),
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
    }
}

#[test]
fn disk_in_plays_a_streamed_tone() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    // Cue a 1-channel stream: 4 chunks of 1024 frames (~85 ms of lookahead).
    let mut producer = controller.buffer_cue(0, 1, SR as f64, 1024, 4).unwrap();
    let mut feeder = ToneFeeder { next_frame: 0 };
    feeder.fill(&mut producer); // prime the queue before playback starts

    controller.add_synthdef(disk_in_def());
    controller
        .synth_new("stream", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // Render ~0.25 s, topping the queue up before each block (the feeder keeps ahead of playback).
    let mut out = Vec::new();
    while out.len() < SR as usize / 4 {
        feeder.fill(&mut producer);
        let mut buf = vec![0.0f32; 512];
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }

    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "the stream played silence"
    );
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "expected the streamed 440 Hz tone"
    );
}

#[test]
fn freeing_a_cued_stream_routes_to_trash() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let _producer = controller.buffer_cue(0, 1, SR as f64, 256, 2).unwrap();
    let _ = render_block(&mut world); // install the stream
    controller.buffer_free(0).unwrap();
    let _ = render_block(&mut world); // apply the free
    assert!(
        nrt.process() >= 1,
        "freeing a cued stream should hand it to the trash ring for off-RT dropping"
    );
}

fn render_block(world: &mut World) -> Vec<f32> {
    let mut buf = vec![0.0f32; 128];
    world.fill(&mut buf, 1);
    buf
}
