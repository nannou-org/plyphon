//! Drive `DiskIn` through the full streaming path: an async `BufferStream` of a continuous tone, fed
//! into a cued plyphon stream by a `StreamFeeder`, played by `DiskIn`. The feeder is driven with a
//! tiny `block_on` here (the in-memory stream resolves immediately); a real app would use a
//! background thread or `spawn_local`.

use std::f32::consts::TAU;
use std::future::Future;

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World, engine,
};
use plyphon_buffers::{BufFuture, BufferStream, LoadError, StreamFeeder, StreamInfo};

const SR: f32 = 48_000.0;

/// An async [`BufferStream`] producing a continuous 440 Hz mono tone.
struct ToneStream {
    next_frame: u64,
}

impl BufferStream for ToneStream {
    fn info(&self) -> StreamInfo {
        StreamInfo {
            num_channels: 1,
            sample_rate: SR as f64,
            total_frames: None,
        }
    }

    fn read<'a>(&'a mut self, out: &'a mut [f32]) -> BufFuture<'a, Result<usize, LoadError>> {
        Box::pin(async move {
            for (f, slot) in out.iter_mut().enumerate() {
                let t = (self.next_frame + f as u64) as f32;
                *slot = (TAU * 440.0 * t / SR).sin() * 0.6;
            }
            self.next_frame += out.len() as u64;
            Ok(out.len())
        })
    }

    fn seek<'a>(&'a mut self, frame: u64) -> BufFuture<'a, Result<(), LoadError>> {
        self.next_frame = frame;
        Box::pin(async { Ok(()) })
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
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

fn disk_in_def() -> SynthDef {
    SynthDef {
        name: "stream".to_string(),
        params: vec![],
        ugens: vec![
            UgenSpec::new("DiskIn", Rate::Audio, vec![InputRef::Constant(0.0)], 1),
            UgenSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn stream_feeder_drives_disk_in() {
    let (mut controller, _nrt, mut world): (_, _, World) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let producer = controller.buffer_cue(0, 1, SR as f64, 1024, 4).unwrap();
    let mut feeder = StreamFeeder::new(producer);
    let mut stream = ToneStream { next_frame: 0 };
    block_on(feeder.fill(&mut stream)).expect("prime"); // fill the queue before playback

    controller.add_synthdef(disk_in_def());
    controller
        .synth_new("stream", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let mut out = Vec::new();
    while out.len() < SR as usize / 4 {
        block_on(feeder.fill(&mut stream)).expect("refill");
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
        "expected the streamed 440 Hz tone through the feeder"
    );
}
