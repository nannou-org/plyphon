//! Drive `DiskOut` through the full write path: a `SinOsc` tone recorded by `DiskOut`, drained off
//! the audio thread by a `StreamDrainer` into a test [`BufferSinkStream`] that collects the samples.
//! The drainer is driven with a tiny `block_on` here (the in-memory sink resolves immediately); a
//! real app would use a background thread or `spawn_local`. The write-side mirror of `stream.rs`.

use std::future::Future;

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};
use plyphon_buffers::{BufFuture, BufferSinkStream, SaveError, StreamDrainer, StreamInfo};

const SR: f32 = 48_000.0;

/// A [`BufferSinkStream`] that just collects everything written to it, and records that it was
/// closed - the write-side counterpart of `stream.rs`'s `ToneStream`.
struct CollectSink {
    info: StreamInfo,
    samples: Vec<f32>,
    closed: bool,
}

impl BufferSinkStream for CollectSink {
    fn info(&self) -> StreamInfo {
        self.info
    }

    fn write<'a>(&'a mut self, samples: &'a [f32]) -> BufFuture<'a, Result<usize, SaveError>> {
        Box::pin(async move {
            let frames = samples.len() / self.info.num_channels.max(1);
            self.samples.extend_from_slice(samples);
            Ok(frames)
        })
    }

    fn close<'a>(&'a mut self) -> BufFuture<'a, Result<(), SaveError>> {
        self.closed = true;
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

/// `SinOsc.ar(440) -> DiskOut.ar(bufnum=0, [SinOsc])`.
fn disk_out_def() -> SynthDef {
    SynthDef {
        name: "rec".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "DiskOut",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                1,
            ),
        ],
    }
}

#[test]
fn stream_drainer_drains_disk_out() {
    let (mut controller, _nrt, mut world): (_, _, World) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let consumer = controller
        .buffer_cue_write(0, 1, SR as f64, 1024, 4)
        .unwrap();
    let mut drainer = StreamDrainer::new(consumer);
    let mut sink = CollectSink {
        info: StreamInfo {
            num_channels: 1,
            sample_rate: SR as f64,
            total_frames: None,
        },
        samples: Vec::new(),
        closed: false,
    };

    controller.add_synthdef(disk_out_def());
    controller
        .synth_new("rec", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    // Render in 512-sample blocks, draining after each so the bounded chunk queue never overruns.
    let mut buf = vec![0.0f32; 512];
    while sink.samples.len() < SR as usize / 4 {
        world.fill(&mut buf, 1);
        block_on(drainer.drain(&mut sink)).expect("drain");
    }
    block_on(drainer.finish(&mut sink)).expect("finish");

    assert!(sink.closed, "finish should close the sink");
    assert!(
        sink.samples.iter().any(|s| s.abs() > 0.1),
        "the drained recording was silent"
    );
    assert!(
        goertzel(&sink.samples, 440.0) > 5.0 * goertzel(&sink.samples, 880.0),
        "expected the recorded 440 Hz tone drained through the sink"
    );
}
