//! `/b_write leaveOpen=1` installs a `DiskOut` recording slot and leaves the host sink open; the
//! dispatcher drains it each `run_pending` tick; `/b_close` flushes the tail, closes the sink, frees
//! the slot, and replies `/done /b_close`.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};
use plyphon_buffers::{BufFuture, BufferSink, BufferSinkStream, SaveError, StreamInfo};
use plyphon_osc::{Host, OscDispatcher};
use rosc::{OscMessage, OscPacket, OscType};

const SR: f64 = 48_000.0;

/// A `BufferSink` collecting every written sample into a shared `Vec`.
struct VecSink {
    samples: Rc<RefCell<Vec<f32>>>,
}

impl VecSink {
    fn new() -> Self {
        VecSink {
            samples: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl BufferSink for VecSink {
    fn open_write<'a>(
        &'a self,
        _key: &'a str,
        info: StreamInfo,
    ) -> BufFuture<'a, Result<Box<dyn BufferSinkStream>, SaveError>> {
        let samples = Rc::clone(&self.samples);
        Box::pin(async move {
            Ok(Box::new(VecSinkStream { samples, info }) as Box<dyn BufferSinkStream>)
        })
    }
}

impl Host for VecSink {
    fn buffer_sink(&self) -> Option<&dyn BufferSink> {
        Some(self)
    }
}

struct VecSinkStream {
    samples: Rc<RefCell<Vec<f32>>>,
    info: StreamInfo,
}

impl BufferSinkStream for VecSinkStream {
    fn info(&self) -> StreamInfo {
        self.info
    }

    fn write<'a>(&'a mut self, samples: &'a [f32]) -> BufFuture<'a, Result<usize, SaveError>> {
        let frames = samples.len() / self.info.num_channels.max(1);
        self.samples.borrow_mut().extend_from_slice(samples);
        Box::pin(async move { Ok(frames) })
    }

    fn close<'a>(&'a mut self) -> BufFuture<'a, Result<(), SaveError>> {
        Box::pin(async move { Ok(()) })
    }
}

fn msg(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    })
}

fn find<'a>(replies: &'a [OscPacket], addr: &str, cmd: &str) -> Option<&'a OscMessage> {
    replies.iter().find_map(|p| match p {
        OscPacket::Message(m)
            if m.addr == addr && m.args.first() == Some(&OscType::String(cmd.to_string())) =>
        {
            Some(m)
        }
        _ => None,
    })
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

/// `DC.ar(0.5) -> DiskOut.ar(bufnum=0)`: writes a constant 0.5 into the recording buffer.
fn diskout_def() -> SynthDef {
    SynthDef {
        name: "rec".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.5)], 1),
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
fn b_write_leave_open_streams_diskout_then_b_close() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    controller.add_synthdef(diskout_def());
    let host = VecSink::new();

    // Allocate a mono recording buffer (its mirror feeds /b_write's geometry), then open it for
    // streaming and start a DiskOut writing 0.5 into it.
    osc.apply(
        &mut controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(64), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    osc.apply(
        &mut controller,
        &msg(
            "/b_write",
            vec![
                OscType::Int(0),
                OscType::String("stream.wav".to_string()),
                OscType::String("wav".to_string()),
                OscType::String("float".to_string()),
                OscType::Int(0), // numFrames
                OscType::Int(0), // startFrame
                OscType::Int(1), // leaveOpen
            ],
        ),
    )
    .expect("/b_write");
    controller
        .synth_new("rec", ROOT_GROUP_ID, AddAction::Tail)
        .expect("start DiskOut");

    // Stream for a while: each tick the engine writes a block and the dispatcher drains it to the sink.
    // `DiskOut` pushes a chunk only once it fills (WRITE_CHUNK_FRAMES = 4096 frames = 64 blocks), so
    // render past a few chunk boundaries to exercise the continuous drain.
    let mut blk = [0.0f32; 64];
    let mut got_write_done = false;
    for _ in 0..200 {
        world.fill(&mut blk, 1);
        nrt.process();
        block_on(osc.run_pending(&mut controller, Some(&host)));
        let replies = osc.take_replies();
        got_write_done |= find(&replies, "/done", "/b_write").is_some();
    }
    assert!(got_write_done, "/b_write leaveOpen=1 should reply /done");
    assert!(
        !host.samples.borrow().is_empty(),
        "the open stream should have drained DiskOut audio to the sink"
    );

    // Close it: a final drain + close, then /done /b_close, and the recording slot is freed.
    osc.apply(&mut controller, &msg("/b_close", vec![OscType::Int(0)]))
        .expect("/b_close");
    world.fill(&mut blk, 1);
    nrt.process();
    block_on(osc.run_pending(&mut controller, Some(&host)));
    let replies = osc.take_replies();
    let done = find(&replies, "/done", "/b_close").expect("/done /b_close");
    assert_eq!(done.args[1], OscType::Int(0));

    let samples = host.samples.borrow();
    assert!(
        samples.iter().all(|&s| (s - 0.5).abs() < 1e-6),
        "DiskOut wrote a constant 0.5 to the stream"
    );
    assert!(
        samples.len() >= 4096,
        "expected at least one full chunk of recorded audio, got {}",
        samples.len()
    );
}

#[test]
fn b_close_of_an_unopened_buffer_fails() {
    let (mut controller, mut nrt, mut world) = engine(Options::default());
    let mut osc = OscDispatcher::new();
    let host = VecSink::new();

    osc.apply(&mut controller, &msg("/b_close", vec![OscType::Int(3)]))
        .expect("/b_close");
    world.fill(&mut [0.0f32; 64], 1);
    nrt.process();
    block_on(osc.run_pending(&mut controller, Some(&host)));
    let replies = osc.take_replies();
    assert!(
        find(&replies, "/fail", "/b_close").is_some(),
        "/b_close of an unopened buffer should fail"
    );
}
