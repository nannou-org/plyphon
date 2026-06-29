//! `/b_write` snapshots an in-memory buffer to a host [`BufferSink`]. The dispatcher queues the
//! copy-out; `run_pending` drives it across ticks (the engine streams the samples out a chunk at a
//! time), then replies `/done /b_write <bufnum>`. An absent sink fails the command.

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;

use plyphon::{Options, engine};
use plyphon_buffers::{BufFuture, BufferSink, BufferSinkStream, SaveError, StreamInfo};
use plyphon_osc::{Host, OscDispatcher};
use rosc::{OscMessage, OscPacket, OscType};

/// A [`BufferSink`] that collects every written sample into a shared `Vec`, so a test can assert what
/// the copy-out produced.
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

/// A host with no capabilities (every action fails).
struct NoHost;
impl Host for NoHost {}

fn osc(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
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

/// Whether any reply at `addr` carries `cmd` as its first (command-name) argument.
fn reply_for(replies: &[OscPacket], addr: &str, cmd: &str) -> bool {
    replies.iter().any(|p| match p {
        OscPacket::Message(m) => {
            m.addr == addr && m.args.first() == Some(&OscType::String(cmd.to_string()))
        }
        _ => false,
    })
}

#[test]
fn b_write_snapshots_a_buffer_to_a_sink() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: 48_000.0,
        output_channels: 1,
        ..Options::default()
    });
    let mut disp = OscDispatcher::new();

    // Allocate a 200-frame mono buffer and fill it with a known ramp through the dispatcher (so the
    // control-side buffer mirror `/b_write` reads is populated, exactly as a real client would).
    let expected: Vec<f32> = (0..200).map(|f| f as f32).collect();
    disp.apply(
        &mut controller,
        &osc(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(200), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    let mut setn = vec![OscType::Int(0), OscType::Int(0), OscType::Int(200)];
    setn.extend(expected.iter().map(|&v| OscType::Float(v)));
    disp.apply(&mut controller, &osc("/b_setn", setn))
        .expect("/b_setn");
    disp.apply(
        &mut controller,
        &osc(
            "/b_write",
            vec![OscType::Int(0), OscType::String("snapshot.wav".to_string())],
        ),
    )
    .expect("/b_write");

    let host = VecSink::new();
    let mut blk = [0.0f32; 64];
    let mut done = false;
    for _ in 0..200 {
        world.fill(&mut blk, 1);
        nrt.process();
        block_on(disp.run_pending(&mut controller, Some(&host)));
        let replies = disp.take_replies();
        assert!(
            !reply_for(&replies, "/fail", "/b_write"),
            "/b_write failed: {replies:?}"
        );
        if reply_for(&replies, "/done", "/b_write") {
            done = true;
            break;
        }
    }

    assert!(done, "/b_write never replied /done");
    assert_eq!(
        *host.samples.borrow(),
        expected,
        "the snapshot did not round-trip the ramp"
    );
}

#[test]
fn b_write_without_a_sink_fails() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: 48_000.0,
        output_channels: 1,
        ..Options::default()
    });
    let mut disp = OscDispatcher::new();
    disp.apply(
        &mut controller,
        &osc(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(64), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    disp.apply(
        &mut controller,
        &osc(
            "/b_write",
            vec![OscType::Int(0), OscType::String("snapshot.wav".to_string())],
        ),
    )
    .expect("/b_write");

    let mut blk = [0.0f32; 64];
    let mut failed = false;
    for _ in 0..16 {
        world.fill(&mut blk, 1);
        nrt.process();
        block_on(disp.run_pending(&mut controller, Some(&NoHost)));
        let replies = disp.take_replies();
        if reply_for(&replies, "/fail", "/b_write") {
            failed = true;
            break;
        }
    }
    assert!(failed, "expected a /fail for /b_write with no sink");
}

#[test]
fn b_write_of_an_unallocated_buffer_fails() {
    let (mut controller, _nrt, _world) = engine(Options::default());
    let mut disp = OscDispatcher::new();
    disp.apply(
        &mut controller,
        &osc(
            "/b_write",
            vec![OscType::Int(5), OscType::String("snapshot.wav".to_string())],
        ),
    )
    .expect("/b_write");
    // The failure is synchronous (the control-side mirror has no such buffer) - no drive needed.
    let replies = disp.take_replies();
    assert!(
        reply_for(&replies, "/fail", "/b_write"),
        "expected a /fail for an unallocated buffer"
    );
}
