//! `/b_read … leaveOpen=1` keeps the file open and streams it off disk into a `DiskIn` (scsynth's
//! cue-for-streaming, the read counterpart to `/b_write leaveOpen=1`): the dispatcher opens a host
//! `BufferStream`, replaces the slot with a playback endpoint, and feeds it each `run_pending` tick.
//! `/b_close` ends it. `/b_readChannel … leaveOpen=1` streams only the selected channels, via a
//! deinterleaving wrapper the dispatcher slips over the file stream.

use std::f32::consts::TAU;
use std::future::Future;

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};
use plyphon_buffers::{
    BufFuture, BufferData, BufferSource, BufferStream, LoadError, ReadRegion, StreamInfo,
};
use plyphon_osc::{Host, OscDispatcher};
use rosc::{OscMessage, OscPacket, OscType};

const SR: f32 = 48_000.0;
/// The streamed tone's length in frames (~0.5 s) - finite, so the stream ends (no looping).
const TONE_FRAMES: u64 = 24_000;

/// A `BufferSource` whose `open` yields a finite 440 Hz mono tone stream at key `"tone"`.
struct ToneSource;

impl BufferSource for ToneSource {
    fn load<'a>(
        &'a self,
        _key: &'a str,
        _region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        Box::pin(async { Err(LoadError::Unsupported("load".to_string())) })
    }

    fn open<'a>(&'a self, key: &'a str) -> BufFuture<'a, Result<Box<dyn BufferStream>, LoadError>> {
        let result = if key == "tone" {
            Ok(Box::new(ToneStream { pos: 0 }) as Box<dyn BufferStream>)
        } else {
            Err(LoadError::NotFound(key.to_string()))
        };
        Box::pin(async move { result })
    }
}

impl Host for ToneSource {
    fn buffer_source(&self) -> Option<&dyn BufferSource> {
        Some(self)
    }
}

/// A finite 440 Hz mono [`BufferStream`]: emits the tone for [`TONE_FRAMES`] frames, then `Ok(0)`
/// forever (non-looping, like the CLI's `FsStream`).
struct ToneStream {
    pos: u64,
}

impl BufferStream for ToneStream {
    fn info(&self) -> StreamInfo {
        StreamInfo {
            num_channels: 1,
            sample_rate: SR as f64,
            total_frames: Some(TONE_FRAMES),
        }
    }

    fn read<'a>(&'a mut self, out: &'a mut [f32]) -> BufFuture<'a, Result<usize, LoadError>> {
        let mut frames = 0;
        while frames < out.len() && self.pos < TONE_FRAMES {
            out[frames] = (TAU * 440.0 * self.pos as f32 / SR).sin() * 0.5;
            self.pos += 1;
            frames += 1;
        }
        Box::pin(async move { Ok(frames) })
    }

    fn seek<'a>(&'a mut self, frame: u64) -> BufFuture<'a, Result<(), LoadError>> {
        self.pos = frame;
        Box::pin(async { Ok(()) })
    }
}

/// A `BufferSource` whose `open` yields a finite 2-channel tone stream at key `"stereo"`: channel 0 is
/// 440 Hz, channel 1 is 880 Hz - so selecting one channel is audibly distinguishable from the other.
struct StereoSource;

impl BufferSource for StereoSource {
    fn load<'a>(
        &'a self,
        _key: &'a str,
        _region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        Box::pin(async { Err(LoadError::Unsupported("load".to_string())) })
    }

    fn open<'a>(&'a self, key: &'a str) -> BufFuture<'a, Result<Box<dyn BufferStream>, LoadError>> {
        let result = if key == "stereo" {
            Ok(Box::new(StereoStream { pos: 0 }) as Box<dyn BufferStream>)
        } else {
            Err(LoadError::NotFound(key.to_string()))
        };
        Box::pin(async move { result })
    }
}

impl Host for StereoSource {
    fn buffer_source(&self) -> Option<&dyn BufferSource> {
        Some(self)
    }
}

/// A finite 2-channel [`BufferStream`]: channel 0 a 440 Hz tone, channel 1 an 880 Hz tone, interleaved
/// for [`TONE_FRAMES`] frames, then `Ok(0)` forever (non-looping).
struct StereoStream {
    pos: u64,
}

impl BufferStream for StereoStream {
    fn info(&self) -> StreamInfo {
        StreamInfo {
            num_channels: 2,
            sample_rate: SR as f64,
            total_frames: Some(TONE_FRAMES),
        }
    }

    fn read<'a>(&'a mut self, out: &'a mut [f32]) -> BufFuture<'a, Result<usize, LoadError>> {
        let mut frames = 0;
        while (frames + 1) * 2 <= out.len() && self.pos < TONE_FRAMES {
            let t = self.pos as f32 / SR;
            out[frames * 2] = (TAU * 440.0 * t).sin() * 0.5;
            out[frames * 2 + 1] = (TAU * 880.0 * t).sin() * 0.5;
            self.pos += 1;
            frames += 1;
        }
        Box::pin(async move { Ok(frames) })
    }

    fn seek<'a>(&'a mut self, frame: u64) -> BufFuture<'a, Result<(), LoadError>> {
        self.pos = frame;
        Box::pin(async { Ok(()) })
    }
}

/// A host with no capabilities.
struct NoHost;
impl Host for NoHost {}

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

/// `DiskIn.ar(bufnum=0) -> Out.ar(0)`.
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
fn b_read_leave_open_streams_through_diskin_then_b_close() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    controller.add_synthdef(disk_in_def());
    let host = ToneSource;

    // Allocate a mono buffer (its mirror feeds /b_read's channel check), then cue it for streaming.
    osc.apply(
        &mut controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(1024), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    osc.apply(
        &mut controller,
        &msg(
            "/b_read",
            vec![
                OscType::Int(0),
                OscType::String("tone".to_string()),
                OscType::Int(0),  // fileStartFrame
                OscType::Int(-1), // numFrames
                OscType::Int(0),  // bufStartFrame
                OscType::Int(1),  // leaveOpen
            ],
        ),
    )
    .expect("/b_read");
    block_on(osc.run_pending(&mut controller, Some(&host))); // opens + cues + primes the queue
    assert!(
        find(&osc.take_replies(), "/done", "/b_read").is_some(),
        "/b_read leaveOpen=1 should reply /done"
    );

    controller
        .synth_new("stream", ROOT_GROUP_ID, AddAction::Tail)
        .expect("start DiskIn");

    // Play the stream, topping the queue up each tick; collect the output.
    let mut out = Vec::new();
    let mut blk = [0.0f32; 512];
    while out.len() < (SR * 0.2) as usize {
        world.fill(&mut blk, 1);
        out.extend_from_slice(&blk);
        nrt.process();
        block_on(osc.run_pending(&mut controller, Some(&host)));
    }
    assert!(out.iter().any(|s| s.abs() > 0.1), "the stream was silent");
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "DiskIn should play the streamed 440 Hz tone"
    );

    // Close the stream: synchronous free + /done.
    osc.apply(&mut controller, &msg("/b_close", vec![OscType::Int(0)]))
        .expect("/b_close");
    block_on(osc.run_pending(&mut controller, Some(&host)));
    let replies = osc.take_replies();
    let done = find(&replies, "/done", "/b_close").expect("/done /b_close");
    assert_eq!(done.args[1], OscType::Int(0));

    // After close the slot is gone; DiskIn underruns to silence (no panic).
    for _ in 0..8 {
        world.fill(&mut blk, 1);
        nrt.process();
        block_on(osc.run_pending(&mut controller, Some(&host)));
    }
}

#[test]
fn b_read_channel_leave_open_streams_only_the_selected_channel() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    controller.add_synthdef(disk_in_def());
    let host = StereoSource;

    // A mono buffer: its mirror is the channel check the selected width (1) must match.
    osc.apply(
        &mut controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(1024), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    // Select channel 1 (the 880 Hz tone) of the stereo stream.
    osc.apply(
        &mut controller,
        &msg(
            "/b_readChannel",
            vec![
                OscType::Int(0),
                OscType::String("stereo".to_string()),
                OscType::Int(0),  // fileStartFrame
                OscType::Int(-1), // numFrames
                OscType::Int(0),  // bufStartFrame
                OscType::Int(1),  // leaveOpen
                OscType::Int(1),  // channel 1
            ],
        ),
    )
    .expect("/b_readChannel");
    block_on(osc.run_pending(&mut controller, Some(&host)));
    assert!(
        find(&osc.take_replies(), "/done", "/b_readChannel").is_some(),
        "/b_readChannel leaveOpen=1 should reply /done"
    );

    controller
        .synth_new("stream", ROOT_GROUP_ID, AddAction::Tail)
        .expect("start DiskIn");

    let mut out = Vec::new();
    let mut blk = [0.0f32; 512];
    while out.len() < (SR * 0.2) as usize {
        world.fill(&mut blk, 1);
        out.extend_from_slice(&blk);
        nrt.process();
        block_on(osc.run_pending(&mut controller, Some(&host)));
    }
    assert!(out.iter().any(|s| s.abs() > 0.1), "the stream was silent");
    assert!(
        goertzel(&out, 880.0) > 5.0 * goertzel(&out, 440.0),
        "DiskIn should play only the selected 880 Hz channel, not the 440 Hz one"
    );
}

#[test]
fn b_read_channel_leave_open_width_must_match_the_buffer() {
    let (mut controller, _nrt, _world) = engine(Options::default());
    let mut osc = OscDispatcher::new();
    let host = StereoSource;

    // A mono buffer, but the selection below is two channels wide - a mismatch.
    osc.apply(
        &mut controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(1024), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    osc.apply(
        &mut controller,
        &msg(
            "/b_readChannel",
            vec![
                OscType::Int(0),
                OscType::String("stereo".to_string()),
                OscType::Int(0),
                OscType::Int(-1),
                OscType::Int(0),
                OscType::Int(1), // leaveOpen
                OscType::Int(0), // channel 0
                OscType::Int(1), // channel 1 -> width 2, into a mono buffer
            ],
        ),
    )
    .expect("/b_readChannel");
    block_on(osc.run_pending(&mut controller, Some(&host)));
    let replies = osc.take_replies();
    let fail = find(&replies, "/fail", "/b_readChannel").expect("/fail /b_readChannel");
    assert_eq!(
        fail.args[1],
        OscType::String("channel mismatch".to_string())
    );
}

#[test]
fn b_free_stops_a_streaming_read_without_a_spurious_fail() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    let host = ToneSource;

    osc.apply(
        &mut controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(1024), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    osc.apply(
        &mut controller,
        &msg(
            "/b_read",
            vec![
                OscType::Int(0),
                OscType::String("tone".to_string()),
                OscType::Int(0),
                OscType::Int(-1),
                OscType::Int(0),
                OscType::Int(1),
            ],
        ),
    )
    .expect("/b_read");
    block_on(osc.run_pending(&mut controller, Some(&host)));
    let _ = osc.take_replies();

    // Free the buffer mid-stream; subsequent ticks must not emit a /fail for the dropped feed.
    osc.apply(&mut controller, &msg("/b_free", vec![OscType::Int(0)]))
        .expect("/b_free");
    let mut blk = [0.0f32; 512];
    for _ in 0..8 {
        world.fill(&mut blk, 1);
        nrt.process();
        block_on(osc.run_pending(&mut controller, Some(&host)));
        assert!(
            find(&osc.take_replies(), "/fail", "/b_read").is_none(),
            "a freed stream should not /fail"
        );
    }
}

/// A `/b_read leaveOpen=1` with no buffer source fails.
#[test]
fn b_read_leave_open_without_a_source_fails() {
    let (mut controller, _nrt, _world) = engine(Options::default());
    let mut osc = OscDispatcher::new();
    osc.apply(
        &mut controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(1024), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    osc.apply(
        &mut controller,
        &msg(
            "/b_read",
            vec![
                OscType::Int(0),
                OscType::String("tone".to_string()),
                OscType::Int(0),
                OscType::Int(-1),
                OscType::Int(0),
                OscType::Int(1),
            ],
        ),
    )
    .expect("/b_read");
    block_on(osc.run_pending(&mut controller, Some(&NoHost)));
    assert!(find(&osc.take_replies(), "/fail", "/b_read").is_some());
}
