//! `/b_read` (and `/b_readChannel`) read a file region INTO an already-allocated buffer at
//! `bufStartFrame`, leaving its dimensions unchanged (scsynth's `BufReadCmd`) - not a whole-buffer
//! replace. The file's (post-channel-selection) channel count must match the buffer's.

use std::future::Future;

use plyphon::{Controller, Nrt, Options, World, engine};
use plyphon_buffers::{BufFuture, BufferData, BufferSource, LoadError, ReadRegion};
use plyphon_osc::{Host, OscDispatcher};
use rosc::{OscMessage, OscPacket, OscType};

const SR: f64 = 48_000.0;
/// The ramp file's sample rate, deliberately != the engine SR, to exercise the SR-update path.
const RAMP_SR: f64 = 22_050.0;

/// A `BufferSource` honoring `region`: `"ramp"` is mono `[1,2,3,4]` at [`RAMP_SR`]; `"stereo"` is a
/// 4-frame, 2-channel file (channel 0 `1x`, channel 1 `2x`).
struct RampSource;

impl BufferSource for RampSource {
    fn load<'a>(
        &'a self,
        key: &'a str,
        region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        let result = match key {
            "ramp" => Ok((vec![1.0, 2.0, 3.0, 4.0], 1, RAMP_SR)),
            "stereo" => Ok((
                (0..4)
                    .flat_map(|f| [10.0 + f as f32, 20.0 + f as f32])
                    .collect(),
                2,
                SR,
            )),
            other => Err(LoadError::NotFound(other.to_string())),
        }
        .map(|(full, channels, sample_rate): (Vec<f32>, usize, f64)| {
            // Apply the requested file region (scsynth's fileStartFrame/numFrames clamp lives here).
            let total = full.len() / channels;
            let start = (region.start_frame as usize).min(total);
            let count = region
                .num_frames
                .map_or(total - start, |n| (n as usize).min(total - start));
            BufferData {
                samples: full[start * channels..(start + count) * channels].to_vec(),
                num_channels: channels,
                sample_rate,
            }
        });
        Box::pin(async move { result })
    }
}

impl Host for RampSource {
    fn buffer_source(&self) -> Option<&dyn BufferSource> {
        Some(self)
    }
}

fn msg(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    })
}

fn find<'a>(replies: &'a [OscPacket], addr: &str) -> Option<&'a OscMessage> {
    replies.iter().find_map(|p| match p {
        OscPacket::Message(m) if m.addr == addr => Some(m),
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

/// Read `count` interleaved samples back from buffer 0 via `/b_getn` (a render applies the pending
/// SetBuffer/WriteBufferRegion + the query, then the `/b_setn` answer is reassembled off the ring).
fn read_buffer(
    osc: &mut OscDispatcher,
    controller: &mut Controller,
    nrt: &mut Nrt,
    world: &mut World,
    count: i32,
) -> Vec<f32> {
    osc.apply(
        controller,
        &msg(
            "/b_getn",
            vec![OscType::Int(0), OscType::Int(0), OscType::Int(count)],
        ),
    )
    .expect("/b_getn");
    let mut blk = [0.0f32; 64];
    world.fill(&mut blk, 1);
    nrt.process();
    while let Some(reply) = nrt.poll_reply() {
        osc.reply(controller, reply);
    }
    let replies = osc.take_replies();
    let setn = find(&replies, "/b_setn").expect("/b_setn answer");
    setn.args[3..]
        .iter()
        .map(|a| match a {
            OscType::Float(f) => *f,
            other => panic!("unexpected /b_setn arg: {other:?}"),
        })
        .collect()
}

fn engine_with_dispatcher() -> (OscDispatcher, Controller, Nrt, World) {
    let (controller, nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    (OscDispatcher::new(), controller, nrt, world)
}

/// Allocate `frames`-frame mono buffer 0 and drop the `/done`.
fn alloc(osc: &mut OscDispatcher, controller: &mut Controller, frames: i32) {
    osc.apply(
        controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(frames), OscType::Int(1)],
        ),
    )
    .expect("/b_alloc");
    let _ = osc.take_replies();
}

/// `/b_read 0 path fileStart numFrames bufStart 0` and drive it.
fn read(
    osc: &mut OscDispatcher,
    controller: &mut Controller,
    path: &str,
    file_start: i32,
    num_frames: i32,
    buf_start: i32,
) {
    osc.apply(
        controller,
        &msg(
            "/b_read",
            vec![
                OscType::Int(0),
                OscType::String(path.to_string()),
                OscType::Int(file_start),
                OscType::Int(num_frames),
                OscType::Int(buf_start),
                OscType::Int(0),
            ],
        ),
    )
    .expect("/b_read");
    block_on(osc.run_pending(controller, Some(&RampSource)));
}

#[test]
fn b_read_splices_at_offset() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    alloc(&mut osc, &mut controller, 8);
    read(&mut osc, &mut controller, "ramp", 0, -1, 3); // whole ramp at frame 3

    let replies = osc.take_replies();
    let done = find(&replies, "/done").expect("/done");
    assert_eq!(
        done.args,
        vec![OscType::String("/b_read".to_string()), OscType::Int(0)]
    );

    // The ramp landed at frame 3; frames 0-2 and 7 are untouched (a splice, not a replace).
    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 8);
    assert_eq!(got, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0]);

    // Dimensions are unchanged.
    osc.apply(&mut controller, &msg("/b_query", vec![OscType::Int(0)]))
        .expect("/b_query");
    let replies = osc.take_replies();
    let info = find(&replies, "/b_info").expect("/b_info");
    assert_eq!(info.args[1], OscType::Int(8)); // frames
    assert_eq!(info.args[2], OscType::Int(1)); // channels
}

#[test]
fn b_read_clamps_to_buffer_end() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    alloc(&mut osc, &mut controller, 6);
    read(&mut osc, &mut controller, "ramp", 0, -1, 4); // 4-frame ramp at frame 4 of a 6-frame buffer

    assert!(find(&osc.take_replies(), "/done").is_some());
    // Only the two frames that fit are written; the rest is left intact.
    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 6);
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn b_read_offset_beyond_buffer_is_noop_success() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    alloc(&mut osc, &mut controller, 4);
    read(&mut osc, &mut controller, "ramp", 0, -1, 10); // offset past the end

    assert!(find(&osc.take_replies(), "/done").is_some()); // scsynth's framesToEnd <= 0 still succeeds
    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 4);
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn b_read_file_region_subset() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    alloc(&mut osc, &mut controller, 4);
    read(&mut osc, &mut controller, "ramp", 1, 2, 0); // file frames [1..3) = [2, 3], at frame 0

    assert!(find(&osc.take_replies(), "/done").is_some());
    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 4);
    assert_eq!(got, vec![2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn b_read_channel_mismatch_fails() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    alloc(&mut osc, &mut controller, 4); // mono buffer
    read(&mut osc, &mut controller, "stereo", 0, -1, 0); // 2-channel file

    let replies = osc.take_replies();
    let fail = find(&replies, "/fail").expect("/fail");
    assert_eq!(
        fail.args,
        vec![
            OscType::String("/b_read".to_string()),
            OscType::String("channel mismatch".to_string()),
        ]
    );
    assert!(find(&replies, "/done").is_none(), "no /done on failure");
    // The buffer is untouched.
    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 4);
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn b_read_into_unallocated_fails() {
    let (mut osc, mut controller, _nrt, _world) = engine_with_dispatcher();
    read(&mut osc, &mut controller, "ramp", 0, -1, 0); // no /b_alloc first

    let replies = osc.take_replies();
    let fail = find(&replies, "/fail").expect("/fail");
    assert_eq!(
        fail.args,
        vec![
            OscType::String("/b_read".to_string()),
            OscType::String("buffer not allocated".to_string()),
        ]
    );
}

#[test]
fn b_read_channel_splices_selected_channel() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    alloc(&mut osc, &mut controller, 8); // mono buffer
    // /b_readChannel: select channel 1 (width 1, matches the mono buffer), splice at frame 2.
    osc.apply(
        &mut controller,
        &msg(
            "/b_readChannel",
            vec![
                OscType::Int(0),
                OscType::String("stereo".to_string()),
                OscType::Int(0),  // fileStartFrame
                OscType::Int(-1), // numFrames
                OscType::Int(2),  // bufStartFrame
                OscType::Int(0),  // leaveOpen
                OscType::Int(1),  // channel 1
            ],
        ),
    )
    .expect("/b_readChannel");
    block_on(osc.run_pending(&mut controller, Some(&RampSource)));
    assert!(find(&osc.take_replies(), "/done").is_some());

    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 8);
    assert_eq!(got, vec![0.0, 0.0, 20.0, 21.0, 22.0, 23.0, 0.0, 0.0]);
}

#[test]
fn b_read_updates_sample_rate() {
    let (mut osc, mut controller, _nrt, _world) = engine_with_dispatcher();
    alloc(&mut osc, &mut controller, 4); // allocated at the engine SR (48 kHz)
    read(&mut osc, &mut controller, "ramp", 0, -1, 0); // ramp file is 22.05 kHz
    let _ = osc.take_replies();

    osc.apply(&mut controller, &msg("/b_query", vec![OscType::Int(0)]))
        .expect("/b_query");
    let replies = osc.take_replies();
    let info = find(&replies, "/b_info").expect("/b_info");
    assert_eq!(info.args[3], OscType::Float(RAMP_SR as f32)); // SR became the file's (scsynth Stage3)
}
