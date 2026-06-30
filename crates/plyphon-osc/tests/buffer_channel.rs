//! `/b_allocReadChannel` and `/b_readChannel` read only a selected subset of a file's channels into a
//! buffer (scsynth's `CopyChannels`): the dispatcher reads the whole file through the `BufferSource`,
//! then deinterleaves the chosen channels (in order; an out-of-range index reads as silence).

use std::future::Future;

use plyphon::{Controller, Nrt, Options, World, engine};
use plyphon_buffers::{BufFuture, BufferData, BufferSource, LoadError, ReadRegion};
use plyphon_osc::{Host, OscDispatcher};
use rosc::{OscMessage, OscPacket, OscType};

const SR: f64 = 48_000.0;

/// A `BufferSource` serving one 4-frame, 2-channel file at key `"stereo"`: channel 0 is `1x`, channel
/// 1 is `2x` (frame-major interleaved `[10,20, 11,21, 12,22, 13,23]`).
struct StereoSource;

impl BufferSource for StereoSource {
    fn load<'a>(
        &'a self,
        key: &'a str,
        _region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        let result = if key == "stereo" {
            let samples = (0..4)
                .flat_map(|f| [10.0 + f as f32, 20.0 + f as f32])
                .collect();
            Ok(BufferData {
                samples,
                num_channels: 2,
                sample_rate: SR,
            })
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

/// Read `count` interleaved samples back from buffer `bufnum` via `/b_getn` (renders a block so the
/// `SetBuffer` install + the range query apply, then reassembles the `/b_setn` answer off the ring).
fn read_buffer(
    osc: &mut OscDispatcher,
    controller: &mut Controller,
    nrt: &mut Nrt,
    world: &mut World,
    bufnum: i32,
    count: i32,
) -> Vec<f32> {
    osc.apply(controller, &osc_get(bufnum, count))
        .expect("/b_getn");
    let mut blk = [0.0f32; 64];
    world.fill(&mut blk, 1);
    nrt.process();
    while let Some(reply) = nrt.poll_reply() {
        osc.reply(controller, reply);
    }
    let replies = osc.take_replies();
    let setn = find(&replies, "/b_setn").expect("/b_setn answer");
    // `/b_setn` = [bufnum, start, count, values...]; the values follow the three header ints.
    setn.args[3..]
        .iter()
        .map(|a| match a {
            OscType::Float(f) => *f,
            other => panic!("unexpected /b_setn arg: {other:?}"),
        })
        .collect()
}

fn osc_get(bufnum: i32, count: i32) -> OscPacket {
    msg(
        "/b_getn",
        vec![OscType::Int(bufnum), OscType::Int(0), OscType::Int(count)],
    )
}

fn engine_with_dispatcher() -> (OscDispatcher, Controller, Nrt, World) {
    let (controller, nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    (OscDispatcher::new(), controller, nrt, world)
}

/// `/b_allocReadChannel 0 "stereo" 0 0 1` keeps only channel 1 (`2x`).
#[test]
fn alloc_read_channel_selects_one_channel() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    osc.apply(
        &mut controller,
        &msg(
            "/b_allocReadChannel",
            vec![
                OscType::Int(0),
                OscType::String("stereo".to_string()),
                OscType::Int(0), // startFrame
                OscType::Int(0), // numFrames (all)
                OscType::Int(1), // channel 1
            ],
        ),
    )
    .expect("/b_allocReadChannel");
    block_on(osc.run_pending(&mut controller, Some(&StereoSource)));
    let replies = osc.take_replies();
    let done = find(&replies, "/done").expect("/done");
    assert_eq!(
        done.args,
        vec![
            OscType::String("/b_allocReadChannel".to_string()),
            OscType::Int(0)
        ]
    );

    // /b_query reports the selected width (1 channel, 4 frames).
    osc.apply(&mut controller, &msg("/b_query", vec![OscType::Int(0)]))
        .expect("/b_query");
    let q = osc.take_replies();
    let info = find(&q, "/b_info").expect("/b_info");
    assert_eq!(info.args[1], OscType::Int(4)); // frames
    assert_eq!(info.args[2], OscType::Int(1)); // channels

    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 0, 4);
    assert_eq!(got, vec![20.0, 21.0, 22.0, 23.0], "channel 1 only");
}

/// `/b_readChannel 0 "stereo" 0 -1 0 0 1 0` selects channels `[1, 0]` (a swap), 2-wide, splicing into
/// a pre-allocated 2-channel buffer (`/b_readChannel` reads into an existing buffer, like `/b_read`).
#[test]
fn read_channel_reorders_channels() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    // The selection is 2-wide, so allocate a matching 2-channel buffer to read into.
    osc.apply(
        &mut controller,
        &msg(
            "/b_alloc",
            vec![OscType::Int(0), OscType::Int(4), OscType::Int(2)],
        ),
    )
    .expect("/b_alloc");
    let _ = osc.take_replies();
    osc.apply(
        &mut controller,
        &msg(
            "/b_readChannel",
            vec![
                OscType::Int(0),
                OscType::String("stereo".to_string()),
                OscType::Int(0),  // fileStartFrame
                OscType::Int(-1), // numFrames (all)
                OscType::Int(0),  // bufStartFrame
                OscType::Int(0),  // leaveOpen
                OscType::Int(1),  // channel 1
                OscType::Int(0),  // channel 0
            ],
        ),
    )
    .expect("/b_readChannel");
    block_on(osc.run_pending(&mut controller, Some(&StereoSource)));
    assert!(find(&osc.take_replies(), "/done").is_some());

    // Swapped interleave: frame f -> [2x, 1x].
    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 0, 8);
    assert_eq!(
        got,
        vec![20.0, 10.0, 21.0, 11.0, 22.0, 12.0, 23.0, 13.0],
        "channels [1, 0] interleaved"
    );
}

/// An out-of-range channel index reads as silence (scsynth's `CopyChannels`).
#[test]
fn alloc_read_channel_out_of_range_is_silent() {
    let (mut osc, mut controller, mut nrt, mut world) = engine_with_dispatcher();
    osc.apply(
        &mut controller,
        &msg(
            "/b_allocReadChannel",
            vec![
                OscType::Int(0),
                OscType::String("stereo".to_string()),
                OscType::Int(0),
                OscType::Int(0),
                OscType::Int(5), // channel 5 - out of range
            ],
        ),
    )
    .expect("/b_allocReadChannel");
    block_on(osc.run_pending(&mut controller, Some(&StereoSource)));
    assert!(find(&osc.take_replies(), "/done").is_some());

    let got = read_buffer(&mut osc, &mut controller, &mut nrt, &mut world, 0, 4);
    assert_eq!(
        got,
        vec![0.0, 0.0, 0.0, 0.0],
        "out-of-range channel is silent"
    );
}
