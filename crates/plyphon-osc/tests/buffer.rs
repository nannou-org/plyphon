//! Drive the buffer commands over OSC: `/b_allocRead` loads through a `BufferSource` (driven by
//! `run_pending`), `/done` is replied, a `PlayBuf` synth plays the loaded buffer, and `/b_query`
//! answers with `/b_info`.

use std::f32::consts::TAU;
use std::future::Future;

use plyphon::{InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine};
use plyphon_buffers::{BufFuture, BufferData, BufferSource, LoadError, ReadRegion};
use plyphon_osc::OscDispatcher;
use rosc::{OscMessage, OscPacket, OscType};

const SR: f32 = 48_000.0;

/// A one-sound `BufferSource`: "tone" -> a seamless 440 Hz mono buffer.
struct ToneSource;

impl BufferSource for ToneSource {
    fn load<'a>(
        &'a self,
        key: &'a str,
        _region: ReadRegion,
    ) -> BufFuture<'a, Result<BufferData, LoadError>> {
        let result = if key == "tone" {
            let samples = (0..4800)
                .map(|i| (TAU * 44.0 * i as f32 / 4800.0).sin() * 0.6)
                .collect();
            Ok(BufferData {
                samples,
                num_channels: 1,
                sample_rate: SR as f64,
            })
        } else {
            Err(LoadError::NotFound(key.to_string()))
        };
        Box::pin(async move { result })
    }
}

fn msg(addr: &str, args: Vec<OscType>) -> Vec<u8> {
    rosc::encoder::encode(&OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    }))
    .expect("encode OSC")
}

fn find<'a>(replies: &'a [OscPacket], addr: &str) -> Option<&'a OscMessage> {
    replies.iter().find_map(|packet| match packet {
        OscPacket::Message(message) if message.addr == addr => Some(message),
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

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = vec![0.0f32; 256];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `PlayBuf.ar(1, 0, 1, ..., loop) -> Out.ar(0)`.
fn player_def() -> SynthDef {
    SynthDef {
        name: "player".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "PlayBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
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
    }
}

#[test]
fn loads_a_buffer_over_osc_and_plays_it() {
    let (controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::with_buffer_source(controller, Box::new(ToneSource));
    osc.controller().add_synthdef(player_def());

    // /b_allocRead queues an async load; nothing happens until run_pending.
    osc.apply_bytes(&msg(
        "/b_allocRead",
        vec![OscType::Int(0), OscType::String("tone".to_string())],
    ))
    .expect("/b_allocRead");
    assert!(
        osc.take_replies().is_empty(),
        "no reply until the load runs"
    );

    block_on(osc.run_pending());
    let replies = osc.take_replies();
    let done = find(&replies, "/done").expect("/done after the load");
    assert_eq!(
        done.args,
        vec![OscType::String("/b_allocRead".to_string()), OscType::Int(0)]
    );

    // Start a PlayBuf on the loaded buffer and confirm it plays the tone.
    osc.apply_bytes(&msg(
        "/s_new",
        vec![
            OscType::String("player".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/s_new");
    let out = render(&mut world, SR as usize / 4);
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "the loaded buffer was silent"
    );
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "expected the 440 Hz loaded buffer"
    );

    // /b_query answers with the mirrored dimensions.
    osc.apply_bytes(&msg("/b_query", vec![OscType::Int(0)]))
        .expect("/b_query");
    let replies = osc.take_replies();
    let info = find(&replies, "/b_info").expect("/b_info reply");
    assert_eq!(
        info.args,
        vec![
            OscType::Int(0),
            OscType::Int(4800),
            OscType::Int(1),
            OscType::Float(SR),
        ]
    );
}

#[test]
fn alloc_read_without_a_source_fails() {
    let (controller, _nrt, _world) = engine(Options::default());
    let mut osc = OscDispatcher::new(controller); // no buffer source
    osc.apply_bytes(&msg(
        "/b_allocRead",
        vec![OscType::Int(0), OscType::String("tone".to_string())],
    ))
    .expect("/b_allocRead");
    block_on(osc.run_pending());
    let replies = osc.take_replies();
    let fail = find(&replies, "/fail").expect("/fail without a source");
    assert_eq!(
        fail.args.first(),
        Some(&OscType::String("/b_allocRead".to_string()))
    );
}

/// A one-channel engine + dispatcher with the `player` def registered.
fn player_engine() -> (OscDispatcher, World) {
    let (controller, _nrt, world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new(controller);
    osc.controller().add_synthdef(player_def());
    (osc, world)
}

/// Start the `player` (node 1000) on buffer 0.
fn start_player(osc: &mut OscDispatcher) {
    osc.apply_bytes(&msg(
        "/s_new",
        vec![
            OscType::String("player".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/s_new");
}

#[test]
fn b_fill_writes_a_constant_buffer() {
    let (mut osc, mut world) = player_engine();
    osc.apply_bytes(&msg(
        "/b_alloc",
        vec![OscType::Int(0), OscType::Int(256), OscType::Int(1)],
    ))
    .expect("/b_alloc");
    // Fill the whole buffer with 0.5 in a single command.
    osc.apply_bytes(&msg(
        "/b_fill",
        vec![
            OscType::Int(0),
            OscType::Int(0),
            OscType::Int(256),
            OscType::Float(0.5),
        ],
    ))
    .expect("/b_fill");
    start_player(&mut osc);

    let out = render(&mut world, SR as usize / 4);
    assert!(
        out[1024..].iter().all(|s| (s - 0.5).abs() < 0.01),
        "a /b_fill'd buffer should play back as a 0.5 constant"
    );
}

#[test]
fn b_set_writes_individual_samples() {
    let (mut osc, mut world) = player_engine();
    osc.apply_bytes(&msg(
        "/b_alloc",
        vec![OscType::Int(0), OscType::Int(4), OscType::Int(1)],
    ))
    .expect("/b_alloc");
    // Set each of the four samples individually to 0.3.
    osc.apply_bytes(&msg(
        "/b_set",
        vec![
            OscType::Int(0),
            OscType::Int(0),
            OscType::Float(0.3),
            OscType::Int(1),
            OscType::Float(0.3),
            OscType::Int(2),
            OscType::Float(0.3),
            OscType::Int(3),
            OscType::Float(0.3),
        ],
    ))
    .expect("/b_set");
    start_player(&mut osc);

    let out = render(&mut world, SR as usize / 4);
    assert!(
        out[1024..].iter().all(|s| (s - 0.3).abs() < 0.01),
        "/b_set samples should play back as a 0.3 constant"
    );
}

#[test]
fn b_setn_writes_a_waveform() {
    let (mut osc, mut world) = player_engine();
    let frames = 240usize; // 2 cycles over 240 frames at 48 kHz -> 400 Hz when looped
    osc.apply_bytes(&msg(
        "/b_alloc",
        vec![
            OscType::Int(0),
            OscType::Int(frames as i32),
            OscType::Int(1),
        ],
    ))
    .expect("/b_alloc");
    let mut args = vec![
        OscType::Int(0),
        OscType::Int(0),
        OscType::Int(frames as i32),
    ];
    for i in 0..frames {
        let v = (TAU * 2.0 * i as f32 / frames as f32).sin() * 0.5;
        args.push(OscType::Float(v));
    }
    osc.apply_bytes(&msg("/b_setn", args)).expect("/b_setn");
    start_player(&mut osc);

    let out = render(&mut world, SR as usize / 4);
    assert!(
        goertzel(&out, 400.0) > 5.0 * goertzel(&out, 800.0),
        "a /b_setn waveform should loop back as a 400 Hz tone"
    );
}

#[test]
fn b_set_sample_rate_updates_the_query_mirror() {
    let (mut osc, _world) = player_engine();
    osc.apply_bytes(&msg(
        "/b_alloc",
        vec![OscType::Int(0), OscType::Int(100), OscType::Int(1)],
    ))
    .expect("/b_alloc");
    let _ = osc.take_replies(); // drop the /b_alloc /done

    osc.apply_bytes(&msg(
        "/b_setSampleRate",
        vec![OscType::Int(0), OscType::Float(22_050.0)],
    ))
    .expect("/b_setSampleRate");
    osc.apply_bytes(&msg("/b_query", vec![OscType::Int(0)]))
        .expect("/b_query");
    let replies = osc.take_replies();
    let info = find(&replies, "/b_info").expect("/b_info");
    assert_eq!(
        info.args,
        vec![
            OscType::Int(0),
            OscType::Int(100),
            OscType::Int(1),
            OscType::Float(22_050.0),
        ]
    );
}

#[test]
fn b_gen_sine1_plays_a_tone() {
    let (mut osc, mut world) = player_engine();
    // sine1 lays one cycle of its (single) partial across the whole table; 120 frames @ 48 kHz looped
    // -> 400 Hz.
    osc.apply_bytes(&msg(
        "/b_alloc",
        vec![OscType::Int(0), OscType::Int(120), OscType::Int(1)],
    ))
    .expect("/b_alloc");
    osc.apply_bytes(&msg(
        "/b_gen",
        vec![
            OscType::Int(0),
            OscType::String("sine1".to_string()),
            OscType::Int(0), // flags
            OscType::Float(1.0),
        ],
    ))
    .expect("/b_gen");
    start_player(&mut osc);
    let out = render(&mut world, SR as usize / 4);
    assert!(
        goertzel(&out, 400.0) > 5.0 * goertzel(&out, 800.0),
        "a /b_gen sine1 buffer should loop back as a 400 Hz tone"
    );
}

#[test]
fn b_gen_copy_duplicates_a_buffer() {
    let (mut osc, mut world) = player_engine();
    // Generate a tone into buffer 1, copy it into buffer 0, then play buffer 0.
    for buf in [0, 1] {
        osc.apply_bytes(&msg(
            "/b_alloc",
            vec![OscType::Int(buf), OscType::Int(120), OscType::Int(1)],
        ))
        .expect("/b_alloc");
    }
    osc.apply_bytes(&msg(
        "/b_gen",
        vec![
            OscType::Int(1),
            OscType::String("sine1".to_string()),
            OscType::Int(0),
            OscType::Float(1.0),
        ],
    ))
    .expect("/b_gen sine1");
    // copy: dstStart 0, srcBuf 1, srcStart 0, count 120.
    osc.apply_bytes(&msg(
        "/b_gen",
        vec![
            OscType::Int(0),
            OscType::String("copy".to_string()),
            OscType::Int(0),
            OscType::Int(0),
            OscType::Int(1),
            OscType::Int(0),
            OscType::Int(120),
        ],
    ))
    .expect("/b_gen copy");
    start_player(&mut osc);
    let out = render(&mut world, SR as usize / 4);
    assert!(
        goertzel(&out, 400.0) > 5.0 * goertzel(&out, 800.0),
        "the copied buffer should play the same 400 Hz tone"
    );
}
