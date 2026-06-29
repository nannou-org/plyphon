//! Demand-rate buffer and post units: `Dbufrd` reads a buffer, `Dbufwr` writes one, `Dpoll` posts a
//! demanded value. These exercise the `DemandWorld` reach threaded into the demand pull - the buffer
//! table and the node-message sink that a demand source could not touch before.
//!
//! A `Duty` consumer drives the pull on the audio thread and holds each demanded value, so a rendered
//! segment reads back the produced value directly (the same harness as `demand.rs`). Writes are
//! confirmed by reading the buffer back through `PlayBuf`.

use plyphon::{
    AddAction, Buffer, InputRef, NodeMsgKind, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// Segment duration fed to `Duty` (seconds); ~96 samples at `SR`, so the middle is clear of edges.
const SEG_DUR: f32 = 0.002;
const SEG: usize = 96;
const MID: usize = SEG / 2;

/// The value held during segment `k` (sampled at its middle).
fn segment(out: &[f32], k: usize) -> f32 {
    out[MID + k * SEG]
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

/// Render `frames` of mono audio, varying the host buffer size to exercise reblocking.
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `Dseries(length, start, step)` as a phase counter source.
fn dseries(length: f32, start: f32, step: f32) -> UnitSpec {
    UnitSpec::new(
        "Dseries",
        Rate::Demand,
        vec![
            InputRef::Constant(length),
            InputRef::Constant(start),
            InputRef::Constant(step),
        ],
        1,
    )
}

/// `Duty.ar(SEG_DUR, 0, level: unit `src`, 0)` driving demand source `src`, then `Out.ar(0, Duty)`.
/// `units` already holds the demand sources; `src` is the index of the source to hold.
fn drive_with_duty(name: &str, mut units: Vec<UnitSpec>, src: u32) -> SynthDef {
    let duty = units.len() as u32;
    units.push(UnitSpec::new(
        "Duty",
        Rate::Audio,
        vec![
            InputRef::Constant(SEG_DUR),
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: src,
                output: 0,
            },
            InputRef::Constant(0.0),
        ],
        1,
    ));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit {
                unit: duty,
                output: 0,
            },
        ],
        0,
    ));
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units,
    }
}

/// `PlayBuf.ar(1, 0, rate=1, trig=0, startPos=0, loop=1) -> Out.ar(0)`; rate 1 over an N-frame loop
/// yields `buffer[i % N]` each sample, so a single block reads the buffer contents back.
fn play_def(name: &str) -> SynthDef {
    SynthDef {
        name: name.to_string(),
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

fn start(def: SynthDef) -> (plyphon::Controller, Nrt, World, i32) {
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let name = def.name.clone();
    controller.add_synthdef(def);
    let node = controller
        .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (controller, nrt, world, node)
}

#[test]
fn dbufrd_reads_a_prefilled_buffer_and_loops() {
    // Buffer [10, 20, 30, 40]; Dbufrd(buf, phase: Dseries(inf, 0, 1), loop: 1) reads 10, 20, 30, 40,
    // then wraps to 10, 20. Proves the demand-rate read reach plus sc_loop wrapping.
    let phase = dseries(f32::INFINITY, 0.0, 1.0);
    let dbufrd = UnitSpec::new(
        "Dbufrd",
        Rate::Demand,
        vec![
            InputRef::Constant(0.0),               // bufnum
            InputRef::Unit { unit: 0, output: 0 }, // phase
            InputRef::Constant(1.0),               // loop
        ],
        1,
    );
    let def = drive_with_duty("rd", vec![phase, dbufrd], 1);

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(
                vec![10.0, 20.0, 30.0, 40.0],
                1,
                SR,
            )),
        )
        .unwrap();
    controller.add_synthdef(def);
    controller
        .synth_new("rd", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let out = render(&mut world, SEG * 6);
    for (k, e) in [10.0f32, 20.0, 30.0, 40.0, 10.0, 20.0]
        .into_iter()
        .enumerate()
    {
        let g = segment(&out, k);
        assert!(approx(g, e), "segment {k}: expected {e}, got {g}");
    }
}

#[test]
fn dbufwr_writes_a_sequence_then_reads_back_through_play_buf() {
    // Dbufwr(input: Dseq([0.5, 0.6, 0.7], 1), buf, phase: Dseries(inf, 0, 1), loop: 0): writes each
    // value at the next frame and passes it through. After the input exhausts, Duty holds the last
    // value (0.7). The buffer then reads back as [0.5, 0.6, 0.7, 0, 0, 0, 0, 0].
    let values = UnitSpec::new(
        "Dseq",
        Rate::Demand,
        vec![
            InputRef::Constant(1.0), // repeats
            InputRef::Constant(0.5),
            InputRef::Constant(0.6),
            InputRef::Constant(0.7),
        ],
        1,
    );
    let phase = dseries(f32::INFINITY, 0.0, 1.0);
    let dbufwr = UnitSpec::new(
        "Dbufwr",
        Rate::Demand,
        vec![
            InputRef::Unit { unit: 0, output: 0 }, // input
            InputRef::Constant(0.0),               // bufnum
            InputRef::Unit { unit: 1, output: 0 }, // phase
            InputRef::Constant(0.0),               // loop
        ],
        1,
    );
    let def = drive_with_duty("wr", vec![values, phase, dbufwr], 2);

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(vec![0.0; 8], 1, SR)))
        .unwrap();
    controller.add_synthdef(def);
    let wr = controller
        .synth_new("wr", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // Drive with block-aligned fills (one control block each) so the buffer readback stays aligned;
    // each ~96-sample segment pulls Dbufwr once, writing the next value at the next frame.
    let mut blk = [0.0f32; 64];
    for _ in 0..8 {
        world.fill(&mut blk, 1);
    }
    controller.free(wr).unwrap();
    world.fill(&mut blk, 1); // apply the free

    // Read the buffer back: PlayBuf at rate 1 over the 8-frame loop yields buffer[i % 8] each sample.
    controller.add_synthdef(play_def("play"));
    controller
        .synth_new("play", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    world.fill(&mut blk, 1);
    let expected = [0.5f32, 0.6, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0];
    for (i, &s) in blk.iter().enumerate() {
        let e = expected[i % 8];
        assert!(approx(s, e), "buffer slot {}: got {s}, expected {e}", i % 8);
    }
}

#[test]
fn dpoll_posts_each_demanded_value_to_the_host() {
    // Dpoll(in: Dseries(3, 0.5, 0.1), trigid: 42, run: 1, label: "dp"): posts 0.5, 0.6, 0.7 as Poll
    // node messages tagged with the node id and trigger id. Proves the node_id + node-message sink
    // reach from a demand source.
    let source = dseries(3.0, 0.5, 0.1);
    // Inputs follow scsynth's layout: [in, trigid, run, labelLen, labelChars...]; label "dp".
    let dpoll = UnitSpec::new(
        "Dpoll",
        Rate::Demand,
        vec![
            InputRef::Unit { unit: 0, output: 0 }, // in
            InputRef::Constant(42.0),              // trigid
            InputRef::Constant(1.0),               // run
            InputRef::Constant(2.0),               // labelLen
            InputRef::Constant(f32::from(b'd')),   // 'd'
            InputRef::Constant(f32::from(b'p')),   // 'p'
        ],
        1,
    );
    let def = drive_with_duty("poll", vec![source, dpoll], 1);
    let (_c, mut nrt, mut world, node) = start(def);

    let _ = render(&mut world, SEG * 5);
    nrt.process();

    let mut posted = Vec::new();
    while let Some(msg) = nrt.poll_node_msg() {
        assert_eq!(
            msg.kind,
            NodeMsgKind::Poll,
            "Dpoll must emit a Poll message"
        );
        assert_eq!(msg.node, node, "Poll tagged with the wrong node");
        assert_eq!(msg.reply_id, 42, "Poll must echo the trigid");
        let label = std::str::from_utf8(&msg.label[..msg.label_len as usize]).unwrap();
        assert_eq!(label, "dp", "Poll label mismatch");
        posted.push(msg.values[0]);
    }

    for e in [0.5f32, 0.6, 0.7] {
        assert!(
            posted.iter().any(|&v| approx(v, e)),
            "expected a Dpoll post of {e}; got {posted:?}"
        );
    }
}
