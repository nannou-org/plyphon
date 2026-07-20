//! `ScopeOut` end to end: it streams every sample of its (multichannel) input off the audio thread
//! into a cued recording stream the app drains (`Controller::cue_scope` → `StreamConsumer`), plyphon's
//! shared-memory-free equivalent of scsynth's `ScopeOut2`. Covers exact sample recovery, multichannel
//! interleaving, several independent taps in one graph, and bounded-overrun safety.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, StreamConsumer, SynthDef, UnitSpec, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 2,
        block_size: BLOCK,
        ..Options::default()
    }
}

/// Drain every filled chunk currently available, returning the concatenated samples (interleaved).
fn drain(consumer: &mut StreamConsumer) -> Vec<f32> {
    let mut out = Vec::new();
    while let Some(chunk) = consumer.pop_filled() {
        out.extend_from_slice(chunk.filled_samples());
        consumer.recycle(chunk);
    }
    out
}

/// `DC.ar(value)` as a unit.
fn dc(value: f32) -> UnitSpec {
    UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(value)], 1)
}

/// `ScopeOut.ar(bufnum, [unit:output, …])`.
fn scope_out(bufnum: f32, channels: &[(u32, u32)]) -> UnitSpec {
    let mut inputs = vec![InputRef::Constant(bufnum)];
    inputs.extend(
        channels
            .iter()
            .map(|&(unit, output)| InputRef::Unit { unit, output }),
    );
    UnitSpec::new("ScopeOut", Rate::Audio, inputs, 0)
}

#[test]
fn scope_streams_every_input_sample() {
    // SinOsc.ar(f) fans out to both ScopeOut.ar(0, sig) and Out.ar(0, sig): the drained scope stream
    // must reproduce the bus output sample-for-sample, in order.
    let (mut controller, _nrt, mut world) = engine(opts());
    // A generous pool so nothing overruns before we drain at the end (10 blocks = 640 frames < 1024).
    let mut consumer = controller
        .cue_scope(0, 1, SR, BLOCK, 16)
        .expect("cue_scope");
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(300.0), InputRef::Constant(0.0)],
                1,
            ),
            scope_out(0.0, &[(0, 0)]),
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
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let blocks = 10;
    let mut expected = Vec::new();
    let mut buf = vec![0.0f32; BLOCK * 2];
    for _ in 0..blocks {
        world.fill(&mut buf, 2);
        // Bus 0 is channel 0 of the interleaved output.
        expected.extend(buf.chunks(2).map(|frame| frame[0]));
    }

    let got = drain(&mut consumer);
    assert_eq!(
        got.len(),
        expected.len(),
        "scope streamed a different length"
    );
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!((g - e).abs() < 1e-6, "sample {i}: scope {g} != bus {e}");
    }
    // It carried real signal, not silence.
    assert!(
        got.iter().any(|&s| s.abs() > 0.1),
        "scope stream was silent"
    );
}

#[test]
fn scope_interleaves_multiple_channels() {
    // ScopeOut.ar(0, [DC(0.25), DC(-0.5)]) -> the stream is [0.25, -0.5, 0.25, -0.5, …] (interleaved).
    let (mut controller, _nrt, mut world) = engine(opts());
    let mut consumer = controller.cue_scope(0, 2, SR, BLOCK, 8).expect("cue_scope");
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units: vec![dc(0.25), dc(-0.5), scope_out(0.0, &[(0, 0), (1, 0)])],
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let mut buf = vec![0.0f32; BLOCK * 2];
    for _ in 0..4 {
        world.fill(&mut buf, 2);
    }
    let got = drain(&mut consumer);
    assert!(!got.is_empty(), "scope produced nothing");
    for (i, frame) in got.chunks(2).enumerate() {
        assert!(
            (frame[0] - 0.25).abs() < 1e-6,
            "frame {i} ch0 = {}",
            frame[0]
        );
        assert!(
            (frame[1] + 0.5).abs() < 1e-6,
            "frame {i} ch1 = {}",
            frame[1]
        );
    }
}

#[test]
fn multiple_scope_taps_stream_independently() {
    // Two ScopeOut units in one SynthDef on distinct bufnums tap two different nodes; each drained
    // consumer must carry only its own tapped signal.
    let (mut controller, _nrt, mut world) = engine(opts());
    let mut a = controller.cue_scope(0, 1, SR, BLOCK, 8).expect("cue a");
    let mut b = controller.cue_scope(1, 1, SR, BLOCK, 8).expect("cue b");
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units: vec![
            dc(0.3),                   // unit 0 -> tap A (bufnum 0)
            dc(0.7),                   // unit 1 -> tap B (bufnum 1)
            scope_out(0.0, &[(0, 0)]), // ScopeOut(0, DC 0.3)
            scope_out(1.0, &[(1, 0)]), // ScopeOut(1, DC 0.7)
        ],
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let mut buf = vec![0.0f32; BLOCK * 2];
    for _ in 0..4 {
        world.fill(&mut buf, 2);
    }
    let (ga, gb) = (drain(&mut a), drain(&mut b));
    assert!(!ga.is_empty() && !gb.is_empty(), "a tap produced nothing");
    assert!(
        ga.iter().all(|&s| (s - 0.3).abs() < 1e-6),
        "tap A not all 0.3"
    );
    assert!(
        gb.iter().all(|&s| (s - 0.7).abs() < 1e-6),
        "tap B not all 0.7"
    );
}

#[test]
fn scope_overruns_without_panic_when_undrained() {
    // A tiny pool that cannot hold the whole run: filling many blocks without draining must not panic
    // (a bounded overrun drops surplus chunks); the engine keeps running.
    let (mut controller, _nrt, mut world) = engine(opts());
    let _consumer = controller.cue_scope(0, 1, SR, BLOCK, 2).expect("cue_scope");
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            scope_out(0.0, &[(0, 0)]),
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
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");

    let mut buf = vec![0.0f32; BLOCK * 2];
    let mut last = 0.0f32;
    for _ in 0..50 {
        world.fill(&mut buf, 2);
        last = buf.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
    }
    // The synth is still producing audio despite the undrained (overrun) scope.
    assert!(
        last > 0.1,
        "engine should keep running through scope overrun, got {last}"
    );
}
