//! Buffer writing: `RecordBuf` captures a signal into a buffer (read back with `PlayBuf`), a
//! non-looping `RecordBuf` frees its synth at the end via a done action, and `BufWr` writes a value
//! at a chosen frame. Exercises the `buffer_at_mut` write seam.

use plyphon::{
    AddAction, Buffer, Controller, Event, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, World, engine,
};

const SR: f32 = 48_000.0;

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

/// Render `frames` samples (mono), exercising several block sizes so cross-block state is tested.
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

fn zeroed_buffer(frames: usize, channels: usize) -> Buffer {
    Buffer::from_interleaved(vec![0.0; frames * channels], channels, SR as f64)
}

/// `SinOsc.ar(freq) -> RecordBuf.ar([sig], bufnum=0, offset=0, recLevel=1, preLevel=0, run=1, loop,
/// trig=0, doneAction)`.
fn record_def(name: &str, freq: f32, looping: f32, done_action: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "RecordBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),         // bufnum
                    InputRef::Constant(0.0),         // offset
                    InputRef::Constant(1.0),         // recLevel
                    InputRef::Constant(0.0),         // preLevel
                    InputRef::Constant(1.0),         // run
                    InputRef::Constant(looping),     // loop
                    InputRef::Constant(0.0),         // trig
                    InputRef::Constant(done_action), // doneAction
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                1,
            ),
        ],
    }
}

/// `PlayBuf.ar(1, 0, rate=1, trig=0, startPos=0, loop=1, doneAction=0) -> Out.ar(0)`.
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

#[test]
fn record_buf_round_trips_through_play_buf() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    // 4800 frames = exactly 44 cycles of 440 Hz: a seamless loop once recorded.
    controller
        .buffer_set(0, Box::new(zeroed_buffer(4800, 1)))
        .unwrap();
    controller.add_synthdef(record_def("rec", 440.0, 1.0, 0.0));
    let rec = controller
        .synth_new("rec", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // Fill the whole buffer with the recorded 440 Hz sine, then free the recorder.
    let _ = render(&mut world, 6000);
    controller.free(rec).unwrap();
    let _ = render(&mut world, 128);

    // Play the recorded buffer back and confirm it is a 440 Hz tone.
    controller.add_synthdef(play_def("play"));
    controller
        .synth_new("play", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, SR as usize / 4);
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "recorded playback was silent"
    );
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "expected the recorded 440 Hz tone on playback"
    );
}

#[test]
fn record_buf_done_action_frees_synth() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    // A short non-looping record: doneAction 2 frees the synth when the buffer fills.
    controller
        .buffer_set(0, Box::new(zeroed_buffer(240, 1)))
        .unwrap();
    controller.add_synthdef(record_def("once", 440.0, 0.0, 2.0));
    let node = controller
        .synth_new("once", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let _ = render(&mut world, 2048); // past the 240-frame buffer
    nrt.process();
    assert!(
        std::iter::from_fn(|| nrt.poll())
            .any(|e| matches!(e, Event::NodeEnded(n) if n.node == node)),
        "expected a NodeEnded for the self-freed RecordBuf synth"
    );
}

/// Drive a `/b_write` copy-out of the buffer at `bufnum` to completion, returning every interleaved
/// frame the engine streamed out. Each tick advances the engine (which copies a poolful of chunks
/// without touching the slot), drops trash (so the finished recording abandons the consumer), and
/// drains whatever was produced - exactly the non-blocking multi-tick loop the host runs.
fn drive_write_out(
    controller: &mut Controller,
    nrt: &mut Nrt,
    world: &mut World,
    bufnum: usize,
    channels: usize,
    frames: usize,
) -> Vec<f32> {
    // A small chunk pool (4 chunks of 64 frames) forces the multi-tick back-pressure path.
    let mut consumer = controller
        .buffer_write_out(bufnum, channels, SR as f64, 64, 4)
        .expect("cue the copy-out");
    let mut blk = [0.0f32; 64];
    let mut got = Vec::new();
    // A generous cap: the copy needs ~frames/256 ticks; far fewer than this bound.
    for _ in 0..(frames + 256) {
        world.fill(&mut blk, 1);
        nrt.process();
        while let Some(chunk) = consumer.pop_filled() {
            got.extend_from_slice(chunk.filled_samples());
            consumer.recycle(chunk);
        }
        if consumer.is_finished() {
            break;
        }
    }
    assert!(consumer.is_finished(), "copy-out did not finish");
    got
}

#[test]
fn buffer_write_out_snapshots_the_whole_buffer() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    // 1000 frames is not a multiple of the 64-frame chunk, so the final partial chunk's flush is
    // exercised (without it the snapshot would come up short by up to a chunk).
    let frames = 1000;
    let ramp: Vec<f32> = (0..frames).map(|f| f as f32).collect();
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(ramp.clone(), 1, SR as f64)),
        )
        .unwrap();
    // Let the SetBuffer land before the copy-out reads the slot.
    world.fill(&mut [0.0f32; 64], 1);

    let got = drive_write_out(&mut controller, &mut nrt, &mut world, 0, 1, frames);
    assert_eq!(got, ramp, "snapshot did not round-trip the buffer");

    // The slot is left in place: a second snapshot still round-trips, proving the copy never consumed
    // or replaced the buffer (RT readers stay intact).
    let again = drive_write_out(&mut controller, &mut nrt, &mut world, 0, 1, frames);
    assert_eq!(again, ramp, "buffer was disturbed by the first copy-out");
}

#[test]
fn buf_wr_writes_at_a_frame() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(zeroed_buffer(8, 1)))
        .unwrap();
    // DC.ar(0.75) -> BufWr.ar([value], bufnum=0, phase=3, loop=1): writes 0.75 into frame 3.
    controller.add_synthdef(SynthDef {
        name: "wr".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.75)], 1),
            UnitSpec::new(
                "BufWr",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),               // bufnum
                    InputRef::Constant(3.0),               // phase
                    InputRef::Constant(1.0),               // loop
                    InputRef::Unit { unit: 0, output: 0 }, // value
                ],
                1,
            ),
        ],
    });
    let wr = controller
        .synth_new("wr", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    // Block-aligned fills (one 64-sample control block each) so each command takes effect cleanly
    // with no reblocking leftover carried between calls.
    let mut blk = [0.0f32; 64];
    world.fill(&mut blk, 1); // write frame 3
    controller.free(wr).unwrap();
    world.fill(&mut blk, 1); // apply the free

    // Read it back: PlayBuf at rate 1 over the 8-frame loop yields buffer[i % 8] each sample.
    controller.add_synthdef(play_def("play"));
    controller
        .synth_new("play", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    world.fill(&mut blk, 1);
    for (i, &s) in blk.iter().enumerate() {
        let expected = if i % 8 == 3 { 0.75 } else { 0.0 };
        assert!(
            (s - expected).abs() < 1e-6,
            "frame {i} (buffer slot {}): got {s}, expected {expected}",
            i % 8
        );
    }
}
