//! Buffer playback: install a buffer, play it with `PlayBuf`, and confirm the looped tone comes out
//! and that a non-looping playback's done action frees its synth. Exercises the buffer table, the
//! `SetBuffer`/`FreeBuffer` install/swap path, and the `Nrt` dropping replaced/freed buffers.

use plyphon::{
    AddAction, Buffer, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World,
    engine,
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

/// `PlayBuf.ar(1, bufnum, rate, trigger, startPos, loop, doneAction) -> Out.ar(0)`.
fn play_buf_def(name: &str, rate: f32, looping: f32, done_action: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        ugens: vec![
            UgenSpec::new(
                "PlayBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),         // bufnum
                    InputRef::Constant(rate),        // rate
                    InputRef::Constant(0.0),         // trigger
                    InputRef::Constant(0.0),         // startPos
                    InputRef::Constant(looping),     // loop
                    InputRef::Constant(done_action), // doneAction
                ],
                1,
            ),
            UgenSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

/// A mono buffer holding `cycles` whole cycles of a sine over `frames` samples - seamless to loop.
fn sine_buffer(frames: usize, cycles: usize) -> Buffer {
    let samples: Vec<f32> = (0..frames)
        .map(|i| (std::f32::consts::TAU * cycles as f32 * i as f32 / frames as f32).sin())
        .collect();
    Buffer::from_interleaved(samples, 1, SR as f64)
}

#[test]
fn play_buf_loops_a_buffer() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    // 4800 frames holding exactly 44 cycles => a seamless 440 Hz loop.
    controller
        .buffer_set(0, Box::new(sine_buffer(4800, 44)))
        .unwrap();
    controller.add_synthdef(play_buf_def("loop", 1.0, 1.0, 0.0));
    controller
        .synth_new("loop", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let out = render(&mut world, SR as usize / 4);
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "buffer playback was silent"
    );
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "expected the 440 Hz buffer loop"
    );
}

#[test]
fn play_buf_done_action_frees_synth() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    // A short non-looping buffer; doneAction 2 frees the synth when it reaches the end.
    controller
        .buffer_set(0, Box::new(sine_buffer(240, 4)))
        .unwrap();
    controller.add_synthdef(play_buf_def("once", 1.0, 0.0, 2.0));
    let node = controller
        .synth_new("once", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // The 240-frame buffer ends within a few blocks; flush past it and confirm the synth freed.
    let _ = render(&mut world, 2048);
    let tail = render(&mut world, SR as usize / 8);
    assert!(
        tail.iter().all(|s| s.abs() < 1e-6),
        "expected silence after the buffer finished and freed its synth"
    );
    assert!(
        nrt.process() >= 1,
        "expected the freed synth to reach the trash ring"
    );
    assert!(
        std::iter::from_fn(|| nrt.poll()).any(|e| e == Event::NodeEnded { id: node }),
        "expected a NodeEnded for the self-freed PlayBuf synth"
    );
}

#[test]
fn buffer_free_routes_to_trash() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(sine_buffer(256, 1)))
        .unwrap();
    let _ = render(&mut world, 128); // install it
    controller.buffer_free(0).unwrap();
    let _ = render(&mut world, 128); // apply the free
    assert!(
        nrt.process() >= 1,
        "freeing a buffer should hand it to the trash ring for off-RT dropping"
    );
}
