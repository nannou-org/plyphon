//! End-to-end test of the real-time engine: build a `SinOsc.ar(freq) -> Out.ar(0, sig)` synth,
//! drive it through `World::fill` at a variety of host buffer sizes (exercising the reblocker), and
//! confirm via a Goertzel detector that the expected tone comes out and responds to `n_set`.

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World, engine,
};

const SR: f32 = 48_000.0;

fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![Param {
            name: "freq".to_string(),
            default: 440.0,
        }],
        ugens: vec![
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
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

/// Goertzel magnitude estimate at `freq` (Hz) over mono `samples` sampled at [`SR`].
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
    let power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    power.max(0.0).sqrt() / n as f32
}

/// Render `frames` of mono audio, cycling through several buffer sizes to exercise the reblocker.
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 100, 128, 480, 512, 333];
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames {
        let size = sizes[i % sizes.len()];
        i += 1;
        buf.clear();
        buf.resize(size, 0.0);
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

#[test]
fn sine_engine_plays_and_responds_to_n_set() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def());
    let node = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // Render ~0.5 s at the default 440 Hz across varying buffer sizes.
    let a = render(&mut world, SR as usize / 2);
    assert!(a.iter().any(|s| s.abs() > 0.1), "engine produced silence");
    assert!(
        a.iter().all(|s| s.abs() <= 1.001),
        "output exceeded full scale"
    );
    let m440 = goertzel(&a, 440.0);
    let m880 = goertzel(&a, 880.0);
    assert!(
        m440 > 5.0 * m880,
        "expected 440 Hz dominant: m440={m440}, m880={m880}"
    );

    // Change the frequency live; drain a little so the command takes effect, then re-analyse.
    controller.set_control(node, 0, 220.0).expect("set_control");
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 2);
    let n220 = goertzel(&b, 220.0);
    let n440 = goertzel(&b, 440.0);
    assert!(
        n220 > 5.0 * n440,
        "expected 220 Hz dominant after n_set: n220={n220}, n440={n440}"
    );

    // Free the node, then flush past the already-computed block still in the reblocker (commands
    // apply at block boundaries, so up to block_size-1 frames of tone are emitted first).
    controller.free(node).expect("free");
    let _ = render(&mut world, 1024);
    // Now the synth is gone; subsequent output is fully silent.
    let c = render(&mut world, SR as usize / 4);
    assert!(
        c.iter().all(|s| s.abs() < 1e-6),
        "expected silence after free"
    );
    // The NRT side drops the freed synth and surfaces the node-ended notification.
    assert!(
        nrt.process() >= 1,
        "expected the freed synth to reach the trash ring"
    );
    assert!(
        std::iter::from_fn(|| nrt.poll()).any(|e| e == plyphon::Event::NodeEnded { id: node }),
        "expected a NodeEnded event for the freed node"
    );
}
