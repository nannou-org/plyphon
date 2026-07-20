//! `PlayBuf` interpolation: 4-point cubic (scsynth's `LOOP_BODY_4`), pinned against the closed
//! form at a known fractional phase.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f64 = 48_000.0;

/// scsynth's `cubicinterp` (the same kernel `plyphon_dsp::interp` ships).
fn cubicinterp(x: f32, y0: f32, y1: f32, y2: f32, y3: f32) -> f32 {
    let c0 = y1;
    let c1 = 0.5 * (y2 - y0);
    let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
    ((c3 * x + c2) * x + c1) * x + c0
}

#[test]
fn playbuf_reads_with_cubic_interpolation() {
    // An 8-frame looped ramp played at rate 0.5: every second output sample sits at phase i+0.5,
    // where linear interpolation would give the midpoint but cubic overshoots per the kernel.
    let data: Vec<f32> = vec![0.0, 1.0, 0.0, -1.0, 0.5, -0.5, 0.25, -0.25];
    let frames = data.len();
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(data.clone(), 1, SR)))
        .unwrap();
    controller.add_synthdef(SynthDef {
        name: "play".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "PlayBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0), // bufnum
                    InputRef::Constant(0.5), // rate
                    InputRef::Constant(0.0), // trig
                    InputRef::Constant(0.0), // startPos
                    InputRef::Constant(1.0), // loop
                    InputRef::Constant(0.0), // doneAction
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
    });
    controller
        .synth_new("play", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let mut out = [0.0f32; 64];
    world.fill(&mut out, 1);

    // Output sample 2k sits exactly on frame k; sample 2k+1 at phase k + 0.5.
    let at = |i: isize| data[(i.rem_euclid(frames as isize)) as usize];
    for k in 0..24 {
        let on = out[2 * k];
        assert!(
            (on - at(k as isize)).abs() < 1e-6,
            "sample {}: on-frame read should be exact, got {on}",
            2 * k
        );
        let mid = out[2 * k + 1];
        let expected = cubicinterp(
            0.5,
            at(k as isize - 1),
            at(k as isize),
            at(k as isize + 1),
            at(k as isize + 2),
        );
        assert!(
            (mid - expected).abs() < 1e-6,
            "sample {}: expected the cubic value {expected}, got {mid}",
            2 * k + 1
        );
    }
}
