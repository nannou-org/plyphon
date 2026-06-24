//! Exercise done actions and the NRT event/trash flow: a `Line.kr(..., doneAction: 2)` amplitude
//! envelope frees its own synth when it finishes; the `Nrt` then drops the freed synth and reports
//! a `NodeEnded` event.

use plyphon::{
    AddAction, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f32 = 48_000.0;

fn render(world: &mut plyphon::World, frames: usize) -> Vec<f32> {
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

/// `SinOsc.ar(440) * Line.kr(1, 0, 0.1, doneAction: 2)` -> `Out`. The line ramps the amplitude to
/// zero over 0.1 s, then frees the enclosing synth.
fn enveloped_sine() -> SynthDef {
    SynthDef {
        name: "env".to_string(),
        params: vec![],
        units: vec![
            // Line.kr(1, 0, 0.1, 2): amplitude 1 -> 0, then doneAction 2 (free self).
            UnitSpec {
                name: "Line".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.1),
                    InputRef::Constant(2.0),
                ],
                num_outputs: 1,
                special_index: 0,
            },
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            // SinOsc * Line (BinaryOpUGen, special index 2 = multiply).
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2,
            },
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn line_done_action_frees_synth() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(enveloped_sine());
    let node = controller
        .synth_new("env", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // It plays before the 0.1 s envelope completes.
    let early = render(&mut world, (SR * 0.05) as usize);
    assert!(
        early.iter().any(|s| s.abs() > 0.05),
        "synth should play before the envelope ends"
    );

    // Render past completion; the done action frees the synth.
    let _ = render(&mut world, (SR * 0.2) as usize);

    // The freed synth's state returns to the rt-pool on the audio thread (no trash); the NRT side
    // still surfaces the lifecycle notifications.
    nrt.process();
    let (mut started, mut ended) = (false, false);
    while let Some(event) = nrt.poll() {
        match event {
            Event::NodeStarted { id } if id == node => started = true,
            Event::NodeEnded { id } if id == node => ended = true,
            _ => {}
        }
    }
    assert!(started, "expected a NodeStarted notification");
    assert!(
        ended,
        "expected a NodeEnded notification from the done action"
    );

    // With the synth gone, the output is silent.
    let late = render(&mut world, (SR * 0.05) as usize);
    assert!(
        late.iter().all(|s| s.abs() < 1e-6),
        "expected silence after the synth freed itself"
    );
}
