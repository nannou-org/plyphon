//! Exercise `SendTrig`: an `Impulse.ar` drives a `SendTrig.ar`, which fires one trigger per rising
//! edge. The triggers flow over the dedicated trigger ring and surface via `Nrt::poll_trigger`,
//! each tagged with the firing synth's node id and carrying the configured trigger id and value.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, Trigger, UnitSpec, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

/// `SendTrig.ar(Impulse.ar(freq), id, value)` - no audio output, fires `/tr` per impulse.
fn send_trig_def(freq: f32, id: f32, value: f32) -> SynthDef {
    SynthDef {
        name: "trig".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "SendTrig",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(id),
                    InputRef::Constant(value),
                ],
                0,
            ),
        ],
    }
}

#[test]
fn send_trig_fires_one_tr_per_rising_edge() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    // Impulse every 48 samples (1000 Hz at 48 kHz).
    controller.add_synthdef(send_trig_def(1000.0, 7.0, 0.5));
    let node = controller
        .synth_new("trig", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // 40 whole control blocks = 2560 samples -> an impulse at 0, 48, 96, ... 2544: 54 rising edges.
    let mut buf = [0.0f32; BLOCK];
    for _ in 0..40 {
        world.fill(&mut buf, 1);
    }

    nrt.process();
    let mut triggers = Vec::new();
    while let Some(trigger) = nrt.poll_trigger() {
        triggers.push(trigger);
    }

    assert!(
        triggers.iter().all(|t| *t
            == Trigger {
                node,
                id: 7,
                value: 0.5
            }),
        "every /tr carries the firing node, the trigger id, and the value: {triggers:?}"
    );
    // ~one per 48-sample period over 2560 samples; per-sample scanning, not per-block (which would be
    // <= 40) or one-shot (which would be 1).
    let count = triggers.len();
    assert!(
        (50..=58).contains(&count),
        "expected ~54 triggers (one per rising edge), got {count}"
    );
}
