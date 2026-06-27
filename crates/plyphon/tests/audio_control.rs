//! AudioControl (rate-aware params): an audio-rate parameter's value is lifted to an audio wire each
//! block, so it feeds audio-rate inputs. `/n_set` and `/n_map` still target the param's value slot
//! (unchanged), and the audio wire follows.

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const BLOCK: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: 48_000.0,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    }
}

fn one(world: &mut World) -> f32 {
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf[0]
}

/// `Out.ar(0, amp)` where `amp` is an audio param. `Out.ar` reads an *audio* input, so this only
/// produces the param's value if the param resolved to an audio wire (a control param would read as
/// an empty audio slice -> silence).
fn audio_param_def() -> SynthDef {
    SynthDef {
        name: "ac".to_string(),
        params: vec![Param::audio("amp", 0.3)],
        units: vec![UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![InputRef::Constant(0.0), InputRef::Param(0)],
            0,
        )],
    }
}

#[test]
fn audio_control_lifts_value_to_an_audio_wire() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(audio_param_def());
    let node = controller
        .synth_new("ac", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    assert!(
        (one(&mut world) - 0.3).abs() < 1e-6,
        "audio param reads its default through an audio wire"
    );
    controller.set_control(node, 0, 0.7).unwrap();
    assert!(
        (one(&mut world) - 0.7).abs() < 1e-6,
        "/n_set updates the audio param's value slot, and the audio wire follows"
    );
}

#[test]
fn audio_control_follows_n_map() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(audio_param_def());
    let node = controller
        .synth_new("ac", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    // /n_map the audio param to control bus 0: the bus fills the value slot, which fills the wire.
    controller.set_control_bus(0, 0.5).unwrap();
    controller.map_control(node, 0, Some(0)).unwrap();

    assert!(
        (one(&mut world) - 0.5).abs() < 1e-6,
        "mapped bus value reaches the audio wire"
    );
    controller.set_control_bus(0, 0.9).unwrap();
    assert!(
        (one(&mut world) - 0.9).abs() < 1e-6,
        "the audio param tracks the control bus"
    );
}
