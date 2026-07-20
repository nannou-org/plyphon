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
        .synth_new("ac", ROOT_GROUP_ID, AddAction::Tail, &[])
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
        .synth_new("ac", ROOT_GROUP_ID, AddAction::Tail, &[])
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

#[test]
fn n_mapa_maps_an_audio_param_to_an_audio_bus() {
    // input_channels: 0 so private audio buses start at 1 (after the single output channel).
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: 48_000.0,
        output_channels: 1,
        input_channels: 0,
        block_size: BLOCK,
        ..Options::default()
    });
    // writer: DC.ar(0.6) -> Out.ar(1) writes private audio bus 1.
    controller.add_synthdef(SynthDef {
        name: "writer".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.6)], 1),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(1.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller.add_synthdef(audio_param_def());
    // The writer must run before the reader, so it is at the head.
    controller
        .synth_new("writer", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let ac = controller
        .synth_new("ac", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    assert!(
        (one(&mut world) - 0.3).abs() < 1e-6,
        "unmapped reads the default"
    );
    // /n_mapa the audio param to audio bus 1.
    controller.map_control_audio(ac, 0, Some(1)).unwrap();
    assert!(
        (one(&mut world) - 0.6).abs() < 1e-6,
        "mapped takes the audio bus block"
    );
    // Unmap -> back to the value slot.
    controller.map_control_audio(ac, 0, None).unwrap();
    assert!(
        (one(&mut world) - 0.3).abs() < 1e-6,
        "unmapped reverts to the value slot"
    );
}
