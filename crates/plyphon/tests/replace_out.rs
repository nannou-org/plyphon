//! `ReplaceOut` overwrites a bus channel instead of summing onto it: an earlier `Out` writing 1.0 to
//! bus 0, followed by a `ReplaceOut` writing 0.5, leaves 0.5 - whereas two `Out`s would leave 1.5.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn dc(v: f32) -> UnitSpec {
    UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(v)], 1)
}

/// `Out.ar(0, DC(1.0)); <second>.ar(0, DC(0.5))` - two writers to bus 0. Returns the steady output.
fn two_writers(second: &str) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "w".to_string(),
        params: vec![],
        units: vec![
            dc(1.0),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
            dc(0.5),
            UnitSpec::new(
                second,
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; 128];
    world.fill(&mut buf, 1);
    buf[64]
}

#[test]
fn replace_out_overwrites_the_bus() {
    // A second `Out` sums (1.0 + 0.5 = 1.5); `ReplaceOut` overwrites (-> 0.5).
    assert!(
        (two_writers("Out") - 1.5).abs() < 1e-4,
        "two Outs should sum to 1.5"
    );
    assert!(
        (two_writers("ReplaceOut") - 0.5).abs() < 1e-4,
        "ReplaceOut should overwrite to 0.5"
    );
}
