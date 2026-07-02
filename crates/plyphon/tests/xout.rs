//! `XOut`: crossfade a signal into a bus against whatever earlier synths wrote there this block -
//! `bus = bus*(1-xfade) + in*xfade`. `xfade = 0` leaves the bus, `xfade = 1` replaces it.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f64 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `DC(a) -> Out(0)` then `DC(b) -> XOut(0, xfade)`, both on the output bus in node order. Returns the
/// resulting output value (steady, so the last sample).
fn out_then_xout(a: f32, b: f32, xfade: f32) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    // Synth A: writes `a` onto bus 0 with a plain Out.
    controller.add_synthdef(SynthDef {
        name: "a".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(a)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    });
    // Synth B: crossfades `b` into bus 0 against A's output. XOut has no signal output.
    controller.add_synthdef(SynthDef {
        name: "b".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(b)], 1),
            UnitSpec::new("XOut", Rate::Audio, vec![c(0.0), c(xfade), u(0)], 0),
        ],
    });
    controller
        .synth_new("a", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller
        .synth_new("b", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = [0.0f32; 64];
    world.fill(&mut buf, 1);
    buf[63]
}

#[test]
fn xout_crossfades_against_earlier_bus_content() {
    // A wrote 1.0; B crossfades 0.5 in: bus = 1.0*(1-xf) + 0.5*xf.
    assert!(
        (out_then_xout(1.0, 0.5, 0.0) - 1.0).abs() < 1e-6,
        "xfade 0 keeps the bus"
    );
    assert!(
        (out_then_xout(1.0, 0.5, 1.0) - 0.5).abs() < 1e-6,
        "xfade 1 replaces the bus"
    );
    assert!(
        (out_then_xout(1.0, 0.5, 0.5) - 0.75).abs() < 1e-6,
        "xfade 0.5 is the half-mix"
    );
    assert!(
        (out_then_xout(1.0, 0.5, 0.25) - 0.875).abs() < 1e-6,
        "xfade 0.25 is a quarter toward the input"
    );
}

#[test]
fn xout_alone_is_first_writer() {
    // With no earlier writer, XOut is the first to touch the channel and treats the bus as zero, so
    // the output is in*xfade.
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "x".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(0.8)], 1),
            UnitSpec::new("XOut", Rate::Audio, vec![c(0.0), c(0.5), u(0)], 0),
        ],
    });
    controller
        .synth_new("x", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = [0.0f32; 64];
    world.fill(&mut buf, 1);
    assert!(
        (buf[63] - 0.4).abs() < 1e-6,
        "a lone XOut lands in*xfade = 0.8*0.5 = 0.4, got {}",
        buf[63]
    );
}

#[test]
fn lone_reblocked_xout_never_mixes_stale_audio() {
    // A sole-first-writer XOut in a *reblocked* def: its first tick's whole-channel clear must
    // cover the later ticks' slices too, so no slice crossfades against a prior block's audio
    // lingering on the (persistent) private bus.
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        input_channels: 0, // private audio channels start at bus 1
        ..Options::default()
    });
    // Pollute private bus 1 with 1.0, then free the writer - the private channel persists.
    controller.add_synthdef(SynthDef {
        name: "w".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(1.0), u(0)], 0),
        ],
    });
    let w = controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = [0.0f32; 64];
    world.fill(&mut buf, 1);
    controller.free(w).unwrap();
    world.fill(&mut buf, 1);

    // A reblocked (16-sample) def whose XOut is the sole writer of bus 1, and a reader after it.
    controller.add_synthdef_reblocked(
        SynthDef {
            name: "x".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("DC", Rate::Audio, vec![c(0.8)], 1),
                UnitSpec::new("XOut", Rate::Audio, vec![c(1.0), c(0.5), u(0)], 0),
            ],
        },
        16,
    );
    controller.add_synthdef(SynthDef {
        name: "r".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    });
    controller
        .synth_new("x", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    world.fill(&mut buf, 1);
    for (i, &s) in buf.iter().enumerate() {
        assert!(
            (s - 0.4).abs() < 1e-6,
            "sample {i}: every tick-slice should land in*xfade = 0.4, got {s} \
             (stale prior-block audio leaked into a later slice)"
        );
    }
}
