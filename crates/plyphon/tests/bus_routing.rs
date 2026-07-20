//! Signal routing through buses: one synth writes a bus, another reads it. Covers `In.ar` over a
//! private audio bus, `Out.kr`/`In.kr` over a control bus, `/c_set`, and `/n_map` control mapping.
//! Each case is verified with a Goertzel tone detector on the rendered output.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f32 = 48_000.0;

/// A single-output mono engine with no hardware inputs, so private audio buses start at channel 1.
fn options() -> Options {
    Options {
        sample_rate: SR as f64,
        output_channels: 1,
        input_channels: 0,
        ..Options::default()
    }
}

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

/// Render `frames` of mono audio across varying buffer sizes (exercising the reblocker).
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

/// Assert `freq` dominates `other` in `samples`.
fn assert_dominant(samples: &[f32], freq: f32, other: f32) {
    let a = goertzel(samples, freq);
    let b = goertzel(samples, other);
    assert!(
        a > 5.0 * b,
        "expected {freq} Hz dominant: m{freq}={a}, m{other}={b}"
    );
}

/// `SinOsc.ar(freq) -> Out.ar(bus)`: writes a tone to audio `bus`.
fn audio_writer(name: &str, freq: f32, bus: f32) -> SynthDef {
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
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(bus),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

/// `In.ar(bus) -> Out.ar(0)`: copies audio `bus` to the output.
fn audio_reader(name: &str, bus: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Audio, vec![InputRef::Constant(bus)], 1),
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

/// `SinOsc.ar(In.kr(bus)) -> Out.ar(0)`: a tone whose frequency is read from control `bus`.
fn control_reader(name: &str, bus: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Control, vec![InputRef::Constant(bus)], 1),
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn audio_bus_routes_between_synths() {
    let (mut controller, _nrt, mut world) = engine(options());
    // Writer puts 440 Hz on private audio bus 16; reader copies bus 16 to output 0.
    controller.add_synthdef(audio_writer("writer", 440.0, 16.0));
    controller.add_synthdef(audio_reader("reader", 16.0));
    // Order matters: the writer must run before the reader within a block (add it first, at the
    // tail), so the reader sees this block's signal rather than the previous block's.
    controller
        .synth_new("writer", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    controller
        .synth_new("reader", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    let out = render(&mut world, SR as usize / 4);
    assert!(
        out.iter().any(|s| s.abs() > 0.1),
        "routed signal was silent"
    );
    assert_dominant(&out, 440.0, 880.0);
}

#[test]
fn control_bus_routes_out_kr_to_in_kr() {
    let (mut controller, _nrt, mut world) = engine(options());
    // Producer writes the constant 330 onto control bus 5 via Out.kr; consumer reads it as a
    // frequency via In.kr.
    let producer = SynthDef {
        name: "kwriter".to_string(),
        params: vec![],
        units: vec![UnitSpec::new(
            "Out",
            Rate::Control,
            vec![InputRef::Constant(5.0), InputRef::Constant(330.0)],
            0,
        )],
    };
    controller.add_synthdef(producer);
    controller.add_synthdef(control_reader("kreader", 5.0));
    controller
        .synth_new("kwriter", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    controller
        .synth_new("kreader", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    let out = render(&mut world, SR as usize / 4);
    assert_dominant(&out, 330.0, 660.0);
}

#[test]
fn c_set_drives_in_kr() {
    let (mut controller, _nrt, mut world) = engine(options());
    controller.add_synthdef(control_reader("creader", 9.0));
    controller
        .synth_new("creader", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    // /c_set the frequency bus to 440, render, expect 440 Hz.
    controller.set_control_bus(9, 440.0).unwrap();
    let a = render(&mut world, SR as usize / 4);
    assert_dominant(&a, 440.0, 220.0);

    // Change the bus to 220; the running synth follows it.
    controller.set_control_bus(9, 220.0).unwrap();
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 4);
    assert_dominant(&b, 220.0, 440.0);
}

#[test]
fn n_map_maps_control_to_bus() {
    let (mut controller, _nrt, mut world) = engine(options());
    // SinOsc.ar(freq) -> Out.ar(0) with a settable `freq` control defaulting to 440.
    let def = SynthDef {
        name: "mapped".to_string(),
        params: vec![plyphon::Param::control("freq", 440.0)],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
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
    };
    controller.add_synthdef(def);
    let node = controller
        .synth_new("mapped", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();

    // Map the freq control to bus 7 and drive that bus: the synth now ignores its 440 default.
    controller.set_control_bus(7, 220.0).unwrap();
    controller.map_control(node, 0, Some(7)).unwrap();
    let _ = render(&mut world, 512);
    let a = render(&mut world, SR as usize / 4);
    assert_dominant(&a, 220.0, 440.0);

    // Move the bus; the mapped control follows.
    controller.set_control_bus(7, 660.0).unwrap();
    let _ = render(&mut world, 512);
    let b = render(&mut world, SR as usize / 4);
    assert_dominant(&b, 660.0, 220.0);

    // Unmap and set the control directly: it no longer follows the bus.
    controller.map_control(node, 0, None).unwrap();
    controller.set_control(node, 0, 440.0).unwrap();
    controller.set_control_bus(7, 660.0).unwrap();
    let _ = render(&mut world, 512);
    let c = render(&mut world, SR as usize / 4);
    assert_dominant(&c, 440.0, 660.0);
}
