//! Info units: route each engine-constant / buffer-info UGen to its own output channel and read the
//! single value back. Constants are broadcast across the block, so frame 0 of each channel carries
//! the reported value.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
const CHANNELS: usize = 6;

/// An audio-rate `Info`/`BufInfo` UGen (`name`) routed to output channel `ch`. Returns the two
/// `UnitSpec`s: the info unit, then an `Out.ar(ch, info)`.
fn route(name: &str, inputs: Vec<InputRef>, ch: f32, info_unit: u32) -> [UnitSpec; 2] {
    [
        UnitSpec::new(name, Rate::Audio, inputs, 1),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(ch),
                InputRef::Unit {
                    unit: info_unit,
                    output: 0,
                },
            ],
            0,
        ),
    ]
}

/// Build a def whose units are `pairs` flattened (each pair is `[info, out]`), assigning info-unit
/// indices automatically (info at 2*i, its Out at 2*i+1).
fn def_from(name: &str, specs: Vec<(&str, Vec<InputRef>, f32)>) -> SynthDef {
    let mut units = Vec::new();
    for (i, (unit_name, inputs, ch)) in specs.into_iter().enumerate() {
        let info_unit = (2 * i) as u32;
        units.extend(route(unit_name, inputs, ch, info_unit));
    }
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units,
    }
}

/// Render one frame of `CHANNELS`-wide output and return it.
fn one_frame(world: &mut World) -> Vec<f32> {
    let mut buf = vec![0.0f32; CHANNELS];
    world.fill(&mut buf, CHANNELS);
    buf
}

fn approx(a: f32, b: f32, what: &str) {
    assert!(
        (a - b).abs() <= 1e-3 * b.abs().max(1.0),
        "{what}: got {a}, expected {b}"
    );
}

#[test]
fn engine_constants() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: CHANNELS,
        block_size: BLOCK,
        ..Options::default()
    });
    controller.add_synthdef(def_from(
        "consts",
        vec![
            ("SampleRate", vec![], 0.0),
            ("SampleDur", vec![], 1.0),
            ("RadiansPerSample", vec![], 2.0),
            ("ControlRate", vec![], 3.0),
            ("ControlDur", vec![], 4.0),
            ("NumOutputBuses", vec![], 5.0),
        ],
    ));
    controller
        .synth_new("consts", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    let f = one_frame(&mut world);
    approx(f[0], SR as f32, "SampleRate");
    approx(f[1], (1.0 / SR) as f32, "SampleDur");
    approx(
        f[2],
        (core::f64::consts::TAU / SR) as f32,
        "RadiansPerSample",
    );
    approx(f[3], (SR / BLOCK as f64) as f32, "ControlRate");
    approx(f[4], (BLOCK as f64 / SR) as f32, "ControlDur");
    approx(f[5], CHANNELS as f32, "NumOutputBuses");
}

#[test]
fn buffer_info() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: CHANNELS,
        block_size: BLOCK,
        ..Options::default()
    });
    let (frames, channels, buf_sr) = (100usize, 2usize, 44_100.0f64);
    controller
        .buffer_alloc(0, frames, channels, buf_sr)
        .expect("buffer_alloc");
    controller.add_synthdef(def_from(
        "bufinfo",
        vec![
            ("BufFrames", vec![InputRef::Constant(0.0)], 0.0),
            ("BufChannels", vec![InputRef::Constant(0.0)], 1.0),
            ("BufSampleRate", vec![InputRef::Constant(0.0)], 2.0),
            ("BufRateScale", vec![InputRef::Constant(0.0)], 3.0),
            ("BufDur", vec![InputRef::Constant(0.0)], 4.0),
            ("BufSamples", vec![InputRef::Constant(0.0)], 5.0),
        ],
    ));
    controller
        .synth_new("bufinfo", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    let f = one_frame(&mut world);
    approx(f[0], frames as f32, "BufFrames");
    approx(f[1], channels as f32, "BufChannels");
    approx(f[2], buf_sr as f32, "BufSampleRate");
    approx(f[3], (buf_sr / SR) as f32, "BufRateScale");
    approx(f[4], (frames as f64 / buf_sr) as f32, "BufDur");
    approx(f[5], (frames * channels) as f32, "BufSamples");
}
