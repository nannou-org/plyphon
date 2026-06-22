//! The demo's synth, built with plyphon's programmatic SynthDef API.
//!
//! A single `SinOsc.ar(freq) -> Out.ar(0, [sig; channels])` graph, driven through a [`World`]. This
//! is the same engine path a real host would use: build a def, instantiate a synth, then pump audio
//! a block at a time.

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World, engine,
};

/// Demo frequency in Hz.
const FREQ: f32 = 440.0;

/// Build a [`World`] already playing a `FREQ`-Hz sine on every output channel.
pub fn build_world(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // `SinOsc.ar(freq)` is UGen 0; `Out.ar(0, [sig; channels])` is UGen 1.
    let mut out_inputs = vec![InputRef::Constant(0.0)]; // input 0: starting bus channel
    for _ in 0..channels {
        out_inputs.push(InputRef::Ugen { ugen: 0, output: 0 }); // one copy per channel
    }
    let def = SynthDef {
        name: "sine".to_string(),
        params: vec![Param {
            name: "freq".to_string(),
            default: FREQ,
        }],
        ugens: vec![
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UgenSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);
    // Queue the synth; the World applies the queued command on its first `fill`. Dropping the
    // controller here is fine - the command persists in the ring for the World to drain.
    let _ = controller.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail);

    world
}
