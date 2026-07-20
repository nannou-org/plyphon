//! The simplest plyphon example: a continuous 440 Hz sine via cpal, natively and on the web.
//!
//! plyphon's `World` plays a one-synth graph (`SinOsc.ar -> Out`) and the cpal callback fills from
//! it. Nothing is ever freed, so there is no NRT work: the `Controller` and `Nrt` are dropped once
//! the synth is queued, leaving only the `World`. (The richer `example-motif` shows the full
//! lifecycle - starting and freeing notes, and running the `Nrt` - and is what the website ships.)
//!
//! The cpal output-stream plumbing (device resolution, sample-format reblocking, and on the web the
//! AudioWorklet backend) lives in the shared [`example_audio`] crate.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// Demo frequency in Hz.
const FREQ: f32 = 440.0;
/// A gentle master gain applied to the full-scale oscillator.
const GAIN: f32 = 0.2;

/// Build a `World` already playing a continuous `FREQ`-Hz sine on every output channel.
fn build(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // SinOsc.ar(FREQ) -> Out, the oscillator copied to every channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)]; // input 0: starting bus channel
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 0, output: 0 });
    }
    let def = SynthDef {
        name: "sine".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("sine", ROOT_GROUP_ID, AddAction::Tail, &[]);

    // The queued synth plays forever and never frees, so there is no NRT cleanup to do: drop the
    // `Controller` and `Nrt` and keep only the `World`.
    world
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // cpal's AudioWorklet backend re-instantiates this module on the audio thread, re-running
    // `main` there; only set up audio on the main browser thread.
    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!("playing a {FREQ} Hz sine for 10s...");

    // The `World` fills the cpal callback; `example_audio` handles device/format/host plumbing.
    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 10);
}
