//! Host-driven control buses: an arpeggio steered from the control plane, via cpal, natively and on
//! the web.
//!
//! Two continuous voices an octave apart both read their frequency from the *same* control bus
//! (their `freq` control is mapped to it with `/n_map`/[`Controller::map_control`]). The control
//! plane then plays a melody by writing that one bus over time with `/c_set`/
//! [`Controller::set_control_bus`] - retuning both voices at once. This is what a control bus buys
//! you over setting each control directly: one value drives many synths (and could equally be driven
//! by an in-engine unit, as in `example-routing`).
//!
//! Retuning a running oscillator is click-free (its phase is continuous), so no per-note envelopes
//! are needed. As in `example-motif`, the only platform-specific part is how the control
//! plane is ticked (a thread loop natively, a timer on the web).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, Controller, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

/// The melody the control plane writes to the frequency bus (Hz) - a pentatonic loop.
const FREQS: [f32; 5] = [220.0, 247.5, 293.3, 330.0, 392.0];
/// One step (note) every this many milliseconds.
const TICK_MS: u32 = 240;
/// The control bus both voices read their frequency from.
const FREQ_BUS: u32 = 0;
/// `freq` is parameter 0, `ratio` is parameter 1, in the voice SynthDef below.
const PARAM_FREQ: usize = 0;
const PARAM_RATIO: usize = 1;
/// A gentle master gain (two voices sum).
const GAIN: f32 = 0.15;

/// SuperCollider binary-operator index for multiply (see `BinaryOpUGen`).
const OP_MUL: i16 = 2;

/// The control plane: holds the `Controller` and steps the arpeggio by writing the frequency bus.
struct Controls {
    controller: Controller,
    step: usize,
}

impl Controls {
    /// One step: write the next note to the frequency bus. Both mapped voices follow it.
    fn tick(&mut self) {
        let freq = FREQS[self.step % FREQS.len()];
        self.step += 1;
        let _ = self.controller.set_control_bus(FREQ_BUS, freq);
    }
}

/// Build the engine: a voice SynthDef, two voices (unison and an octave up) whose `freq` is mapped
/// to the shared control bus, and the control plane that will drive that bus. Identical on native
/// and web.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // voice := SinOsc.ar(freq * ratio) -> Out, with `freq` and `ratio` controls.
    //   unit 0: BinaryOpUGen freq * ratio (control rate)
    //   unit 1: SinOsc.ar(freq * ratio)
    //   unit 2: Out, copied to every channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 1, output: 0 });
    }
    let def = SynthDef {
        name: "voice".to_string(),
        params: vec![Param::control("freq", 220.0), Param::control("ratio", 1.0)],
        units: vec![
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Param(PARAM_FREQ as u32),
                    InputRef::Param(PARAM_RATIO as u32),
                ],
                num_outputs: 1,
                special_index: OP_MUL,
            },
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);

    // Two voices, an octave apart, both reading frequency from the shared control bus.
    for ratio in [1.0, 2.0] {
        if let Ok(node) = controller.synth_new("voice", ROOT_GROUP_ID, AddAction::Tail) {
            let _ = controller.set_control(node, PARAM_RATIO, ratio);
            let _ = controller.map_control(node, PARAM_FREQ, Some(FREQ_BUS));
        }
    }

    (
        Controls {
            controller,
            step: 0,
        },
        world,
    )
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");
    let config = device
        .default_output_config()
        .expect("no default output config");

    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
        format => panic!("unsupported sample format: {format}"),
    }
}

/// Play the demo: the `World` feeds the cpal stream while the `Controls` step the arpeggio off the
/// audio thread by writing the frequency bus.
fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let (controls, mut source) = build(sample_rate, channels);
    let mut scratch: Vec<f32> = Vec::new();

    let stream = device
        .build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                source.fill(&mut scratch, channels);
                for (out, sample) in output.iter_mut().zip(scratch.iter()) {
                    *out = T::from_sample(*sample * GAIN);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream");
    stream.play().expect("failed to start audio stream");

    run_control_plane(controls, stream);
}

/// Tick the control plane off the audio thread for the demo's lifetime, holding the stream alive.
#[cfg(not(target_arch = "wasm32"))]
fn run_control_plane(mut controls: Controls, _stream: cpal::Stream) {
    use std::time::Duration;
    println!("playing a bus-driven arpeggio for 10s...");
    let ticks = 10_000 / TICK_MS;
    for _ in 0..ticks {
        controls.tick();
        std::thread::sleep(Duration::from_millis(u64::from(TICK_MS)));
    }
}

/// On the web, `main` returns immediately, so run the control plane on a periodic timer and keep
/// both it and the audio stream alive.
#[cfg(target_arch = "wasm32")]
fn run_control_plane(mut controls: Controls, stream: cpal::Stream) {
    let interval = gloo_timers::callback::Interval::new(TICK_MS, move || controls.tick());
    interval.forget();
    std::mem::forget(stream);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

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

    fn render(world: &mut World, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);
        out
    }

    /// Writing the shared bus should retune both mapped voices: a single `/c_set` makes the unison
    /// voice sound at the bus frequency and the octave voice one octave up. Exercises `/n_map` +
    /// `/c_set` end to end, headlessly.
    #[test]
    fn bus_retunes_both_mapped_voices() {
        let (mut controls, mut world) = build(SR, 1);
        // Before any write the bus reads 0, so the voices are silent; the first tick sets FREQS[0].
        controls.tick();
        let note = FREQS[0];
        let out = render(&mut world, (SR / 4.0) as usize);

        assert!(out.iter().any(|s| s.abs() > 0.05), "the voices were silent");
        let fundamental = goertzel(&out, note);
        let octave = goertzel(&out, note * 2.0);
        let off = goertzel(&out, note * 1.5);
        assert!(
            fundamental > 5.0 * off && octave > 5.0 * off,
            "expected both mapped voices ({note} Hz and {} Hz) to follow the bus",
            note * 2.0
        );
    }
}
