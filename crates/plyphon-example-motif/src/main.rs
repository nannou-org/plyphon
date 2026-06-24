//! Minimal cpal example, working natively and on the web from one identical engine.
//!
//! A `cpal` output stream's callback asks the engine's `plyphon::World` to fill an interleaved
//! `f32` buffer (`World::fill`). The control plane - a `Controls` value bundling a `Controller` and
//! an `Nrt` - is kept alive and ticked off the audio thread: it starts a looping motif of
//! self-freeing notes and runs the `Nrt` to drop the freed synths and react to notifications.
//!
//! The engine is pure Rust with no platform-specific paths, so [`build`] is identical on both
//! targets. The *only* difference is how the control plane is ticked (a thread loop natively, a
//! timer on the web), in [`run_control_plane`].

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UgenSpec, World, engine,
};

/// A short looping motif (Hz).
const FREQS: [f32; 4] = [440.0, 550.0, 660.0, 550.0];
/// How often to tick the control plane, in milliseconds.
const TICK_MS: u32 = 50;
/// Start a new note every this many ticks (~500 ms at `TICK_MS`).
const SPAWN_EVERY: u32 = 10;
/// Cap on simultaneously-playing notes, enforced from node notifications.
const MAX_VOICES: usize = 6;

/// The demo's control plane: kept alive by the host and ticked on an NRT cadence. It starts notes
/// (via the `Controller`) and runs the `Nrt` to drop freed synths and react to notifications - all
/// off the audio thread.
struct Controls {
    controller: Controller,
    nrt: Nrt,
    ticks: u32,
    next_freq: usize,
    /// Voices currently playing, tracked from `Event` notifications.
    playing: usize,
}

impl Controls {
    /// One NRT tick: drop synths the audio thread has finished with, react to node notifications,
    /// and periodically start a new note. This is the work the `Nrt` exists to do.
    fn tick(&mut self) {
        // Drop the `Box`es of freed synths here, never on the audio thread.
        self.nrt.process();
        // React to node notifications - here, track how many voices are currently playing.
        while let Some(event) = self.nrt.poll() {
            match event {
                Event::NodeStarted { .. } => self.playing += 1,
                Event::NodeEnded { .. } => self.playing = self.playing.saturating_sub(1),
                Event::NodePaused { .. } | Event::NodeResumed { .. } => {}
                Event::SynthFailed { .. } => {}
            }
        }

        self.ticks += 1;
        if self.ticks.is_multiple_of(SPAWN_EVERY) && self.playing < MAX_VOICES {
            self.spawn_note();
        }
    }

    /// Start one note. Its `Line.kr` envelope frees it ~0.4 s later, giving the `Nrt` work to do.
    fn spawn_note(&mut self) {
        let freq = FREQS[self.next_freq % FREQS.len()];
        self.next_freq += 1;
        if let Ok(node) = self
            .controller
            .synth_new("note", ROOT_GROUP_ID, AddAction::Tail)
        {
            let _ = self.controller.set_control(node, 0, freq); // parameter 0 = freq
        }
    }
}

/// Build the engine: a `Controls` (kept alive and ticked by the host) and the `World` (the audio
/// source). Registers the `note` SynthDef; notes are started later via `Controls::tick`. Identical
/// on native and web.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // note := SinOsc.ar(freq) * Line.kr(0.2, 0, 0.4, doneAction: 2) -> Out
    //   ugen 0: Line.kr - amplitude envelope that frees the synth when it reaches the end.
    //   ugen 1: SinOsc.ar(freq)
    //   ugen 2: SinOsc * Line (BinaryOpUGen multiply)
    //   ugen 3: Out, the product copied to each channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Ugen { ugen: 2, output: 0 });
    }
    let def = SynthDef {
        name: "note".to_string(),
        params: vec![Param {
            name: "freq".to_string(),
            default: 440.0,
        }],
        ugens: vec![
            UgenSpec {
                name: "Line".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Constant(0.2), // start amplitude
                    InputRef::Constant(0.0), // end amplitude
                    InputRef::Constant(0.4), // duration (s)
                    InputRef::Constant(2.0), // doneAction 2 = free the synth
                ],
                num_outputs: 1,
                special_index: 0,
            },
            UgenSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UgenSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Ugen { ugen: 1, output: 0 },
                    InputRef::Ugen { ugen: 0, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            UgenSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);

    (
        Controls {
            controller,
            nrt,
            ticks: 0,
            next_freq: 0,
            playing: 0,
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

/// Play the demo: the `World` feeds the cpal stream, while the `Controls` are ticked off the audio
/// thread to start notes and run the NRT cleanup.
fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let (controls, mut source) = build(sample_rate, channels);
    // Reused interleaved `f32` scratch buffer; the source fills it, then we convert to `T`.
    let mut scratch: Vec<f32> = Vec::new();

    let stream = device
        .build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                source.fill(&mut scratch, channels);
                for (out, sample) in output.iter_mut().zip(scratch.iter()) {
                    *out = T::from_sample(*sample);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream");
    stream.play().expect("failed to start audio stream");

    run_control_plane(controls, stream);
}

/// Tick the control plane off the audio thread for the demo's lifetime, holding the stream alive
/// meanwhile.
#[cfg(not(target_arch = "wasm32"))]
fn run_control_plane(mut controls: Controls, _stream: cpal::Stream) {
    use std::time::Duration;
    println!("playing a looping motif for 10s...");
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
