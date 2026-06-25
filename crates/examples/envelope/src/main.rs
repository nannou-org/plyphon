//! Percussive plucks shaped by `EnvGen`, working natively and on the web from one identical engine.
//!
//! Where the motif example shapes each note with a straight-line `Line.kr` ramp, this one uses a
//! multi-segment `EnvGen` percussive envelope - a near-instant linear attack into an *exponential*
//! decay (curve type 5, a negative curvature) - so the notes ring and fade like a plucked string.
//! The envelope's `doneAction` frees each note when it has faded, and the control plane spawns a new
//! note from a pentatonic scale every so often.
//!
//! As in the other cpal examples the `plyphon` engine is pure Rust with no platform-specific paths;
//! only the control-plane ticking differs between native (a thread loop) and web (a timer).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, World, engine,
};

/// A C-major pentatonic scale (Hz) the plucks walk through.
const SCALE: [f32; 6] = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25];
/// Peak amplitude of each pluck.
const AMP: f32 = 0.25;
/// Attack time to the peak (s) - near-instant, as a pluck's onset is.
const ATTACK: f32 = 0.005;
/// Exponential decay time back to silence (s).
const RELEASE: f32 = 0.6;
/// scsynth curve type 5 (custom curvature); a negative value gives the exponential pluck decay.
const CURVE_TYPE: f32 = 5.0;
/// The curvature: more negative is a sharper initial fall.
const CURVE: f32 = -4.0;
/// How often to tick the control plane, in milliseconds.
const TICK_MS: u32 = 50;
/// Start a new pluck every this many ticks (~350 ms at `TICK_MS`).
const SPAWN_EVERY: u32 = 7;
/// Cap on simultaneously-ringing plucks, enforced from node notifications.
const MAX_VOICES: usize = 8;

/// The demo's control plane: kept alive by the host and ticked on an NRT cadence. It starts plucks
/// (via the `Controller`) and runs the `Nrt` to drop freed synths and react to notifications - all
/// off the audio thread.
struct Controls {
    controller: Controller,
    nrt: Nrt,
    ticks: u32,
    next_note: usize,
    /// Voices currently ringing, tracked from `Event` notifications.
    playing: usize,
}

impl Controls {
    /// One NRT tick: drop synths the audio thread has finished with, react to node notifications,
    /// and periodically start a new pluck.
    fn tick(&mut self) {
        // Drop the `Box`es of freed synths here, never on the audio thread.
        self.nrt.process();
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
            self.spawn_pluck();
        }
    }

    /// Start one pluck. Its `EnvGen` perc envelope frees it once it has faded.
    fn spawn_pluck(&mut self) {
        let freq = SCALE[self.next_note % SCALE.len()];
        self.next_note += 1;
        if let Ok(node) = self
            .controller
            .synth_new("pluck", ROOT_GROUP_ID, AddAction::Tail)
        {
            let _ = self.controller.set_control(node, 0, freq); // parameter 0 = freq
        }
    }
}

/// The flat `EnvGen` input array for a percussive envelope: a linear attack to [`AMP`] then an
/// exponential decay to silence, freeing the synth at the end. Mirrors SuperCollider's
/// `Env.perc(ATTACK, RELEASE, AMP, CURVE)` unrolled for the `EnvGen` unit.
fn perc_env_inputs() -> Vec<InputRef> {
    let values = [
        1.0,   // gate (held open; a perc with no release node just plays through)
        1.0,   // levelScale
        0.0,   // levelBias
        1.0,   // timeScale
        2.0,   // doneAction = 2 (free the synth when faded)
        0.0,   // initialLevel
        2.0,   // numSegments
        -99.0, // releaseNode (none)
        -99.0, // loopNode (none)
        AMP, ATTACK, 1.0, 0.0, // attack: -> AMP over ATTACK s, linear
        0.0, RELEASE, CURVE_TYPE, CURVE, // decay: -> 0 over RELEASE s, exponential
    ];
    values.into_iter().map(InputRef::Constant).collect()
}

/// Build the engine: a `Controls` (kept alive and ticked by the host) and the `World` (the audio
/// source). Registers the `pluck` SynthDef; plucks are started later via `Controls::tick`. Identical
/// on native and web.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // pluck := SinOsc.ar(freq) * EnvGen.kr(Env.perc, doneAction: 2) -> Out
    //   unit 0: EnvGen.kr - the percussive amplitude envelope that frees the synth when faded.
    //   unit 1: SinOsc.ar(freq)
    //   unit 2: SinOsc * EnvGen (BinaryOpUGen multiply)
    //   unit 3: Out, the product copied to each channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 2, output: 0 });
    }
    let def = SynthDef {
        name: "pluck".to_string(),
        params: vec![Param {
            name: "freq".to_string(),
            default: 440.0,
        }],
        units: vec![
            UnitSpec::new("EnvGen", Rate::Control, perc_env_inputs(), 1),
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);

    (
        Controls {
            controller,
            nrt,
            ticks: 0,
            next_note: 0,
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
/// thread to start plucks and run the NRT cleanup.
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
    println!("playing percussive plucks for 10s...");
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
