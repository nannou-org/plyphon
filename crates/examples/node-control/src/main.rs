//! In-graph node-to-node control: one node pauses, resumes, and frees *another* by id.
//!
//! A chord of sustained drone voices plays continuously, but each one is driven by a *separate*
//! `gater` node that addresses it **by id** - pausing and resuming it (so the chord shimmers as
//! voices cut in and out at their own rates) and ultimately freeing it. None of this touches the
//! host's bookkeeping; the gaters drive the drones entirely from inside the engine, the way
//! scsynth's `Pause`/`Free` do.
//!
//! - Each `gater` runs `Pause.kr(LFPulse.kr(rate), droneId)`: the square wave pauses the drone while
//!   it is low and resumes it while it is high, reported as `/n_off` and `/n_on`. A different rate
//!   per drone gives a polyrhythmic gate.
//! - Each `gater` also holds `Free.kr(In.kr(panicBus), droneId)`: when the host raises the panic
//!   control bus (natively, at the end of the run), every drone is freed by id - reported as
//!   `/n_end`. On the web the bus stays low, so the gated texture loops forever.
//!
//! Phase 1 also added the engine-info ugens (`SampleRate`, `ControlRate`, `BufFrames`, ...) and the
//! rate bridges (`DC`/`K2A`/`A2K`/`T2A`); this demo focuses on the new in-graph node control.
//!
//! The engine is identical on native and web; only the control-plane tick differs (a thread loop vs
//! a timer), exactly as in the `triggers` and `motif` examples.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, World, engine,
};

/// The drones: `(frequency Hz, gate rate Hz)`. The frequencies spell an A-major chord; the gate
/// rates are deliberately unrelated, so the voices pulse against each other.
const DRONES: [(f32, f32); 4] = [
    (220.00, 1.5), // A3
    (277.18, 2.0), // C#4
    (329.63, 3.0), // E4
    (440.00, 4.0), // A4
];
/// Each drone's amplitude (they sum into one output, so keep headroom).
const DRONE_AMP: f32 = 0.16;
/// The control bus the host raises to make every `gater` free its drone (the "panic").
const PANIC_BUS: u32 = 0;
/// How often to tick the control plane, in milliseconds.
const TICK_MS: u32 = 25;
/// Seconds of gating before the native run frees the drones; the web run never frees them.
#[cfg(not(target_arch = "wasm32"))]
const GATE_SECS: u32 = 10;
/// Seconds to keep ticking after the panic, so the `/n_end`s drain.
#[cfg(not(target_arch = "wasm32"))]
const DRAIN_SECS: u32 = 2;

/// The demo's control plane: kept alive by the host and ticked on an NRT cadence. It runs the `Nrt`
/// cleanup and logs the gate (`/n_off`/`/n_on`) and free (`/n_end`) notifications the gaters cause.
struct Controls {
    /// Only used (natively) to raise the panic bus; on the web the texture loops without it.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    controller: Controller,
    nrt: Nrt,
    /// Each drone's node id, so a notification can be labelled with its pitch.
    drones: Vec<(i32, f32)>,
}

impl Controls {
    /// One NRT tick: drop finished synths and log the gate/free notifications.
    fn tick(&mut self) {
        // Drop the `Box`es of freed synths here, never on the audio thread.
        self.nrt.process();
        while let Some(event) = self.nrt.poll() {
            match event {
                Event::NodePaused { id } => self.log(id, "/n_off: paused"),
                Event::NodeResumed { id } => self.log(id, "/n_on : resumed"),
                Event::NodeEnded { id } => self.log(id, "/n_end: freed by id"),
                _ => {}
            }
        }
    }

    /// Print a notification labelled with the drone's pitch, if it is one of ours.
    fn log(&self, id: i32, what: &str) {
        if let Some((_, freq)) = self.drones.iter().find(|(node, _)| *node == id) {
            println!("  {what}  (drone {id}, {freq:.1} Hz)");
        }
    }

    /// Raise the panic bus so every `gater`'s `Free.kr` fires on its drone.
    #[cfg(not(target_arch = "wasm32"))]
    fn panic(&mut self) {
        println!("panic: raising the control bus - every gater frees its drone by id");
        let _ = self.controller.set_control_bus(PANIC_BUS, 1.0);
    }
}

/// Build the engine: a `drone` SynthDef (a sustained tone) and a `gater` SynthDef (pauses/resumes
/// and can free a drone by id). One drone + one gater is spawned per entry in [`DRONES`]. Identical
/// on native and web.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    controller.add_synthdef(drone_def(channels));
    controller.add_synthdef(gater_def());

    // Spawn each drone, then a gater that drives it. The gater's `id` parameter is the drone's node
    // id - the whole point of the demo: a node controlling another node by id.
    let mut drones = Vec::with_capacity(DRONES.len());
    for (freq, rate) in DRONES {
        let Ok(drone) = controller.synth_new("drone", ROOT_GROUP_ID, AddAction::Tail) else {
            continue;
        };
        let _ = controller.set_control(drone, 0, freq); // parameter 0 = freq
        drones.push((drone, freq));

        if let Ok(gater) = controller.synth_new("gater", ROOT_GROUP_ID, AddAction::Tail) {
            let _ = controller.set_control(gater, 0, rate); // parameter 0 = gate rate
            let _ = controller.set_control(gater, 1, drone as f32); // parameter 1 = target id
        }
    }

    (
        Controls {
            controller,
            nrt,
            drones,
        },
        world,
    )
}

/// `drone := SinOsc.ar(freq) * DRONE_AMP -> Out` - a sustained tone, one parameter (`freq`).
fn drone_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 1, output: 0 });
    }
    SynthDef {
        name: "drone".to_string(),
        params: vec![Param::control("freq", 220.0)],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            // MulAdd.ar(sin, DRONE_AMP, 0): scale to the per-voice amplitude.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(DRONE_AMP),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// `gater` drives a drone by id and makes no sound itself:
/// - `Pause.kr(LFPulse.kr(rate), id)` - gate the drone on/off at `rate`.
/// - `Free.kr(In.kr(PANIC_BUS), id)` - free the drone when the host raises the panic bus.
///
/// Parameters: `rate` (0) and `id` (1, the target drone's node id).
fn gater_def() -> SynthDef {
    SynthDef {
        name: "gater".to_string(),
        params: vec![Param::control("rate", 2.0), Param::control("id", -1.0)],
        units: vec![
            // unit 0: In.kr(PANIC_BUS) - the host-raised "free everything" signal.
            UnitSpec::new(
                "In",
                Rate::Control,
                vec![InputRef::Constant(PANIC_BUS as f32)],
                1,
            ),
            // unit 1: LFPulse.kr(rate, 0, 0.5) - the gate (1 = run, 0 = pause).
            UnitSpec::new(
                "LFPulse",
                Rate::Control,
                vec![
                    InputRef::Param(0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.5),
                ],
                1,
            ),
            // unit 2: Pause.kr(gate, id) - pause/resume the target drone on each gate edge.
            UnitSpec::new(
                "Pause",
                Rate::Control,
                vec![InputRef::Unit { unit: 1, output: 0 }, InputRef::Param(1)],
                0,
            ),
            // unit 3: Free.kr(panic, id) - free the target drone on a rising panic edge.
            UnitSpec::new(
                "Free",
                Rate::Control,
                vec![InputRef::Unit { unit: 0, output: 0 }, InputRef::Param(1)],
                0,
            ),
        ],
    }
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
/// thread to run the NRT cleanup and log the gate/free notifications.
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

/// Native: gate for [`GATE_SECS`], then raise the panic bus so the gaters free the drones by id, and
/// keep ticking briefly so the `/n_end`s drain.
#[cfg(not(target_arch = "wasm32"))]
fn run_control_plane(mut controls: Controls, _stream: cpal::Stream) {
    use std::time::Duration;
    println!(
        "gated drones: {} voices pausing/resuming by id...",
        DRONES.len()
    );
    let tick = || std::thread::sleep(Duration::from_millis(u64::from(TICK_MS)));
    for _ in 0..(GATE_SECS * 1000 / TICK_MS) {
        controls.tick();
        tick();
    }
    controls.panic();
    for _ in 0..(DRAIN_SECS * 1000 / TICK_MS) {
        controls.tick();
        tick();
    }
}

/// Web: `main` returns immediately, so tick the control plane on a periodic timer and keep both it
/// and the audio stream alive. The panic bus is never raised, so the gated texture loops forever.
#[cfg(target_arch = "wasm32")]
fn run_control_plane(mut controls: Controls, stream: cpal::Stream) {
    let interval = gloo_timers::callback::Interval::new(TICK_MS, move || controls.tick());
    interval.forget();
    std::mem::forget(stream);
}
