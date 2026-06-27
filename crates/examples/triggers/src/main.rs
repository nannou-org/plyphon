//! `SendTrig` as a clock: the engine fires a `/tr` on every beat and the host turns each into a note.
//!
//! This demonstrates the new server -> client trigger path and two related notifications:
//!
//! - A silent **clock** synth runs `SendTrig.ar(Impulse.ar(BEAT_HZ), 0, SinOsc.ar(LFO_HZ))`. It makes
//!   no sound; it just fires a `/tr` on every beat, carrying the slow LFO's value sampled at that
//!   instant. The control plane drains those triggers with [`Nrt::poll_trigger`] and maps each value
//!   to a pitch.
//! - Each beat spawns a note in its own one-shot **voice group**. The note's `Line.kr` envelope ends
//!   with done action 14 (free the enclosing group), so the synth and its group are freed together -
//!   reported as two `/n_end`s. (Codes 3-14 are the variants that touch neighbours or the group; 14
//!   is the natural fit for "a voice is a group".)
//! - At startup the clock is moved to the head of the root group, so control runs before the voices.
//!   The engine reports the reorder as a `/n_move`.
//!
//! The engine is identical on native and web; only the control-plane tick differs (a thread loop vs
//! a timer), exactly as in the `motif` example.

use std::collections::HashSet;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    Trigger, UnitSpec, World, engine,
};

/// Beats per second the clock fires.
const BEAT_HZ: f32 = 3.0;
/// The LFO whose value each `/tr` carries (cycles slowly, so the melody drifts).
const LFO_HZ: f32 = 0.2;
/// Each note's decay time (s); its `Line.kr` then frees the voice group.
const NOTE_DUR: f32 = 0.35;
/// Each note's starting amplitude (it decays to zero).
const NOTE_AMP: f32 = 0.25;
/// How often to tick the control plane, in milliseconds.
const TICK_MS: u32 = 25;
/// Major-pentatonic ratios over two octaves, indexed by the `/tr` value.
const SCALE: [f32; 5] = [1.0, 9.0 / 8.0, 5.0 / 4.0, 3.0 / 2.0, 5.0 / 3.0];
/// Lowest pitch (Hz), at the bottom of the LFO sweep.
const BASE_FREQ: f32 = 220.0;

/// The demo's control plane: kept alive by the host and ticked on an NRT cadence. It reacts to the
/// clock's `/tr` triggers (spawning notes) and to node notifications, all off the audio thread.
struct Controls {
    controller: Controller,
    nrt: Nrt,
    /// Beats counted from the clock's `/tr`s.
    beat: u32,
    /// Live one-shot voice groups, so an `/n_end` can be recognised as a done-action free.
    voice_groups: HashSet<i32>,
}

impl Controls {
    /// One NRT tick: drop finished synths, spawn a note per `/tr` beat, and log the notifications.
    fn tick(&mut self) {
        // Drop the `Box`es of freed synths here, never on the audio thread.
        self.nrt.process();
        // The clock's `/tr`s: one note per beat (the new SendTrig -> /tr path).
        while let Some(trigger) = self.nrt.poll_trigger() {
            self.on_beat(trigger);
        }
        // Node notifications: the startup `/n_move`, and the `/n_end`s from done action 14.
        while let Some(event) = self.nrt.poll() {
            match event {
                Event::NodeMoved {
                    node, parent, next, ..
                } => {
                    println!("/n_move: node {node} now heads group {parent} (next sibling {next})")
                }
                Event::NodeEnded { id } if self.voice_groups.remove(&id) => {
                    println!("  /n_end: voice group {id} freed by done action 14");
                }
                _ => {}
            }
        }
    }

    /// Turn one `/tr` beat into a note: a fresh one-shot voice group holding a single decaying tone
    /// whose envelope frees the whole group (done action 14).
    fn on_beat(&mut self, trigger: Trigger) {
        self.beat += 1;
        let freq = value_to_freq(trigger.value);
        let Ok(group) = self.controller.new_group(ROOT_GROUP_ID, AddAction::Tail) else {
            return;
        };
        self.voice_groups.insert(group);
        if let Ok(note) = self.controller.synth_new("note", group, AddAction::Tail) {
            let _ = self.controller.set_control(note, 0, freq); // parameter 0 = freq
        }
        println!(
            "beat {:>3}: /tr value {:+.3} -> {:6.1} Hz (voice group {group})",
            self.beat, trigger.value, freq,
        );
    }
}

/// Map a `/tr` value (the LFO, in `-1..1`) to a pitch: a major-pentatonic degree over two octaves.
fn value_to_freq(value: f32) -> f32 {
    let t = ((value + 1.0) * 0.5).clamp(0.0, 0.999);
    let steps = SCALE.len() * 2;
    let i = (t * steps as f32) as usize;
    let octave = (i / SCALE.len()) as u32;
    BASE_FREQ * SCALE[i % SCALE.len()] * 2.0f32.powi(octave as i32)
}

/// Build the engine: the `clock` synth (which fires the `/tr`s) and the `note` SynthDef (spawned per
/// beat). The clock is moved to the head of the root group, so it runs before the voices - a reorder
/// the engine reports as a `/n_move`. Identical on native and web.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // clock := SendTrig.ar(Impulse.ar(BEAT_HZ), 0, SinOsc.ar(LFO_HZ)) - no audio output.
    //   unit 0: SinOsc.ar(LFO_HZ) - the slow LFO whose value each /tr reports.
    //   unit 1: Impulse.ar(BEAT_HZ) - the beat.
    //   unit 2: SendTrig.ar(in: impulse, id: 0, value: lfo) - fires /tr on each rising edge.
    let clock = SynthDef {
        name: "clock".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(LFO_HZ), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(BEAT_HZ), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "SendTrig",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    };
    controller.add_synthdef(clock);

    // note := SinOsc.ar(freq) * Line.kr(NOTE_AMP, 0, NOTE_DUR, doneAction: 14) -> Out
    //   unit 0: Line.kr - decay envelope that frees the enclosing voice group when it ends.
    //   unit 1: SinOsc.ar(freq)
    //   unit 2: SinOsc * Line (BinaryOpUGen multiply)
    //   unit 3: Out, the product copied to each channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 2, output: 0 });
    }
    let note = SynthDef {
        name: "note".to_string(),
        params: vec![Param::control("freq", BASE_FREQ)],
        units: vec![
            UnitSpec {
                name: "Line".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Constant(NOTE_AMP),
                    InputRef::Constant(0.0),
                    InputRef::Constant(NOTE_DUR),
                    InputRef::Constant(14.0), // doneAction 14 = free the enclosing voice group
                ],
                num_outputs: 1,
                special_index: 0,
            },
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
    controller.add_synthdef(note);

    // Start the clock, then move it to the head of the root group so it runs before the voices we
    // append at the tail. The reorder is reported as a `/n_move`.
    if let Ok(clock_node) = controller.synth_new("clock", ROOT_GROUP_ID, AddAction::Tail) {
        let _ = controller.move_node(clock_node, ROOT_GROUP_ID, AddAction::Head);
    }

    (
        Controls {
            controller,
            nrt,
            beat: 0,
            voice_groups: HashSet::new(),
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
/// thread to react to `/tr`s and run the NRT cleanup.
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
    println!("SendTrig clock driving notes for 12s...");
    let ticks = 12_000 / TICK_MS;
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
