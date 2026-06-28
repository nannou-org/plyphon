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

use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UnitSpec, World, engine,
};

/// A short looping motif (Hz).
const FREQS: [f32; 4] = [440.0, 550.0, 660.0, 550.0];
/// How often to tick the control plane, in milliseconds.
const TICK_MS: u32 = 50;
/// Start a new note every this many ticks (~500 ms at `TICK_MS`).
const SPAWN_EVERY: u32 = 10;
/// Cap on simultaneously-playing notes, enforced from node notifications.
const MAX_VOICES: usize = 6;
/// Master gain applied in the audio callback (each note is already scaled by its `Line.kr`).
const GAIN: f32 = 1.0;

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
                Event::NodeMoved { .. } => {}
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
    //   unit 0: Line.kr - amplitude envelope that frees the synth when it reaches the end.
    //   unit 1: SinOsc.ar(freq)
    //   unit 2: SinOsc * Line (BinaryOpUGen multiply)
    //   unit 3: Out, the product copied to each channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 2, output: 0 });
    }
    let def = SynthDef {
        name: "note".to_string(),
        params: vec![Param::control("freq", 440.0)],
        units: vec![
            UnitSpec {
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
            next_freq: 0,
            playing: 0,
        },
        world,
    )
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
    println!("playing a looping motif for 10s...");

    let (stream, mut controls) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (controls, mut world) = build(sample_rate as f32, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            controls,
        )
    });
    example_audio::run_control(stream, 10_000, TICK_MS, move || controls.tick());
}
