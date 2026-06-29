//! Record into a buffer, then loop it back - `RecordBuf` writing what `PlayBuf` later reads.
//!
//! A `record` synth plays a rising pitch sweep, monitors it live, and captures it into a buffer with
//! `RecordBuf` (non-looping, `doneAction: 2`). When the buffer fills, `RecordBuf` frees the recorder;
//! the control plane sees the `NodeEnded` and starts a `play` synth that loops the recorded buffer
//! with `PlayBuf`. So you hear the sweep once, live, then it repeats from the buffer. The engine is
//! identical on native and web; only the control plane's idle upkeep differs.

use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec,
    World, engine,
};

/// Sweep start/end pitch (Hz).
const TONE_LO: f32 = 220.0;
const TONE_HI: f32 = 660.0;
/// How long to record (seconds) - also the buffer length and the sweep duration.
const RECORD_SECS: f32 = 2.5;
/// Amplitude of the recorded/monitored tone.
const AMP: f32 = 0.5;
/// Master gain in the cpal callback.
const GAIN: f32 = 0.6;
/// Control-plane idle tick (ms).
const TICK_MS: u32 = 50;
/// How long to run (ms): record once, then loop for the rest.
const RUN_MS: u32 = 12_000;

/// The control plane: when the recorder frees itself (its `NodeEnded`), start looping playback.
struct Controls {
    controller: Controller,
    nrt: Nrt,
    recorder: i32,
    playing: bool,
}

impl Controls {
    fn tick(&mut self) {
        self.nrt.process();
        let mut recorder_ended = false;
        while let Some(event) = self.nrt.poll() {
            if event == (Event::NodeEnded { id: self.recorder }) {
                recorder_ended = true;
            }
        }
        if recorder_ended && !self.playing {
            let _ = self
                .controller
                .synth_new("play", ROOT_GROUP_ID, AddAction::Tail);
            self.playing = true;
            #[cfg(not(target_arch = "wasm32"))]
            println!("recording finished - looping it back");
        }
    }
}

/// `record`: a swept tone, monitored live and captured into buffer 0 by `RecordBuf` (non-looping,
/// `doneAction: 2`, so it frees this synth when the buffer fills).
fn record_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 2, output: 0 });
    }
    SynthDef {
        name: "record".to_string(),
        params: vec![],
        units: vec![
            // 0: Line.kr(lo, hi, dur) - the pitch sweep.
            UnitSpec::new(
                "Line",
                Rate::Control,
                vec![
                    InputRef::Constant(TONE_LO),
                    InputRef::Constant(TONE_HI),
                    InputRef::Constant(RECORD_SECS),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 1: SinOsc.ar(sweep).
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 2: tone = SinOsc * AMP.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(AMP),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 3: RecordBuf.ar([tone], buf=0, offset=0, recLevel=1, preLevel=0, run=1, loop=0, trig=0,
            //    doneAction=2) - capture the tone; frees this synth when the buffer fills.
            UnitSpec::new(
                "RecordBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(2.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                1,
            ),
            // 4: Out.ar(0, tone) - monitor live while recording.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// `play`: loop buffer 0 with `PlayBuf` and route it to every channel.
fn play_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 0, output: 0 });
    }
    SynthDef {
        name: "play".to_string(),
        params: vec![],
        units: vec![
            // PlayBuf.ar(1, buf=0, rate=1, trig=0, startPos=0, loop=1, doneAction=0).
            UnitSpec::new(
                "PlayBuf",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// Build the engine with a mono record buffer and the `record`/`play` defs, start the recorder, and
/// return the control plane plus the `World`.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    let frames = (RECORD_SECS * sample_rate) as usize;
    let _ = controller.buffer_alloc(0, frames, 1, sample_rate as f64);
    controller.add_synthdef(record_def(channels));
    controller.add_synthdef(play_def(channels));
    let recorder = controller
        .synth_new("record", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap_or(-1);

    (
        Controls {
            controller,
            nrt,
            recorder,
            playing: false,
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
    println!("recording a {RECORD_SECS}s pitch sweep into a buffer, then looping it...");

    let (stream, mut controls) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (controls, mut world) = build(sample_rate as f32, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            controls,
        )
    });
    example_audio::run_control(stream, RUN_MS, TICK_MS, move || controls.tick());
}
