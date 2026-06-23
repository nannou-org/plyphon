//! The demo's control plane.
//!
//! [`build`] returns a [`Controls`] (a [`Controller`] + [`Nrt`]) that the host keeps alive and ticks
//! on an NRT cadence, plus the [`World`] that goes to the audio thread. It demonstrates the full
//! engine lifecycle: the `Controller` starts notes, the `World` plays them and frees them via a
//! `Line.kr` done action, and the `Nrt` drops the freed synths and drains notifications - all off
//! the audio thread. Crucially the `Nrt` is *run*, not dropped.

use plyphon::{
    AddAction, Controller, Event, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    UgenSpec, World, engine,
};

/// A short looping motif (Hz).
const FREQS: [f32; 4] = [440.0, 550.0, 660.0, 550.0];
/// How often the host should tick [`Controls::tick`], in milliseconds.
pub const TICK_MS: u32 = 50;
/// Start a new note every this many ticks (~500 ms at [`TICK_MS`]).
const SPAWN_EVERY: u32 = 10;
/// Cap on simultaneously-playing notes, enforced from node notifications.
const MAX_VOICES: usize = 6;

/// The demo's control plane: kept alive by the host and ticked on an NRT cadence.
pub struct Controls {
    controller: Controller,
    nrt: Nrt,
    ticks: u32,
    next_freq: usize,
    /// Voices currently playing, tracked from [`Event`] notifications.
    playing: usize,
}

impl Controls {
    /// One NRT tick: drop synths the audio thread has finished with, drain node notifications, and
    /// periodically start a new note. This is the work the `Nrt` exists to do, run off the audio
    /// thread on whatever cadence the host provides.
    pub fn tick(&mut self) {
        // Drop the `Box`es of freed synths here, never on the audio thread.
        self.nrt.process();
        // React to node notifications - here, track how many voices are currently playing.
        while let Some(event) = self.nrt.poll() {
            match event {
                Event::NodeStarted { .. } => self.playing += 1,
                Event::NodeEnded { .. } => self.playing = self.playing.saturating_sub(1),
                Event::NodePaused { .. } | Event::NodeResumed { .. } => {}
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

/// Build the demo: a [`Controls`] (kept alive and ticked by the host) and the [`World`] (the audio
/// source). Registers the `note` SynthDef; notes are started later via [`Controls::tick`].
pub fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
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
