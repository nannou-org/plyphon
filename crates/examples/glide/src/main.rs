//! The control family: a portamento lead driven entirely by `/n_set` on special parameters.
//!
//! One synth plays a looping melody. The host never restarts it - it just retunes two parameters
//! each beat:
//!
//! - `freq` is a **`LagControl`**: a `/n_set` glides to the new pitch over `GLIDE` seconds (a one-pole
//!   de-zipper, one step per control block) instead of jumping, so the line *slides* between notes.
//! - `trig` is a **`TrigControl`**: a `/n_set` is seen for exactly one control block, then resets to
//!   0. Gated against a noise source it makes a short percussive *tick* at each note onset.
//!
//! (The third member of the family, `AudioControl` + `/n_mapa`, is an audio-rate parameter mapped to
//! an audio bus; it is exercised in the `audio_control` tests rather than here.)
//!
//! The whole phrase is **scheduled up front**: each beat's `/n_set`s are time-tagged with the exact
//! moment they should sound and handed to the engine's scheduler, which fires them on its
//! free-running clock (the same `CommandTime::At` machinery the `schedule` example drives over OSC).
//! So the rhythm is locked to the audio block grid rather than to whenever a wall-clock tick happened
//! to dispatch it - which matters precisely because the line is so rhythmic. The engine is identical
//! on native and web; only the control plane's idle upkeep differs (a thread loop vs a timer).

use plyphon::{
    AddAction, CommandTime, Controller, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate,
    SynthDef, UnitSpec, World, engine,
};

/// OSC/NTP fixed-point units in one second (the engine clock is 32.32 fixed point, starting at 0 on
/// the first `fill`); a beat at second `t` is tagged `(t * OSC_UNITS_PER_SEC)`.
const OSC_UNITS_PER_SEC: f64 = 4_294_967_296.0;
/// The looping melody (Hz) - an A-minor-pentatonic phrase the lead glides through.
const MELODY: [f32; 8] = [
    220.00, 277.18, 329.63, 277.18, 440.00, 329.63, 392.00, 293.66,
];
/// Portamento time for the `freq` `LagControl`, in seconds.
const GLIDE: f32 = 0.07;
/// The lead's low-pass cutoff (Hz).
const CUTOFF: f32 = 1500.0;
/// Lead amplitude (the sustained, gliding saw).
const LEAD_AMP: f32 = 0.22;
/// Accent amplitude (the one-block noise tick from the `TrigControl`).
const TICK_AMP: f32 = 0.18;
/// Master gain applied in the cpal callback.
const GAIN: f32 = 0.9;
/// Seconds between beats (~230 notes/min).
const BEAT_SECS: f64 = 0.26;
/// Beats in the phrase - a whole number of melody loops, scheduled up front.
const NUM_BEATS: usize = 64;
/// How often to tick the control plane's idle upkeep (ms).
const TICK_MS: u32 = 50;

/// The demo's control plane: it owns the engine's control side for the run, doing off-audio-thread
/// upkeep. The phrase itself is already queued in the scheduler, so nothing is dispatched per tick.
struct Controls {
    // Held to keep the command ring's producer alive for the run; the phrase is already queued.
    #[allow(dead_code)]
    controller: Controller,
    nrt: Nrt,
}

impl Controls {
    /// One idle tick: drop anything the audio thread has freed (nothing here - the lead never frees).
    fn tick(&mut self) {
        self.nrt.process();
    }
}

/// Schedule the whole phrase up front: each beat glides `freq` to the next note and fires the accent
/// `trig`, time-tagged so the engine applies it on the exact block. The clock starts at 0 on the
/// first `fill`, so beat `k` is tagged `k * BEAT_SECS` seconds in - beat 0 lands on the first block,
/// no lead-in, the same length as every other note.
fn schedule_phrase(controller: &mut Controller, node: i32) {
    for beat in 0..NUM_BEATS {
        let time = (beat as f64 * BEAT_SECS * OSC_UNITS_PER_SEC) as u64;
        let freq = MELODY[beat % MELODY.len()];
        controller.begin_scheduled(CommandTime::At(time));
        let _ = controller.set_control(node, 0, freq); // 0 = freq (LagControl): glides to the new pitch
        let _ = controller.set_control(node, 1, 1.0); // 1 = trig (TrigControl): a one-block accent
        controller.end_scheduled();
    }
}

/// Build the engine with one `glider` synth, schedule the phrase, and return the control plane plus
/// the `World`.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    controller.add_synthdef(glider_def(channels));
    let node = controller
        .synth_new("glider", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap_or(-1);
    schedule_phrase(&mut controller, node);

    (Controls { controller, nrt }, world)
}

/// `glider`: a low-passed saw whose `freq` glides (`LagControl`), plus a noise tick gated by a
/// `TrigControl`. Parameters: `freq` (0, lagged) and `trig` (1, one-block).
fn glider_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 6, output: 0 });
    }
    SynthDef {
        name: "glider".to_string(),
        params: vec![
            Param::lag("freq", MELODY[0], GLIDE),
            Param::trig("trig", 0.0),
        ],
        units: vec![
            // 0: Saw.ar(freq) - the lagged frequency makes it glide between notes.
            UnitSpec::new("Saw", Rate::Audio, vec![InputRef::Param(0)], 1),
            // 1: LPF(saw, CUTOFF) - tame the saw.
            UnitSpec::new(
                "LPF",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(CUTOFF),
                ],
                1,
            ),
            // 2: MulAdd.ar(lead, LEAD_AMP, 0) - the sustained lead level.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(LEAD_AMP),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 3: WhiteNoise.ar - the accent source.
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            // 4: noise * trig - the TrigControl gates one block of noise per /n_set.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Unit { unit: 3, output: 0 }, InputRef::Param(1)],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 5: MulAdd.ar(tick, TICK_AMP, 0) - the accent level.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 4, output: 0 },
                    InputRef::Constant(TICK_AMP),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 6: lead + tick.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Unit { unit: 5, output: 0 },
                ],
                num_outputs: 1,
                special_index: 0, // add
            },
            // 7: Out.ar(0, mix) on every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // cpal's AudioWorklet backend re-instantiates this module on the audio thread, re-running
    // `main` there; only set up audio on the main browser thread.
    if example_audio::on_worklet_thread() {
        return;
    }

    // The whole phrase is scheduled up front; the control plane only does idle upkeep until it has
    // played out (with a little tail so the last note rings).
    let total_secs = NUM_BEATS as f64 * BEAT_SECS + 0.5;

    #[cfg(not(target_arch = "wasm32"))]
    println!(
        "gliding lead with trigger accents: {NUM_BEATS} pre-scheduled beats over ~{total_secs:.1}s..."
    );

    let (stream, mut controls) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (controls, mut world) = build(sample_rate as f32, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            controls,
        )
    });
    let total_ms = (total_secs * 1000.0) as u32;
    example_audio::run_control(stream, total_ms, TICK_MS, move || controls.tick());
}
