//! Audio→host emission: an amplitude follower reports its level to the host with `SendReply`.
//!
//! A tremolo tone plays; an `Amplitude` follower tracks its envelope, and a `SendReply` fires
//! `/amp [nodeID, replyID, level]` four times a second. The control plane drains those messages off
//! the audio thread with [`Nrt::poll_node_msg`] and logs each - to the browser console on the web,
//! or stdout natively. This is the inverse direction of most examples: data flowing *out* of the
//! engine. The engine is identical on native and web; only how the log line is printed differs.

use plyphon::{
    AddAction, Controller, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

/// Carrier pitch (Hz).
const FREQ: f32 = 330.0;
/// Tremolo rate (Hz) - the amplitude swings up and down this often.
const TREM_HZ: f32 = 0.5;
/// How often `SendReply` reports the level (Hz).
const POLL_HZ: f32 = 4.0;
/// The reply id echoed in each `/amp` message.
const REPLY_ID: f32 = 1.0;
/// Master gain in the cpal callback.
const GAIN: f32 = 0.3;
/// Control-plane idle tick (ms).
const TICK_MS: u32 = 50;
/// How long to run (ms).
const RUN_MS: u32 = 12_000;

/// `BinaryOpUGen` special index for multiply.
const OP_MUL: i16 = 2;

/// Print a line to the host: the browser console on the web, stdout natively.
fn log(line: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    println!("{line}");
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(line));
}

/// The control plane: it owns the engine's control side and, each idle tick, drains the `SendReply`
/// messages the audio thread emitted and logs them.
struct Controls {
    // Held to keep the command ring's producer alive for the run.
    #[allow(dead_code)]
    controller: Controller,
    nrt: Nrt,
}

impl Controls {
    /// One idle tick: drop freed nodes (none here) and surface every `/amp` reply.
    fn tick(&mut self) {
        self.nrt.process();
        while let Some(msg) = self.nrt.poll_node_msg() {
            let path = core::str::from_utf8(&msg.label[..msg.label_len as usize]).unwrap_or("?");
            let level = msg.values[0];
            log(&format!("{path}: {level:.3}  (node {})", msg.node));
        }
    }
}

/// Build the engine with one tremolo `amp` synth and return the control plane plus the `World`.
fn build(sample_rate: f32, channels: usize) -> (Controls, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // Out.ar(0, signal) on every channel (signal is unit 3).
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 3, output: 0 });
    }
    let def = SynthDef {
        name: "amp".to_string(),
        params: vec![],
        units: vec![
            // 0: SinOsc.ar(FREQ) - the carrier.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
                1,
            ),
            // 1: SinOsc.ar(TREM_HZ) - a slow LFO in [-1, 1].
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(TREM_HZ), InputRef::Constant(0.0)],
                1,
            ),
            // 2: amp = LFO * 0.45 + 0.5 -> a tremolo depth in [0.05, 0.95].
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.45),
                    InputRef::Constant(0.5),
                ],
                1,
            ),
            // 3: signal = carrier * amp.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                num_outputs: 1,
                special_index: OP_MUL,
            },
            // 4: env = Amplitude.ar(signal) - follows the tremolo envelope.
            UnitSpec::new(
                "Amplitude",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Constant(0.01),
                    InputRef::Constant(0.1),
                ],
                1,
            ),
            // 5: Impulse.ar(POLL_HZ) - the reporting clock.
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(POLL_HZ), InputRef::Constant(0.0)],
                1,
            ),
            // 6: SendReply.ar(trig, replyID, "/amp", [env]) - report the level on each tick.
            UnitSpec::send_reply(
                Rate::Audio,
                InputRef::Unit { unit: 5, output: 0 },
                InputRef::Constant(REPLY_ID),
                "/amp",
                &[InputRef::Unit { unit: 4, output: 0 }],
            ),
            // 7: Out.ar(0, signal) on every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("amp", ROOT_GROUP_ID, AddAction::Tail, &[]);

    (Controls { controller, nrt }, world)
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
    println!("tremolo tone reporting its level via SendReply (/amp, {POLL_HZ} Hz) for 12s...");

    let (stream, mut controls) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (controls, mut world) = build(sample_rate as f32, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            controls,
        )
    });
    example_audio::run_control(stream, RUN_MS, TICK_MS, move || controls.tick());
}
