//! A feedback echo - the new `DelayN` delay line in a one-block `LocalIn`/`LocalOut` feedback loop.
//!
//! `DelayN` is the first unit to use plyphon's per-instance *auxiliary memory*: a delay line whose
//! length is sized at compile time from its scalar `maxdelaytime` and folded into the synth's single
//! pool block (the safe analogue of scsynth's `RTAlloc`'d delay buffer). Here it carries the echo
//! tail for `ECHO_SECS` while a `LocalIn`/`LocalOut` bus recirculates it, attenuated by `FEEDBACK`:
//!
//! ```text
//!   dry  = SinOsc * pluck-envelope                 (a short ping ~every 1/BEEP_HZ seconds)
//!   fb   = LocalIn (last block's wet) * FEEDBACK
//!   wet  = DelayN(dry + fb, ECHO_SECS)             (delayed by the aux line)
//!   LocalOut(wet)                                  (feed wet back for next block)
//!   out  = dry + wet                               (ping, then decaying echoes ECHO_SECS apart)
//! ```
//!
//! The whole patch plays forever and frees nothing, so - like `example-sine` - there is no NRT work:
//! the `Controller` and `Nrt` are dropped once the synth is queued, leaving only the `World`. The
//! engine is identical on native and web; the cpal output plumbing lives in [`example_audio`].

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// Pitch of the plucked ping (Hz).
const TONE: f32 = 330.0;
/// How often a ping sounds (Hz) - one every ~2.2 s, leaving room for the echo tail.
const BEEP_HZ: f32 = 0.45;
/// Pulse duty cycle of the ping gate (fraction of the period it is open).
const WIDTH: f32 = 0.06;
/// One-pole lag on the gate, in seconds, to round its edges into a pluck (no hard-gate click).
const LAG: f32 = 0.015;
/// Echo spacing: the `DelayN` delay time, in seconds.
const ECHO_SECS: f32 = 0.33;
/// Maximum delay the line is sized for (must exceed `ECHO_SECS`); a compile-time constant.
const MAX_DELAY: f32 = 1.0;
/// Echo feedback: each repeat is this fraction of the previous (< 1 so the tail decays).
const FEEDBACK: f32 = 0.5;
/// Master scale on the dry+wet mix before the cpal gain (keeps stacked echoes off the ceiling).
const MASTER: f32 = 0.4;
/// Master gain applied in the cpal callback.
const GAIN: f32 = 0.7;

/// `BinaryOpUGen` special index for multiply.
const OP_MUL: i16 = 2;
/// `BinaryOpUGen` special index for add.
const OP_ADD: i16 = 0;

/// A binary-op unit (`a <op> b`) at audio rate.
fn binop(op: i16, a: u32, b: u32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![
            InputRef::Unit { unit: a, output: 0 },
            InputRef::Unit { unit: b, output: 0 },
        ],
        num_outputs: 1,
        special_index: op,
    }
}

/// Build a `World` already playing the feedback-echo patch on every output channel.
fn build(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        ..Options::default()
    });

    // Out.ar(0, master) copied to every channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit {
            unit: 10,
            output: 0,
        });
    }
    let def = SynthDef {
        name: "echo".to_string(),
        params: vec![],
        units: vec![
            // 0: SinOsc.ar(TONE) - the ping's carrier.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(TONE), InputRef::Constant(0.0)],
                1,
            ),
            // 1: LFPulse.ar(BEEP_HZ, 0, WIDTH) - a short gate, once per ping period.
            UnitSpec::new(
                "LFPulse",
                Rate::Audio,
                vec![
                    InputRef::Constant(BEEP_HZ),
                    InputRef::Constant(0.0),
                    InputRef::Constant(WIDTH),
                ],
                1,
            ),
            // 2: Lag.ar(gate, LAG) - round the gate edges into a pluck envelope.
            UnitSpec::new(
                "Lag",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(LAG),
                ],
                1,
            ),
            // 3: dry = SinOsc * env.
            binop(OP_MUL, 0, 2),
            // 4: LocalIn.ar(1) - last block's wet signal (the one-block feedback delay).
            UnitSpec::new("LocalIn", Rate::Audio, vec![], 1),
            // 5: fb = LocalIn * FEEDBACK.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 4, output: 0 },
                    InputRef::Constant(FEEDBACK),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 6: mixed = dry + fb - what enters the delay line.
            binop(OP_ADD, 3, 5),
            // 7: wet = DelayN.ar(mixed, MAX_DELAY, ECHO_SECS) - the aux-memory delay line.
            UnitSpec::new(
                "DelayN",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 6, output: 0 },
                    InputRef::Constant(MAX_DELAY),
                    InputRef::Constant(ECHO_SECS),
                ],
                1,
            ),
            // 8: LocalOut.ar(wet) - feed the wet signal back for next block.
            UnitSpec::new(
                "LocalOut",
                Rate::Audio,
                vec![InputRef::Unit { unit: 7, output: 0 }],
                0,
            ),
            // 9: mix = dry + wet (the ping plus its decaying echoes).
            binop(OP_ADD, 3, 7),
            // 10: master = mix * MASTER.
            UnitSpec::new(
                "MulAdd",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 9, output: 0 },
                    InputRef::Constant(MASTER),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 11: Out.ar(0, master) on every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("echo", ROOT_GROUP_ID, AddAction::Tail);

    // The synth plays forever and never frees, so there is no NRT cleanup: drop the `Controller`
    // and `Nrt`, keep only the `World`.
    world
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
    println!("a feedback echo through DelayN: a ping every ~2s, then decaying repeats, for 15s...");

    let stream = example_audio::play(GAIN, |sample_rate, channels| {
        let mut world = build(sample_rate as f32, channels);
        move |out: &mut [f32], channels: usize| world.fill(out, channels)
    });
    example_audio::keep_alive(stream, 15);
}
