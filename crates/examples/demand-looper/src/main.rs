//! A self-driving step sequencer whose melody lives in a buffer.
//!
//! A buffer is pre-filled (off the audio thread) with a row of pitches. A single synth then sequences
//! itself entirely on the audio thread: `Duty.kr` clocks the steps, and on each step a demand-rate
//! `Dbufrd` reads the next pitch out of the buffer (looping with `Dseries` as the phase), which feeds
//! `SinOsc`. A `Dpoll` in the chain posts each pitch to the console as it is demanded.
//!
//! This exercises the demand-rate **buffer reach** - a demand source touching the world's buffer
//! table, which the pull context could not do before. The write side (`Dbufwr`) is covered by the
//! `demand_buffer` round-trip test; here the buffer is filled control-side so the demo stays a single
//! self-driving synth, like `duty-seq`.

use plyphon::{
    AddAction, Buffer, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

/// Peak amplitude of the oscillator.
const AMP: f32 = 0.2;
/// Master gain applied in the audio callback.
const GAIN: f32 = 1.0;
/// Seconds per step (the `Duty` duration).
const STEP: f32 = 0.22;
/// The melody written into the buffer, one pitch per step (a minor-pentatonic riff in A).
const MELODY: [f32; 8] = [220.0, 262.0, 294.0, 330.0, 392.0, 330.0, 294.0, 262.0];

/// The sequencer synth:
///
/// ```text
///   phase = Dseries(inf, 0, 1)                       // 0, 1, 2, ... (wrapped by Dbufrd's loop)
///   note  = Dpoll(Dbufrd(buf: 0, phase, loop: 1), "note")   // read the melody, posting each pitch
///   freq  = Duty.kr(STEP, 0, note, 0)                // clock the steps, hold each pitch
///   out   = SinOsc.ar(freq) * AMP
/// ```
fn seq_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 5, output: 0 });
    }
    SynthDef {
        name: "demand-looper".to_string(),
        params: vec![],
        units: vec![
            // 0: Dseries(length: inf, start: 0, step: 1) - the running phase into the buffer.
            UnitSpec::new(
                "Dseries",
                Rate::Demand,
                vec![
                    InputRef::Constant(f32::INFINITY),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            // 1: Dbufrd(bufnum: 0, phase: unit 0, loop: 1) - read the next melody pitch, wrapping.
            UnitSpec::new(
                "Dbufrd",
                Rate::Demand,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            // 2: Dpoll(in: unit 1, trigid: -1, run: 1, label: "note") - post each demanded pitch.
            UnitSpec::new(
                "Dpoll",
                Rate::Demand,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(-1.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(4.0),
                    InputRef::Constant(f32::from(b'n')),
                    InputRef::Constant(f32::from(b'o')),
                    InputRef::Constant(f32::from(b't')),
                    InputRef::Constant(f32::from(b'e')),
                ],
                1,
            ),
            // 3: Duty.kr(dur: STEP, reset: 0, level: unit 2, doneAction: 0) - the self-clock.
            UnitSpec::new(
                "Duty",
                Rate::Control,
                vec![
                    InputRef::Constant(STEP),
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 4: SinOsc.ar(freq = Duty output).
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 5: SinOsc * AMP.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 4, output: 0 },
                    InputRef::Constant(AMP),
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 6: Out.ar(0, osc) - the same voice into every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// Build the engine, install the melody buffer, register the def, and start the self-driving synth.
fn build(sample_rate: f64, channels: usize) -> (Nrt, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate,
        output_channels: channels,
        ..Options::default()
    });
    // The melody lives in buffer 0; the synth reads it on the audio thread via Dbufrd.
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(MELODY.to_vec(), 1, sample_rate)),
        )
        .expect("buffer_set");
    controller.add_synthdef(seq_def(channels));
    controller
        .synth_new("demand-looper", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (nrt, world)
}

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    // cpal's AudioWorklet backend re-instantiates this module on the audio thread, re-running `main`
    // there; only set up audio on the main browser thread.
    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!(
        "playing a demand-rate buffer sequencer (~9s); each note is posted as it is demanded..."
    );

    let (stream, mut nrt) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (nrt, mut world) = build(sample_rate, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            nrt,
        )
    });
    // The synth sequences itself on the audio thread; the control thread just ticks NRT cleanup and
    // drains the posted Dpoll notes (surfaced as node messages off the audio thread).
    example_audio::run_control(stream, 9_000, 50, move || {
        nrt.process();
        while let Some(msg) = nrt.poll_node_msg() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = std::str::from_utf8(&msg.label[..msg.label_len as usize]).unwrap_or("");
                println!("{label}: {}", msg.values[0]);
            }
            #[cfg(target_arch = "wasm32")]
            let _ = msg;
        }
        while nrt.poll().is_some() {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// Goertzel magnitude of `freq` in `samples` - a single-bin DTFT for cheap pitch checks.
    fn goertzel(samples: &[f32], freq: f32) -> f32 {
        let n = samples.len();
        let k = (0.5 + n as f32 * freq / SR as f32).floor();
        let w = 2.0 * std::f32::consts::PI * k / n as f32;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in samples {
            let s = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
    }

    #[test]
    fn first_step_sounds_the_first_melody_pitch() {
        // The first step reads MELODY[0] (220 Hz) out of the buffer with Dbufrd and holds it for the
        // first beat - so a window inside that beat should be dominated by 220 Hz.
        let (_nrt, mut world) = build(SR, 1);
        let mut out = vec![0.0f32; 4096];
        world.fill(&mut out, 1);

        assert!(
            out.iter().any(|s| s.abs() > 0.01),
            "the sequencer was silent"
        );
        assert!(out.iter().all(|s| s.abs() <= 1.0), "output left [-1, 1]");
        let fundamental = goertzel(&out, MELODY[0]);
        let other = goertzel(&out, MELODY[3]);
        assert!(
            fundamental > 5.0 * other,
            "first step should sound {} Hz (got {}={fundamental:.4}, {}={other:.4})",
            MELODY[0],
            MELODY[0],
            MELODY[3],
        );
    }
}
