//! A self-driving melodic sequencer built from demand-rate units.
//!
//! Every other sequencing example drives note onsets from the control plane - a thread loop, a web
//! timer, or a batch of time-tagged OSC bundles. This one has *no* per-note control-plane traffic at
//! all: a single synth, created once, sequences itself on the audio thread. `Duty.kr` clocks the
//! sequence - each time its current note's duration elapses it *demands* the next duration and the
//! next note pitch from two demand-rate sources, entirely on the RT thread.
//!
//! The melody source shows demand-rate units nesting: an outer `Dseq` whose items are a `Dseries`
//! (a rising arpeggio), a `Dwhite` (two random notes), and two fixed pitches - so one line walks
//! through `Dseq`, `Dseries`, and `Dwhite`. The only off-RT work is compiling the `SynthDef`; the
//! pulling, sequencing, and randomness all happen in the audio callback.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// Peak amplitude of the oscillator.
const AMP: f32 = 0.2;

/// The sequencer synth, built entirely from demand-rate units:
///
/// ```text
///   freq = Duty.kr(
///       dur:   Dseq([0.15, 0.15, 0.3], inf),                       // the rhythm
///       level: Dseq([ Dseries(4, 220, 55),  // 220 275 330 385     // the melody, nesting
///                     Dwhite(2, 300, 500),   // two random notes
///                     440, 330 ], inf))
///   out  = SinOsc.ar(freq) * AMP
/// ```
fn seq_def(channels: usize) -> SynthDef {
    // Build the multi-channel `Out` inputs: bus 0, then the (amplified) oscillator into each channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 6, output: 0 });
    }
    SynthDef {
        name: "duty-seq".to_string(),
        params: vec![],
        units: vec![
            // 0: Dseries(length: 4, start: 220, step: 55) - a four-note rising arpeggio.
            UnitSpec::new(
                "Dseries",
                Rate::Demand,
                vec![
                    InputRef::Constant(4.0),
                    InputRef::Constant(220.0),
                    InputRef::Constant(55.0),
                ],
                1,
            ),
            // 1: Dwhite(length: 2, lo: 300, hi: 500) - two random notes per pass.
            UnitSpec::new(
                "Dwhite",
                Rate::Demand,
                vec![
                    InputRef::Constant(2.0),
                    InputRef::Constant(300.0),
                    InputRef::Constant(500.0),
                ],
                1,
            ),
            // 2: Dseq([Dseries, Dwhite, 440, 330], inf) - the melody, nesting the two sources above.
            UnitSpec::new(
                "Dseq",
                Rate::Demand,
                vec![
                    InputRef::Constant(f32::INFINITY),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(440.0),
                    InputRef::Constant(330.0),
                ],
                1,
            ),
            // 3: Dseq([0.15, 0.15, 0.3], inf) - the rhythm (beat durations in seconds).
            UnitSpec::new(
                "Dseq",
                Rate::Demand,
                vec![
                    InputRef::Constant(f32::INFINITY),
                    InputRef::Constant(0.15),
                    InputRef::Constant(0.15),
                    InputRef::Constant(0.3),
                ],
                1,
            ),
            // 4: Duty.kr(dur: rhythm, reset: 0, level: melody) - pulls the next note when each beat
            // elapses. This is the only clock; there is no control-plane tick.
            UnitSpec::new(
                "Duty",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 5: SinOsc.ar(freq = Duty.kr output).
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 4, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 6: SinOsc * AMP.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 5, output: 0 },
                    InputRef::Constant(AMP),
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 7: Out.ar(0, osc) - the same voice into every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// Build the engine, register the sequencer def, and start the single self-driving synth. Returns the
/// NRT cleanup side and the audio [`World`]; the `Controller` is dropped (its queued commands live on
/// in the ring until the audio thread applies them).
fn build(sample_rate: f64, channels: usize) -> (Nrt, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(seq_def(channels));
    controller
        .synth_new("duty-seq", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    (nrt, world)
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

fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f64;

    let (nrt, mut world) = build(sample_rate, channels);

    let mut scratch: Vec<f32> = Vec::new();
    let stream = device
        .build_output_stream(
            config,
            move |output: &mut [T], _info: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                world.fill(&mut scratch, channels);
                for (out, sample) in output.iter_mut().zip(scratch.iter()) {
                    *out = T::from_sample(*sample);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream");
    stream.play().expect("failed to start audio stream");

    run_control_plane(nrt, stream);
}

/// The synth sequences itself on the audio thread, so the control plane has nothing to schedule - it
/// just ticks NRT cleanup and holds the stream alive for the demo's duration.
#[cfg(not(target_arch = "wasm32"))]
fn run_control_plane(mut nrt: Nrt, _stream: cpal::Stream) {
    use std::time::Duration;
    println!("playing a self-driving demand-rate sequence (~12s); no per-note control messages...");
    for _ in 0..240 {
        nrt.process();
        while nrt.poll().is_some() {}
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// On the web, tick NRT cleanup on a periodic timer and keep it and the stream alive.
#[cfg(target_arch = "wasm32")]
fn run_control_plane(mut nrt: Nrt, stream: cpal::Stream) {
    let interval = gloo_timers::callback::Interval::new(50, move || {
        nrt.process();
        while nrt.poll().is_some() {}
    });
    interval.forget();
    std::mem::forget(stream);
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
    fn first_beat_sounds_the_first_arpeggio_note() {
        // The melody's first note is the Dseries start (220 Hz), held for the first beat (0.15 s).
        // Render a window inside that first beat and confirm 220 Hz dominates - i.e. Duty.kr pulled
        // the nested Dseq -> Dseries on the audio thread with no control-plane help.
        let (_nrt, mut world) = build(SR, 1);
        let mut out = vec![0.0f32; 4096];
        world.fill(&mut out, 1);

        assert!(
            out.iter().any(|s| s.abs() > 0.01),
            "the sequencer was silent"
        );
        assert!(out.iter().all(|s| s.abs() <= 1.0), "output left [-1, 1]");
        let fundamental = goertzel(&out, 220.0);
        let other = goertzel(&out, 330.0);
        assert!(
            fundamental > 5.0 * other,
            "first beat should sound 220 Hz (got 220={fundamental:.4}, 330={other:.4})"
        );
    }
}
