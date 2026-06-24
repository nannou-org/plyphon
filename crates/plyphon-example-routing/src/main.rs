//! A bus-routing plyphon example: an LFO-swept low-pass filter on a noise source, via cpal.
//!
//! This shows signals flowing *between* synths through buses - the thing a single synth graph can't
//! do. Three synths are wired together by two buses:
//!
//! 1. `lfo`: a slow control-rate sine, scaled to a cutoff range, written to a **control bus** with
//!    `Out.kr`.
//! 2. `source`: `WhiteNoise.ar` written to a private **audio bus** with `Out.ar`.
//! 3. `filter`: reads the noise bus with `In.ar` and low-passes it at the cutoff read from the
//!    control bus with `In.kr`, then writes the result to the output.
//!
//! Nothing is ever freed, so (as in `plyphon-example-sine`) the `Controller`/`Nrt` are dropped once
//! the graph is queued, leaving only the `World` for the cpal callback to fill. The whole patch is
//! static - the sweep is driven by the in-engine LFO, not the host - so it needs no control plane.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// LFO rate in Hz (how fast the filter sweeps).
const LFO_FREQ: f32 = 0.25;
/// Centre cutoff frequency in Hz.
const CUTOFF_CENTRE: f32 = 1200.0;
/// Cutoff sweep depth in Hz (the LFO's -1..1 is scaled by this around the centre).
const CUTOFF_DEPTH: f32 = 1000.0;
/// A gentle master gain.
const GAIN: f32 = 0.15;

/// SuperCollider binary-operator indices (see `BinaryOpUGen`): multiply and add.
const OP_MUL: i16 = 2;
const OP_ADD: i16 = 0;

/// Build a `World` already playing LFO-swept filtered noise routed through buses.
fn build(sample_rate: f32, channels: usize) -> World {
    let channels = channels.max(1);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: channels,
        input_channels: 0,
        ..Options::default()
    });

    // The noise source lives on the first private audio bus, just past the output channels.
    let noise_bus = channels as f32;
    // The LFO writes, and the filter reads, this control bus for the cutoff frequency.
    let cutoff_bus = 0.0;

    // lfo: SinOsc.kr(LFO_FREQ) * CUTOFF_DEPTH + CUTOFF_CENTRE -> Out.kr(cutoff_bus)
    let lfo = SynthDef {
        name: "lfo".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Control,
                vec![InputRef::Constant(LFO_FREQ), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(CUTOFF_DEPTH),
                ],
                num_outputs: 1,
                special_index: OP_MUL,
            },
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Control,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(CUTOFF_CENTRE),
                ],
                num_outputs: 1,
                special_index: OP_ADD,
            },
            UnitSpec::new(
                "Out",
                Rate::Control,
                vec![
                    InputRef::Constant(cutoff_bus),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    };

    // source: WhiteNoise.ar -> Out.ar(noise_bus)
    let source = SynthDef {
        name: "source".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(noise_bus),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    };

    // filter: LPF(In.ar(noise_bus), In.kr(cutoff_bus)) -> Out.ar(0, ..), copied to every channel.
    let mut out_inputs = vec![InputRef::Constant(0.0)]; // input 0: starting output channel
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 2, output: 0 });
    }
    let filter = SynthDef {
        name: "filter".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Audio, vec![InputRef::Constant(noise_bus)], 1),
            UnitSpec::new("In", Rate::Control, vec![InputRef::Constant(cutoff_bus)], 1),
            UnitSpec::new(
                "LPF",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    };

    controller.add_synthdef(lfo);
    controller.add_synthdef(source);
    controller.add_synthdef(filter);
    // Order matters: the LFO and source must run before the filter within a block, so the filter
    // reads this block's cutoff and noise. Adding each at the tail yields process order
    // lfo -> source -> filter.
    let _ = controller.synth_new("lfo", ROOT_GROUP_ID, AddAction::Tail);
    let _ = controller.synth_new("source", ROOT_GROUP_ID, AddAction::Tail);
    let _ = controller.synth_new("filter", ROOT_GROUP_ID, AddAction::Tail);

    // The graph plays forever and never frees, so there is no NRT cleanup: drop the `Controller`
    // and `Nrt`, keeping only the `World`.
    world
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

/// Build and play an output stream fed by the engine `World`.
fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0 as f32;

    let mut source = build(sample_rate, channels);
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
                    *out = T::from_sample(*sample * GAIN);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream");
    stream.play().expect("failed to start audio stream");

    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("playing LFO-swept filtered noise (routed through buses) for 10s...");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    // On the web `main` returns immediately; keep the stream (and its callback) alive.
    #[cfg(target_arch = "wasm32")]
    std::mem::forget(stream);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    /// Render `frames` of mono audio from `world`.
    fn render(world: &mut World, frames: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames + 512);
        let mut buf = vec![0.0f32; 512];
        while out.len() < frames {
            world.fill(&mut buf, 1);
            out.extend_from_slice(&buf);
        }
        out.truncate(frames);
        out
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// The graph should make sound, and the in-engine LFO should audibly sweep the filter via the
    /// control bus: a window near the LFO peak (bright, high cutoff) carries more energy than one
    /// near the trough (dark, low cutoff). Exercises the whole routing chain headlessly.
    #[test]
    fn sweeps_filtered_noise_through_buses() {
        let mut world = build(SR, 1);
        // SinOsc.kr starts at phase 0, so cutoff = sin(2*pi*LFO_FREQ*t)*DEPTH + CENTRE: brightest a
        // quarter-period in (peak), darkest three-quarters in (trough). At 0.25 Hz that is t=1s and
        // t=3s. Render past the trough and slice a 0.2 s window around each.
        let all = render(&mut world, (SR * 3.5) as usize);
        assert!(
            all.iter().any(|s| s.abs() > 0.05),
            "the graph produced no sound"
        );

        let half_window = (SR * 0.1) as usize;
        let window = |centre_secs: f32| {
            let c = (SR * centre_secs) as usize;
            &all[c - half_window..c + half_window]
        };
        let bright = rms(window(1.0));
        let dark = rms(window(3.0));
        assert!(
            bright > 1.5 * dark,
            "expected the LFO to brighten the filter via the control bus: bright={bright}, dark={dark}"
        );
    }
}
