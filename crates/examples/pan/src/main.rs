//! A tone auto-panned across the stereo field with `Pan2`, via cpal, natively and on the web.
//!
//! A `Saw` oscillator is panned by `Pan2`, whose position is driven by a slow `SinOsc.kr` LFO, so
//! the tone drifts left and right. The whole patch is in-engine (no control plane), like the sine
//! example. Best heard in stereo; on a mono device you only hear the left channel.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The panned tone (Hz).
const FREQ: f32 = 220.0;
/// How fast the tone pans across the field (Hz).
const PAN_RATE: f32 = 0.3;
/// A gentle master gain.
const GAIN: f32 = 0.2;

/// Build a `World` playing an auto-panned saw across the first two channels.
fn build(sample_rate: f32, channels: usize) -> World {
    // Pan2 writes two channels, so give the engine at least two output channels.
    let out_channels = channels.max(2);
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: sample_rate as f64,
        output_channels: out_channels,
        ..Options::default()
    });

    // SinOsc.kr(PAN_RATE) -> pan position; Saw.ar(FREQ) -> Pan2 -> Out.ar(0, [left, right]).
    let def = SynthDef {
        name: "pan".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Control,
                vec![InputRef::Constant(PAN_RATE), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new("Saw", Rate::Audio, vec![InputRef::Constant(FREQ)], 1),
            UnitSpec::new(
                "Pan2",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 1, output: 0 }, // in
                    InputRef::Unit { unit: 0, output: 0 }, // pos = LFO
                    InputRef::Constant(1.0),               // level
                ],
                2,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Unit { unit: 2, output: 1 },
                ],
                0,
            ),
        ],
    };
    controller.add_synthdef(def);
    let _ = controller.synth_new("pan", ROOT_GROUP_ID, AddAction::Tail);

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

fn run<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: &cpal::StreamConfig) {
    let channels = config.channels as usize;
    let sample_rate = config.sample_rate as f32;

    let mut source = build(sample_rate, channels);
    let mut scratch: Vec<f32> = Vec::new();

    let stream = device
        .build_output_stream(
            *config,
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
        println!("auto-panning a {FREQ} Hz saw for 10s...");
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
    #[cfg(target_arch = "wasm32")]
    std::mem::forget(stream);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 48_000.0;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
    }

    /// Over a pan cycle the tone should reach both channels, and the balance should shift over time.
    #[test]
    fn the_tone_pans_across_both_channels() {
        let mut world = build(SR, 2);
        // Render ~2 s (more than half a pan period) in stereo.
        let frames = (SR * 2.0) as usize;
        let mut out = vec![0.0f32; frames * 2];
        world.fill(&mut out, 2);
        let left: Vec<f32> = out.iter().step_by(2).copied().collect();
        let right: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
        assert!(rms(&left) > 0.05, "the tone should reach the left channel");
        assert!(
            rms(&right) > 0.05,
            "the tone should reach the right channel"
        );

        // The balance early vs late should differ (the pan moved).
        let early = rms(&left[..4800]) / (rms(&right[..4800]) + 1e-6);
        let late = rms(&left[frames - 4800..]) / (rms(&right[frames - 4800..]) + 1e-6);
        assert!(
            (early - late).abs() > 0.1,
            "the stereo balance should change as the tone pans (early={early}, late={late})"
        );
    }
}
