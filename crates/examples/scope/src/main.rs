//! Tap a live signal with `ScopeOut` and draw a level meter from the streamed samples - native only.
//!
//! `ScopeOut` is plyphon's shared-memory-free take on scsynth's `ScopeOut2`: it streams every sample
//! of its input off the audio thread into a cued chunk ring the app drains. Here a tremolo'd tone is
//! both played (`Out`) and tapped (`ScopeOut`); a background thread drains the scope stream and prints
//! a live peak meter, so you see the exact samples the engine produced arriving in the app. The audio
//! thread only copies each block into a pre-allocated chunk over a wait-free ring - it never blocks,
//! allocates, or touches the terminal.
//!
//! ```console
//! cargo run -p example-scope
//! ```

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, StreamConsumer, SynthDef, UnitSpec, engine,
};

/// A gentle master gain on playback.
const GAIN: f32 = 0.4;
/// The tapped (and played) tone, in Hz.
const FREQ: f32 = 220.0;
/// The tremolo rate, in Hz (a slow amplitude wobble so the meter visibly moves).
const TREMOLO_HZ: f32 = 0.7;
/// The scope tap is mono.
const SCOPE_CHANNELS: usize = 1;
/// Chunk size and queue depth: 8 x 2048 frames is ~340 ms of headroom at 48 kHz.
const CHUNK_FRAMES: usize = 2048;
const NUM_CHUNKS: usize = 8;
/// How often the drainer wakes to empty the queue and redraw the meter, in milliseconds.
const TICK_MS: u64 = 60;
/// How long to run, in seconds.
const RUN_SECS: u64 = 12;

fn main() {
    // Build the engine on the output device; `play_with` hands the controller back so we can cue the
    // scope stream and start the synth from this (the main) thread.
    let (stream, (mut controller, sample_rate, channels)) =
        example_audio::play_with(GAIN, |sample_rate, channels| {
            let (controller, _nrt, mut world) = engine(Options {
                sample_rate,
                output_channels: channels.max(1),
                ..Options::default()
            });
            (
                move |out: &mut [f32], ch: usize| world.fill(out, ch),
                (controller, sample_rate, channels),
            )
        });

    // Cue a mono scope stream and start a tremolo'd tone that is both played and tapped.
    let consumer = controller
        .cue_scope(0, SCOPE_CHANNELS, sample_rate, CHUNK_FRAMES, NUM_CHUNKS)
        .expect("failed to cue the scope stream");
    controller.add_synthdef(tone_def(channels));
    controller
        .synth_new("scope", ROOT_GROUP_ID, AddAction::Tail)
        .expect("failed to start the synth");

    // Drain the scope stream on a background thread and draw a peak meter from the streamed samples.
    let stop = Arc::new(AtomicBool::new(false));
    let meter = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || run_meter(consumer, &stop))
    };

    println!(
        "scoping a {FREQ} Hz tremolo tone for {RUN_SECS}s (peak meter from the ScopeOut stream):"
    );
    example_audio::keep_alive(stream, RUN_SECS);
    stop.store(true, Ordering::Relaxed);
    meter.join().expect("the meter thread panicked");
    println!();
}

/// Poll the scope stream until `stop`, drawing a decaying peak meter from every drained sample.
fn run_meter(mut consumer: StreamConsumer, stop: &AtomicBool) {
    let mut level = 0.0f32;
    while !stop.load(Ordering::Relaxed) {
        // Drain every chunk the audio thread has queued since the last tick, tracking the peak.
        let mut peak = 0.0f32;
        while let Some(chunk) = consumer.pop_filled() {
            for &s in chunk.filled_samples() {
                peak = peak.max(s.abs());
            }
            consumer.recycle(chunk);
        }
        // A simple attack/release so the bar rises fast and falls smoothly.
        level = if peak > level {
            peak
        } else {
            level * 0.8 + peak * 0.2
        };
        print!("\r{}", bar(level));
        let _ = std::io::stdout().flush();
        std::thread::sleep(std::time::Duration::from_millis(TICK_MS));
    }
}

/// A 40-cell ASCII level bar for `level` in `[0, 1]`, e.g. `[############        ] 0.62`.
fn bar(level: f32) -> String {
    const WIDTH: usize = 40;
    let filled = (level.clamp(0.0, 1.0) * WIDTH as f32) as usize;
    let mut s = String::with_capacity(WIDTH + 12);
    s.push('[');
    for i in 0..WIDTH {
        s.push(if i < filled { '#' } else { ' ' });
    }
    s.push_str(&format!("] {level:.2}"));
    s
}

/// `(SinOsc.ar(FREQ) * tremolo)` played to the speakers (`Out`) and tapped (`ScopeOut`), where the
/// tremolo is a slow `SinOsc.kr` mapped to a `[0.1, 0.9]` gain.
fn tone_def(device_channels: usize) -> SynthDef {
    // A control-rate MulAdd (`in * mul + add`).
    let mul_add = |src: u32, mul: f32, add: f32| UnitSpec {
        name: "MulAdd".to_string(),
        rate: Rate::Control,
        inputs: vec![
            InputRef::Unit {
                unit: src,
                output: 0,
            },
            InputRef::Constant(mul),
            InputRef::Constant(add),
        ],
        num_outputs: 1,
        special_index: 0,
    };
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..device_channels.max(1) {
        out_inputs.push(InputRef::Unit { unit: 3, output: 0 });
    }
    SynthDef {
        name: "scope".to_string(),
        params: vec![],
        units: vec![
            // 0: SinOsc.ar(FREQ) - the tone.
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(FREQ), InputRef::Constant(0.0)],
                1,
            ),
            // 1: SinOsc.kr(TREMOLO_HZ) - a slow LFO.
            UnitSpec::new(
                "SinOsc",
                Rate::Control,
                vec![InputRef::Constant(TREMOLO_HZ), InputRef::Constant(0.0)],
                1,
            ),
            // 2: map the LFO to a [0.1, 0.9] gain.
            mul_add(1, 0.4, 0.5),
            // 3: tone * gain.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 4: ScopeOut.ar(0, voice) - the tap the app drains.
            UnitSpec::new(
                "ScopeOut",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 3, output: 0 },
                ],
                0,
            ),
            // 5: Out.ar(0, voice) - to the speakers.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plyphon::World;

    const SR: f32 = 48_000.0;

    fn goertzel(samples: &[f32], freq: f32) -> f32 {
        let n = samples.len();
        let k = (0.5 + n as f32 * freq / SR).floor();
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

    /// Render the synth offline, drain the scope stream directly, and confirm it carries the tapped
    /// tone (dominant at `FREQ`) with a tremolo (its windowed peak varies). Exercises the whole
    /// ScopeOut -> StreamConsumer path the example draws from, minus cpal.
    #[test]
    fn scope_stream_carries_the_tapped_tone() {
        let (mut controller, _nrt, mut world): (_, _, World) = engine(Options {
            sample_rate: SR as f64,
            output_channels: 1,
            ..Options::default()
        });
        let mut consumer = controller
            .cue_scope(0, SCOPE_CHANNELS, SR as f64, 1024, NUM_CHUNKS)
            .unwrap();
        controller.add_synthdef(tone_def(1));
        controller
            .synth_new("scope", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();

        // Render a couple of seconds (enough to see the tremolo), draining as we go so nothing overruns.
        let mut buf = vec![0.0f32; 512];
        let mut got = Vec::new();
        for _ in 0..((SR * 2.0) as usize / 512) {
            world.fill(&mut buf, 1);
            while let Some(chunk) = consumer.pop_filled() {
                got.extend_from_slice(chunk.filled_samples());
                consumer.recycle(chunk);
            }
        }

        assert!(!got.is_empty(), "the scope stream was empty");
        assert!(
            goertzel(&got, FREQ) > 5.0 * goertzel(&got, FREQ * 2.0),
            "expected the tapped {FREQ} Hz tone in the scope stream"
        );
        // The tremolo makes the windowed peak vary across the stream.
        let win = SR as usize / 4;
        let peaks: Vec<f32> = got
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, &s| m.max(s.abs())))
            .collect();
        let loud = peaks.iter().cloned().fold(0.0f32, f32::max);
        let quiet = peaks.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            loud > 1.3 * quiet.max(1e-3),
            "expected a tremolo (loud {loud}, quiet {quiet})"
        );
    }
}
