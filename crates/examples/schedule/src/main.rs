//! Sample-accurate rhythm from time-tagged OSC bundles, scheduled ahead of the audio clock.
//!
//! Every other cpal example drives the engine from a control-plane tick - a thread loop or a web
//! timer - whose timing is only as good as the host's scheduler (milliseconds of jitter). This one
//! instead hands the engine a batch of time-tagged OSC bundles *up front* (and deliberately out of
//! order), one `/s_new` per beat, each tagged with the exact OSC/NTP time it should sound. The
//! engine holds them in its scheduler and fires each at its precise sample - via `OffsetOut`, on a
//! drift-corrected clock - so the rhythm is sample-accurate and decoupled from any tick.
//!
//! Natively the bundles carry real wall-clock times and the audio callback drives [`World::fill_at`]
//! with each buffer's host time, so scheduling stays accurate even as the audio device clock drifts
//! against the system clock. On the web (no wall clock here) the engine clock free-runs at the
//! nominal rate via [`World::fill`] and the times are relative to engine start; the rhythm is
//! identical.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use plyphon::{
    InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};
use plyphon_osc::OscDispatcher;
use rosc::{OscBundle, OscMessage, OscPacket, OscTime, OscType};

/// OSC/NTP fixed-point units in one second (OSC time is 32.32 fixed point).
const OSC_UNITS_PER_SEC: f64 = 4_294_967_296.0;
/// A pentatonic scale (Hz) the beats walk through.
const SCALE: [f32; 5] = [261.63, 311.13, 349.23, 392.00, 466.16];
/// Seconds between beats.
const BEAT_SECS: f64 = 0.16;
/// Number of beats in the phrase.
const NUM_BEATS: usize = 24;
/// Lead time before the first beat (s) - covers startup and output latency.
const LEAD_SECS: f64 = 0.3;
/// Peak amplitude of each blip.
const AMP: f32 = 0.3;

/// The control side: the OSC front-end that schedules the phrase, and the NRT cleanup.
struct Sched {
    dispatcher: OscDispatcher,
    nrt: Nrt,
}

impl Sched {
    /// Off-audio-thread upkeep: drop freed synths and drain notifications.
    fn tick(&mut self) {
        self.nrt.process();
        while self.nrt.poll().is_some() {}
    }
}

/// `SinOsc.ar(freq) * EnvGen.kr(Env.perc, doneAction: 2)` written via `OffsetOut` so a scheduled
/// note onsets at exactly its sample. `freq` is a parameter set per beat.
fn blip_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 2, output: 0 });
    }
    SynthDef {
        name: "blip".to_string(),
        params: vec![Param {
            name: "freq".to_string(),
            default: 440.0,
        }],
        units: vec![
            UnitSpec::new("EnvGen", Rate::Control, perc_env_inputs(), 1),
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            UnitSpec::new("OffsetOut", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// `Env.perc(0.001, 0.13, AMP)` unrolled for `EnvGen`: a near-instant attack into a short
/// exponential decay, freeing the synth when it has faded.
fn perc_env_inputs() -> Vec<InputRef> {
    let values = [
        1.0,   // gate
        1.0,   // levelScale
        0.0,   // levelBias
        1.0,   // timeScale
        2.0,   // doneAction = 2 (free when faded)
        0.0,   // initialLevel
        2.0,   // numSegments
        -99.0, // releaseNode (none)
        -99.0, // loopNode (none)
        AMP, 0.001, 1.0, 0.0, // attack: -> AMP over 1 ms, linear
        0.0, 0.13, 5.0, -4.0, // decay: -> 0 over 130 ms, exponential
    ];
    values.into_iter().map(InputRef::Constant).collect()
}

/// Build the engine: a [`Sched`] (kept on the control side) and the [`World`] (the audio source).
fn build(sample_rate: f64, channels: usize) -> (Sched, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(blip_def(channels));
    (
        Sched {
            dispatcher: OscDispatcher::new(controller),
            nrt,
        },
        world,
    )
}

/// Split a packed 32.32 OSC/NTP `u64` into the [`OscTime`] a bundle carries.
fn unpack_ntp(ntp: u64) -> OscTime {
    OscTime {
        seconds: (ntp >> 32) as u32,
        fractional: ntp as u32,
    }
}

/// A time-tagged bundle that starts one `blip` at `freq`, scheduled for OSC/NTP `time`.
fn beat_bundle(time: u64, id: i32, freq: f32) -> OscPacket {
    OscPacket::Bundle(OscBundle {
        timetag: unpack_ntp(time),
        content: vec![OscPacket::Message(OscMessage {
            addr: "/s_new".to_string(),
            args: vec![
                OscType::String("blip".to_string()),
                OscType::Int(id),
                OscType::Int(1),             // addAction: tail
                OscType::Int(ROOT_GROUP_ID), // target: root group
                OscType::String("freq".to_string()),
                OscType::Float(freq),
            ],
        })],
    })
}

/// Schedule the whole phrase as a batch of time-tagged bundles relative to `base`, submitted in
/// reverse to show the engine fires them by time tag, not arrival order.
fn schedule_phrase(sched: &mut Sched, base: u64) {
    let beat_units = (BEAT_SECS * OSC_UNITS_PER_SEC) as u64;
    for k in (0..NUM_BEATS).rev() {
        let time = base + k as u64 * beat_units;
        let freq = SCALE[k % SCALE.len()];
        let _ = sched
            .dispatcher
            .apply(&beat_bundle(time, 1000 + k as i32, freq));
    }
}

/// The OSC/NTP wall clock, native only (the web build has no `SystemTime` and uses the engine's own
/// free-running clock instead).
#[cfg(not(target_arch = "wasm32"))]
mod clock {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    /// Seconds between the NTP epoch (1900) and the Unix epoch (1970).
    const NTP_UNIX_OFFSET: u64 = 2_208_988_800;

    /// The current wall-clock time, plus `extra`, as packed OSC/NTP.
    pub fn now(extra: Duration) -> u64 {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            + extra;
        let secs = d.as_secs() + NTP_UNIX_OFFSET;
        let frac = ((d.subsec_nanos() as u64) << 32) / 1_000_000_000;
        (secs << 32) | frac
    }

    /// The OSC/NTP time at which this callback's first output frame is heard: now + output latency.
    pub fn buffer_time(info: &cpal::OutputCallbackInfo) -> u64 {
        let ts = info.timestamp();
        let ahead = ts
            .playback
            .duration_since(&ts.callback)
            .unwrap_or(Duration::ZERO);
        now(ahead)
    }
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

    let (mut sched, mut world) = build(sample_rate, channels);

    // The phrase's base time: real wall clock + lead natively; relative to engine start on the web.
    #[cfg(not(target_arch = "wasm32"))]
    let base = clock::now(std::time::Duration::from_secs_f64(LEAD_SECS));
    #[cfg(target_arch = "wasm32")]
    let base = (LEAD_SECS * OSC_UNITS_PER_SEC) as u64;
    schedule_phrase(&mut sched, base);

    let mut scratch: Vec<f32> = Vec::new();
    let stream = device
        .build_output_stream(
            config,
            move |output: &mut [T], info: &cpal::OutputCallbackInfo| {
                scratch.clear();
                scratch.resize(output.len(), 0.0);
                // Natively, drift-correct the engine clock to the buffer's host time; on the web,
                // let the engine clock free-run at the nominal rate.
                #[cfg(not(target_arch = "wasm32"))]
                world.fill_at(&mut scratch, channels, clock::buffer_time(info));
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = info;
                    world.fill(&mut scratch, channels);
                }
                for (out, sample) in output.iter_mut().zip(scratch.iter()) {
                    *out = T::from_sample(*sample);
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .expect("failed to build output stream");
    stream.play().expect("failed to start audio stream");

    run_control_plane(sched, stream);
}

/// Tick the NRT cleanup off the audio thread for the phrase's lifetime, holding the stream alive.
#[cfg(not(target_arch = "wasm32"))]
fn run_control_plane(mut sched: Sched, _stream: cpal::Stream) {
    use std::time::Duration;
    let total = LEAD_SECS + NUM_BEATS as f64 * BEAT_SECS + 0.5;
    println!(
        "scheduled {NUM_BEATS} beats up front; playing ~{total:.1}s of sample-accurate rhythm..."
    );
    let tick_ms = 50u64;
    for _ in 0..((total * 1000.0 / tick_ms as f64) as u32) {
        sched.tick();
        std::thread::sleep(Duration::from_millis(tick_ms));
    }
}

/// On the web, run the NRT cleanup on a periodic timer and keep it and the stream alive.
#[cfg(target_arch = "wasm32")]
fn run_control_plane(mut sched: Sched, stream: cpal::Stream) {
    let interval = gloo_timers::callback::Interval::new(50, move || sched.tick());
    interval.forget();
    std::mem::forget(stream);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;
    const BLOCK: usize = 64;

    /// A test voice with a clean, sample-exact onset: a constant 0.5 for 5 ms, freed by its
    /// `doneAction` - so the output is exactly `0.0` before the onset and `0.5` at it.
    fn click_def() -> SynthDef {
        SynthDef {
            name: "click".to_string(),
            params: vec![],
            units: vec![
                // Line.ar(0.5, 0.5, 0.005, doneAction: 2): hold 0.5 for 5 ms, then free.
                UnitSpec::new(
                    "Line",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.5),
                        InputRef::Constant(0.5),
                        InputRef::Constant(0.005),
                        InputRef::Constant(2.0),
                    ],
                    1,
                ),
                UnitSpec::new(
                    "OffsetOut",
                    Rate::Audio,
                    vec![
                        InputRef::Constant(0.0),
                        InputRef::Unit { unit: 0, output: 0 },
                    ],
                    0,
                ),
            ],
        }
    }

    /// The engine's nominal OSC-units-per-block increment (matching `Clock::new`).
    fn nominal_increment() -> u64 {
        (BLOCK as f64 * OSC_UNITS_PER_SEC / SR) as u64
    }

    /// The OSC/NTP tag that fires a beat at exactly global sample `s` on the free-running clock: the
    /// clock starts at 0 and advances by the nominal increment per block, so a beat in block
    /// `s / BLOCK` at offset `s % BLOCK` lands at this absolute time. `s` must not be block-aligned
    /// (a zero offset would round into the previous block).
    fn time_for_sample(s: usize) -> u64 {
        let block = (s / BLOCK) as u64;
        let off = (s % BLOCK) as f64;
        block * nominal_increment() + (off * (OSC_UNITS_PER_SEC / SR)).round() as u64
    }

    fn click_bundle(time: u64, id: i32) -> OscPacket {
        OscPacket::Bundle(OscBundle {
            timetag: unpack_ntp(time),
            content: vec![OscPacket::Message(OscMessage {
                addr: "/s_new".to_string(),
                args: vec![
                    OscType::String("click".to_string()),
                    OscType::Int(id),
                    OscType::Int(1),
                    OscType::Int(ROOT_GROUP_ID),
                ],
            })],
        })
    }

    /// Build an engine and schedule a click at each `targets[i]`, submitting them in `order` (an
    /// index permutation). Returns the `World`; the dispatcher's queued commands outlive it in the
    /// ring, so it can be dropped.
    fn scheduled_world(targets: &[usize], order: &[usize]) -> World {
        let (mut controller, _nrt, world) = engine(Options {
            sample_rate: SR,
            output_channels: 1,
            ..Options::default()
        });
        controller.add_synthdef(click_def());
        let mut dispatcher = OscDispatcher::new(controller);
        for &i in order {
            let bundle = click_bundle(time_for_sample(targets[i]), 1000 + i as i32);
            dispatcher.apply(&bundle).expect("schedule a click");
        }
        world
    }

    /// The first sample index at or after `from` where the output departs from exact silence.
    fn first_onset(out: &[f32], from: usize) -> Option<usize> {
        (from..out.len()).find(|&i| out[i] != 0.0)
    }

    /// Assert each beat onsets at its exact target sample, in time order.
    fn assert_onsets(out: &[f32], targets: &[usize]) {
        let mut from = 0;
        for (k, &s) in targets.iter().enumerate() {
            let onset = first_onset(out, from).unwrap_or_else(|| panic!("beat {k} never sounded"));
            assert_eq!(onset, s, "beat {k} should onset at sample {s}, got {onset}");
            // Skip past this beat's tone (the contiguous non-silent run) to find the next onset.
            from = onset;
            while from < out.len() && out[from] != 0.0 {
                from += 1;
            }
        }
    }

    #[test]
    fn beats_onset_at_their_exact_scheduled_sample() {
        // Non-block-aligned targets, spaced wider than a tone (240 samples) so each is isolated.
        let targets = [600usize, 1503, 2305, 3100];
        // Submit out of order on purpose: the engine fires by time tag, not arrival order.
        let order = [2usize, 0, 3, 1];
        let mut out = vec![0.0f32; 4096];
        scheduled_world(&targets, &order).fill(&mut out, 1);
        assert_onsets(&out, &targets);
    }

    #[test]
    fn fill_at_with_a_nominal_clock_keeps_the_same_exact_onsets() {
        // Drive fill_at block by block with perfectly-nominal buffer times: the DLL converges to the
        // nominal rate and onsets stay identical to the free-running fill path.
        let targets = [600usize, 1503, 2305, 3100];
        let order = [0usize, 1, 2, 3];
        let mut world = scheduled_world(&targets, &order);
        let inc = nominal_increment();
        let total = 4096;
        let mut out = vec![0.0f32; total];
        for (n, block) in out.chunks_mut(BLOCK).enumerate() {
            world.fill_at(block, 1, n as u64 * inc);
        }
        assert_onsets(&out, &targets);
    }
}
