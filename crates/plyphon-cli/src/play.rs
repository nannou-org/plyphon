//! `plyphon play` - real-time playback of a time-tagged OSC score to the audio device.
//!
//! The `World` plays on the audio thread; this thread feeds the score's time-tagged bundles to an
//! [`OscDispatcher`] as each bundle's onset approaches, a little ahead of time so the engine's
//! sample-accurate scheduler places it exactly. It is `render`'s real-time sibling - a scripted local
//! client instead of a socket.

use std::time::{Duration, Instant};

use plyphon::{Nrt, engine};
use plyphon_osc::{OscDispatcher, parse_score};

use crate::audio;
use crate::cli::PlayArgs;
use crate::defs::load_dir;
use crate::options::engine_options;

/// OSC/NTP fixed-point units in one second (OSC time is 32.32 fixed point).
const OSC_UNITS_PER_SEC: f64 = 4_294_967_296.0;
/// Schedule each bundle this far ahead of its onset so the engine can place it sample-accurately.
const LEAD: Duration = Duration::from_millis(50);
/// How often to service NRT cleanup while idling between score events.
const TICK: Duration = Duration::from_millis(10);

pub fn run(args: PlayArgs) -> Result<(), String> {
    let score_bytes = std::fs::read(&args.score).map_err(|e| format!("reading score: {e}"))?;
    let (score, max_time) = parse_score(&score_bytes).map_err(|e| e.to_string())?;

    let audio = audio::resolve(&args.audio)?;
    let options = engine_options(&args.engine, audio.sample_rate, audio.channels, 0);
    let (mut controller, mut nrt, world) = engine(options);
    if let Some(dir) = &args.engine.load_dir {
        load_dir(&mut controller, dir)?;
    }
    let mut dispatcher = OscDispatcher::new(controller);

    // The World plays on the audio thread on its free-running clock; that is all scripted playback
    // needs (no wall-clock resync). Keep the stream alive for the run.
    let mut world = world;
    let _stream = audio::play_output(&audio, move |output, channels| world.fill(output, channels))?;

    let started = Instant::now();
    for entry in &score {
        let onset = Duration::from_secs_f64(entry.osc_time as f64 / OSC_UNITS_PER_SEC);
        wait_until(onset.saturating_sub(LEAD), started, &mut nrt);
        dispatcher.apply(&entry.packet).map_err(|e| e.to_string())?;
    }

    // Let the tail ring out, still servicing NRT cleanup.
    let end = Duration::from_secs_f64(max_time as f64 / OSC_UNITS_PER_SEC + args.tail);
    wait_until(end, started, &mut nrt);

    println!(
        "played {} commands (~{:.2}s)",
        score.len(),
        end.as_secs_f64()
    );
    Ok(())
}

/// Sleep in short ticks until `target` has elapsed since `started`, running NRT cleanup each tick.
fn wait_until(target: Duration, started: Instant, nrt: &mut Nrt) {
    while started.elapsed() < target {
        service_nrt(nrt);
        let remaining = target - started.elapsed();
        std::thread::sleep(remaining.min(TICK));
    }
    service_nrt(nrt);
}

/// Drop synths the audio thread has finished with and drain (ignore) node notifications.
fn service_nrt(nrt: &mut Nrt) {
    nrt.process();
    while nrt.poll().is_some() {}
}
