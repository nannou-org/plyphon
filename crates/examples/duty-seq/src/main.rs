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

use plyphon::{AddAction, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, World, engine};
use plyphon_synthdef::{DSeq, Out, SinOsc, SynthDefBuilder, mce};

/// Peak amplitude of the oscillator.
const AMP: f32 = 0.2;
/// Master gain applied in the audio callback (the voice is already scaled by `AMP`).
const GAIN: f32 = 1.0;

plyphon_synthdef::ugen!(
    /// Demand rate arithmetic series.
    "Dseries" => DSeries [new: Rate::Demand](
        /// Start value (default 1).
        length = 1,
        /// Step value (default 1).
        start = 1,
        /// Number of values to create (default infinity).
        step = f32::INFINITY,
    ) -> 1
);

plyphon_synthdef::ugen!(
    /// DWhite returns numbers in the continuous range between lo and hi. Returns integer values.
    "Dwhite" => DWhite [new: Rate::Demand](
        /// Number of values to create (default infinity).
        length = f32::INFINITY,
        /// Minimum value (default 0).
        lo = 0,
        /// Maximum value (default 1).
        hi = 1,
    ) -> 1
);

plyphon_synthdef::ugen!(
    "Duty" => Duty [ar: Rate::Audio, kr: Rate::Control](
        /// Time values. Can be a demand UGen or any signal.
        /// The next level is acquired after duration.
        dur = 1.0,
        /// Trigger or reset time values. Resets the list of UGens
        /// and the duration UGen when triggered. The reset input
        /// may also be a demand UGen, providing a stream of reset times.
        reset = 0.0,
        /// A doneAction that is evaluated when the duration stream ends.
        done_action = 0,
        /// Demand UGen providing the output values.
        level = 0.0,
    ) -> 1
);

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
    SynthDefBuilder::build_with("duty-seq", |g| {
        // Pulls the next note when each beat elapses. This is the only clock;
        // there is no control-plane tick.
        let freq = Duty::kr(g)
            // The rhythm (beat durations in seconds)
            .dur(DSeq::new(g).list([0.15, 0.15, 0.3]).repeats(f32::INFINITY))
            // The melody, nesting several sources
            .level(
                DSeq::new(g)
                    .list(mce![
                        // A four-note rising arpeggio: 220 275 330 385
                        DSeries::new(g).length(4).start(220).step(55),
                        // Two random notes per pass
                        DWhite::new(g).length(2).lo(300).hi(500),
                        440,
                        330
                    ])
                    .repeats(f32::INFINITY),
            );

        let osc = SinOsc::ar(g).freq(freq) * AMP;

        // Send to all output channels
        Out::ar(g).channels(osc.repeat(channels)).emit();
    })
    .0
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

    // cpal's AudioWorklet backend re-instantiates this module on the audio thread, re-running
    // `main` there; only set up audio on the main browser thread.
    if example_audio::on_worklet_thread() {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    println!("playing a self-driving demand-rate sequence (~12s); no per-note control messages...");

    // The synth sequences itself on the audio thread, so the control plane has nothing to schedule -
    // it just ticks NRT cleanup (dropping any boxes the audio thread has freed) off the audio thread.
    let (stream, mut nrt) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (nrt, mut world) = build(sample_rate, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            nrt,
        )
    });
    example_audio::run_control(stream, 12_000, 50, move || {
        nrt.process();
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
