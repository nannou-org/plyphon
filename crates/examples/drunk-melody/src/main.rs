//! A self-driving drunk-walk melody from the generator demand sources, via cpal.
//!
//! Like `duty-seq`, the whole sequence runs on the audio thread with no per-note control traffic: a
//! `Duty.kr` clock pulls the next note each beat. But where `duty-seq` walks a fixed `Dseq`/`Dseries`
//! arpeggio, here the melody is a `Dibrown` - an *integer* bounded random walk over MIDI note numbers,
//! so the line meanders by a few semitones each step and never repeats (a "drunk walk"). `midicps`
//! turns the held note into a frequency, `Lag.kr` glides between notes, and a `Dseq` sets the rhythm.
//! Showcases the generator demand sources `Dgeom`/`Diwhite`/`Dbrown`/`Dibrown`.
//!
//! The whole patch is in-engine (no control plane), so only NRT cleanup is ticked off the audio
//! thread. Plays in mono or stereo.

use plyphon::{
    AddAction, InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

/// The `midicps` unary operator's `special_index` (MIDI note number -> frequency in Hz).
const MIDICPS: i16 = 17;
/// The random walk's MIDI note bounds and maximum step (semitones).
const NOTE_LO: f32 = 45.0;
const NOTE_HI: f32 = 72.0;
const NOTE_STEP: f32 = 5.0;
/// Peak amplitude of the voice.
const AMP: f32 = 0.18;
/// Master gain applied in the audio callback.
const GAIN: f32 = 1.0;

/// The drunk-melody synth, built entirely from demand-rate units plus a saw voice.
fn melody_def(channels: usize) -> SynthDef {
    let mut out_inputs = vec![InputRef::Constant(0.0)];
    for _ in 0..channels {
        out_inputs.push(InputRef::Unit { unit: 8, output: 0 });
    }
    SynthDef {
        name: "drunk-melody".to_string(),
        params: vec![],
        units: vec![
            // 0: Dibrown(inf, NOTE_LO, NOTE_HI, NOTE_STEP) - the integer random-walk melody (MIDI notes).
            UnitSpec::new(
                "Dibrown",
                Rate::Demand,
                vec![
                    InputRef::Constant(f32::INFINITY),
                    InputRef::Constant(NOTE_LO),
                    InputRef::Constant(NOTE_HI),
                    InputRef::Constant(NOTE_STEP),
                ],
                1,
            ),
            // 1: Dseq([0.16, 0.16, 0.16, 0.32], inf) - the rhythm (beat durations in seconds).
            UnitSpec::new(
                "Dseq",
                Rate::Demand,
                vec![
                    InputRef::Constant(f32::INFINITY),
                    InputRef::Constant(0.16),
                    InputRef::Constant(0.16),
                    InputRef::Constant(0.16),
                    InputRef::Constant(0.32),
                ],
                1,
            ),
            // 2: Duty.kr(dur: rhythm, 0, level: melody) - pulls the next MIDI note each beat.
            UnitSpec::new(
                "Duty",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            // 3: midicps(note) -> frequency.
            UnitSpec {
                name: "UnaryOpUGen".to_string(),
                rate: Rate::Control,
                inputs: vec![InputRef::Unit { unit: 2, output: 0 }],
                num_outputs: 1,
                special_index: MIDICPS,
            },
            // 4: Lag.kr(freq, 0.04) - a short glide between notes.
            UnitSpec::new(
                "Lag",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Constant(0.04),
                ],
                1,
            ),
            // 5: Saw.ar(freq) - the voice.
            UnitSpec::new(
                "Saw",
                Rate::Audio,
                vec![InputRef::Unit { unit: 4, output: 0 }],
                1,
            ),
            // 6: RLPF(saw, 1400, 0.4) - a little resonance for character.
            UnitSpec::new(
                "RLPF",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 5, output: 0 },
                    InputRef::Constant(1400.0),
                    InputRef::Constant(0.4),
                ],
                1,
            ),
            // 7: scale to the voice amplitude.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 6, output: 0 },
                    InputRef::Constant(AMP),
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 8: a gentle soft-clip so any resonance peak stays polite (UnaryOpUGen softclip = 43).
            UnitSpec {
                name: "UnaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![InputRef::Unit { unit: 7, output: 0 }],
                num_outputs: 1,
                special_index: 43, // softclip
            },
            // 9: Out.ar(0, voice) into every channel.
            UnitSpec::new("Out", Rate::Audio, out_inputs, 0),
        ],
    }
}

/// Build the engine, register the def, and start the single self-driving synth.
fn build(sample_rate: f64, channels: usize) -> (Nrt, World) {
    let channels = channels.max(1);
    let (mut controller, nrt, world) = engine(Options {
        sample_rate,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(melody_def(channels));
    controller
        .synth_new("drunk-melody", ROOT_GROUP_ID, AddAction::Tail, &[])
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
    println!("a self-driving drunk-walk melody (~15s); no per-note control messages...");

    let (stream, mut nrt) = example_audio::play_with(GAIN, |sample_rate, channels| {
        let (nrt, mut world) = build(sample_rate, channels);
        (
            move |out: &mut [f32], channels: usize| world.fill(out, channels),
            nrt,
        )
    });
    example_audio::run_control(stream, 15_000, 50, move || {
        nrt.process();
        while nrt.poll().is_some() {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// The melody should sound, stay bounded, be pitched (a saw voice has many zero crossings), and -
    /// because the pitch is a random walk - not sit on one note: sampling the pitch at several beats
    /// should turn up more than one distinct frequency.
    #[test]
    fn drunk_melody_sounds_and_wanders() {
        let (_nrt, mut world) = build(SR, 1);
        let frames = (SR * 3.0) as usize;
        let mut out = vec![0.0f32; frames];
        world.fill(&mut out, 1);

        assert!(out.iter().all(|s| s.is_finite()), "output must stay finite");
        assert!(
            out.iter().all(|&s| s.abs() < 1.5),
            "output should stay bounded"
        );
        let rms = (out.iter().map(|&s| s * s).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.01, "the melody should be audible, rms {rms}");

        // Zero crossings per second: a saw at ~100-500 Hz crosses zero hundreds of times.
        let crossings = out.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        assert!(
            crossings > 200,
            "should be pitched, got {crossings} crossings"
        );

        // The walk should visit different pitches: compare the dominant zero-cross rate in the first
        // vs the last third of the render - a static note would match, a walk very likely won't.
        let third = frames / 3;
        let cross = |s: &[f32]| s.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let early = cross(&out[..third]);
        let late = cross(&out[2 * third..]);
        assert!(
            early.abs_diff(late) * 20 > early.max(late),
            "the pitch should wander over time (early {early}, late {late})"
        );
    }
}
