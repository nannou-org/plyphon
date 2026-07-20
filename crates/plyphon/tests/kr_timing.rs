//! Control-rate timing semantics: the kr timing family steps once per control period, so its
//! seconds-to-steps conversions must use the *control* rate (scsynth's `unit->mRate`). These
//! regressions pin the wall-clock behaviour of `Trig1.kr`, `Timer.kr`, `Dust.kr` and
//! `DetectSilence.kr` - each was ~`block_size`x off when converted at the audio rate.

use plyphon::{
    AddAction, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 1,
        ..Options::default()
    }
}

/// Render `frames` mono samples in whole control blocks.
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames.next_multiple_of(BLOCK)];
    for chunk in out.chunks_mut(BLOCK) {
        world.fill(chunk, 1);
    }
    out.truncate(frames);
    out
}

/// A def whose kr unit `name` (inputs: an `Impulse.ar(freq)` trigger then `extra` constants) is
/// lifted to the output via `K2A`.
fn kr_via_k2a(def_name: &str, name: &str, freq: f32, extra: &[f32]) -> SynthDef {
    let mut inputs = vec![InputRef::Unit { unit: 0, output: 0 }];
    inputs.extend(extra.iter().map(|&v| InputRef::Constant(v)));
    SynthDef {
        name: def_name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(name, Rate::Control, inputs, 1),
            UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 1, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn trig1_kr_holds_dur_seconds() {
    let (mut controller, _nrt, mut world) = engine(opts());
    // One trigger at t=0 (0.1 Hz never refires within the render), held for 0.1 s.
    controller.add_synthdef(kr_via_k2a("t", "Trig1", 0.1, &[0.1]));
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let out = render(&mut world, (SR * 0.3) as usize);
    let held = out.iter().filter(|s| **s > 0.5).count();
    let expected = (SR * 0.1) as usize;
    assert!(
        held.abs_diff(expected) <= 2 * BLOCK,
        "Trig1.kr(dur: 0.1) should hold ~{expected} samples, held {held}"
    );
}

#[test]
fn dust_kr_fires_at_density_per_second() {
    let (mut controller, _nrt, mut world) = engine(opts());
    // Dust.kr(100): ~100 events/second on average. The audio-rate bug fired ~100/BLOCK.
    controller.add_synthdef(SynthDef {
        name: "d".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("Dust", Rate::Control, vec![InputRef::Constant(100.0)], 1),
            UnitSpec::new(
                "K2A",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    // Count quiet->loud transitions (K2A's ramp smears each event across two blocks, so raw
    // nonzero-block counting would double-count; back-to-back events merge, slightly
    // undercounting instead).
    let out = render(&mut world, SR as usize * 2);
    let mut events = 0;
    let mut prev_quiet = true;
    for b in out.chunks(BLOCK) {
        let quiet = b.iter().all(|s| *s < 1e-6);
        if prev_quiet && !quiet {
            events += 1;
        }
        prev_quiet = quiet;
    }
    // ~200 expected over 2 s; generous statistical bounds (the engine is deterministically
    // seeded, so this is stable run to run). The audio-rate bug fired only ~4 times in 2 s.
    assert!(
        (120..=280).contains(&events),
        "Dust.kr(100) over 2 s should fire ~200 times, fired {events}"
    );
}

/// `unit.kr(inputs) -> DC.ar -> Out.ar(0)`: lifts a kr signal to audio with a zero-order hold (no
/// K2A ramp), so each block's samples all equal the kr value.
fn kr_dc_def(def_name: &str, name: &str, inputs: Vec<InputRef>) -> SynthDef {
    SynthDef {
        name: def_name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(name, Rate::Control, inputs, 1),
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn lfsaw_kr_matches_ar_at_block_starts() {
    // The decimation-equivalence invariant: a kr oscillator stepping once per control period must
    // trace the same waveform its ar twin passes through at each block start.
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 2,
        ..opts()
    });
    controller.add_synthdef(SynthDef {
        name: "pair".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "LFSaw",
                Rate::Audio,
                vec![InputRef::Constant(3.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
            UnitSpec::new(
                "LFSaw",
                Rate::Control,
                vec![InputRef::Constant(3.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 2, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(1.0),
                    InputRef::Unit { unit: 3, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("pair", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let frames = (SR * 0.2) as usize;
    let mut out = vec![0.0f32; frames * 2];
    for chunk in out.chunks_mut(BLOCK * 2) {
        world.fill(chunk, 2);
    }
    for block in 0..frames / BLOCK {
        let ar = out[block * BLOCK * 2]; // channel 0, block's first sample
        let kr = out[block * BLOCK * 2 + 1]; // channel 1, ZOH'd kr value
        assert!(
            (ar - kr).abs() < 1e-3,
            "block {block}: LFSaw.ar at block start ({ar}) != LFSaw.kr ({kr})"
        );
    }
}

#[test]
fn impulse_kr_emits_one_impulse_per_period() {
    let (mut controller, _nrt, mut world) = engine(opts());
    // Impulse.kr(10) over 2 s: exactly one 1.0-valued control period per cycle, ~20 total.
    // (Before the 1-sample kr calc, the impulse landed mid-block and was discarded.)
    controller.add_synthdef(kr_dc_def(
        "imp",
        "Impulse",
        vec![InputRef::Constant(10.0), InputRef::Constant(0.0)],
    ));
    controller
        .synth_new("imp", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let out = render(&mut world, SR as usize * 2);
    let ones = out
        .chunks(BLOCK)
        .filter(|b| (b[0] - 1.0).abs() < 1e-6)
        .count();
    assert!(
        (19..=21).contains(&ones),
        "Impulse.kr(10) over 2 s should emit ~20 impulses, emitted {ones}"
    );
}

#[test]
fn lag_kr_settles_in_lag_time() {
    let (mut controller, _nrt, mut world) = engine(opts());
    // Lag.kr(param, 0.1): seeded at the param's initial 0, then stepped to 1 - the -60 dB
    // convention puts it at ~0.968 after 0.05 s and ~0.999 after 0.15 s.
    controller.add_synthdef(SynthDef {
        name: "lag".to_string(),
        params: vec![plyphon::Param::control("x", 0.0)],
        units: vec![
            UnitSpec::new(
                "Lag",
                Rate::Control,
                vec![InputRef::Param(0), InputRef::Constant(0.1)],
                1,
            ),
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    });
    let node = controller
        .synth_new("lag", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let _ = render(&mut world, BLOCK); // seed the lag at the initial 0
    controller.set_control(node, 0, 1.0).unwrap();
    let out = render(&mut world, (SR * 0.15) as usize);
    let at_50ms = out[(SR * 0.05) as usize];
    let at_150ms = *out.last().unwrap();
    assert!(
        (0.93..=0.99).contains(&at_50ms),
        "Lag.kr(0.1) should sit near 0.968 at 50 ms, got {at_50ms}"
    );
    assert!(
        at_150ms > 0.995,
        "Lag.kr(0.1) should have settled by 150 ms, got {at_150ms}"
    );
}

#[test]
fn timer_kr_measures_seconds_between_triggers() {
    let (mut controller, _nrt, mut world) = engine(opts());
    // Timer.kr clocked by Impulse.kr(10): reports the 0.1 s period (in whole control periods).
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Impulse",
                Rate::Control,
                vec![InputRef::Constant(10.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Timer",
                Rate::Control,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 1, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let out = render(&mut world, (SR * 0.5) as usize);
    let last = *out.last().unwrap();
    assert!(
        (last - 0.1).abs() < 3e-3,
        "Timer.kr should report the 0.1 s trigger period, got {last}"
    );
}

#[test]
fn amplitude_kr_consumes_the_whole_audio_block() {
    let (mut controller, _nrt, mut world) = engine(opts());
    // Amplitude.kr over a full-scale 440 Hz sine: the follower must consume every audio sample
    // (scsynth's atok), not just each block's first - which for a sine near a zero crossing
    // would leave the follower far below the peak.
    controller.add_synthdef(SynthDef {
        name: "amp".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Amplitude",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.01),
                    InputRef::Constant(0.01),
                ],
                1,
            ),
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 1, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 2, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("amp", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let out = render(&mut world, (SR * 0.5) as usize);
    let last = *out.last().unwrap();
    assert!(
        (0.6..=1.01).contains(&last),
        "Amplitude.kr should track the sine's envelope near full scale, got {last}"
    );
}

#[test]
fn detect_silence_kr_frees_after_time_seconds() {
    let (mut controller, mut nrt, mut world) = engine(opts());
    // Sound for ~0.02 s, then silence (DetectSilence waits for signal before it counts): with
    // time=0.05, doneAction 2 should free the synth by ~0.07 s.
    controller.add_synthdef(SynthDef {
        name: "ds".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Line",
                Rate::Audio,
                vec![
                    InputRef::Constant(1.0),  // start
                    InputRef::Constant(0.0),  // end
                    InputRef::Constant(0.02), // dur
                    InputRef::Constant(0.0),  // doneAction
                ],
                1,
            ),
            UnitSpec::new(
                "DetectSilence",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.001), // amp threshold
                    InputRef::Constant(0.05),  // time
                    InputRef::Constant(2.0),   // doneAction: free self
                ],
                1,
            ),
        ],
    });
    let node = controller
        .synth_new("ds", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    // 0.12 s comfortably covers sound + silence window; the audio-rate bug needed ~3.2 s.
    let _ = render(&mut world, (SR * 0.12) as usize);
    nrt.process();
    assert!(
        std::iter::from_fn(|| nrt.poll())
            .any(|e| matches!(e, Event::NodeEnded(n) if n.node == node)),
        "DetectSilence.kr(time: 0.05) should have freed the synth within 0.12 s"
    );
}
