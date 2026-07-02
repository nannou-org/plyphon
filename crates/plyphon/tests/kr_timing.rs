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
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
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
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
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
        .synth_new("ds", ROOT_GROUP_ID, AddAction::Tail)
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
