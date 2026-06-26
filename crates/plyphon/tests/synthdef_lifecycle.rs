//! Tests for the synth-def lifecycle: redefining or freeing a def retires its compiled `GraphDef`,
//! and the controller reclaims each retired def once the audio thread is done with it - so the
//! retired set stays bounded by live def state instead of leaking for the engine's lifetime.

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;

/// `SinOsc.ar(freq) -> Out.ar(0, sig)`: one audio wire, one control parameter.
fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![Param {
            name: "freq".to_string(),
            default: 440.0,
        }],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
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
        ],
    }
}

/// Drive the engine for `frames` of mono audio so queued commands are applied (a freed synth's
/// `GraphDef` clone, and any replaced/cleared def-table slot, drop on the audio thread here).
fn render(world: &mut World, frames: usize) {
    let mut buf = vec![0.0f32; 64];
    let mut done = 0;
    while done < frames {
        world.fill(&mut buf, 1);
        done += buf.len();
    }
}

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

/// Repeatedly spawn, free, and redefine one name. Each redefinition retires the prior compiled
/// form, but once the audio thread has replaced the def-table slot and freed the synth that used
/// it, reaping reclaims it - so the retired set stays O(1), never growing with the iteration count.
#[test]
fn redefining_a_name_does_not_leak_retired_defs() {
    let (mut controller, mut nrt, mut world) = engine(opts());
    controller.add_synthdef(sine_def());

    for i in 0..32 {
        let node = controller
            .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();
        render(&mut world, 64); // install the def + build the synth
        controller.free(node).unwrap();
        render(&mut world, 64); // free the synth: its def clone drops on the audio thread
        nrt.process();

        // Redefine: retires the current compiled form and reaps any def the audio thread finished
        // with (its slot replaced, its synth freed). `add_synthdef` reaps opportunistically.
        controller.add_synthdef(sine_def());
        assert!(
            controller.retired_defs_len() <= 1,
            "retired defs must stay bounded under churn (iter {i}), got {}",
            controller.retired_defs_len(),
        );
    }

    // Freeing the def clears its resident slot; once that lands, the last retired def reclaims.
    controller.free_def("sine").unwrap();
    render(&mut world, 64);
    controller.reap_retired_defs();
    assert_eq!(
        controller.retired_defs_len(),
        0,
        "every retired def is reclaimed once the audio thread is done with it",
    );
}

/// A live synth pins exactly the def it was built from across arbitrarily many redefinitions; the
/// intermediate defs (used only by since-freed throwaway synths) are reclaimed, and freeing the
/// pinning synth finally releases the def it held.
#[test]
fn a_live_synth_pins_exactly_its_def() {
    let (mut controller, mut nrt, mut world) = engine(opts());
    controller.add_synthdef(sine_def());

    // A long-lived synth on the original def.
    let pinned = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 64);

    // Churn redefinitions, each instantiated and freed via a throwaway synth so the def-table slot
    // is actually replaced - without ever freeing `pinned`.
    for _ in 0..16 {
        controller.add_synthdef(sine_def());
        let tmp = controller
            .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();
        render(&mut world, 64);
        controller.free(tmp).unwrap();
        render(&mut world, 64);
        nrt.process();
        controller.reap_retired_defs();
    }

    assert_eq!(
        controller.retired_defs_len(),
        1,
        "only the def the live synth was built from stays retired",
    );

    // Freeing the pinning synth drops its def clone on the audio thread; the def then reclaims.
    controller.free(pinned).unwrap();
    render(&mut world, 64);
    nrt.process();
    controller.reap_retired_defs();
    assert_eq!(
        controller.retired_defs_len(),
        0,
        "freeing the last user lets the pinned def reclaim",
    );
}
