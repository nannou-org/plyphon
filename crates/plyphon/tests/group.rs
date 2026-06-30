//! Node-tree manipulation: deeply freeing a group frees its whole subtree, `/g_freeAll` empties a
//! group but keeps it, and moving a node relocates it (so it survives its old group being freed).

use plyphon::{
    AddAction, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

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

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + 256);
    let mut buf = vec![0.0f32; 256];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// `SinOsc.ar(freq) -> Out.ar(0)`.
fn sine_def(freq: f32) -> SynthDef {
    SynthDef {
        name: format!("sine{freq}"),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
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

fn drain_ended(nrt: &mut plyphon::Nrt) -> Vec<i32> {
    std::iter::from_fn(|| nrt.poll())
        .filter_map(|e| match e {
            Event::NodeEnded(n) => Some(n.node),
            _ => None,
        })
        .collect()
}

#[test]
fn freeing_a_group_frees_its_subtree() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def(440.0));
    let group = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let a = controller
        .synth_new("sine440", group, AddAction::Tail)
        .unwrap();
    let b = controller
        .synth_new("sine440", group, AddAction::Tail)
        .unwrap();

    let _ = render(&mut world, 512);
    assert!(
        render(&mut world, 256).iter().any(|s| s.abs() > 0.1),
        "group should play"
    );

    controller.free(group).unwrap();
    let _ = render(&mut world, 1024); // apply + flush
    assert!(
        render(&mut world, SR as usize / 8)
            .iter()
            .all(|s| s.abs() < 1e-6),
        "the whole group should be silent after a deep free"
    );
    // Both child synths free on the audio thread (state back to the pool, no trash); their
    // notifications still flow to the NRT side.
    nrt.process();
    let ended = drain_ended(&mut nrt);
    for id in [group, a, b] {
        assert!(ended.contains(&id), "expected NodeEnded for {id}");
    }
}

#[test]
fn free_all_empties_a_group_but_keeps_it() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def(440.0));
    let group = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let a = controller
        .synth_new("sine440", group, AddAction::Tail)
        .unwrap();

    controller.free_all(group).unwrap();
    let _ = render(&mut world, 1024);
    assert!(
        render(&mut world, SR as usize / 8)
            .iter()
            .all(|s| s.abs() < 1e-6),
        "the group's contents should be gone"
    );
    nrt.process();
    let ended = drain_ended(&mut nrt);
    assert!(ended.contains(&a), "the child synth should have ended");
    assert!(
        !ended.contains(&group),
        "the group itself should survive /g_freeAll"
    );

    // The group is still usable: add a fresh synth to it and confirm it plays.
    controller
        .synth_new("sine440", group, AddAction::Tail)
        .unwrap();
    let _ = render(&mut world, 512);
    let out = render(&mut world, SR as usize / 8);
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "the kept group should still accept and play synths"
    );
}

#[test]
fn moving_a_synth_out_of_a_group_lets_it_survive() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(sine_def(440.0));
    let group = controller
        .new_group(ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let synth = controller
        .synth_new("sine440", group, AddAction::Tail)
        .unwrap();

    // Move the synth out to the root group, then free the (now empty) group.
    controller
        .move_node(synth, ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller.free(group).unwrap();
    let _ = render(&mut world, 1024);

    let out = render(&mut world, SR as usize / 4);
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "the moved synth should survive its old group being freed"
    );
    nrt.process();
    let ended = drain_ended(&mut nrt);
    assert!(ended.contains(&group), "the freed group should have ended");
    assert!(
        !ended.contains(&synth),
        "the moved-out synth should not have been freed"
    );
}
