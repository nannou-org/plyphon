//! Node-tree commands over OSC: `/g_new` + `/s_new` build a group, `/g_freeAll` empties it, and
//! `/g_tail` moves a synth between groups (so it survives its old group being freed).

use plyphon::{Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine};
use plyphon_osc::OscDispatcher;
use rosc::{OscMessage, OscPacket, OscType};

const SR: f32 = 48_000.0;

fn msg(addr: &str, args: Vec<OscType>) -> Vec<u8> {
    rosc::encoder::encode(&OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    }))
    .expect("encode OSC")
}

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

fn drain_ended(nrt: &mut plyphon::Nrt) -> Vec<i32> {
    std::iter::from_fn(|| nrt.poll())
        .filter_map(|e| match e {
            Event::NodeEnded { id } => Some(id),
            _ => None,
        })
        .collect()
}

fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
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

fn dispatcher() -> (OscDispatcher, plyphon::Nrt, World) {
    let (controller, nrt, world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new(controller);
    osc.controller().add_synthdef(sine_def());
    (osc, nrt, world)
}

#[test]
fn g_free_all_over_osc() {
    let (mut osc, mut nrt, mut world) = dispatcher();
    // /g_new id 1 addToTail target root; /s_new "sine" 1000 addToTail target group 1.
    osc.apply_bytes(&msg(
        "/g_new",
        vec![
            OscType::Int(1),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .unwrap();
    osc.apply_bytes(&msg(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(1),
        ],
    ))
    .unwrap();
    let _ = render(&mut world, 512);
    assert!(
        render(&mut world, 256).iter().any(|s| s.abs() > 0.1),
        "group should play"
    );

    osc.apply_bytes(&msg("/g_freeAll", vec![OscType::Int(1)]))
        .unwrap();
    let _ = render(&mut world, 1024);
    assert!(
        render(&mut world, SR as usize / 8)
            .iter()
            .all(|s| s.abs() < 1e-6),
        "the group's synth should be gone after /g_freeAll"
    );
    nrt.process();
    let ended = drain_ended(&mut nrt);
    assert!(ended.contains(&1000), "the child synth should have ended");
    assert!(!ended.contains(&1), "the group should survive /g_freeAll");
}

#[test]
fn g_tail_moves_a_synth_between_groups_over_osc() {
    let (mut osc, mut nrt, mut world) = dispatcher();
    // Two groups under the root; a synth in group 1.
    osc.apply_bytes(&msg(
        "/g_new",
        vec![
            OscType::Int(1),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
            OscType::Int(2),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .unwrap();
    osc.apply_bytes(&msg(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(1),
        ],
    ))
    .unwrap();

    // /g_tail group=2 node=1000: move the synth to group 2's tail, then free group 1.
    osc.apply_bytes(&msg("/g_tail", vec![OscType::Int(2), OscType::Int(1000)]))
        .unwrap();
    osc.apply_bytes(&msg("/n_free", vec![OscType::Int(1)]))
        .unwrap();
    let _ = render(&mut world, 1024);

    let out = render(&mut world, SR as usize / 4);
    assert!(
        goertzel(&out, 440.0) > 5.0 * goertzel(&out, 880.0),
        "the moved synth should survive group 1 being freed"
    );
    nrt.process();
    let ended = drain_ended(&mut nrt);
    assert!(ended.contains(&1), "group 1 should have ended");
    assert!(
        !ended.contains(&1000),
        "the moved synth should not have been freed"
    );
}

/// Forward the engine's node notifications into the dispatcher, then take the resulting OSC messages
/// (the way a host loop would after `poll`).
fn drain_notifications(osc: &mut OscDispatcher, nrt: &mut plyphon::Nrt) -> Vec<OscMessage> {
    nrt.process();
    while let Some(event) = nrt.poll() {
        osc.notify(event);
    }
    osc.take_replies()
        .into_iter()
        .filter_map(|p| match p {
            OscPacket::Message(m) => Some(m),
            OscPacket::Bundle(_) => None,
        })
        .collect()
}

#[test]
fn n_after_emits_n_move_over_osc() {
    let (mut osc, mut nrt, mut world) = dispatcher();
    // root -> [1000, 1001].
    for id in [1000, 1001] {
        osc.apply_bytes(&msg(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(id),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .unwrap();
    }
    // /n_after node=1000 target=1001: move 1000 to just after 1001 -> root -> [1001, 1000].
    osc.apply_bytes(&msg(
        "/n_after",
        vec![OscType::Int(1000), OscType::Int(1001)],
    ))
    .unwrap();
    let _ = render(&mut world, 256);

    let msgs = drain_notifications(&mut osc, &mut nrt);
    let mv = msgs
        .iter()
        .find(|m| m.addr == "/n_move")
        .expect("/n_move emitted");
    // node, parent, prev, next, isGroup: 1000 is now the tail, after 1001.
    assert_eq!(
        mv.args,
        vec![
            OscType::Int(1000),
            OscType::Int(ROOT_GROUP_ID),
            OscType::Int(1001),
            OscType::Int(-1),
            OscType::Int(0),
        ]
    );
}

#[test]
fn moving_a_group_emits_n_move_with_head_tail() {
    let (mut osc, mut nrt, mut world) = dispatcher();
    // Two empty groups under the root -> [1, 2].
    osc.apply_bytes(&msg(
        "/g_new",
        vec![
            OscType::Int(1),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
            OscType::Int(2),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .unwrap();
    // /n_after node=1 target=2: move group 1 to just after group 2 -> root -> [2, 1].
    osc.apply_bytes(&msg("/n_after", vec![OscType::Int(1), OscType::Int(2)]))
        .unwrap();
    let _ = render(&mut world, 256);

    let msgs = drain_notifications(&mut osc, &mut nrt);
    let mv = msgs
        .iter()
        .find(|m| m.addr == "/n_move")
        .expect("/n_move emitted");
    // A group also reports head/tail (both -1 - the group is empty).
    assert_eq!(
        mv.args,
        vec![
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
            OscType::Int(2),
            OscType::Int(-1),
            OscType::Int(1),
            OscType::Int(-1),
            OscType::Int(-1),
        ]
    );
}
