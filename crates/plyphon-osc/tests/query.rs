//! Drive the getter commands in-process: issue a query over OSC, render a block so the engine
//! answers over the reply ring, drain the answers into the dispatcher, and assert the exact OSC
//! reply layout. Getter replies are asynchronous, so each test renders before draining.

use std::cell::RefCell;
use std::rc::Rc;

use plyphon::{
    InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};
use plyphon_osc::OscDispatcher;
use rosc::{OscMessage, OscPacket, OscType};

const SR: f64 = 48_000.0;

/// `SinOsc.ar(freq) -> Out.ar(0)`, with one named control `freq` (default 440).
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

fn osc(addr: &str, args: Vec<OscType>) -> Vec<u8> {
    rosc::encoder::encode(&OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    }))
    .expect("encode OSC")
}

fn engine_1ch() -> (OscDispatcher, Nrt, World) {
    let (controller, nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    (OscDispatcher::new(controller), nrt, world)
}

/// Render a couple of control blocks, so any queued query is processed and its reply pushed.
fn render(world: &mut World, blocks: usize) {
    let mut buf = vec![0.0f32; 64];
    for _ in 0..blocks {
        world.fill(&mut buf, 1);
    }
}

/// Forward the engine's queued query answers into the dispatcher, then take the resulting OSC
/// messages (the way a host loop would after `poll_reply`).
fn drain_replies(dispatcher: &mut OscDispatcher, nrt: &mut Nrt) -> Vec<OscMessage> {
    nrt.process();
    while let Some(reply) = nrt.poll_reply() {
        dispatcher.reply(reply);
    }
    dispatcher
        .take_replies()
        .into_iter()
        .filter_map(|packet| match packet {
            OscPacket::Message(message) => Some(message),
            OscPacket::Bundle(_) => None,
        })
        .collect()
}

/// Apply one OSC message, render, and drain the replies.
fn query(
    dispatcher: &mut OscDispatcher,
    nrt: &mut Nrt,
    world: &mut World,
    bytes: &[u8],
) -> Vec<OscMessage> {
    dispatcher.apply_bytes(bytes).expect("apply getter");
    render(world, 2);
    drain_replies(dispatcher, nrt)
}

fn find<'a>(msgs: &'a [OscMessage], addr: &str) -> Option<&'a OscMessage> {
    msgs.iter().find(|m| m.addr == addr)
}

#[test]
fn sync_replies_with_synced() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/sync", vec![OscType::Int(42)]),
    );
    let synced = find(&replies, "/synced").expect("/synced");
    assert_eq!(synced.args, vec![OscType::Int(42)]);
}

#[test]
fn status_reports_counts() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.controller().add_synthdef(sine_def());
    d.apply_bytes(&osc(
        "/g_new",
        vec![
            OscType::Int(2000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/g_new");
    for id in [1000, 1001] {
        d.apply_bytes(&osc(
            "/s_new",
            vec![
                OscType::String("sine".to_string()),
                OscType::Int(id),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ))
        .expect("/s_new");
    }
    let replies = query(&mut d, &mut nrt, &mut world, &osc("/status", vec![]));
    let status = find(&replies, "/status.reply").expect("/status.reply");
    // 1 reserved, ugens (2 synths * 2 units), synths, groups (root + new), synthdefs, 2x cpu, 2x sr.
    assert_eq!(
        status.args,
        vec![
            OscType::Int(1),
            OscType::Int(4),
            OscType::Int(2),
            OscType::Int(2),
            OscType::Int(1),
            OscType::Float(0.0),
            OscType::Float(0.0),
            OscType::Double(SR),
            OscType::Double(SR),
        ]
    );
}

#[test]
fn rt_memory_status_is_consistent() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/rtMemoryStatus", vec![]),
    );
    let mem = find(&replies, "/rtMemoryStatus.reply").expect("/rtMemoryStatus.reply");
    assert_eq!(mem.args.len(), 2);
    let (total, largest) = match (&mem.args[0], &mem.args[1]) {
        (OscType::Int(t), OscType::Int(l)) => (*t, *l),
        _ => panic!("expected two ints, got {:?}", mem.args),
    };
    assert!(total > 0 && largest > 0 && largest <= total);
}

#[test]
fn n_query_describes_node_and_group() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.controller().add_synthdef(sine_def());
    d.apply_bytes(&osc(
        "/g_new",
        vec![
            OscType::Int(2000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/g_new");
    d.apply_bytes(&osc(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(2000),
        ],
    ))
    .expect("/s_new");

    // The synth: parent 2000, no siblings, isGroup 0.
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/n_query", vec![OscType::Int(1000)]),
    );
    let info = find(&replies, "/n_info").expect("/n_info");
    assert_eq!(
        info.args,
        vec![
            OscType::Int(1000),
            OscType::Int(2000),
            OscType::Int(-1),
            OscType::Int(-1),
            OscType::Int(0),
        ]
    );

    // The group: parent root, head==tail==1000, isGroup 1.
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/n_query", vec![OscType::Int(2000)]),
    );
    let info = find(&replies, "/n_info").expect("/n_info group");
    assert_eq!(
        info.args,
        vec![
            OscType::Int(2000),
            OscType::Int(ROOT_GROUP_ID),
            OscType::Int(-1),
            OscType::Int(-1),
            OscType::Int(1),
            OscType::Int(1000),
            OscType::Int(1000),
        ]
    );

    // An unknown id: the not-found shape.
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/n_query", vec![OscType::Int(7777)]),
    );
    let info = find(&replies, "/n_info").expect("/n_info missing");
    assert_eq!(
        info.args,
        vec![
            OscType::Int(7777),
            OscType::Int(-1),
            OscType::Int(-1),
            OscType::Int(-1),
            OscType::Int(-1),
        ]
    );
}

#[test]
fn c_get_and_getn_read_buses() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.apply_bytes(&osc(
        "/c_setn",
        vec![
            OscType::Int(4),
            OscType::Int(3),
            OscType::Float(0.25),
            OscType::Float(0.5),
            OscType::Float(0.75),
        ],
    ))
    .expect("/c_setn");

    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/c_get", vec![OscType::Int(4), OscType::Int(6)]),
    );
    let set = find(&replies, "/c_set").expect("/c_set");
    assert_eq!(
        set.args,
        vec![
            OscType::Int(4),
            OscType::Float(0.25),
            OscType::Int(6),
            OscType::Float(0.75),
        ]
    );

    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/c_getn", vec![OscType::Int(4), OscType::Int(3)]),
    );
    let setn = find(&replies, "/c_setn").expect("/c_setn");
    assert_eq!(
        setn.args,
        vec![
            OscType::Int(4),
            OscType::Int(3),
            OscType::Float(0.25),
            OscType::Float(0.5),
            OscType::Float(0.75),
        ]
    );
}

#[test]
fn s_get_echoes_control_token() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.controller().add_synthdef(sine_def());
    d.apply_bytes(&osc(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/s_new");
    d.apply_bytes(&osc(
        "/n_set",
        vec![
            OscType::Int(1000),
            OscType::String("freq".to_string()),
            OscType::Float(330.0),
        ],
    ))
    .expect("/n_set");

    // By name: the reply echoes the string token.
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc(
            "/s_get",
            vec![OscType::Int(1000), OscType::String("freq".to_string())],
        ),
    );
    let set = find(&replies, "/n_set").expect("/n_set by name");
    assert_eq!(
        set.args,
        vec![
            OscType::Int(1000),
            OscType::String("freq".to_string()),
            OscType::Float(330.0),
        ]
    );

    // By index: the reply echoes the int token.
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/s_get", vec![OscType::Int(1000), OscType::Int(0)]),
    );
    let set = find(&replies, "/n_set").expect("/n_set by index");
    assert_eq!(
        set.args,
        vec![OscType::Int(1000), OscType::Int(0), OscType::Float(330.0)]
    );

    // Unknown node -> /fail.
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/s_get", vec![OscType::Int(9999), OscType::Int(0)]),
    );
    assert!(
        find(&replies, "/fail").is_some(),
        "expected /fail, got {replies:?}"
    );
}

#[test]
fn b_get_and_getn_read_samples() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.apply_bytes(&osc(
        "/b_alloc",
        vec![OscType::Int(0), OscType::Int(8), OscType::Int(1)],
    ))
    .expect("/b_alloc");
    d.apply_bytes(&osc(
        "/b_setn",
        vec![
            OscType::Int(0),
            OscType::Int(0),
            OscType::Int(3),
            OscType::Float(0.1),
            OscType::Float(0.2),
            OscType::Float(0.3),
        ],
    ))
    .expect("/b_setn");

    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc(
            "/b_get",
            vec![OscType::Int(0), OscType::Int(1), OscType::Int(2)],
        ),
    );
    let set = find(&replies, "/b_set").expect("/b_set");
    assert_eq!(
        set.args,
        vec![
            OscType::Int(0),
            OscType::Int(1),
            OscType::Float(0.2),
            OscType::Int(2),
            OscType::Float(0.3),
        ]
    );

    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc(
            "/b_getn",
            vec![OscType::Int(0), OscType::Int(0), OscType::Int(3)],
        ),
    );
    let setn = find(&replies, "/b_setn").expect("/b_setn");
    assert_eq!(
        setn.args,
        vec![
            OscType::Int(0),
            OscType::Int(0),
            OscType::Int(3),
            OscType::Float(0.1),
            OscType::Float(0.2),
            OscType::Float(0.3),
        ]
    );
}

#[test]
fn g_query_tree_streams_the_subtree() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.controller().add_synthdef(sine_def());
    d.apply_bytes(&osc(
        "/g_new",
        vec![
            OscType::Int(2000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/g_new");
    d.apply_bytes(&osc(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(2000),
        ],
    ))
    .expect("/s_new");

    // flag 0: structure only (root -> group 2000 -> synth 1000).
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc(
            "/g_queryTree",
            vec![OscType::Int(ROOT_GROUP_ID), OscType::Int(0)],
        ),
    );
    let tree = find(&replies, "/g_queryTree.reply").expect("/g_queryTree.reply");
    assert_eq!(
        tree.args,
        vec![
            OscType::Int(0),                     // flag
            OscType::Int(ROOT_GROUP_ID),         // root
            OscType::Int(1),                     // root has 1 child
            OscType::Int(2000),                  // group
            OscType::Int(1),                     // group has 1 child
            OscType::Int(1000),                  // synth
            OscType::Int(-1),                    // synth marker
            OscType::String("sine".to_string()), // synth def name
        ]
    );

    // flag 1: include the synth's control (freq = 440 default).
    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc("/g_queryTree", vec![OscType::Int(2000), OscType::Int(1)]),
    );
    let tree = find(&replies, "/g_queryTree.reply").expect("/g_queryTree.reply flag1");
    assert_eq!(
        tree.args,
        vec![
            OscType::Int(1),
            OscType::Int(2000),
            OscType::Int(1),
            OscType::Int(1000),
            OscType::Int(-1),
            OscType::String("sine".to_string()),
            OscType::Int(1), // 1 control
            OscType::String("freq".to_string()),
            OscType::Float(440.0),
        ]
    );
}

#[test]
fn g_dump_tree_feeds_the_sink_not_osc() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.controller().add_synthdef(sine_def());
    d.apply_bytes(&osc(
        "/s_new",
        vec![
            OscType::String("sine".to_string()),
            OscType::Int(1000),
            OscType::Int(1),
            OscType::Int(ROOT_GROUP_ID),
        ],
    ))
    .expect("/s_new");

    let captured = Rc::new(RefCell::new(String::new()));
    let sink = captured.clone();
    d.set_dump_sink(Box::new(move |text: &str| sink.borrow_mut().push_str(text)));

    let replies = query(
        &mut d,
        &mut nrt,
        &mut world,
        &osc(
            "/g_dumpTree",
            vec![OscType::Int(ROOT_GROUP_ID), OscType::Int(0)],
        ),
    );
    assert!(
        replies.is_empty(),
        "/g_dumpTree should queue no OSC reply, got {replies:?}"
    );
    let text = captured.borrow();
    assert!(
        text.contains("synth sine"),
        "dump text should name the synth, got {text:?}"
    );
    assert!(
        text.contains("1000"),
        "dump text should list the synth id, got {text:?}"
    );
}

#[test]
fn replies_keep_fifo_order() {
    let (mut d, mut nrt, mut world) = engine_1ch();
    d.apply_bytes(&osc("/c_set", vec![OscType::Int(0), OscType::Float(1.5)]))
        .expect("/c_set");

    // Three getters in one burst: /c_get, /sync, /c_get. Their replies must come back in order.
    d.apply_bytes(&osc("/c_get", vec![OscType::Int(0)]))
        .expect("/c_get 1");
    d.apply_bytes(&osc("/sync", vec![OscType::Int(7)]))
        .expect("/sync");
    d.apply_bytes(&osc("/c_get", vec![OscType::Int(0)]))
        .expect("/c_get 2");
    render(&mut world, 2);
    let replies = drain_replies(&mut d, &mut nrt);
    let addrs: Vec<&str> = replies.iter().map(|m| m.addr.as_str()).collect();
    assert_eq!(
        addrs,
        vec!["/c_set", "/synced", "/c_set"],
        "FIFO order, got {addrs:?}"
    );
}
