//! `/n_trace` dumps a synth's per-unit inputs and outputs (first sample each) for one block to a host
//! text sink, resolving each calc unit's index to its UGen name via the node's def. No OSC reply.

use std::cell::RefCell;
use std::rc::Rc;

use plyphon::{InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};
use plyphon_osc::{OscDispatcher, ReplyTarget};
use rosc::{OscMessage, OscPacket, OscType};

const SR: f64 = 48_000.0;

fn msg(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    })
}

/// `DC.ar(0.5) -> Out.ar(0)`: a two-unit synth with predictable first-sample I/O.
fn trace_def() -> SynthDef {
    SynthDef {
        name: "tr".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.5)], 1),
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

#[test]
fn n_trace_dumps_per_unit_io_to_the_sink() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    let captured = Rc::new(RefCell::new(String::new()));
    {
        let sink = Rc::clone(&captured);
        osc.set_trace_sink(Box::new(move |text| sink.borrow_mut().push_str(text)));
    }
    controller.add_synthdef(trace_def());

    // Start the synth through the dispatcher (so node 1000 -> "tr" is tracked for name resolution).
    osc.apply(
        &mut controller,
        &msg(
            "/s_new",
            vec![
                OscType::String("tr".to_string()),
                OscType::Int(1000),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ),
    )
    .expect("/s_new");
    let mut blk = [0.0f32; 64];
    world.fill(&mut blk, 1); // run a block so the synth is warm

    // Trace it: the command applies, the walk dumps this block, the records stream back over the ring.
    osc.apply(&mut controller, &msg("/n_trace", vec![OscType::Int(1000)]))
        .expect("/n_trace");
    world.fill(&mut blk, 1);
    nrt.process();
    while let Some(reply) = nrt.poll_reply() {
        osc.reply(&controller, reply);
    }

    // No OSC reply for /n_trace (like /g_dumpTree); only the text sink is fed.
    assert!(
        osc.take_replies().is_empty(),
        "/n_trace produces no OSC reply"
    );

    let text = captured.borrow();
    assert!(
        text.contains("TRACE node 1000"),
        "trace header names the node: {text}"
    );
    assert!(text.contains("(tr)"), "trace header names the def: {text}");
    // DC: input 0.5 -> output 0.5; Out: [bus 0.0, the DC signal 0.5] -> no outputs.
    assert!(
        text.contains("DC") && text.contains("in: [0.5]") && text.contains("out: [0.5]"),
        "DC's I/O is dumped: {text}"
    );
    assert!(
        text.contains("Out") && text.contains("in: [0.0, 0.5]") && text.contains("out: []"),
        "Out's I/O is dumped: {text}"
    );
}

#[test]
fn n_trace_of_an_unknown_node_is_a_no_op() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    osc.set_reply_target(ReplyTarget::Requester(1));
    let captured = Rc::new(RefCell::new(String::new()));
    {
        let sink = Rc::clone(&captured);
        osc.set_trace_sink(Box::new(move |text| sink.borrow_mut().push_str(text)));
    }

    osc.apply(&mut controller, &msg("/n_trace", vec![OscType::Int(42)]))
        .expect("/n_trace");
    let mut blk = [0.0f32; 64];
    world.fill(&mut blk, 1);
    nrt.process();
    while let Some(reply) = nrt.poll_reply() {
        osc.reply(&controller, reply);
    }
    assert!(
        captured.borrow().is_empty(),
        "tracing an unknown node dumps nothing"
    );
    assert!(osc.take_replies().is_empty(), "and produces no OSC reply");
}
