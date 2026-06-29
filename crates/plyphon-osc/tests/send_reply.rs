//! `SendReply` over OSC: a synth with `SendReply.ar(Impulse.ar(...))` broadcasts its custom message
//! `/<path> [node, replyID, values...]` to notification subscribers, the same broadcast path as `/tr`.

use plyphon::{InputRef, Nrt, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};
use plyphon_osc::OscDispatcher;
use rosc::{OscMessage, OscPacket, OscType};

const SR: f64 = 48_000.0;

fn msg(addr: &str, args: Vec<OscType>) -> Vec<u8> {
    rosc::encoder::encode(&OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    }))
    .expect("encode OSC")
}

fn rep_def() -> SynthDef {
    SynthDef {
        name: "rep".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(1000.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::send_reply(
                Rate::Audio,
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(42.0),
                "/myreply",
                &[InputRef::Constant(11.0), InputRef::Constant(22.0)],
            ),
        ],
    }
}

fn drain(osc: &mut OscDispatcher, nrt: &mut Nrt) -> Vec<OscMessage> {
    nrt.process();
    while let Some(m) = nrt.poll_node_msg() {
        osc.notify_node_msg(m);
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
fn send_reply_broadcasts_over_osc() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    controller.add_synthdef(rep_def());
    osc.apply_bytes(
        &mut controller,
        &msg(
            "/s_new",
            vec![
                OscType::String("rep".to_string()),
                OscType::Int(1000),
                OscType::Int(1),
                OscType::Int(ROOT_GROUP_ID),
            ],
        ),
    )
    .expect("/s_new");

    let mut buf = [0.0f32; 64];
    for _ in 0..16 {
        world.fill(&mut buf, 1);
    }

    let msgs = drain(&mut osc, &mut nrt);
    let rep = msgs
        .iter()
        .find(|m| m.addr == "/myreply")
        .expect("/myreply broadcast");
    assert_eq!(
        rep.args,
        vec![
            OscType::Int(1000),
            OscType::Int(42),
            OscType::Float(11.0),
            OscType::Float(22.0),
        ]
    );
}
