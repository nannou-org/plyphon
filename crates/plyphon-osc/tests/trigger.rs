//! `SendTrig` over OSC: a synth with `SendTrig.ar(Impulse.ar(...))` broadcasts `/tr [node, id, value]`
//! to notification subscribers, the same broadcast path as the node notifications.

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

fn trig_def() -> SynthDef {
    SynthDef {
        name: "trig".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(1000.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "SendTrig",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(7.0),
                    InputRef::Constant(0.5),
                ],
                0,
            ),
        ],
    }
}

fn drain_triggers(osc: &mut OscDispatcher, nrt: &mut Nrt) -> Vec<OscMessage> {
    nrt.process();
    while let Some(trigger) = nrt.poll_trigger() {
        osc.notify_trigger(trigger);
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
fn send_trig_broadcasts_tr_over_osc() {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let mut osc = OscDispatcher::new();
    controller.add_synthdef(trig_def());
    osc.apply_bytes(
        &mut controller,
        &msg(
            "/s_new",
            vec![
                OscType::String("trig".to_string()),
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

    let msgs = drain_triggers(&mut osc, &mut nrt);
    let tr = msgs
        .iter()
        .find(|m| m.addr == "/tr")
        .expect("/tr broadcast");
    assert_eq!(
        tr.args,
        vec![OscType::Int(1000), OscType::Int(7), OscType::Float(0.5)]
    );
}
