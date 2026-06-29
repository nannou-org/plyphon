//! `/d_load` and `/d_loadDir` load SynthDef files through a host [`DefSource`] (driven by
//! `run_pending`): the def is parsed, registered, and `/done /<command>` is replied; an absent
//! `DefSource` fails the command.

use std::future::Future;

use plyphon::{Controller, Options, engine};
use plyphon_buffers::{BufFuture, DefSource, LoadError};
use plyphon_osc::{Host, OscDispatcher, ReplyTarget};
use rosc::{OscMessage, OscPacket, OscType};
use scgf::{Input, ParamName, Rate, SynthDef, SynthDefFile, Ugen};

/// SCgf bytes of a one-control `sine` def (`SinOsc(freq) -> Out`).
fn sine_scgf() -> Vec<u8> {
    let file = SynthDefFile {
        version: 2,
        defs: vec![SynthDef {
            name: "sine".to_string(),
            constants: vec![0.0, 0.0],
            param_values: vec![440.0],
            param_names: vec![ParamName {
                name: "freq".to_string(),
                index: 0,
            }],
            ugens: vec![
                Ugen {
                    name: "Control".to_string(),
                    rate: Rate::Control,
                    special_index: 0,
                    inputs: vec![],
                    outputs: vec![Rate::Control],
                },
                Ugen {
                    name: "SinOsc".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Ugen { ugen: 0, output: 0 },
                        Input::Constant { index: 0 },
                    ],
                    outputs: vec![Rate::Audio],
                },
                Ugen {
                    name: "Out".to_string(),
                    rate: Rate::Audio,
                    special_index: 0,
                    inputs: vec![
                        Input::Constant { index: 1 },
                        Input::Ugen { ugen: 1, output: 0 },
                    ],
                    outputs: vec![],
                },
            ],
            variants: vec![],
        }],
    };
    scgf::encode(&file).expect("encode SCgf")
}

/// A [`DefSource`] serving the `sine` def at key `"sine.scsyndef"` (and any directory).
struct DefStub;

impl DefSource for DefStub {
    fn read_def<'a>(&'a self, key: &'a str) -> BufFuture<'a, Result<Vec<u8>, LoadError>> {
        let result = if key == "sine.scsyndef" {
            Ok(sine_scgf())
        } else {
            Err(LoadError::NotFound(key.to_string()))
        };
        Box::pin(async move { result })
    }

    fn read_def_dir<'a>(&'a self, _key: &'a str) -> BufFuture<'a, Result<Vec<Vec<u8>>, LoadError>> {
        Box::pin(async move { Ok(vec![sine_scgf()]) })
    }
}

impl Host for DefStub {
    fn def_source(&self) -> Option<&dyn DefSource> {
        Some(self)
    }
}

/// A host with no capabilities (every action fails).
struct NoHost;
impl Host for NoHost {}

fn osc(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    })
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
    }
}

/// The first `/done`/`/fail` reply's command-name argument.
fn reply_for<'a>(replies: &'a [OscPacket], addr: &str) -> Option<&'a OscMessage> {
    replies.iter().find_map(|p| match p {
        OscPacket::Message(m) if m.addr == addr => Some(m),
        _ => None,
    })
}

fn controller() -> Controller {
    let (controller, _nrt, _world) = engine(Options::default());
    controller
}

#[test]
fn d_load_registers_a_def_and_replies_done() {
    let mut controller = controller();
    let mut osc_disp = OscDispatcher::new();
    osc_disp.set_reply_target(ReplyTarget::Requester(7));

    osc_disp
        .apply(
            &mut controller,
            &osc(
                "/d_load",
                vec![OscType::String("sine.scsyndef".to_string())],
            ),
        )
        .expect("/d_load");
    // Nothing happens until run_pending drives the host DefSource.
    assert!(controller.synthdef("sine").is_none());

    block_on(osc_disp.run_pending(&mut controller, Some(&DefStub)));

    assert!(
        controller.synthdef("sine").is_some(),
        "def should be registered"
    );
    let replies = osc_disp.take_replies();
    let done = reply_for(&replies, "/done").expect("/done reply");
    assert_eq!(done.args, vec![OscType::String("/d_load".to_string())]);
}

#[test]
fn d_load_dir_loads_every_def() {
    let mut controller = controller();
    let mut osc_disp = OscDispatcher::new();
    osc_disp
        .apply(
            &mut controller,
            &osc("/d_loadDir", vec![OscType::String("defs".to_string())]),
        )
        .expect("/d_loadDir");
    block_on(osc_disp.run_pending(&mut controller, Some(&DefStub)));
    assert!(controller.synthdef("sine").is_some());
    let replies = osc_disp.take_replies();
    let done = reply_for(&replies, "/done").expect("/done reply");
    assert_eq!(done.args, vec![OscType::String("/d_loadDir".to_string())]);
}

#[test]
fn d_load_without_a_def_source_fails() {
    let mut controller = controller();
    let mut osc_disp = OscDispatcher::new();
    osc_disp
        .apply(
            &mut controller,
            &osc(
                "/d_load",
                vec![OscType::String("sine.scsyndef".to_string())],
            ),
        )
        .expect("/d_load");
    block_on(osc_disp.run_pending(&mut controller, Some(&NoHost)));
    assert!(controller.synthdef("sine").is_none());
    let replies = osc_disp.take_replies();
    assert!(reply_for(&replies, "/fail").is_some(), "expected a /fail");
}
