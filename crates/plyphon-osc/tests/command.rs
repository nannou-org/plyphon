//! `/cmd` and `/u_cmd` route plugin/unit commands to the host's [`CommandHost`] (driven by
//! `run_pending`): the handler interprets each command - including its argument layout - and owns any
//! reply (scsynth sends no automatic `/done`); an absent handler fails the command.

use std::future::Future;

use plyphon::{Controller, Options, engine};
use plyphon_buffers::BufFuture;
use plyphon_osc::{CmdTarget, CommandHost, Host, OscDispatcher, ReplyTarget};
use rosc::{OscMessage, OscPacket, OscType};

/// A [`CommandHost`] that echoes each command's target, name, and payload back in a `/cmd.reply`
/// packet - enough to assert the dispatcher parsed `/cmd`/`/u_cmd`'s argument layout faithfully.
struct EchoHost;

impl CommandHost for EchoHost {
    fn command<'a>(
        &'a self,
        target: CmdTarget,
        name: &'a str,
        args: &'a [OscType],
    ) -> BufFuture<'a, Result<Option<OscPacket>, String>> {
        let mut reply = match target {
            CmdTarget::Plugin => vec![OscType::String("plugin".to_string())],
            CmdTarget::Unit { node, index } => vec![
                OscType::String("unit".to_string()),
                OscType::Int(node),
                OscType::Int(index),
            ],
        };
        reply.push(OscType::String(name.to_string()));
        reply.extend(args.iter().cloned());
        Box::pin(async move {
            Ok(Some(OscPacket::Message(OscMessage {
                addr: "/cmd.reply".to_string(),
                args: reply,
            })))
        })
    }
}

impl Host for EchoHost {
    fn commands(&self) -> Option<&dyn CommandHost> {
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

/// The first reply at `addr`, if any.
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
fn cmd_routes_a_plugin_command_to_the_host() {
    let mut controller = controller();
    let mut osc_disp = OscDispatcher::new();
    osc_disp.set_reply_target(ReplyTarget::Requester(7));

    osc_disp
        .apply(
            &mut controller,
            &osc(
                "/cmd",
                vec![OscType::String("ping".to_string()), OscType::Float(0.5)],
            ),
        )
        .expect("/cmd");
    // Nothing happens until run_pending drives the host CommandHost.
    assert!(osc_disp.take_replies().is_empty());

    block_on(osc_disp.run_pending(&mut controller, Some(&EchoHost)));

    let replies = osc_disp.take_replies();
    let reply = reply_for(&replies, "/cmd.reply").expect("/cmd.reply");
    assert_eq!(
        reply.args,
        vec![
            OscType::String("plugin".to_string()),
            OscType::String("ping".to_string()),
            OscType::Float(0.5),
        ]
    );
}

#[test]
fn u_cmd_routes_a_unit_command_with_node_and_index() {
    let mut controller = controller();
    let mut osc_disp = OscDispatcher::new();

    osc_disp
        .apply(
            &mut controller,
            &osc(
                "/u_cmd",
                vec![
                    OscType::Int(1000),
                    OscType::Int(2),
                    OscType::String("set".to_string()),
                    OscType::Float(0.25),
                ],
            ),
        )
        .expect("/u_cmd");
    block_on(osc_disp.run_pending(&mut controller, Some(&EchoHost)));

    let replies = osc_disp.take_replies();
    let reply = reply_for(&replies, "/cmd.reply").expect("/cmd.reply");
    assert_eq!(
        reply.args,
        vec![
            OscType::String("unit".to_string()),
            OscType::Int(1000),
            OscType::Int(2),
            OscType::String("set".to_string()),
            OscType::Float(0.25),
        ]
    );
}

#[test]
fn cmd_without_a_command_host_fails() {
    let mut controller = controller();
    let mut osc_disp = OscDispatcher::new();
    osc_disp
        .apply(
            &mut controller,
            &osc("/cmd", vec![OscType::String("ping".to_string())]),
        )
        .expect("/cmd");
    block_on(osc_disp.run_pending(&mut controller, Some(&NoHost)));
    let replies = osc_disp.take_replies();
    let fail = reply_for(&replies, "/fail").expect("expected a /fail");
    assert_eq!(
        fail.args.first(),
        Some(&OscType::String("/cmd".to_string()))
    );
}
