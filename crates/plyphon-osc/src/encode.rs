//! Pure engine-event -> OSC encoders - the controller-free, dispatcher-free half of producing replies.
//!
//! Each function maps one engine value ([`Event`], [`Trigger`], [`NodeMsg`], or a self-contained
//! getter [`Reply`]) to the OSC packet a SuperCollider client expects, holding no state and touching
//! neither the engine nor the dispatcher. The [`OscDispatcher`](crate::OscDispatcher) calls these and
//! adds only the genuinely stateful bits (which client to route to, def-name bookkeeping, the
//! multi-message getter reassembly that these one-shot encoders deliberately do *not* cover). A host
//! that owns its own transport can use them directly - e.g. `encode_event(Event::NodeEnded { id })`
//! yields `/n_end [id]` with no dispatcher in sight.

use alloc::string::ToString;
use alloc::vec::Vec;

use plyphon::{Event, NodeMsg, NodeMsgKind, Reply, Trigger};
use rosc::{OscMessage, OscPacket, OscType};

/// Build an `OscPacket::Message` from an address and its arguments.
fn message(addr: &str, args: Vec<OscType>) -> OscPacket {
    OscPacket::Message(OscMessage {
        addr: addr.to_string(),
        args,
    })
}

/// A node-lifecycle [`Event`] as its SuperCollider notification: `/n_go` (started), `/n_end` (freed),
/// `/n_off` (paused), `/n_on` (resumed), `/n_move` (moved, carrying the full `/n_info`-shaped
/// position), or `/fail` for a failed `/s_new`. Pure: the dispatcher handles the `node_defs`
/// bookkeeping and broadcast tagging around this.
pub fn encode_event(event: Event) -> OscPacket {
    let (addr, args) = match event {
        Event::NodeStarted { id } => ("/n_go", vec![OscType::Int(id)]),
        Event::NodeEnded { id } => ("/n_end", vec![OscType::Int(id)]),
        Event::NodePaused { id } => ("/n_off", vec![OscType::Int(id)]),
        Event::NodeResumed { id } => ("/n_on", vec![OscType::Int(id)]),
        Event::NodeMoved {
            node,
            parent,
            prev,
            next,
            is_group,
            head,
            tail,
        } => (
            "/n_move",
            node_info_args(node, parent, prev, next, is_group, head, tail),
        ),
        Event::SynthFailed { id } => (
            "/fail",
            vec![OscType::String("/s_new".to_string()), OscType::Int(id)],
        ),
    };
    message(addr, args)
}

/// A `SendTrig` [`Trigger`] as `/tr [nodeID, id, value]`.
pub fn encode_trigger(trigger: Trigger) -> OscPacket {
    message(
        "/tr",
        vec![
            OscType::Int(trigger.node),
            OscType::Int(trigger.id),
            OscType::Float(trigger.value),
        ],
    )
}

/// A `SendReply` [`NodeMsg`] as `/<label> [nodeID, replyID, values...]`; `None` for a kind that has
/// no OSC form. (A non-UTF-8 label degrades to an empty path, as scsynth-style hosts tolerate.)
pub fn encode_node_msg(msg: NodeMsg) -> Option<OscPacket> {
    match msg.kind {
        NodeMsgKind::Reply => {
            let path = core::str::from_utf8(&msg.label[..msg.label_len as usize]).unwrap_or("");
            let mut args = Vec::with_capacity(2 + msg.num_values as usize);
            args.push(OscType::Int(msg.node));
            args.push(OscType::Int(msg.reply_id));
            for &v in &msg.values[..msg.num_values as usize] {
                args.push(OscType::Float(v));
            }
            Some(message(path, args))
        }
    }
}

/// Build the arguments shared by `/n_info` (the `/n_query` answer) and `/n_move` (the node-move
/// notification): node, parent, prev, next, isGroup, plus head/tail when the node is a group.
pub fn node_info_args(
    node: i32,
    parent: i32,
    prev: i32,
    next: i32,
    is_group: i32,
    head: i32,
    tail: i32,
) -> Vec<OscType> {
    let mut args = vec![
        OscType::Int(node),
        OscType::Int(parent),
        OscType::Int(prev),
        OscType::Int(next),
        OscType::Int(is_group),
    ];
    if is_group == 1 {
        args.push(OscType::Int(head));
        args.push(OscType::Int(tail));
    }
    args
}

/// `/synced [id]` - the answer to `/sync`.
pub fn encode_synced(id: i32) -> OscPacket {
    message("/synced", vec![OscType::Int(id)])
}

/// [`Reply::Status`] as `/status.reply`; `None` for any other reply.
pub fn encode_status(reply: &Reply) -> Option<OscPacket> {
    if let Reply::Status {
        num_ugens,
        num_synths,
        num_groups,
        num_synthdefs,
        avg_cpu,
        peak_cpu,
        nominal_sr,
        actual_sr,
    } = reply
    {
        Some(message(
            "/status.reply",
            vec![
                OscType::Int(1),
                OscType::Int(*num_ugens),
                OscType::Int(*num_synths),
                OscType::Int(*num_groups),
                OscType::Int(*num_synthdefs),
                OscType::Float(*avg_cpu),
                OscType::Float(*peak_cpu),
                OscType::Double(*nominal_sr),
                OscType::Double(*actual_sr),
            ],
        ))
    } else {
        None
    }
}

/// [`Reply::RtMemoryStatus`] as `/rtMemoryStatus.reply`; `None` for any other reply.
pub fn encode_rt_memory(reply: &Reply) -> Option<OscPacket> {
    if let Reply::RtMemoryStatus {
        total_free,
        largest_free,
    } = reply
    {
        Some(message(
            "/rtMemoryStatus.reply",
            vec![OscType::Int(*total_free), OscType::Int(*largest_free)],
        ))
    } else {
        None
    }
}

/// [`Reply::NodeInfo`] as `/n_info`; `None` for any other reply (e.g. `NodeNotFound`, which the
/// dispatcher reports as a `/fail`).
pub fn encode_node_info(reply: &Reply) -> Option<OscPacket> {
    if let Reply::NodeInfo {
        node,
        parent,
        prev,
        next,
        is_group,
        head,
        tail,
    } = reply
    {
        Some(message(
            "/n_info",
            node_info_args(*node, *parent, *prev, *next, *is_group, *head, *tail),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(packet: OscPacket) -> OscMessage {
        match packet {
            OscPacket::Message(m) => m,
            OscPacket::Bundle(_) => panic!("expected a message"),
        }
    }

    #[test]
    fn lifecycle_events_map_to_their_addresses() {
        for (event, addr) in [
            (Event::NodeStarted { id: 5 }, "/n_go"),
            (Event::NodeEnded { id: 5 }, "/n_end"),
            (Event::NodePaused { id: 5 }, "/n_off"),
            (Event::NodeResumed { id: 5 }, "/n_on"),
        ] {
            let m = msg(encode_event(event));
            assert_eq!(m.addr, addr);
            assert_eq!(m.args, vec![OscType::Int(5)]);
        }
    }

    #[test]
    fn node_moved_carries_position_and_group_head_tail() {
        // A non-group omits head/tail: node, parent, prev, next, isGroup.
        let synth = msg(encode_event(Event::NodeMoved {
            node: 10,
            parent: 1,
            prev: -1,
            next: 11,
            is_group: 0,
            head: -1,
            tail: -1,
        }));
        assert_eq!(synth.addr, "/n_move");
        assert_eq!(synth.args.len(), 5);

        // A group appends head/tail.
        let group = msg(encode_event(Event::NodeMoved {
            node: 2,
            parent: 1,
            prev: -1,
            next: -1,
            is_group: 1,
            head: 3,
            tail: 4,
        }));
        assert_eq!(group.args.len(), 7);
    }

    #[test]
    fn synth_failed_is_a_fail_for_s_new() {
        let m = msg(encode_event(Event::SynthFailed { id: 9 }));
        assert_eq!(m.addr, "/fail");
        assert_eq!(
            m.args,
            vec![OscType::String("/s_new".to_string()), OscType::Int(9)]
        );
    }

    #[test]
    fn trigger_maps_to_tr() {
        let m = msg(encode_trigger(Trigger {
            node: 100,
            id: 7,
            value: 0.5,
        }));
        assert_eq!(m.addr, "/tr");
        assert_eq!(
            m.args,
            vec![OscType::Int(100), OscType::Int(7), OscType::Float(0.5)]
        );
    }

    #[test]
    fn node_msg_uses_its_label_as_the_path() {
        let mut label = [0u8; 32];
        label[..4].copy_from_slice(b"/amp");
        let mut values = [0.0f32; 32];
        values[0] = 0.25;
        let m = msg(encode_node_msg(NodeMsg {
            node: 100,
            reply_id: 3,
            kind: NodeMsgKind::Reply,
            label,
            label_len: 4,
            values,
            num_values: 1,
        })
        .expect("Reply kind always encodes"));
        assert_eq!(m.addr, "/amp");
        assert_eq!(
            m.args,
            vec![OscType::Int(100), OscType::Int(3), OscType::Float(0.25)]
        );
    }

    #[test]
    fn status_reply_shape() {
        let m = msg(encode_status(&Reply::Status {
            num_ugens: 1,
            num_synths: 2,
            num_groups: 3,
            num_synthdefs: 4,
            avg_cpu: 0.0,
            peak_cpu: 0.0,
            nominal_sr: 48_000.0,
            actual_sr: 48_000.0,
        })
        .unwrap());
        assert_eq!(m.addr, "/status.reply");
        // Leading int 1 (scsynth's unused "1"), then the four counts, two cpu floats, two sr doubles.
        assert_eq!(m.args[0], OscType::Int(1));
        assert_eq!(m.args[8], OscType::Double(48_000.0));
        assert!(encode_status(&Reply::Synced { id: 0 }).is_none());
    }

    #[test]
    fn standalone_use_needs_no_dispatcher() {
        // The stated payoff: encode an event to OSC with nothing but this function.
        let m = msg(encode_event(Event::NodeEnded { id: 1000 }));
        assert_eq!(m.addr, "/n_end");
        assert_eq!(m.args, vec![OscType::Int(1000)]);
    }
}
