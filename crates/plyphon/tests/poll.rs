//! `Poll`: on each rising trigger it posts `label: value` to the host (a `NodeMsg`/`Poll`) and, when
//! `trigid >= 0`, a `/tr` trigger, while passing its `in` input straight through to its output.

use plyphon::{
    AddAction, InputRef, NodeMsgKind, Options, ROOT_GROUP_ID, Rate, SynthDef, Trigger, UnitSpec,
    engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `Poll.ar(trig, in, trigid, label)` - inputs `[trig, in, trigid, labelLen, labelChars...]`.
fn poll_unit(trig: u32, in_u: u32, trigid: f32, label: &str) -> UnitSpec {
    let mut inputs = vec![u(trig), u(in_u), c(trigid), c(label.len() as f32)];
    inputs.extend(label.bytes().map(|b| c(b as f32)));
    UnitSpec::new("Poll", Rate::Audio, inputs, 1)
}

/// A synth: `Impulse(1000) -> Poll(trig, DC(value), trigid, "amp") -> Out(0)`. Runs 40 blocks and
/// returns `(node id, output, posts, triggers)`.
fn run(trigid: f32, value: f32) -> (i32, Vec<f32>, Vec<plyphon::NodeMsg>, Vec<Trigger>) {
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "p".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("Impulse", Rate::Audio, vec![c(1000.0), c(0.0)], 1),
            UnitSpec::new("DC", Rate::Audio, vec![c(value)], 1),
            poll_unit(0, 1, trigid, "amp"),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
        ],
    });
    let node = controller
        .synth_new("p", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    let mut out = Vec::new();
    let mut buf = [0.0f32; BLOCK];
    for _ in 0..40 {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    nrt.process();
    let posts: Vec<_> = std::iter::from_fn(|| nrt.poll_node_msg()).collect();
    let triggers: Vec<_> = std::iter::from_fn(|| nrt.poll_trigger()).collect();
    (node, out, posts, triggers)
}

#[test]
fn poll_posts_value_and_passes_input_through() {
    let (node, out, posts, triggers) = run(5.0, 0.75);

    // Pass-through: the output equals `in` (0.75) every sample.
    assert!(
        out.iter().all(|&s| (s - 0.75).abs() < 1e-6),
        "Poll passes its input through to the output"
    );

    // Each post is a `Poll` message carrying the label and the polled value.
    assert!(!posts.is_empty(), "expected Poll posts");
    for m in &posts {
        assert_eq!(m.node, node);
        assert_eq!(m.kind, NodeMsgKind::Poll);
        assert_eq!(m.reply_id, 5);
        let label = core::str::from_utf8(&m.label[..m.label_len as usize]).unwrap();
        assert_eq!(label, "amp");
        assert_eq!(m.num_values, 1);
        assert!((m.values[0] - 0.75).abs() < 1e-6);
    }

    // With trigid >= 0, each edge also sends a `/tr`.
    assert!(!triggers.is_empty(), "expected /tr triggers");
    assert!(
        triggers.iter().all(|t| *t
            == Trigger {
                node,
                id: 5,
                value: 0.75
            }),
        "each /tr carries the node, trigid and value: {triggers:?}"
    );
    // Per-rising-edge (not per-block): ~54 over 2560 samples at 1000 Hz.
    assert!(
        (50..=58).contains(&posts.len()),
        "one post per edge: {}",
        posts.len()
    );
    assert_eq!(posts.len(), triggers.len(), "one /tr per post");
}

#[test]
fn poll_negative_trigid_posts_but_sends_no_tr() {
    // trigid = -1 posts to the console but sends no `/tr`.
    let (_node, _out, posts, triggers) = run(-1.0, 0.5);
    assert!(!posts.is_empty(), "still posts with trigid = -1");
    assert!(triggers.is_empty(), "no /tr when trigid < 0");
}
