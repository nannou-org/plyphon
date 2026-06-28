//! `SendReply`: an `Impulse.ar` drives a `SendReply.ar`, which on each rising edge emits a custom OSC
//! message `/<path> [nodeID, replyID, values...]` over the dedicated node-message ring, surfaced via
//! `Nrt::poll_node_msg`. Also checks that `build()` rejects an over-bound or non-constant path.

use plyphon::{
    AddAction, BuildError, InputRef, NodeMsgKind, Options, Param, ROOT_GROUP_ID, Rate, RateInfo,
    SynthDef, UnitRegistry, UnitSpec, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
/// Mirrors `plyphon_unit::unit::MAX_LABEL`/`MAX_VALUES` (the inline carrier bounds).
const MAX_LABEL: usize = 32;
const MAX_VALUES: usize = 32;

fn try_compile(def: &SynthDef) -> Result<(), BuildError> {
    let rate = RateInfo::new(SR, BLOCK);
    def.compile(&UnitRegistry::with_builtins(), &rate, &rate, 64, 32)
        .map(|_| ())
}

#[test]
fn emits_one_reply_per_rising_edge() {
    let (reply_id, path, v0, v1) = (42.0f32, "/myreply", 11.0f32, 22.0f32);
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "rep".to_string(),
        params: vec![],
        units: vec![
            // Impulse every 48 samples (1000 Hz @ 48 kHz).
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(1000.0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::send_reply(
                Rate::Audio,
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(reply_id),
                path,
                &[InputRef::Constant(v0), InputRef::Constant(v1)],
            ),
        ],
    });
    let node = controller
        .synth_new("rep", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // 40 control blocks = 2560 samples -> impulses at 0, 48, ... 2544: ~54 rising edges.
    let mut buf = [0.0f32; BLOCK];
    for _ in 0..40 {
        world.fill(&mut buf, 1);
    }

    nrt.process();
    let mut msgs = Vec::new();
    while let Some(m) = nrt.poll_node_msg() {
        msgs.push(m);
    }

    assert!(!msgs.is_empty(), "expected SendReply messages");
    for m in &msgs {
        assert_eq!(m.node, node);
        assert_eq!(m.reply_id, 42);
        assert_eq!(m.kind, NodeMsgKind::Reply);
        let label = core::str::from_utf8(&m.label[..m.label_len as usize]).unwrap();
        assert_eq!(label, path);
        assert_eq!(m.num_values, 2);
        assert_eq!(&m.values[..2], &[v0, v1]);
    }
    let count = msgs.len();
    assert!(
        (50..=58).contains(&count),
        "expected ~54 replies (one per rising edge), got {count}"
    );
}

#[test]
fn non_constant_path_rejected() {
    // The cmdNameLen input (index 2) is a control parameter, not a constant.
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![Param::control("n", 0.0)],
        units: vec![UnitSpec::new(
            "SendReply",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0), // trig
                InputRef::Constant(0.0), // replyID
                InputRef::Param(0),      // cmdNameLen (non-constant)
            ],
            0,
        )],
    };
    assert_eq!(try_compile(&def), Err(BuildError::EmitBadLabel));
}

#[test]
fn over_long_path_rejected() {
    let path = "/".to_string() + &"x".repeat(MAX_LABEL); // MAX_LABEL + 1 bytes
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![],
        units: vec![UnitSpec::send_reply(
            Rate::Audio,
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
            &path,
            &[],
        )],
    };
    assert_eq!(
        try_compile(&def),
        Err(BuildError::EmitLabelTooLong {
            len: MAX_LABEL + 1,
            limit: MAX_LABEL,
        })
    );
}

#[test]
fn too_many_values_rejected() {
    let values = vec![InputRef::Constant(0.0); MAX_VALUES + 1];
    let def = SynthDef {
        name: "bad".to_string(),
        params: vec![],
        units: vec![UnitSpec::send_reply(
            Rate::Audio,
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
            "/r",
            &values,
        )],
    };
    assert_eq!(
        try_compile(&def),
        Err(BuildError::EmitTooManyValues {
            count: MAX_VALUES + 1,
            limit: MAX_VALUES,
        })
    );
}
