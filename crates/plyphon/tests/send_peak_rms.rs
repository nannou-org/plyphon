//! `SendPeakRMS`: each expected reply is the bit pattern scsynth's `SendPeakRMS` (with nova-simd's
//! four-lane peak meter, as scsynth builds it for NEON or SSE3) reports at 48 kHz with 64-sample
//! blocks, measuring `WhiteNoise.ar` on a fresh stream 0 and a constant control-rate channel.

use plyphon::{
    AddAction, BuildError, InputRef, NodeMsgKind, Options, ROOT_GROUP_ID, Rate, RateInfo, SynthDef,
    UnitRegistry, UnitSpec, engine,
};

const BLOCK: usize = 64;
const PATH: &str = "/peak";

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `SendPeakRMS.<rate>([WhiteNoise.ar, DC.kr(dc)], replyRate, peakLag, PATH, 7)`.
fn def(rate: Rate, reply_rate: f32, peak_lag: f32, dc: f32) -> SynthDef {
    let mut inputs = vec![c(reply_rate), c(peak_lag), c(7.0), c(2.0), u(0), u(1)];
    inputs.push(c(PATH.len() as f32));
    inputs.extend(PATH.bytes().map(|b| c(b as f32)));
    SynthDef {
        name: "meter".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new("DC", Rate::Control, vec![c(dc)], 1),
            UnitSpec::new("SendPeakRMS", rate, inputs, 0),
        ],
    }
}

/// Run `def` for `blocks` blocks and return each reply's values as bit patterns.
fn replies(def: SynthDef, blocks: usize) -> Vec<[u32; 4]> {
    let (mut controller, mut nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("meter", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * blocks];
    world.fill(&mut buf, 1);
    nrt.process();
    let mut out = Vec::new();
    while let Some(m) = nrt.poll_node_msg() {
        assert_eq!(m.node, node);
        assert_eq!(m.reply_id, 7);
        assert_eq!(m.kind, NodeMsgKind::Reply);
        assert_eq!(&m.label[..m.label_len as usize], PATH.as_bytes());
        assert_eq!(m.num_values, 4);
        out.push(
            m.values[..4]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
    }
    out
}

#[test]
fn audio_rate_reports_mid_block_where_each_interval_ends() {
    // replyRate 700: a reply every 68 samples, so the interval ends 4, 8, 12... samples into a
    // block and the parts either side mix the scalar and four-lane sums.
    assert_eq!(
        replies(def(Rate::Audio, 700.0, 0.01, -0.25), 6),
        [
            [0x3f6b9288, 0x3f0da2fc, 0x3e800000, 0x3eb504f3],
            [0x3f7cc8bc, 0x3f19b2c4, 0x3e800000, 0x3e800000],
            [0x3f7c21c8, 0x3f0dd1e4, 0x3e800000, 0x3e800000],
            [0x3f7f1670, 0x3f0e92d3, 0x3e800000, 0x3e800000],
            [0x3f7e4171, 0x3f1a8608, 0x3e800000, 0x3e800000],
        ]
    );
}

#[test]
fn audio_rate_with_whole_block_intervals_takes_the_four_lane_sums() {
    // replyRate 500: a reply every 96 samples, analysed in 32- and 64-sample parts. The control
    // channel's RMS divides by the one-block control interval, so it reads 0.25 * sqrt(2) when two
    // of its parts fell in one interval.
    assert_eq!(
        replies(def(Rate::Audio, 500.0, 0.002, -0.25), 6),
        [
            [0x3f7cc8bc, 0x3f0e93e1, 0x3e800000, 0x3eb504f3],
            [0x3f7bbed4, 0x3f13f95d, 0x3e800000, 0x3e800000],
            [0x3f7f1670, 0x3f12c57e, 0x3e800000, 0x3eb504f3],
        ]
    );
}

#[test]
fn control_rate_reports_every_few_blocks_before_analysing() {
    // replyRate 250 at a 750 Hz control rate: a reply every 3 blocks, sent before that block is
    // analysed, so the first covers blocks 1 and 2 only.
    assert_eq!(
        replies(def(Rate::Control, 250.0, 0.05, 0.5), 10),
        [
            [0x3f7cc8bc, 0x3ef108c5, 0x3f000000, 0x3ed105ec],
            [0x3f7f1670, 0x3f132614, 0x3f000000, 0x3f000000],
            [0x3f7e377e, 0x3f13705f, 0x3f000000, 0x3f000000],
        ]
    );
}

#[test]
fn rejects_more_channels_than_the_carrier_holds() {
    let mut inputs = vec![c(20.0), c(3.0), c(-1.0), c(17.0)];
    inputs.extend((0..17).map(|_| c(0.0)));
    inputs.push(c(1.0));
    inputs.push(c(b'/' as f32));
    let def = SynthDef {
        name: "wide".to_string(),
        params: vec![],
        units: vec![UnitSpec::new("SendPeakRMS", Rate::Control, inputs, 0)],
    };
    let rate = RateInfo::new(48_000.0, BLOCK);
    let err = def
        .compile(
            &UnitRegistry::with_builtins(),
            &rate,
            &rate,
            64,
            32,
            None,
            1,
        )
        .err();
    assert_eq!(
        err,
        Some(BuildError::EmitTooManyValues {
            count: 34,
            limit: 32
        })
    );
}
