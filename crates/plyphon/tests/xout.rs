//! `XOut`: crossfade a signal into a bus against whatever earlier synths wrote there this block.
//! `xfade = 0` leaves the bus, `xfade = 1` replaces it. The bit-exact tests pin scsynth's own
//! `XOut` (with nova-simd's NEON kernels at block sizes that are multiples of 16) over blocks that
//! hold, ramp and hit the 0 and 1 branches, at audio and control rate and in a reblocked graph.

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f64 = 48_000.0;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `DC(a) -> Out(0)` then `DC(b) -> XOut(0, xfade)`, both on the output bus in node order. Returns the
/// resulting output value (steady, so the last sample).
fn out_then_xout(a: f32, b: f32, xfade: f32) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    // Synth A: writes `a` onto bus 0 with a plain Out.
    controller.add_synthdef(SynthDef {
        name: "a".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(a)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    });
    // Synth B: crossfades `b` into bus 0 against A's output. XOut has no signal output.
    controller.add_synthdef(SynthDef {
        name: "b".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(b)], 1),
            UnitSpec::new("XOut", Rate::Audio, vec![c(0.0), c(xfade), u(0)], 0),
        ],
    });
    controller
        .synth_new("a", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller
        .synth_new("b", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = [0.0f32; 64];
    world.fill(&mut buf, 1);
    buf[63]
}

#[test]
fn xout_crossfades_against_earlier_bus_content() {
    // A wrote 1.0; B crossfades 0.5 in: bus = 1.0*(1-xf) + 0.5*xf.
    assert!(
        (out_then_xout(1.0, 0.5, 0.0) - 1.0).abs() < 1e-6,
        "xfade 0 keeps the bus"
    );
    assert!(
        (out_then_xout(1.0, 0.5, 1.0) - 0.5).abs() < 1e-6,
        "xfade 1 replaces the bus"
    );
    assert!(
        (out_then_xout(1.0, 0.5, 0.5) - 0.75).abs() < 1e-6,
        "xfade 0.5 is the half-mix"
    );
    assert!(
        (out_then_xout(1.0, 0.5, 0.25) - 0.875).abs() < 1e-6,
        "xfade 0.25 is a quarter toward the input"
    );
}

#[test]
fn xout_alone_is_first_writer() {
    // With no earlier writer, XOut is the first to touch the channel and treats the bus as zero, so
    // the output is in*xfade.
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "x".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(0.8)], 1),
            UnitSpec::new("XOut", Rate::Audio, vec![c(0.0), c(0.5), u(0)], 0),
        ],
    });
    controller
        .synth_new("x", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = [0.0f32; 64];
    world.fill(&mut buf, 1);
    assert!(
        (buf[63] - 0.4).abs() < 1e-6,
        "a lone XOut lands in*xfade = 0.8*0.5 = 0.4, got {}",
        buf[63]
    );
}

#[test]
fn lone_reblocked_xout_never_mixes_stale_audio() {
    // A sole-first-writer XOut in a *reblocked* def: its first tick's whole-channel clear must
    // cover the later ticks' slices too, so no slice crossfades against a prior block's audio
    // lingering on the (persistent) private bus.
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        input_channels: 0, // private audio channels start at bus 1
        ..Options::default()
    });
    // Pollute private bus 1 with 1.0, then free the writer - the private channel persists.
    controller.add_synthdef(SynthDef {
        name: "w".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(1.0), u(0)], 0),
        ],
    });
    let w = controller
        .synth_new("w", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = [0.0f32; 64];
    world.fill(&mut buf, 1);
    controller.free(w).unwrap();
    world.fill(&mut buf, 1);

    // A reblocked (16-sample) def whose XOut is the sole writer of bus 1, and a reader after it.
    controller.add_synthdef_reblocked(
        SynthDef {
            name: "x".to_string(),
            params: vec![],
            units: vec![
                UnitSpec::new("DC", Rate::Audio, vec![c(0.8)], 1),
                UnitSpec::new("XOut", Rate::Audio, vec![c(1.0), c(0.5), u(0)], 0),
            ],
        },
        16,
    );
    controller.add_synthdef(SynthDef {
        name: "r".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("In", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    });
    controller
        .synth_new("x", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller
        .synth_new("r", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    world.fill(&mut buf, 1);
    for (i, &s) in buf.iter().enumerate() {
        assert!(
            (s - 0.4).abs() < 1e-6,
            "sample {i}: every tick-slice should land in*xfade = 0.4, got {s} \
             (stale prior-block audio leaked into a later slice)"
        );
    }
}

/// The `xfade` each block of [`xout_blocks`] runs at: steady, a change (a ramp), steady, a change
/// to 1, steady at 1 (a copy), a change to 0, steady at 0 (nothing written), a change to 0.5.
const XFADES: [f32; 8] = [0.3, 0.8, 0.8, 1.0, 1.0, 0.0, 0.0, 0.5];

/// `Out.ar(0, DC.ar(0.25))`, then `XOut.ar(0, xfade, [noise, noise])` with `noise` a
/// `WhiteNoise.ar` on a fresh stream 0, so output 0 is crossfaded over a touched channel and output
/// 1 over an untouched one. `xfade` steps through [`XFADES`], one value per block; each block
/// returns samples 0, 1, 5, `bs/2` and `bs-1` of both outputs.
fn xout_blocks(block_size: usize, reblock: Option<usize>) -> Vec<[u32; 10]> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 2,
        block_size,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "a".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(0.25)], 1),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
    });
    let x = SynthDef {
        name: "x".to_string(),
        params: vec![Param::control("xfade", XFADES[0])],
        units: vec![
            UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1),
            UnitSpec::new(
                "XOut",
                Rate::Audio,
                vec![c(0.0), InputRef::Param(0), u(0), u(0)],
                0,
            ),
        ],
    };
    match reblock {
        Some(block) => controller.add_synthdef_reblocked(x, block),
        None => controller.add_synthdef(x),
    }
    controller
        .synth_new("a", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let node = controller
        .synth_new("x", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = vec![0.0f32; block_size * 2];
    XFADES
        .iter()
        .map(|&xfade| {
            controller.set_control(node, 0, xfade).unwrap();
            world.fill(&mut buf, 2);
            let mut picked = [0u32; 10];
            for ch in 0..2 {
                for (p, j) in [0, 1, 5, block_size / 2, block_size - 1].iter().enumerate() {
                    picked[ch * 5 + p] = buf[j * 2 + ch].to_bits();
                }
            }
            picked
        })
        .collect()
}

#[test]
fn xout_matches_scsynth_nova_kernels_at_a_block_multiple_of_16() {
    // `XOut_next_a_nova`: `bus * (1 - xfade) + in * xfade`, with nova-simd's four-lane ramp on a
    // change.
    assert_eq!(xout_blocks(64, None), NOVA_64);
}

#[test]
fn xout_matches_scsynth_plain_kernels_at_other_block_sizes() {
    // `XOut_next_a`: `bus + xfade * (in - bus)`, with a per-sample ramp on a change. The ramp from
    // 0 starts `in * 0.0`, `-0.0` for a negative `in`, which the untouched output keeps.
    assert_eq!(xout_blocks(24, None), PLAIN_24);
}

#[test]
fn reblocked_xout_matches_scsynth() {
    // `XOut_next_a_reblock`: each 16-sample tick ramps from the last tick's `xfade`, and at
    // `xfade` 1 sums onto the channel, which here still holds the block before's crossfade.
    assert_eq!(xout_blocks(64, Some(16)), REBLOCK_16);
}

/// `Out.kr(5, 0.25)` (unless `touched` is false), then `XOut.kr(5, 0.3, 0.9)`, read back by `In.kr`.
fn xout_kr(touched: bool) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    let mut units = vec![UnitSpec::new("DC", Rate::Control, vec![c(0.25)], 1)];
    if touched {
        units.push(UnitSpec::new("Out", Rate::Control, vec![c(5.0), u(0)], 0));
    }
    units.push(UnitSpec::new("DC", Rate::Control, vec![c(0.9)], 1));
    let signal = units.len() as u32 - 1;
    units.push(UnitSpec::new(
        "XOut",
        Rate::Control,
        vec![c(5.0), c(0.3), u(signal)],
        0,
    ));
    let reader = units.len() as u32;
    units.push(UnitSpec::new("In", Rate::Control, vec![c(5.0)], 1));
    units.push(UnitSpec::new("DC", Rate::Audio, vec![u(reader)], 1));
    units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![c(0.0), u(reader + 1)],
        0,
    ));
    controller.add_synthdef(SynthDef {
        name: "k".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("k", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let mut buf = [0.0f32; 64];
    world.fill(&mut buf, 1);
    buf[0]
}

#[test]
fn control_rate_xout_crossfades_as_scsynth() {
    // `XOut_next_k`: `bus + xfade * (in - bus)` over a written channel, `xfade * in` otherwise.
    let (bus, xfade, value) = (0.25f32, 0.3f32, 0.9f32);
    assert_eq!(
        xout_kr(true).to_bits(),
        (bus + xfade * (value - bus)).to_bits()
    );
    assert_eq!(xout_kr(false).to_bits(), (xfade * value).to_bits());
}

const NOVA_64: [[u32; 10]; 8] = [
    [
        0xbda6dfba, 0xbd05bbc4, 0x3e82d043, 0xbdb5af9e, 0x3e9c830e, 0xbe835188, 0xbe54a224,
        0x3da4daa7, 0xbe870581, 0x3e05d2ea,
    ],
    [
        0x3cfdf618, 0x3e67d2bd, 0xbe259e7d, 0x3edf2b24, 0xbf37ad07, 0xbe137470, 0x3d5a7e27,
        0xbea768d8, 0x3ea5918b, 0xbf44f9d4,
    ],
    [
        0xbefc0460, 0x3e9be7a6, 0xbf0a088a, 0x3d53bbcc, 0xbe133734, 0xbf0acefd, 0x3e824e0d,
        0xbf16d557, 0x3adde000, 0xbe466a67,
    ],
    [
        0x3f540674, 0x3e11f70b, 0x3eab7a6e, 0x3c859caf, 0x3f4f28cf, 0x3f4739a7, 0x3dbf214a,
        0x3e93e0d4, 0xbc0e6032, 0x3f4ef59c,
    ],
    [
        0x3f5cb810, 0xbe9adf78, 0x3eabb8d8, 0x3db9e6e0, 0xbf5caf64, 0x3f5cb810, 0xbe9adf78,
        0x3eabb8d8, 0x3db9e6e0, 0xbf5caf64,
    ],
    [
        0xbe6e6f60, 0xbf2963a8, 0xbd0dff4a, 0x3f039342, 0x3e6f8838, 0xbe6e6f60, 0xbf2a63a8,
        0xbd5dff4a, 0x3ec72684, 0xbc477c7c,
    ],
    [
        0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000,
    ],
    [
        0x3e800000, 0x3e7d635a, 0x3e658b53, 0x3e9b4794, 0xbe091c68, 0x00000000, 0xba1ca640,
        0xbc83a568, 0x3ded1e50, 0xbe858e34,
    ],
];

const PLAIN_24: [[u32; 10]; 8] = [
    [
        0xbda6dfbc, 0xbd05bbc0, 0x3e82d043, 0x3ec4c64e, 0x3edb0e05, 0xbe835188, 0xbe54a224,
        0x3da4daa7, 0x3e56596a, 0x3e81746c,
    ],
    [
        0x3d096810, 0x3e3e99d0, 0x3e9c3e81, 0xbd43afd8, 0xbdb21f88, 0xbe10d92f, 0x3c85df94,
        0x3e1ff47b, 0xbe241f29, 0xbe11984f,
    ],
    [
        0x3eb1b986, 0xbf2f9320, 0xbf22f157, 0xbf1b9e03, 0xbe12a7ce, 0x3e981fed, 0xbf3c5fed,
        0xbf2fbe23, 0xbf286ad0, 0xbe45db00,
    ],
    [
        0xbec7bfda, 0x3e1aad98, 0xbee75c16, 0xbeb27106, 0x3f47f747, 0xbee15973, 0x3dd3390e,
        0xbefba05a, 0xbebf3dd4, 0x3f476ebe,
    ],
    [
        0x3f168450, 0x3f6db814, 0xbf0e87c0, 0x3f67bd44, 0x3d134040, 0x3f168450, 0x3f6db814,
        0xbf0e87c0, 0x3f67bd44, 0x3d134040,
    ],
    [
        0xbf134a40, 0xbf25be4f, 0x3ee6c606, 0x3ea24f2f, 0x3e619ae0, 0xbf134a40, 0xbf2868fa,
        0x3ecc1b5b, 0x3e449e5b, 0xbc9dd3c7,
    ],
    [
        0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000,
    ],
    [
        0x3e800000, 0x3e7c5244, 0x3e5afc47, 0x3dc367e2, 0x3c67d980, 0x80000000, 0x3ad3cc80,
        0xbc2590e6, 0xbdbc981f, 0xbdedaf78,
    ],
];

const REBLOCK_16: [[u32; 10]; 8] = [
    [
        0xbda6dfbc, 0xbd05bbc0, 0x3e82d043, 0xbdb5afa0, 0x3e9c830e, 0xbe835188, 0xbe54a224,
        0x3da4daa7, 0xbe870581, 0x3e05d2ea,
    ],
    [
        0x3cfdf618, 0x3e65fb79, 0xbe9bab84, 0x3f0536a6, 0xbf3a1e53, 0xbe137470, 0x3d6b2119,
        0xbee1451e, 0x3ef0d3b3, 0xbf46eb20,
    ],
    [
        0xbefc0460, 0x3e9be7a6, 0xbf0a088a, 0x3d53bbcc, 0xbe133734, 0xbf0acefd, 0x3e824e0d,
        0xbf16d557, 0x3adde000, 0xbe466a67,
    ],
    [
        0x3f540673, 0x3e10ae39, 0x3eadfa1b, 0x3e761ce0, 0x3f87cdda, 0x3f4739a7, 0x3dc15c72,
        0x3e9c6081, 0xbc1e3200, 0x3f4f9bb4,
    ],
    [
        0x3f8e5c08, 0xbd56fbc0, 0x3f15dc6c, 0x3eae79b8, 0xbf1caf64, 0x3fd1f8dc, 0xbe5510b7,
        0x3f240cac, 0x3da620a0, 0xbd513b00,
    ],
    [
        0xbe6e6f60, 0xbf1e4688, 0x3d1a7158, 0x3e800000, 0x3e800000, 0xbe6e6f60, 0xbf224688,
        0xbd258ea8, 0x00000000, 0x00000000,
    ],
    [
        0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x00000000,
    ],
    [
        0x3e800000, 0x3e758d67, 0x3e162d4c, 0x3eb68f28, 0xbe0f59d0, 0x00000000, 0xbb1ca640,
        0xbd83a568, 0x3e6d1e50, 0xbe87ace8,
    ],
];
