//! The World's random streams (scsynth's `world->mRGen`): every synth draws from one of them -
//! stream 0 unless `RandID` selects another - so synths share them, and a `RandSeed` restarts a
//! stream for every synth on it. Stream `i` is seeded as scsynth's `RGen::init(i)`, so each
//! expected value below is the float bit pattern scsynth's own `RGen` draws.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

/// Samples per control block at the default engine options.
const BLOCK: usize = 64;

/// `frand()` draws from scsynth's `RGen` after `init(seed)`.
const INIT_0: [u32; 2] = [0x3f5b_887e, 0x3d94_8b20];
const INIT_3: u32 = 0x3f5c_55c6;
const INIT_42: u32 = 0x3ecb_6f74;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// `Rand.ir(0, 1)`.
fn rand() -> UnitSpec {
    UnitSpec::new("Rand", Rate::Scalar, vec![c(0.0), c(1.0)], 1)
}

/// `units`, then unit `src` held into `Out.ar(bus)` by `K2A`.
fn to_bus(mut units: Vec<UnitSpec>, src: u32, bus: f32) -> Vec<UnitSpec> {
    let k2a = units.len() as u32;
    units.push(UnitSpec::new("K2A", Rate::Audio, vec![u(src)], 1));
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(bus), u(k2a)], 0));
    units
}

/// Start one synth per def, in order, and render one block of two output channels: the first
/// sample of each channel.
fn render(options: Options, defs: Vec<Vec<UnitSpec>>) -> [f32; 2] {
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 2,
        ..options
    });
    for (i, units) in defs.into_iter().enumerate() {
        let name = format!("t{i}");
        controller.add_synthdef(SynthDef {
            name: name.clone(),
            params: vec![],
            units,
        });
        controller
            .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail)
            .expect("synth_new");
    }
    let mut buf = vec![0.0f32; BLOCK * 2];
    world.fill(&mut buf, 2);
    [buf[0], buf[1]]
}

#[test]
fn synths_share_stream_zero() {
    // Both synths draw from stream 0, in node order: the second gets the stream's second value.
    let [a, b] = render(
        Options::default(),
        vec![to_bus(vec![rand()], 0, 0.0), to_bus(vec![rand()], 0, 1.0)],
    );
    assert_eq!([a.to_bits(), b.to_bits()], INIT_0);
}

#[test]
fn rand_id_selects_a_world_stream() {
    let rand_id = |id: f32| UnitSpec::new("RandID", Rate::Scalar, vec![c(id)], 1);
    let [a, _] = render(
        Options::default(),
        vec![to_bus(vec![rand_id(3.0), rand()], 1, 0.0)],
    );
    assert_eq!(a.to_bits(), INIT_3, "RandID(3) draws from stream 3");

    // A stream the World does not have selects nothing (scsynth's `iid < mNumRGens` check).
    let [a, _] = render(
        Options {
            num_rgens: 3,
            ..Options::default()
        },
        vec![to_bus(vec![rand_id(3.0), rand()], 1, 0.0)],
    );
    assert_eq!(
        a.to_bits(),
        INIT_0[0],
        "RandID(3) of 3 streams keeps stream 0"
    );
}

#[test]
fn rand_seed_restarts_the_stream_for_every_synth_on_it() {
    // The first synth only re-seeds stream 0; the second synth's `Rand` then draws the first value
    // of the re-seeded stream.
    let seed = UnitSpec::new("RandSeed", Rate::Scalar, vec![c(1.0), c(42.0)], 1);
    let [_, b] = render(
        Options::default(),
        vec![to_bus(vec![seed], 0, 0.0), to_bus(vec![rand()], 0, 1.0)],
    );
    assert_eq!(b.to_bits(), INIT_42);
}

#[test]
fn an_audio_rate_rand_seed_seeds_in_its_constructor() {
    // `RandSeed_Ctor` runs its calc for one sample, so an `Impulse.ar` trigger - high on its first
    // sample - re-seeds before the `Rand` after it draws.
    let impulse = UnitSpec::new("Impulse", Rate::Audio, vec![c(0.0), c(0.0)], 1);
    let seed = UnitSpec::new("RandSeed", Rate::Audio, vec![u(0), c(42.0)], 1);
    let [a, _] = render(
        Options::default(),
        vec![to_bus(vec![impulse, seed, rand()], 2, 0.0)],
    );
    assert_eq!(a.to_bits(), INIT_42);
}

/// Render one block of the audio-rate unit `units[src]` on output bus 0.
fn block_of(units: Vec<UnitSpec>, src: u32) -> Vec<f32> {
    let mut units = units;
    units.push(UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0));
    let (mut controller, _nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK];
    world.fill(&mut buf, 1);
    buf
}

fn bits(samples: &[f32]) -> Vec<u32> {
    samples.iter().map(|s| s.to_bits()).collect()
}

#[test]
fn white_noise_draws_from_the_synths_stream() {
    // `WhiteNoise_Ctor` draws one sample, so the first block starts at the stream's second value.
    let white = UnitSpec::new("WhiteNoise", Rate::Audio, vec![], 1);
    let buf = block_of(vec![white.clone()], 0);
    assert_eq!(bits(&buf[..3]), [0xbf5a_dd38, 0xbf31_31c8, 0x3da0_d840]);

    // A `RandSeed` before it restarts the stream the noise draws from.
    let seed = UnitSpec::new("RandSeed", Rate::Scalar, vec![c(1.0), c(42.0)], 1);
    let buf = block_of(vec![seed, white], 1);
    assert_eq!(bits(&buf[..3]), [0x3e47_45b0, 0xbf22_3a44, 0x3f1a_eb28]);
}

#[test]
fn brown_noise_draws_its_starting_level_in_its_constructor() {
    // `BrownNoise_Ctor` draws `mLevel = frand2()`; the block steps it by `frand8` from there.
    let brown = UnitSpec::new("BrownNoise", Rate::Audio, vec![], 1);
    let buf = block_of(vec![brown], 0);
    assert_eq!(bits(&buf[..3]), [0x3f1b_b555, 0x3f05_8f1c, 0x3f08_127d]);
}

/// `Duty.kr(1, 0, level)`, with `level` the demand unit at `level`.
fn duty(level: u32) -> UnitSpec {
    UnitSpec::new(
        "Duty",
        Rate::Control,
        vec![c(1.0), c(0.0), c(0.0), u(level)],
        1,
    )
}

#[test]
fn demand_randoms_draw_from_the_synths_stream() {
    // `Duty_Ctor` pulls the first level, so `Dwhite(inf, 0, 1)` gives the stream's first value.
    let dwhite = UnitSpec::new(
        "Dwhite",
        Rate::Demand,
        vec![c(f32::INFINITY), c(0.0), c(1.0)],
        1,
    );
    let [a, _] = render(
        Options::default(),
        vec![to_bus(vec![dwhite, duty(0)], 1, 0.0)],
    );
    assert_eq!(a.to_bits(), INIT_0[0]);

    // `Drand` draws on its constructor reset and after every pick, even with one item to choose:
    // its reset, then `Duty`'s constructor pull, then the `Rand` - the stream's third value.
    let drand = UnitSpec::new("Drand", Rate::Demand, vec![c(f32::INFINITY), c(5.0)], 1);
    let [a, _] = render(
        Options::default(),
        vec![to_bus(vec![drand, duty(0), rand()], 2, 0.0)],
    );
    assert_eq!(a.to_bits(), 0x3e1d_9c70);
}

#[test]
fn audio_rate_texprand_redraws_on_every_high_sample() {
    // scsynth's `TExpRand_next_a` compares each sample with the trigger before the block, not the
    // previous sample, so once a block that starts low goes high it redraws on every sample.
    // `TRand_next_a` compares with the previous sample and draws once.
    let ramp = UnitSpec::new(
        "Line",
        Rate::Audio,
        vec![c(-1.0), c(1.0), c(BLOCK as f32 / 48_000.0)],
        1,
    );
    let t_rand = |name: &str| UnitSpec::new(name, Rate::Audio, vec![c(1.0), c(2.0), u(0)], 1);
    let distinct = |buf: &[f32]| {
        let high = &buf[BLOCK / 2 + 1..];
        high.windows(2).filter(|w| w[0] != w[1]).count()
    };
    let exp = block_of(vec![ramp.clone(), t_rand("TExpRand")], 1);
    assert_eq!(
        distinct(&exp),
        BLOCK / 2 - 2,
        "TExpRand redraws every high sample"
    );
    let uniform = block_of(vec![ramp, t_rand("TRand")], 1);
    assert_eq!(distinct(&uniform), 0, "TRand draws once on the rising edge");
}
