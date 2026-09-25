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
