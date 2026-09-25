//! `PSinGrain`: each expected value is the bit pattern scsynth's `PSinGrain_Ctor`/`PSinGrain_next`
//! produce at 48 kHz with 64-sample blocks for `PSinGrain.ar(1000, 0.003, 0.5)`: a 144-sample
//! grain that ends 16 samples into the third block, which frees the synth.

use plyphon::{
    AddAction, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const BLOCK: usize = 64;

#[test]
fn grain_matches_scsynth_and_frees_its_synth() {
    let c = InputRef::Constant;
    let (mut controller, mut nrt, mut world) = engine(Options {
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "grain".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "PSinGrain",
                Rate::Audio,
                vec![c(1000.0), c(0.003), c(0.5)],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![c(0.0), InputRef::Unit { unit: 0, output: 0 }],
                0,
            ),
        ],
    });
    let node = controller
        .synth_new("grain", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    let mut buf = vec![0.0f32; BLOCK * 2];
    world.fill(&mut buf, 1);
    nrt.process();
    while let Some(event) = nrt.poll() {
        assert!(
            !matches!(event, Event::NodeEnded(n) if n.node == node),
            "the grain is still sounding after two blocks"
        );
    }
    let mut third = vec![0.0f32; BLOCK];
    world.fill(&mut third, 1);
    buf.extend_from_slice(&third);

    let picks = [0, 1, 2, 3, 62, 63, 64, 100, 127, 128, 143, 144, 150];
    let got: Vec<u32> = picks.iter().map(|&i| buf[i].to_bits()).collect();
    assert_eq!(
        got,
        [
            0x80000000, 0x3aebf72a, 0x3be84f91, 0x3c7fcd77, 0x3ef28216, 0x3ee8d1b6, 0x3edaf7a6,
            0x3e5946e8, 0xbe292af7, 0xbe2f2bab, 0xbaec07f3, 0x00000000, 0x00000000,
        ]
    );
    assert!(
        buf[144..].iter().all(|&s| s == 0.0),
        "silent after the grain"
    );

    nrt.process();
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        ended |= matches!(event, Event::NodeEnded(n) if n.node == node);
    }
    assert!(ended, "the block the grain ends in frees the synth");
}
