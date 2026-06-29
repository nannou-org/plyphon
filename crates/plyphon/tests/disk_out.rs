//! `DiskOut`: streams audio from the audio thread out to a cued recording buffer, drained off-RT
//! through a [`StreamConsumer`]. Confirms the recorded signal crosses the RT→NRT ring intact - a
//! clean tone for a sine source, and exact interleaved samples (channel mapping + ordering) for a
//! multi-channel DC source. Exercises the `buffer_cue_write` / `recording_at_mut` write seam.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, StreamConsumer, SynthDef, UnitSpec, World,
    engine,
};

const SR: f32 = 48_000.0;
/// Frames per recording chunk; a multiple of the 64-sample control block so block-aligned fills land
/// on chunk boundaries and every recorded frame is flushed once `frames` is a multiple of it.
const CHUNK_FRAMES: usize = 256;

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR).floor();
    let w = 2.0 * std::f32::consts::PI * k / n as f32;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f32, 0.0f32);
    for &x in samples {
        let s = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt() / n as f32
}

/// Render `frames` of audio in block-aligned 64-sample fills, draining the recorder into a `Vec`
/// after each block (and recycling the emptied chunks) so the bounded chunk queue never overruns.
/// Returns the recorded interleaved samples.
fn record(world: &mut World, consumer: &mut StreamConsumer, frames: usize) -> Vec<f32> {
    let mut recorded = Vec::new();
    let mut blk = [0.0f32; 64];
    let mut produced = 0;
    while produced < frames {
        world.fill(&mut blk, 1);
        produced += blk.len();
        while let Some(chunk) = consumer.pop_filled() {
            recorded.extend_from_slice(chunk.filled_samples());
            consumer.recycle(chunk);
        }
    }
    recorded
}

/// `SinOsc.ar(freq) -> DiskOut.ar(bufnum=0, [SinOsc])`. DiskOut's one output (a running frame count)
/// is left unconnected, as scsynth allows.
fn tone_def(name: &str, freq: f32) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "DiskOut",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),               // bufnum
                    InputRef::Unit { unit: 0, output: 0 }, // ch0
                ],
                1,
            ),
        ],
    }
}

#[test]
fn disk_out_records_a_tone() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut consumer = controller
        .buffer_cue_write(0, 1, SR as f64, CHUNK_FRAMES, 4)
        .unwrap();
    controller.add_synthdef(tone_def("rec", 440.0));
    controller
        .synth_new("rec", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // 4608 = 18 whole chunks of 256: a clean tone with every frame flushed.
    let recorded = record(&mut world, &mut consumer, 4608);
    assert_eq!(
        recorded.len(),
        4608,
        "every recorded frame should be drained"
    );
    assert!(
        recorded.iter().any(|s| s.abs() > 0.1),
        "the recording was silent"
    );
    assert!(
        goertzel(&recorded, 440.0) > 5.0 * goertzel(&recorded, 880.0),
        "expected the recorded 440 Hz tone to cross the ring intact"
    );
}

#[test]
fn disk_out_preserves_channels_and_samples_exactly() {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    let mut consumer = controller
        .buffer_cue_write(0, 2, SR as f64, CHUNK_FRAMES, 4)
        .unwrap();
    // DC.ar(0.25) -> ch0, DC.ar(0.75) -> ch1, both into DiskOut(bufnum=0, [ch0, ch1]).
    controller.add_synthdef(SynthDef {
        name: "rec2".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.25)], 1),
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(0.75)], 1),
            UnitSpec::new(
                "DiskOut",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),               // bufnum
                    InputRef::Unit { unit: 0, output: 0 }, // ch0
                    InputRef::Unit { unit: 1, output: 0 }, // ch1
                ],
                1,
            ),
        ],
    });
    controller
        .synth_new("rec2", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let recorded = record(&mut world, &mut consumer, 512);
    assert_eq!(
        recorded.len(),
        512 * 2,
        "expected 512 interleaved stereo frames"
    );
    for (i, &s) in recorded.iter().enumerate() {
        let expected = if i % 2 == 0 { 0.25 } else { 0.75 };
        assert!(
            (s - expected).abs() < 1e-6,
            "interleaved sample {i}: got {s}, expected {expected}"
        );
    }
}
