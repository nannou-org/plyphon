//! Spectral (`PV_*`) operators inserted into an FFT -> PV -> IFFT chain: `PV_MagAbove` gates the whole
//! spectrum away above a huge threshold (and passes at threshold 0), and `PV_BrickWall` high/low-passes
//! by zeroing a fraction of the bins. Requires the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const FFT_SIZE: usize = 1024;

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 128, 512, 256];
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let mut i = 0;
    while out.len() < frames {
        buf.clear();
        buf.resize(sizes[i % sizes.len()], 0.0);
        i += 1;
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

fn rms(s: &[f32]) -> f32 {
    (s.iter().map(|x| x * x).sum::<f32>() / s.len().max(1) as f32).sqrt()
}

/// `SinOsc(freq)*0.5 -> FFT(buf 0) -> pv(fbufnum, ..extra) -> IFFT -> Out`, rendered; returns the
/// steady-state tail's RMS.
fn tail_rms(freq: f32, pv: &str, extra: Vec<InputRef>) -> f32 {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(vec![0.0; FFT_SIZE], 1, SR)),
        )
        .unwrap();

    let mut pv_inputs = vec![InputRef::Unit { unit: 2, output: 0 }];
    pv_inputs.extend(extra);
    controller.add_synthdef(SynthDef {
        name: "pvc".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.5),
                ],
                num_outputs: 1,
                special_index: 2,
            },
            // 2: FFT(buf 0, in, 0.5, 0, 1, FFT_SIZE).
            UnitSpec::new(
                "FFT",
                Rate::Control,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.5),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(FFT_SIZE as f32),
                ],
                1,
            ),
            // 3: the PV op under test.
            UnitSpec::new(pv, Rate::Control, pv_inputs, 1),
            // 4: IFFT(fbufnum, 0, FFT_SIZE).
            UnitSpec::new(
                "IFFT",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 3, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(FFT_SIZE as f32),
                ],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 4, output: 0 },
                ],
                0,
            ),
        ],
    });
    controller
        .synth_new("pvc", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, 12_288);
    rms(&out[8_192..])
}

#[test]
fn pv_mag_above_gates_the_spectrum() {
    // A bin-aligned tone (20 bins).
    let freq = 20.0 * (SR as f32 / FFT_SIZE as f32);
    // thresh 0: nothing is below 0, so every bin passes -> the tone survives.
    let passed = tail_rms(freq, "PV_MagAbove", vec![InputRef::Constant(0.0)]);
    // A huge threshold zeroes every bin (all magnitudes are below it) -> silence.
    let gated = tail_rms(freq, "PV_MagAbove", vec![InputRef::Constant(1.0e9)]);
    assert!(
        passed > 0.05,
        "PV_MagAbove(0) should pass the tone, rms {passed}"
    );
    assert!(
        gated < 0.01,
        "PV_MagAbove(huge) should gate everything, rms {gated}"
    );
}

#[test]
fn pv_brick_wall_high_and_low_passes() {
    // A low-ish tone (bin 20). BrickWall(+0.5) zeroes the lower half of bins (high-pass) -> removed;
    // BrickWall(-0.5) zeroes the upper half (low-pass) -> the low tone survives.
    let freq = 20.0 * (SR as f32 / FFT_SIZE as f32);
    let highpassed = tail_rms(freq, "PV_BrickWall", vec![InputRef::Constant(0.5)]);
    let lowpassed = tail_rms(freq, "PV_BrickWall", vec![InputRef::Constant(-0.5)]);
    assert!(
        lowpassed > 0.05,
        "a low-pass should keep the low tone, rms {lowpassed}"
    );
    assert!(
        highpassed < 0.2 * lowpassed,
        "a high-pass should remove the low tone (high={highpassed}, low={lowpassed})"
    );
}
