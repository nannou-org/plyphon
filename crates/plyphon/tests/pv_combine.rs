//! Two-buffer spectral ops on two FFT chains: `PV_Add` sums two spectra (both tones survive), and
//! `PV_Copy` overwrites its target buffer with the source (only the source tone survives). Requires
//! the default `fft` feature.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const FFT_SIZE: usize = 1024;
const BIN: f32 = SR as f32 / FFT_SIZE as f32; // 46.875 Hz

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

fn goertzel(samples: &[f32], freq: f32) -> f32 {
    let n = samples.len();
    let k = (0.5 + n as f32 * freq / SR as f32).floor();
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

fn sin(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

/// `<sine unit> * 0.5`, referencing the SinOsc at index `src`.
fn half(src: u32) -> UnitSpec {
    UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![
            InputRef::Unit {
                unit: src,
                output: 0,
            },
            InputRef::Constant(0.5),
        ],
        num_outputs: 1,
        special_index: 2,
    }
}

fn fft(bufnum: f32, in_unit: u32) -> UnitSpec {
    UnitSpec::new(
        "FFT",
        Rate::Control,
        vec![
            InputRef::Constant(bufnum),
            InputRef::Unit {
                unit: in_unit,
                output: 0,
            },
            InputRef::Constant(0.5),
            InputRef::Constant(0.0),
            InputRef::Constant(1.0),
            InputRef::Constant(FFT_SIZE as f32),
        ],
        1,
    )
}

/// Two FFT chains (tone A on buf 0, tone B on buf 1), combined by `pv`, resynthesised. Returns the
/// steady-state tail.
fn combined(pv: &str, freq_a: f32, freq_b: f32) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    for b in 0..2 {
        controller
            .buffer_set(
                b,
                Box::new(Buffer::from_interleaved(vec![0.0; FFT_SIZE], 1, SR)),
            )
            .unwrap();
    }
    // Units: 0,1 = sine A -> 2 = FFT(buf0); 3,4 = sine B -> 5 = FFT(buf1); 6 = pv(fftA, fftB);
    // 7 = IFFT; 8 = Out.
    let units = vec![
        sin(freq_a), // 0
        half(0),     // 1
        fft(0.0, 1), // 2
        sin(freq_b), // 3
        half(3),     // 4
        fft(1.0, 4), // 5
        // 6: pv(fftA=unit2, fftB=unit5).
        UnitSpec::new(
            pv,
            Rate::Control,
            vec![
                InputRef::Unit { unit: 2, output: 0 },
                InputRef::Unit { unit: 5, output: 0 },
            ],
            1,
        ),
        // 7: IFFT(the pv output).
        UnitSpec::new(
            "IFFT",
            Rate::Audio,
            vec![
                InputRef::Unit { unit: 6, output: 0 },
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
                InputRef::Unit { unit: 7, output: 0 },
            ],
            0,
        ),
    ];
    controller.add_synthdef(SynthDef {
        name: "pvc".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("pvc", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, 12_288);
    out[8_192..].to_vec()
}

#[test]
fn pv_add_sums_two_spectra() {
    // Tones at bins 20 and 40; PV_Add keeps both.
    let (fa, fb) = (20.0 * BIN, 40.0 * BIN);
    let out = combined("PV_Add", fa, fb);
    assert!(goertzel(&out, fa) > 0.05, "PV_Add should keep tone A");
    assert!(goertzel(&out, fb) > 0.05, "PV_Add should keep tone B");
}

#[test]
fn pv_max_keeps_the_louder_bins() {
    // Both tones live in different bins, so the per-bin max keeps each.
    let (fa, fb) = (20.0 * BIN, 40.0 * BIN);
    let out = combined("PV_Max", fa, fb);
    assert!(goertzel(&out, fa) > 0.05, "PV_Max should keep tone A");
    assert!(goertzel(&out, fb) > 0.05, "PV_Max should keep tone B");
}

#[test]
fn pv_copy_overwrites_the_target() {
    // PV_Copy copies A (bin 20) into B and continues with B, so only tone A survives; B's own tone
    // (bin 40) is discarded.
    let (fa, fb) = (20.0 * BIN, 40.0 * BIN);
    let out = combined("PV_Copy", fa, fb);
    assert!(
        goertzel(&out, fa) > 5.0 * goertzel(&out, fb),
        "PV_Copy should keep only the source tone"
    );
}
