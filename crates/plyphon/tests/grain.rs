//! Granular synthesis: `GrainSin` spawns windowed sine grains on a trigger and pans them across the
//! output channels.

use plyphon::{
    AddAction, Buffer, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f32 = 48_000.0;

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

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

/// Render `units` (interleaved) into `channels` output channels.
fn render(units: Vec<UnitSpec>, channels: usize, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "g".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("g", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; frames * channels];
    world.fill(&mut out, channels);
    out
}

/// Render `units` into `channels` output channels with a mono sine buffer installed at `bufnum`.
fn render_with_buffer(
    units: Vec<UnitSpec>,
    channels: usize,
    frames: usize,
    bufnum: usize,
    buf: Buffer,
) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: channels,
        ..Options::default()
    });
    controller
        .buffer_set(bufnum, Box::new(buf))
        .expect("buffer_set");
    controller.add_synthdef(SynthDef {
        name: "g".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("g", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; frames * channels];
    world.fill(&mut out, channels);
    out
}

/// A mono buffer of `cycles` whole sine cycles over `frames` samples - seamless to loop.
fn sine_buffer(frames: usize, cycles: usize) -> Buffer {
    let samples: Vec<f32> = (0..frames)
        .map(|i| (std::f32::consts::TAU * cycles as f32 * i as f32 / frames as f32).sin())
        .collect();
    Buffer::from_interleaved(samples, 1, SR as f64)
}

fn impulse(freq: f32) -> UnitSpec {
    UnitSpec::new("Impulse", Rate::Audio, vec![c(freq), c(0.0)], 1)
}

/// `GrainSin(numChannels, Impulse(unit 0), dur, freq, pan, envbufnum=-1, maxGrains=32)`.
fn grain_sin(num_ch: usize, freq: f32, pan: f32) -> UnitSpec {
    UnitSpec::new(
        "GrainSin",
        Rate::Audio,
        vec![u(0), c(0.03), c(freq), c(pan), c(-1.0), c(32.0)],
        num_ch,
    )
}

#[test]
fn grain_sin_rings_at_the_grain_frequency() {
    // A 20 Hz impulse train triggers 440 Hz windowed sine grains.
    let out = render(
        vec![
            impulse(20.0),
            grain_sin(1, 440.0, 0.0),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 4,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "GrainSin must stay finite"
    );
    assert!(
        out.iter().all(|&s| s.abs() < 3.0),
        "windowed grains stay bounded"
    );
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 660.0),
        "grains should ring at 440 (440={at})"
    );
    // The grains actually sound (they are not all silent).
    assert!(at > 0.02, "grains should be audible (440={at})");
}

/// Per-channel energy at 440 Hz for a stereo `GrainSin` panned by `pan`.
fn stereo_energy(pan: f32) -> (f32, f32) {
    let out = render(
        vec![
            impulse(20.0),
            grain_sin(2, 440.0, pan),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    c(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 1, output: 1 },
                ],
                0,
            ),
        ],
        2,
        SR as usize / 4,
    );
    let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
    let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
    (goertzel(&ch0, 440.0), goertzel(&ch1, 440.0))
}

#[test]
fn grain_sin_pans_across_channels() {
    // pan = -1 puts the grains in channel 0; pan = +1 in channel 1.
    let (l0, l1) = stereo_energy(-1.0);
    let (r0, r1) = stereo_energy(1.0);
    assert!(
        l0 > 5.0 * l1,
        "pan -1 favours channel 0 (ch0={l0}, ch1={l1})"
    );
    assert!(
        r1 > 5.0 * r0,
        "pan +1 favours channel 1 (ch0={r0}, ch1={r1})"
    );
}

#[test]
fn grain_fm_has_carrier_and_sidebands() {
    // A long FM grain (carrier 440, modulator 110, index 2) resolves the full FM sideband series at
    // 440 +/- k*110; at index 2 the first sidebands (330, 550) exceed the carrier (Bessel J1 > J0).
    let out = render(
        vec![
            impulse(1.0), // a single, long grain over the render
            UnitSpec::new(
                "GrainFM",
                Rate::Audio,
                vec![
                    u(0),
                    c(0.2),
                    c(440.0),
                    c(110.0),
                    c(2.0),
                    c(0.0),
                    c(-1.0),
                    c(32.0),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 4,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "GrainFM must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 3.0), "GrainFM stays bounded");
    // 495 Hz sits between the 440 and 550 components (an ~empty bin).
    let off = goertzel(&out, 495.0);
    let carrier = goertzel(&out, 440.0);
    let side = goertzel(&out, 550.0);
    assert!(
        carrier > 10.0 * off,
        "the carrier should sound (440={carrier}, 495={off})"
    );
    assert!(
        side > 10.0 * off,
        "the 550 Hz FM sideband should appear (550={side}, 495={off})"
    );
    assert!(
        side > carrier,
        "at index 2 the first sideband exceeds the carrier (550={side}, 440={carrier})"
    );
}

#[test]
fn grain_in_windows_the_input() {
    // GrainIn windows a live 440 Hz sine, so its grains ring at 440.
    let out = render(
        vec![
            impulse(20.0),
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            UnitSpec::new(
                "GrainIn",
                Rate::Audio,
                vec![u(0), c(0.03), u(1), c(0.0), c(-1.0), c(32.0)],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(2)], 0),
        ],
        1,
        SR as usize / 4,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "GrainIn must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 3.0), "GrainIn stays bounded");
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 660.0),
        "windows the 440 input (440={at})"
    );
    assert!(at > 0.02, "the windowed grains should sound");
}

#[test]
fn grain_buf_plays_windowed_buffer_grains() {
    // A 4800-frame buffer holding 44 seamless cycles is a 440 Hz loop; GrainBuf reads it at rate 1
    // from pos 0, so its grains ring at 440.
    let out = render_with_buffer(
        vec![
            impulse(20.0),
            UnitSpec::new(
                "GrainBuf",
                Rate::Audio,
                vec![
                    u(0),    // trigger
                    c(0.03), // dur
                    c(0.0),  // sndbuf
                    c(1.0),  // rate
                    c(0.0),  // pos
                    c(2.0),  // interp (linear)
                    c(0.0),  // pan
                    c(-1.0), // envbufnum (default window)
                    c(32.0), // maxGrains
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 4,
        0,
        sine_buffer(4800, 44),
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "GrainBuf must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 3.0), "GrainBuf stays bounded");
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 660.0),
        "buffer grains ring at 440 (440={at})"
    );
    assert!(at > 0.02, "buffer grains should sound (440={at})");
}

#[test]
fn grain_buf_rate_transposes() {
    // Reading the 440 Hz buffer at rate 2 transposes the grains up an octave to 880.
    let out = render_with_buffer(
        vec![
            impulse(20.0),
            UnitSpec::new(
                "GrainBuf",
                Rate::Audio,
                vec![
                    u(0),
                    c(0.03),
                    c(0.0),
                    c(2.0), // rate 2 -> +1 octave
                    c(0.0),
                    c(2.0),
                    c(0.0),
                    c(-1.0),
                    c(32.0),
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 4,
        0,
        sine_buffer(4800, 44),
    );
    let up = goertzel(&out, 880.0);
    assert!(
        up > 8.0 * goertzel(&out, 440.0),
        "rate 2 transposes to 880 (880={up})"
    );
}

#[test]
fn tgrains_centres_buffer_grains() {
    // TGrains reads the same 440 Hz buffer, always with the default sin^2 window, amp folded into the
    // gain. centerPos in seconds; rate 1 keeps the pitch at 440.
    let out = render_with_buffer(
        vec![
            impulse(20.0),
            UnitSpec::new(
                "TGrains",
                Rate::Audio,
                vec![
                    u(0),    // trigger
                    c(0.0),  // bufnum
                    c(1.0),  // rate
                    c(0.05), // centerPos (seconds)
                    c(0.03), // dur
                    c(0.0),  // pan
                    c(0.5),  // amp
                    c(2.0),  // interp (linear)
                ],
                1,
            ),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(1)], 0),
        ],
        1,
        SR as usize / 4,
        0,
        sine_buffer(4800, 44),
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "TGrains must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 3.0), "TGrains stays bounded");
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 660.0),
        "TGrains rings at 440 (440={at})"
    );
    assert!(at > 0.02, "TGrains should sound (440={at})");
}

/// `Warp1(numChannels, bufnum, pointer, freqScale, windowSize, envbufnum, overlaps, windowRandRatio,
/// interp)` reading the 440 Hz buffer at `freq_scale`.
fn warp1(num_ch: usize, freq_scale: f32, window_rand: f32) -> UnitSpec {
    UnitSpec::new(
        "Warp1",
        Rate::Audio,
        vec![
            c(0.0),         // bufnum
            c(0.2),         // pointer (into the buffer)
            c(freq_scale),  // freqScale
            c(0.1),         // windowSize (seconds)
            c(-1.0),        // envbufnum (default window)
            c(8.0),         // overlaps
            c(window_rand), // windowRandRatio
            c(2.0),         // interp (linear)
        ],
        num_ch,
    )
}

#[test]
fn warp1_reads_the_buffer_at_pitch() {
    // Self-triggering grains read the seamless 440 Hz buffer at freqScale 1, so the output rings at
    // 440; freqScale 2 transposes it up an octave to 880.
    let out = render_with_buffer(
        vec![
            warp1(1, 1.0, 0.0),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
        1,
        SR as usize / 4,
        0,
        sine_buffer(4800, 44),
    );
    assert!(out.iter().all(|s| s.is_finite()), "Warp1 must stay finite");
    assert!(out.iter().all(|&s| s.abs() < 3.0), "Warp1 stays bounded");
    let at = goertzel(&out, 440.0);
    assert!(
        at > 8.0 * goertzel(&out, 660.0),
        "Warp1 rings at 440 (440={at})"
    );
    assert!(at > 0.02, "Warp1 should sound (440={at})");

    let up = render_with_buffer(
        vec![
            warp1(1, 2.0, 0.0),
            UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(0)], 0),
        ],
        1,
        SR as usize / 4,
        0,
        sine_buffer(4800, 44),
    );
    assert!(
        goertzel(&up, 880.0) > 8.0 * goertzel(&up, 440.0),
        "freqScale 2 transposes to 880 (880={})",
        goertzel(&up, 880.0)
    );
}

#[test]
fn warp1_spreads_channels_and_randomises_windows() {
    // Two Warp1 channels run independent grain clouds; with windowRandRatio > 0 the per-channel window
    // sizes decorrelate, so the channels are not identical, yet both still carry the 440 Hz tone.
    let out = render_with_buffer(
        vec![
            warp1(2, 1.0, 0.5),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    c(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 0, output: 1 },
                ],
                0,
            ),
        ],
        2,
        SR as usize / 4,
        0,
        sine_buffer(4800, 44),
    );
    assert!(out.iter().all(|s| s.is_finite()), "Warp1 must stay finite");
    let ch0: Vec<f32> = out.iter().step_by(2).copied().collect();
    let ch1: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
    assert!(
        goertzel(&ch0, 440.0) > 0.02 && goertzel(&ch1, 440.0) > 0.02,
        "both channels carry the 440 tone"
    );
    // The channels are decorrelated (independent random window sizes), so they are not sample-identical.
    assert!(
        ch0.iter().zip(&ch1).any(|(a, b)| (a - b).abs() > 1e-4),
        "the two grain clouds should differ"
    );
}
