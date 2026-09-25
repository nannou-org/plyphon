//! The FFT/IFFT spectral chain: a sine analyzed by `FFT` into a packed-spectrum buffer and
//! resynthesized by `IFFT` should reconstruct the sine (after the analysis latency). Exercises the
//! `FftTables` resource threaded into `ProcessCtx` and the per-unit `aux` analysis/resynthesis rings.
//!
//! Requires the default `fft` feature (the FFT units are gated on it).

use plyphon::{
    AddAction, Buffer, Event, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World,
    engine,
};

const SR: f64 = 48_000.0;
const FFT_SIZE: usize = 1024;

/// Render `frames` of mono audio across a few block sizes (exercising reblocking and cross-block
/// FFT/IFFT state).
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    // FFT/IFFT alignment wants whole control blocks per hop; keep host buffers block-multiples.
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

/// Goertzel magnitude of `freq` in `samples` - a single-bin DTFT for cheap pitch checks.
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

/// `SinOsc.ar(freq) * amp -> FFT(buf, _, 0.5, 0, 1, FFT_SIZE) -> IFFT(_, 0, FFT_SIZE) -> Out`.
fn chain_def(freq: f32, amp: f32) -> SynthDef {
    chain_def_winsize(freq, amp, FFT_SIZE as f32)
}

/// [`chain_def`] with an explicit `winsize` (0 = "use the chain buffer's size", sclang's default).
fn chain_def_winsize(freq: f32, amp: f32, winsize: f32) -> SynthDef {
    SynthDef {
        name: "fft-chain".to_string(),
        params: vec![],
        units: vec![
            // 0: SinOsc.ar(freq).
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            // 1: SinOsc * amp.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(amp),
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            // 2: FFT(buffer 0, in, hop 0.5, wintype 0, active 1, winsize FFT_SIZE) - control-rate fbufnum.
            UnitSpec::new(
                "FFT",
                Rate::Control,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.5),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(winsize),
                ],
                1,
            ),
            // 3: IFFT(fbufnum, wintype 0, winsize FFT_SIZE) - audio-rate resynthesis.
            UnitSpec::new(
                "IFFT",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(winsize),
                ],
                1,
            ),
            // 4: Out.ar(0, resynthesis).
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 3, output: 0 },
                ],
                0,
            ),
        ],
    }
}

fn start(def: SynthDef) -> (plyphon::Controller, World) {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    // The packed-spectrum chain buffer: FFT_SIZE mono frames.
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(vec![0.0; FFT_SIZE], 1, SR)),
        )
        .unwrap();
    controller.add_synthdef(def);
    controller
        .synth_new("fft-chain", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    (controller, world)
}

#[test]
fn fft_ifft_reconstructs_a_sine() {
    // A bin-aligned tone (20 bins = 937.5 Hz) so the analysis is exact; amplitude 0.5.
    let bin = SR as f32 / FFT_SIZE as f32; // 46.875 Hz
    let freq = 20.0 * bin;
    let amp = 0.5;
    let (_c, mut world) = start(chain_def(freq, amp));

    // Render past the analysis/overlap-add latency, then analyze the steady-state tail.
    let out = render(&mut world, 12_288);
    let tail = &out[8_192..];

    assert!(
        tail.iter().any(|s| s.abs() > 0.05),
        "resynthesis was silent"
    );
    assert!(
        tail.iter().all(|s| s.abs() <= 1.0),
        "resynthesis left [-1, 1]"
    );
    // The reconstructed tone dominates its neighbours and is near the input amplitude.
    let at = goertzel(tail, freq);
    let off = goertzel(tail, freq * 2.0);
    assert!(
        at > 8.0 * off,
        "expected {freq} Hz to dominate (at={at:.4}, off={off:.4})"
    );
    assert!(
        (0.1..1.0).contains(&at),
        "reconstruction amplitude {at:.4} unreasonable for input amp {amp}"
    );
}

#[test]
fn fft_winsize_zero_resolves_from_the_chain_buffer() {
    // sclang's FFT default passes winsize = 0 ("use the buffer size"); the chain must build and,
    // once the FFT_SIZE chain buffer is installed, reconstruct the tone exactly as the explicit
    // winsize does.
    let bin = SR as f32 / FFT_SIZE as f32;
    let freq = 20.0 * bin;
    let (_c, mut world) = start(chain_def_winsize(freq, 0.5, 0.0));

    let out = render(&mut world, 12_288);
    let tail = &out[8_192..];
    assert!(
        tail.iter().any(|s| s.abs() > 0.05),
        "winsize-0 resynthesis was silent"
    );
    let at = goertzel(tail, freq);
    let off = goertzel(tail, freq * 2.0);
    assert!(
        at > 8.0 * off,
        "expected {freq} Hz to dominate (at={at:.4}, off={off:.4})"
    );
}

/// `FFT(buffer 0, SinOsc.ar(440), 0.5, 0, 1, winsize)` written to the output through `K2A`, so the
/// test reads the chain signal: the buffer number on a frame, `-1` otherwise.
fn fft_signal_def(winsize: f32, watch_done: bool) -> SynthDef {
    let mut units = vec![
        UnitSpec::new(
            "SinOsc",
            Rate::Audio,
            vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
            1,
        ),
        UnitSpec::new(
            "FFT",
            Rate::Control,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 0, output: 0 },
                InputRef::Constant(0.5),
                InputRef::Constant(0.0),
                InputRef::Constant(1.0),
                InputRef::Constant(winsize),
            ],
            1,
        ),
        UnitSpec::new(
            "K2A",
            Rate::Audio,
            vec![InputRef::Unit { unit: 1, output: 0 }],
            1,
        ),
        UnitSpec::new(
            "Out",
            Rate::Audio,
            vec![
                InputRef::Constant(0.0),
                InputRef::Unit { unit: 2, output: 0 },
            ],
            0,
        ),
    ];
    if watch_done {
        units.push(UnitSpec::new(
            "FreeSelfWhenDone",
            Rate::Control,
            vec![InputRef::Unit { unit: 1, output: 0 }],
            1,
        ));
    }
    SynthDef {
        name: "fft-signal".to_string(),
        params: vec![],
        units,
    }
}

/// Start [`fft_signal_def`] on an engine with `options`, installing a `buffer_frames` chain buffer
/// before (`buffer_first`) or after the synth starts, and render `frames` of its chain signal.
fn fft_signal(
    options: Options,
    def: SynthDef,
    buffer_frames: usize,
    buffer_first: bool,
    frames: usize,
) -> (Vec<f32>, Vec<Event>) {
    let (mut controller, mut nrt, mut world) = engine(options);
    let chain = || Box::new(Buffer::from_interleaved(vec![0.0; buffer_frames], 1, SR));
    if buffer_first {
        controller.buffer_set(0, chain()).unwrap();
    }
    controller.add_synthdef(def);
    controller
        .synth_new("fft-signal", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    if !buffer_first {
        world.fill(&mut [0.0; 64], 1);
        controller.buffer_set(0, chain()).unwrap();
    }
    let out = render(&mut world, frames);
    let events = std::iter::from_fn(|| nrt.poll()).collect();
    (out, events)
}

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

#[test]
fn fft_frames_carry_the_chain_buffer_number() {
    // The control: with a usable chain buffer the FFT emits buffer 0 on its frames.
    let (out, _) = fft_signal(opts(), fft_signal_def(0.0, false), FFT_SIZE, true, 4096);
    assert!(out.contains(&0.0), "the FFT emits frames");
}

#[test]
fn a_window_other_than_the_chain_buffer_leaves_the_fft_idle() {
    // scsynth zero-pads a window smaller than the buffer; plyphon does not, so the FFT never
    // starts and outputs `-1` (no frame) like scsynth's failed `FFTBase_Ctor`. The same holds for a
    // window that is not a power of two.
    for winsize in [512.0, 1000.0] {
        let (out, _) = fft_signal(opts(), fft_signal_def(winsize, false), FFT_SIZE, true, 4096);
        assert!(
            out[64..].iter().all(|&s| s == -1.0),
            "winsize {winsize}: no frames"
        );
    }
}

#[test]
fn an_fft_started_before_its_chain_buffer_stays_idle() {
    // scsynth's `FFT_Ctor` reads the chain buffer when the synth starts; a buffer installed later
    // is never picked up.
    let (out, _) = fft_signal(opts(), fft_signal_def(0.0, false), FFT_SIZE, false, 4096);
    assert!(out[64..].iter().all(|&s| s == -1.0), "no frames");
}

#[test]
fn a_failed_fft_allocation_outputs_no_frames_and_marks_the_unit_done() {
    // A unit pool too small for the FFT's rings: scsynth's `ClearFFTUnitIfMemFailed` switches the
    // unit to `FFT_ClearUnitOutputs` (`-1`, so no downstream unit sees a frame) and sets its done
    // flag, which `FreeSelfWhenDone` acts on.
    let small = Options {
        unit_pool_bytes: 1024,
        ..opts()
    };
    let (out, _) = fft_signal(small, fft_signal_def(0.0, false), FFT_SIZE, true, 4096);
    assert!(out[64..].iter().all(|&s| s == -1.0), "no frames");

    let (_, events) = fft_signal(small, fft_signal_def(0.0, true), FFT_SIZE, true, 256);
    assert!(
        events.iter().any(|e| matches!(e, Event::NodeEnded(_))),
        "the watcher frees the synth"
    );
}

#[test]
fn fft_ifft_size_themselves_from_a_16384_frame_chain_buffer() {
    // The largest supported size, with `winsize = 0`: each unit allocates exactly what the buffer
    // needs, so no size ceiling applies below the engine's largest FFT plan.
    const BIG: usize = 16_384;
    let freq = 20.0 * SR as f32 / FFT_SIZE as f32; // bin-aligned for both 1024 and 16384
    let (mut controller, _nrt, mut world) = engine(opts());
    controller
        .buffer_set(0, Box::new(Buffer::from_interleaved(vec![0.0; BIG], 1, SR)))
        .unwrap();
    controller.add_synthdef(chain_def_winsize(freq, 0.5, 0.0));
    controller
        .synth_new("fft-chain", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, 4 * BIG);
    let tail = &out[2 * BIG..];
    let at = goertzel(tail, freq);
    let off = goertzel(tail, freq * 2.0);
    assert!(
        at > 8.0 * off && at > 0.05,
        "expected {freq} Hz to dominate (at={at:.4}, off={off:.4})"
    );
}

#[test]
fn pv_magmul_filters_a_tone_through_the_chain() {
    // `FFT(sine) -> PV_MagMul(A, B) -> IFFT`, where B (buffer 1) is a flat magnitude mask. PV_MagMul
    // reads B while rewriting A in place (the two-buffer `buffer_pair_mut` seam), so the chain still
    // resynthesizes the tone. Proves the cross-buffer PV pattern end to end.
    let bin = SR as f32 / FFT_SIZE as f32;
    let freq = 16.0 * bin;
    let def = SynthDef {
        name: "pv-chain".to_string(),
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
            // 2: FFT into buffer 0 (A).
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
            // 3: PV_MagMul(A = fbufnum from FFT, B = buffer 1, a static mask).
            UnitSpec::new(
                "PV_MagMul",
                Rate::Control,
                vec![
                    InputRef::Unit { unit: 2, output: 0 },
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            // 4: IFFT of the filtered buffer.
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
    };

    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    });
    // A: the live FFT chain buffer; B: a flat magnitude mask (all ones) at buffer 1.
    controller
        .buffer_set(
            0,
            Box::new(Buffer::from_interleaved(vec![0.0; FFT_SIZE], 1, SR)),
        )
        .unwrap();
    controller
        .buffer_set(
            1,
            Box::new(Buffer::from_interleaved(vec![1.0; FFT_SIZE], 1, SR)),
        )
        .unwrap();
    controller.add_synthdef(def);
    controller
        .synth_new("pv-chain", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let out = render(&mut world, 12_288);
    let tail = &out[8_192..];
    assert!(tail.iter().any(|s| s.abs() > 0.01), "PV chain was silent");
    assert!(
        tail.iter().all(|s| s.abs() <= 4.0),
        "PV chain output ran away"
    );
    let at = goertzel(tail, freq);
    let off = goertzel(tail, freq * 3.0);
    assert!(
        at > 8.0 * off,
        "PV_MagMul chain should still sound {freq} Hz (at={at:.4}, off={off:.4})"
    );
}

#[test]
fn pv_magsquared_keeps_the_tone_through_the_polar_path() {
    // `FFT(sine) -> PV_MagSquared -> IFFT`. PV_MagSquared converts the frame to polar, squares each
    // bin's magnitude, and leaves the phases; IFFT then converts back to complex (the coord round
    // trip). The tone's bin stays the only non-null one, so the chain still resynthesizes `freq`.
    // This is the first *polar* `PV_*` unit, exercising the shared `pv::to_polar`/`to_complex` seam.
    let bin = SR as f32 / FFT_SIZE as f32;
    let freq = 16.0 * bin;
    let def = SynthDef {
        name: "pv-sq".to_string(),
        params: vec![],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
                1,
            ),
            // A small amplitude keeps the squared magnitudes bounded.
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.02),
                ],
                num_outputs: 1,
                special_index: 2,
            },
            // 2: FFT into buffer 0.
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
            // 3: PV_MagSquared on the frame (the polar conversion).
            UnitSpec::new(
                "PV_MagSquared",
                Rate::Control,
                vec![InputRef::Unit { unit: 2, output: 0 }],
                1,
            ),
            // 4: IFFT (converts back to complex first).
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
    };

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
    controller.add_synthdef(def);
    controller
        .synth_new("pv-sq", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    let out = render(&mut world, 12_288);
    let tail = &out[8_192..];
    assert!(
        tail.iter().all(|s| s.is_finite()),
        "the polar round trip produced non-finite output"
    );
    assert!(
        tail.iter().any(|s| s.abs() > 1e-4),
        "PV_MagSquared chain was silent"
    );
    let at = goertzel(tail, freq);
    let off = goertzel(tail, freq * 3.0);
    assert!(
        at > 8.0 * off,
        "PV_MagSquared chain should still sound {freq} Hz (at={at:.5}, off={off:.5})"
    );
}
