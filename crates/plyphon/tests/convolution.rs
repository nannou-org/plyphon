//! `Convolution`: two live audio signals convolved frame by frame through the FFT, with overlap-add
//! between frames. These tests pin the two properties the frame machinery is easy to get wrong - the
//! exact latency (`framesize - block_size`, because a block is collected *before* the frame boundary
//! is checked) and the fact that the kernel is re-transformed every frame - plus the frame-size
//! handling when the synth starts.
//!
//! Requires the default `fft` feature (the unit is gated on it).

use plyphon::{
    AddAction, Buffer, BuildError, Event, GraphDef, InputRef, Options, Param, ROOT_GROUP_ID, Rate,
    RateInfo, SynthDef, UnitRegistry, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
/// The World control block. Every rendering test also runs reblocked, which is what makes the
/// block-size dependence of the latency visible.
const WORLD_BLOCK: usize = 64;
/// The frame size the rendering tests convolve at: small enough for short renders, and divisible by
/// both `WORLD_BLOCK` and the reblocked sub-block below.
const FRAMESIZE: usize = 128;
/// The sub-block the reblocked case runs its graph at.
const REBLOCK: usize = 16;
/// Samples rendered per case - eight frames, so several overlap-add boundaries are crossed.
const RENDER: usize = 8 * FRAMESIZE;
/// Where the delayed impulse-train case puts its impulse inside the frame. Not a multiple of either
/// block size, and far enough into the frame that every emitted window straddles a frame boundary.
const OFFSET: usize = 37;
/// The first sample every latency comparison starts from: past the largest lag under test, so each
/// candidate lag is scored over the same window.
const COMPARE_FROM: usize = 2 * FRAMESIZE;

/// The input buffer number; the kernel buffer follows it.
const IN_BUF: f32 = 0.0;
const KERNEL_BUF: f32 = 1.0;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        block_size: WORLD_BLOCK,
        ..Options::default()
    }
}

/// A deterministic, noise-like test signal in `[-0.5, 0.5)`. Neighbouring samples are uncorrelated,
/// so a latency assertion at the wrong lag fails loudly instead of nearly matching.
fn test_signal(len: usize) -> Vec<f32> {
    let mut state = 0x1234_5678u32;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

/// A kernel of `len` samples that is a unit impulse at sample `at` and silent elsewhere. Looped by
/// `PlayBuf`, `len == FRAMESIZE` makes it an impulse *train* whose impulses land at the same offset
/// in every frame; a longer `len` leaves every frame after the first with a silent kernel.
fn impulse_kernel(len: usize, at: usize) -> Vec<f32> {
    let mut k = vec![0.0f32; len];
    k[at] = 1.0;
    k
}

/// `PlayBuf.ar(1, in) -> Convolution.ar(_, PlayBuf.ar(1, kernel), framesize) -> Out.ar(0)`.
///
/// Both `PlayBuf`s run at rate 1 with `loop = 1`, so each reads its buffer sample for sample (the
/// cubic interpolator is exact on integer phases) and wraps at the end.
fn conv_def() -> SynthDef {
    let play = |bufnum: f32| {
        UnitSpec::new(
            "PlayBuf",
            Rate::Audio,
            vec![
                InputRef::Constant(bufnum),
                InputRef::Constant(1.0), // rate
                InputRef::Constant(0.0), // trigger
                InputRef::Constant(0.0), // startPos
                InputRef::Constant(1.0), // loop
                InputRef::Constant(0.0), // doneAction
            ],
            1,
        )
    };
    SynthDef {
        name: "conv".to_string(),
        params: vec![],
        units: vec![
            play(IN_BUF),
            play(KERNEL_BUF),
            UnitSpec::new(
                "Convolution",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(FRAMESIZE as f32),
                ],
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
        ],
    }
}

/// An engine with the input and kernel buffers installed and `conv_def` added, optionally reblocked.
fn start(input: &[f32], kernel: &[f32], reblock: Option<usize>) -> (plyphon::Controller, World) {
    let (mut controller, _nrt, world) = engine(opts());
    controller
        .buffer_set(
            IN_BUF as usize,
            Box::new(Buffer::from_interleaved(input.to_vec(), 1, SR)),
        )
        .unwrap();
    controller
        .buffer_set(
            KERNEL_BUF as usize,
            Box::new(Buffer::from_interleaved(kernel.to_vec(), 1, SR)),
        )
        .unwrap();
    match reblock {
        Some(block) => controller.add_synthdef_reblocked(conv_def(), block),
        None => controller.add_synthdef(conv_def()),
    }
    (controller, world)
}

/// Render `RENDER` samples of `conv_def` over the given input and kernel buffers, optionally
/// reblocking the graph to `reblock` samples.
fn render_conv(input: &[f32], kernel: &[f32], reblock: Option<usize>) -> Vec<f32> {
    let (mut controller, mut world) = start(input, kernel, reblock);
    controller
        .synth_new("conv", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, RENDER)
}

/// As [`render_conv`], but from a second voice that reclaims a freed voice's pool block - so the
/// cold-start clear is exercised over a previous tenant's data rather than over fresh memory.
fn render_recycled(input: &[f32], kernel: &[f32]) -> Vec<f32> {
    let (mut controller, mut world) = start(input, kernel, None);
    let first = controller
        .synth_new("conv", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    // Several frames, so the freed voice leaves real signal in its output and overlap spans.
    render(&mut world, 4 * FRAMESIZE);
    controller.free(first).unwrap();
    // The `PlayBuf`s restart from frame 0, so the new voice sees the same input as a fresh one.
    controller
        .synth_new("conv", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, RENDER)
}

/// Render `frames` of mono audio in one-World-block host buffers.
fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + WORLD_BLOCK);
    let mut buf = vec![0.0f32; WORLD_BLOCK];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// The largest `|out[t] - input[t - lag]|` over the samples past every candidate lag under test, so
/// the same window is compared at every lag.
fn max_err_at_lag(out: &[f32], input: &[f32], lag: usize) -> f32 {
    (COMPARE_FROM..out.len())
        .map(|t| (out[t] - input[t - lag]).abs())
        .fold(0.0f32, f32::max)
}

/// Compile with the built-in registry at the World block, or at `reblock` when given.
fn compile(def: &SynthDef, reblock: Option<usize>) -> Result<GraphDef, BuildError> {
    let rate = RateInfo::new(SR, WORLD_BLOCK);
    def.compile(
        &UnitRegistry::with_builtins(),
        &rate,
        &rate,
        64,
        32,
        reblock,
        1,
    )
}

/// A `Convolution` def with an arbitrary `framesize` input, for the build-time contract tests.
fn framesize_def(framesize: InputRef, params: Vec<Param>) -> SynthDef {
    SynthDef {
        name: "conv-framesize".to_string(),
        params,
        units: vec![
            UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(1.0)], 1),
            UnitSpec::new(
                "Convolution",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                    framesize,
                ],
                1,
            ),
        ],
    }
}

#[test]
fn convolution_impulse_identity_and_latency() {
    let input = test_signal(RENDER);
    assert!(
        input.iter().any(|s| s.abs() > 0.1),
        "the test signal was silent, so the identity assertion would be vacuous"
    );

    // An impulse *train* of period `framesize`: every frame's kernel is a unit impulse, so every
    // frame convolves to itself and the unit is an exact delay line. (A single impulse would only
    // be the identity for the first frame - the case below.)
    //
    // Two impulse positions per block size. At offset 0 the kernel spectrum is real and every
    // frame's convolution fits the emitted half, so nothing lands in the overlap; at a non-zero
    // offset the kernel spectrum carries a full phase ramp and the last `OFFSET` samples of every
    // frame arrive through the overlap-add instead.
    for (label, reblock, block) in [
        ("world block", None, WORLD_BLOCK),
        ("reblocked", Some(REBLOCK), REBLOCK),
    ] {
        for offset in [0, OFFSET] {
            let train = impulse_kernel(FRAMESIZE, offset);
            let out = render_conv(&input, &train, reblock);
            // The block is collected into the frame before the boundary is checked, so the block
            // that completes a frame already emits that frame's first samples; the impulse's own
            // position inside the frame then delays it further.
            let lag = FRAMESIZE - block + offset;

            // Nothing has been transformed before the first frame completes, so those samples are
            // *exactly* zero - which is also what proves the un-zeroed aux arena is cleared on the
            // first block. The rest of the latency window is silent to transform precision.
            let cold = FRAMESIZE - block;
            assert!(
                out[..cold].iter().all(|s| *s == 0.0),
                "{label} (impulse at {offset}): the first {cold} samples must be exactly zero"
            );
            let leak = out[..lag].iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!(
                leak < 1e-4,
                "{label} (impulse at {offset}): output before the {lag}-sample latency must be \
                 silent (peak {leak})"
            );
            let err = max_err_at_lag(&out, &input, lag);
            assert!(
                err < 1e-4,
                "{label} (impulse at {offset}): an impulse-train kernel must reproduce the input at \
                 {lag} samples of latency (max error {err})"
            );
            // The lag is exact, not approximate: neighbouring lags, and a whole-block error either
            // way, all mismatch by a wide margin on this uncorrelated signal.
            for wrong in [lag - 1, lag + 1, lag - block, lag + block] {
                let wrong_err = max_err_at_lag(&out, &input, wrong);
                assert!(
                    wrong_err > 0.1,
                    "{label} (impulse at {offset}): lag {wrong} must not also match (max error \
                     {wrong_err}); the latency assertion is not discriminating"
                );
            }
        }
    }

    // The aux arena is deliberately not zeroed at instantiation. Run one voice long enough to leave
    // signal in its output and overlap spans, free it, then start an identical voice that reclaims
    // the same (still dirty) region: its cold-start clear must give it the same silent latency
    // window, and the same identity, as a fresh voice.
    let recycled = render_recycled(&input, &impulse_kernel(FRAMESIZE, OFFSET));
    assert!(
        recycled[..FRAMESIZE - WORLD_BLOCK]
            .iter()
            .all(|s| *s == 0.0),
        "recycled voice: a previous tenant's aux leaked into the latency window"
    );
    let recycled_err = max_err_at_lag(&recycled, &input, FRAMESIZE - WORLD_BLOCK + OFFSET);
    assert!(
        recycled_err < 1e-4,
        "recycled voice: the identity must survive a reused pool block (max error {recycled_err})"
    );

    // A single impulse, in a kernel buffer long enough not to loop within the render: only the first
    // frame's kernel is an impulse, and the kernel is re-transformed every frame, so exactly one
    // frame of the input comes through and everything after it is silent.
    let single = impulse_kernel(RENDER, 0);
    let out = render_conv(&input, &single, None);
    let lag = FRAMESIZE - WORLD_BLOCK;
    for t in lag..lag + FRAMESIZE {
        let want = input[t - lag];
        assert!(
            (out[t] - want).abs() < 1e-4,
            "single impulse: sample {t} should carry input sample {} ({want}), got {}",
            t - lag,
            out[t]
        );
    }
    let tail_peak = out[lag + FRAMESIZE..]
        .iter()
        .fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(
        tail_peak < 1e-4,
        "single impulse: the kernel is re-transformed every frame, so frames after the first must \
         be silent (peak {tail_peak})"
    );
}

/// Run `framesize_def` with an `Out` and a `FreeSelfWhenDone` watching the `Convolution`, reblocked
/// to `reblock` when given, and return the rendered output and whether the synth ended.
fn run_framesize(mut def: SynthDef, reblock: Option<usize>) -> (Vec<f32>, bool) {
    def.units.push(UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![
            InputRef::Constant(0.0),
            InputRef::Unit { unit: 1, output: 0 },
        ],
        0,
    ));
    def.units.push(UnitSpec::new(
        "FreeSelfWhenDone",
        Rate::Control,
        vec![InputRef::Unit { unit: 1, output: 0 }],
        1,
    ));
    let (mut controller, mut nrt, mut world) = engine(opts());
    let name = def.name.clone();
    match reblock {
        Some(block) => controller.add_synthdef_reblocked(def, block),
        None => controller.add_synthdef(def),
    }
    controller
        .synth_new(&name, ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, 4 * FRAMESIZE);
    let ended = std::iter::from_fn(|| nrt.poll()).any(|e| matches!(e, Event::NodeEnded(_)));
    (out, ended)
}

#[test]
fn convolution_with_an_unusable_framesize_is_silenced_and_done() {
    // The reference reads `framesize` when the synth starts and, when its FFT setup fails, clears
    // the unit's output and marks it done (`ClearUnitIfMemFailed`). Here that happens for any frame
    // the engine has no plan for, and for one the unit's calc length does not divide. Each case is
    // isolated at a graph block that carries only its own condition.
    for (why, framesize, reblock) in [
        ("not a power of two", 100usize, Some(1usize)),
        ("twice the frame below the plan range", 16, Some(16)),
        ("twice the frame above the plan range", 16_384, None),
        ("not divisible by the graph block", 32, None),
    ] {
        let def = framesize_def(InputRef::Constant(framesize as f32), vec![]);
        let (out, ended) = run_framesize(def, reblock);
        assert!(
            ended,
            "framesize {framesize} ({why}): the unit reports done"
        );
        assert!(
            out.iter().all(|&s| s == 0.0),
            "framesize {framesize} ({why}): silent"
        );
    }
}

#[test]
fn convolution_reads_framesize_when_the_synth_starts() {
    // A `framesize` wired from a parameter works like the constant: the reference's constructor
    // reads it with `ZIN0`.
    let def = framesize_def(
        InputRef::Param(0),
        vec![Param::control("framesize", FRAMESIZE as f32)],
    );
    let (out, ended) = run_framesize(def, None);
    assert!(!ended, "a usable wired framesize runs");
    assert!(
        out.iter().any(|&s| s.abs() > 1e-3),
        "the DC self-convolution is audible"
    );

    // A control-rate instance calcs one sample per block, and one divides every frame, so a
    // 32-sample frame runs at control rate even though the world block does not divide it.
    let mut def = framesize_def(InputRef::Constant(32.0), vec![]);
    for unit in &mut def.units {
        if unit.name == "Convolution" {
            unit.rate = Rate::Control;
        }
    }
    let (_, ended) = run_framesize(def, None);
    assert!(!ended, "the control-rate instance runs");
}

#[test]
fn new_units_reject_wrong_input_counts() {
    // Only `Convolution` is covered here; `tests/pack_fft.rs` and `tests/chaos_l.rs` assert the
    // spectral, demand-operator, and chaos arities
    // in their own test files.
    for inputs in [
        vec![InputRef::Constant(0.0), InputRef::Constant(0.0)],
        vec![
            InputRef::Constant(0.0),
            InputRef::Constant(0.0),
            InputRef::Constant(FRAMESIZE as f32),
            InputRef::Constant(0.0),
        ],
    ] {
        let count = inputs.len();
        let def = SynthDef {
            name: "conv-arity".to_string(),
            params: vec![],
            units: vec![UnitSpec::new("Convolution", Rate::Audio, inputs, 1)],
        };
        assert_eq!(
            compile(&def, None).map(|_| ()),
            Err(BuildError::WrongInputCount),
            "Convolution takes exactly 3 inputs, so {count} must be rejected",
        );
    }
}
