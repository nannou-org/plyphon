//! The buffer-backed delay family (`BufDelayN/L/C`, `BufCombN/L/C`, `BufAllpassN/L/C`): the delay line
//! is a `/b_alloc`'d buffer at `bufnum` rather than per-instance aux memory. Only the buffer's
//! power-of-two prefix is used (scsynth's `BUFMASK`), so an odd-sized buffer clamps the maximum delay.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;
const BLOCK: usize = 64;

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn u(i: u32) -> InputRef {
    InputRef::Unit { unit: i, output: 0 }
}

/// Allocate a mono buffer of `buf_frames` at `bufnum`, run `units` for `frames_out` samples of mono
/// output, and return the output.
fn render(units: Vec<UnitSpec>, frames_out: usize, bufnum: usize, buf_frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        block_size: BLOCK,
        ..Options::default()
    });
    controller
        .buffer_alloc(bufnum, buf_frames, 1, SR as f64)
        .expect("buffer_alloc");
    controller.add_synthdef(SynthDef {
        name: "d".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("d", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut out = vec![0.0f32; frames_out];
    world.fill(&mut out, 1);
    out
}

fn out_unit(src: u32) -> UnitSpec {
    UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0)
}

#[test]
fn buf_delay_dc_by_n_samples_across_blocks() {
    // A constant 1.0 fed through BufDelayN reads back as a step from 0 to 1.0 exactly at the delay
    // length: silence while the read tap is still behind the start of writing, then the signal. The
    // delay spans more than one control block, proving the line (a buffer) persists across blocks.
    let delay_secs = 100.0 / SR;
    let out = render(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new(
                "BufDelayN",
                Rate::Audio,
                vec![c(0.0), u(0), c(delay_secs)],
                1,
            ),
            out_unit(1),
        ],
        3 * BLOCK,
        0,
        1024,
    );

    let k = (delay_secs * SR) as usize;
    assert!(k > BLOCK, "delay must span >1 block (k = {k})");
    for (i, &s) in out.iter().enumerate().take(k) {
        assert!(s.abs() < 1e-6, "pre-delay silence at {i}: {s}");
    }
    for (i, &s) in out.iter().enumerate().skip(k) {
        assert!((s - 1.0).abs() < 1e-6, "delayed signal at {i}: {s}");
    }
}

#[test]
fn buf_comb_echoes_and_decays() {
    // A single impulse through BufCombC recirculates: an echo every `delay` samples, each scaled by the
    // feedback coefficient, so successive echoes shrink. With decaytime > 0 the coefficient is in
    // (0, 1), so the second echo is smaller than the first and everything stays bounded.
    let delay = 50usize;
    let delay_secs = delay as f32 / SR;
    let out = render(
        vec![
            // A one-sample impulse train slow enough to fire once over the render.
            UnitSpec::new("Impulse", Rate::Audio, vec![c(0.0), c(0.0)], 1),
            UnitSpec::new(
                "BufCombC",
                Rate::Audio,
                vec![c(0.0), u(0), c(delay_secs), c(0.2)],
                1,
            ),
            out_unit(1),
        ],
        4 * BLOCK,
        0,
        1024,
    );

    assert!(
        out.iter().all(|s| s.is_finite()),
        "BufComb must stay finite"
    );
    assert!(out.iter().all(|&s| s.abs() < 4.0), "BufComb stays bounded");
    // The impulse is at sample 0; echoes land near multiples of `delay`.
    let peak_near = |center: usize| {
        let lo = center.saturating_sub(3);
        let hi = (center + 4).min(out.len());
        out[lo..hi].iter().fold(0.0f32, |m, &s| m.max(s.abs()))
    };
    let e1 = peak_near(delay);
    let e2 = peak_near(2 * delay);
    assert!(e1 > 0.05, "first echo should be audible (e1 = {e1})");
    assert!(
        e2 > 0.0 && e2 < e1,
        "the comb should decay: second echo {e2} < first {e1}"
    );
}

#[test]
fn buf_allpass_passes_signal() {
    // An allpass has a flat magnitude response, so a sine through BufAllpassC comes out at full level
    // (just phase-shifted), finite and bounded.
    let out = render(
        vec![
            UnitSpec::new("SinOsc", Rate::Audio, vec![c(440.0), c(0.0)], 1),
            UnitSpec::new(
                "BufAllpassC",
                Rate::Audio,
                vec![c(0.0), u(0), c(0.01), c(0.3)],
                1,
            ),
            out_unit(1),
        ],
        4 * BLOCK,
        0,
        2048,
    );
    assert!(
        out.iter().all(|s| s.is_finite()),
        "BufAllpass must stay finite"
    );
    assert!(
        out.iter().all(|&s| s.abs() < 4.0),
        "BufAllpass stays bounded"
    );
    // After the line warms, the signal is present (an allpass does not attenuate).
    let tail = &out[out.len() - BLOCK..];
    let rms = (tail.iter().map(|&s| s * s).sum::<f32>() / tail.len() as f32).sqrt();
    assert!(rms > 0.1, "the allpass should pass the sine (rms = {rms})");
}

#[test]
fn odd_buffer_clamps_delay_to_power_of_two_prefix() {
    // A 1500-frame buffer uses only its 1024-sample power-of-two prefix; a requested delay of 1200
    // samples exceeds it and clamps to 1023, so the step still appears (at 1023) rather than the line
    // reading out of range or staying silent forever.
    let want = 1200usize;
    let out = render(
        vec![
            UnitSpec::new("DC", Rate::Audio, vec![c(1.0)], 1),
            UnitSpec::new(
                "BufDelayN",
                Rate::Audio,
                vec![c(0.0), u(0), c(want as f32 / SR)],
                1,
            ),
            out_unit(1),
        ],
        // enough to reach the clamped 1023-sample step and confirm the plateau after it
        1024 + 4 * BLOCK,
        0,
        1500,
    );
    let clamped = 1023usize;
    // Silent up to the clamped delay, then the constant.
    for (i, &s) in out.iter().enumerate().take(clamped) {
        assert!(
            s.abs() < 1e-6,
            "silence before the clamped delay at {i}: {s}"
        );
    }
    let after = &out[clamped + BLOCK..];
    assert!(
        after.iter().all(|&s| (s - 1.0).abs() < 1e-6),
        "the delayed constant appears at the clamped (1023) delay"
    );
}
