//! `PV_Diffuser` (spectral phase diffusion), `Gendy1` (dynamic stochastic synthesis), `IEnvGen`
//! (index-driven envelope reading), and `TDuty` (demand-driven trigger stream).

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;
const MAX_BUFFERS: usize = 32;

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        max_buffers: MAX_BUFFERS,
        ..Options::default()
    }
}

fn u(unit: u32) -> InputRef {
    InputRef::Unit { unit, output: 0 }
}

fn c(v: f32) -> InputRef {
    InputRef::Constant(v)
}

fn out(src: u32) -> UnitSpec {
    UnitSpec::new("Out", Rate::Audio, vec![c(0.0), u(src)], 0)
}

fn render(units: Vec<UnitSpec>, blocks: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    let mut buf = vec![0.0f32; BLOCK * blocks];
    world.fill(&mut buf, 1);
    buf
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

// ---------------------------------------------------------------------------
// PV_Diffuser
// ---------------------------------------------------------------------------

/// The FFT chain `IFFT(PV_Diffuser(FFT(LocalBuf(n), tone), trig))` with a sine at bin
/// `bin_index`, or the same chain without the diffuser when `trig` is `None`. The diffuser adds a
/// fixed random phase to the first `trig * numbins` bins.
fn diffuser_chain(trig: Option<f32>, bin_index: f32, blocks: usize) -> Vec<f32> {
    const FFT_SIZE: usize = 1024;
    let bin = SR as f32 / FFT_SIZE as f32;
    let freq = bin_index * bin;
    let mut units = vec![
        UnitSpec::new(
            "LocalBuf",
            Rate::Scalar,
            vec![c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
        UnitSpec::new("SinOsc", Rate::Audio, vec![c(freq), c(0.0)], 1),
        UnitSpec {
            name: "BinaryOpUGen".to_string(),
            rate: Rate::Audio,
            inputs: vec![u(1), c(0.5)],
            num_outputs: 1,
            special_index: 2,
        },
        UnitSpec::new(
            "FFT",
            Rate::Control,
            vec![u(0), u(2), c(0.5), c(0.0), c(1.0), c(FFT_SIZE as f32)],
            1,
        ),
    ];
    let chain = if let Some(trig) = trig {
        // PV_Diffuser(fbufnum, trig).
        units.push(UnitSpec::new(
            "PV_Diffuser",
            Rate::Control,
            vec![u(3), c(trig)],
            1,
        ));
        4
    } else {
        3
    };
    units.push(UnitSpec::new(
        "IFFT",
        Rate::Audio,
        vec![u(chain), c(0.0), c(FFT_SIZE as f32)],
        1,
    ));
    units.push(out(chain + 1));
    render(units, blocks)
}

#[test]
fn pv_diffuser_preserves_magnitude_but_alters_the_waveform() {
    const FFT_SIZE: usize = 1024;
    let bin = SR as f32 / FFT_SIZE as f32;
    let freq = 20.0 * bin;
    let diffused = diffuser_chain(Some(1.0), 20.0, 12_288 / BLOCK);
    let tail = &diffused[8_192..];

    assert!(
        tail.iter().any(|s| s.abs() > 0.02),
        "diffuser output silent"
    );
    // The tone's magnitude survives the phase diffusion (a fixed per-bin phase does not move energy
    // between bins).
    let at = goertzel(tail, freq);
    let off = goertzel(tail, freq * 2.0);
    assert!(
        at > 8.0 * off,
        "diffused tone should still dominate (at={at:.4}, off={off:.4})"
    );

    // The plain reconstruction (no diffuser) at the same bin: the diffuser changes the waveform
    // (per-bin phase shift) so the two time series differ, even though both hold the tone.
    let plain = diffuser_chain(None, 20.0, 12_288 / BLOCK);
    let plain_tail = &plain[8_192..];
    let diff: f32 = tail
        .iter()
        .zip(plain_tail)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1.0,
        "diffuser did not alter the waveform (diff={diff:.4})"
    );
}

/// scsynth's `PV_Diffuser_next` shifts only the first `clip(trig * numbins, 0, numbins)` bins, so
/// a zero trig leaves every phase untouched: the chain reconstructs the same waveform as one with
/// no diffuser at all (up to the polar round-trip's float error).
#[test]
fn pv_diffuser_zero_trig_passes_the_frame_through() {
    let diffused = diffuser_chain(Some(0.0), 20.0, 12_288 / BLOCK);
    let plain = diffuser_chain(None, 20.0, 12_288 / BLOCK);
    for (i, (a, b)) in diffused[8_192..].iter().zip(&plain[8_192..]).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "zero-trig diffuser altered sample {i}: {a} vs {b}"
        );
    }
}

/// A `0.5` trig shifts only the lower half of the spectrum: a tone in an upper bin passes through
/// unshifted, while the same partial trig does alter a low-bin tone.
#[test]
fn pv_diffuser_partial_trig_shifts_only_the_lower_bins() {
    // Bin 400 of 511 lies above `0.5 * numbins`, so its phase is never offset.
    let high = diffuser_chain(Some(0.5), 400.0, 12_288 / BLOCK);
    let high_plain = diffuser_chain(None, 400.0, 12_288 / BLOCK);
    for (i, (a, b)) in high[8_192..].iter().zip(&high_plain[8_192..]).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "high-bin tone altered by half-trig diffuser at sample {i}: {a} vs {b}"
        );
    }

    // Bin 20 lies below the cut, so the half-trig diffuser still shifts it.
    let low = diffuser_chain(Some(0.5), 20.0, 12_288 / BLOCK);
    let low_plain = diffuser_chain(None, 20.0, 12_288 / BLOCK);
    let diff: f32 = low[8_192..]
        .iter()
        .zip(&low_plain[8_192..])
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1.0,
        "half-trig diffuser did not alter a low-bin tone (diff={diff:.4})"
    );
}

#[test]
fn pv_diffuser_is_deterministic_under_the_same_seed() {
    // Two fresh engines render bit-identically: the per-bin phases come from the seeded stream.
    let a = diffuser_chain(Some(1.0), 20.0, 12_288 / BLOCK);
    let b = diffuser_chain(Some(1.0), 20.0, 12_288 / BLOCK);
    assert_eq!(a, b, "diffuser output should reproduce across engines");
}

// ---------------------------------------------------------------------------
// Gendy1
// ---------------------------------------------------------------------------

fn gendy(blocks: usize) -> Vec<f32> {
    // Gendy1(ampdist, durdist, adparam, ddparam, minfreq, maxfreq, ampscale, durscale, initCPs,
    // knum).
    let units = vec![
        UnitSpec::new(
            "Gendy1",
            Rate::Audio,
            vec![
                c(1.0),   // ampdist CAUCHY
                c(1.0),   // durdist CAUCHY
                c(1.0),   // adparam
                c(1.0),   // ddparam
                c(220.0), // minfreq
                c(440.0), // maxfreq
                c(0.5),   // ampscale
                c(0.5),   // durscale
                c(12.0),  // initCPs
                c(12.0),  // knum
            ],
            1,
        ),
        out(0),
    ];
    render(units, blocks)
}

#[test]
fn gendy1_produces_bounded_nonconstant_audio() {
    let output = gendy(64);
    assert!(
        output.iter().all(|s| s.abs() <= 1.5),
        "Gendy1 output should stay bounded"
    );
    let first = output[0];
    assert!(
        output.iter().any(|s| (s - first).abs() > 1e-4),
        "Gendy1 output should vary"
    );
}

#[test]
fn gendy1_is_deterministic_across_engines() {
    let a = gendy(32);
    let b = gendy(32);
    assert_eq!(a, b, "per-instance seeding makes Gendy1 reproducible");
}

// ---------------------------------------------------------------------------
// IEnvGen
// ---------------------------------------------------------------------------

/// `IEnvGen` reading a 2-segment linear envelope: levels 0 -> 1 -> 0.5 over durations 1 s + 1 s.
/// Returns the output level at a constant `index` (seconds).
fn ienv_linear_at(index: f32) -> f32 {
    let units = vec![
        UnitSpec::new(
            "IEnvGen",
            Rate::Audio,
            vec![
                c(index), // index
                c(0.0),   // offset
                c(0.0),   // startLevel
                c(2.0),   // numSegments
                c(2.0),   // totalDur
                c(1.0),
                c(1.0),
                c(0.0),
                c(1.0), // seg0: dur=1, shape=lin, curve=0, endLevel=1
                c(1.0),
                c(1.0),
                c(0.0),
                c(0.5), // seg1: dur=1, shape=lin, curve=0, endLevel=0.5
            ],
            1,
        ),
        out(0),
    ];
    render(units, 1)[0]
}

#[test]
fn ienvgen_reads_a_linear_envelope_by_index() {
    let cases = [
        (-1.0, 0.0), // before the start clamps to the initial level
        (0.0, 0.0),  // start
        (0.5, 0.5),  // halfway up segment 0
        (1.0, 1.0),  // the peak (segment boundary)
        (1.5, 0.75), // halfway down segment 1 (1.0 -> 0.5)
        (2.0, 0.5),  // the end
        (3.0, 0.5),  // past the end clamps to the final level
    ];
    for (index, expected) in cases {
        let got = ienv_linear_at(index);
        assert!(
            (got - expected).abs() < 1e-4,
            "IEnvGen at index {index}: expected {expected}, got {got}"
        );
    }
}

// ---------------------------------------------------------------------------
// TDuty
// ---------------------------------------------------------------------------

/// `TDuty(dur, reset, doneAction, level, gapFirst)` with a constant one-block duration and level.
fn tduty(gap_first: f32, blocks: usize) -> Vec<f32> {
    let dur = BLOCK as f32 / SR as f32; // exactly one control block
    let units = vec![
        UnitSpec::new(
            "TDuty",
            Rate::Audio,
            vec![c(dur), c(0.0), c(0.0), c(1.0), c(gap_first)],
            1,
        ),
        out(0),
    ];
    render(units, blocks)
}

#[test]
fn tduty_emits_one_frame_impulses_at_each_boundary() {
    // No gap-first: impulses at samples 0, BLOCK, 2*BLOCK, ..., zeros between.
    let output = tduty(0.0, 4);
    for (i, &s) in output.iter().enumerate() {
        if i % BLOCK == 0 {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "expected an impulse at sample {i}, got {s}"
            );
        } else {
            assert_eq!(s, 0.0, "expected silence at sample {i}, got {s}");
        }
    }
}

#[test]
fn tduty_gap_first_delays_the_first_impulse() {
    // Gap-first pulls one duration up front, so the first impulse lands at sample BLOCK, not 0.
    let output = tduty(1.0, 4);
    assert_eq!(output[0], 0.0, "gap-first should not fire at sample 0");
    assert!(
        (output[BLOCK] - 1.0).abs() < 1e-5,
        "gap-first's first impulse should be at sample {BLOCK}, got {}",
        output[BLOCK]
    );
}

#[test]
fn gendy1_rejects_a_non_constant_init_cps() {
    // The breakpoint arrays are aux memory sized at compile, so a wired `initCPs` input must be
    // refused like any other non-constant aux size (a delay's `maxdelaytime`).
    use plyphon::{BuildError, UnitRegistry};
    use plyphon_dsp::rate::RateInfo;
    let dc = UnitSpec::new("DC", Rate::Scalar, vec![c(12.0)], 1);
    let mut inputs = vec![
        c(1.0),
        c(1.0),
        c(1.0),
        c(1.0),
        c(440.0),
        c(660.0),
        c(0.5),
        c(0.5),
    ];
    inputs.push(u(0)); // initCPs wired to the DC unit.
    inputs.push(c(12.0)); // knum.
    let gendy = UnitSpec::new("Gendy1", Rate::Audio, inputs, 1);
    let def = SynthDef {
        name: "bad_gendy".to_string(),
        params: vec![],
        units: vec![dc, gendy, out(1)],
    };
    let rate = RateInfo::new(SR, BLOCK);
    let result = def.compile(
        &UnitRegistry::with_builtins(),
        &rate,
        &rate,
        64,
        32,
        None,
        1,
    );
    assert!(
        matches!(
            result.as_ref().map(|_| ()),
            Err(BuildError::AuxRequiresConstant { input: 8 })
        ),
        "expected AuxRequiresConstant for a wired initCPs"
    );
}

#[test]
fn ienvgen_clamps_an_oversized_stage_count_to_its_inputs() {
    // A def declaring a huge numSegments with only one actual segment must clamp the walk to the
    // inputs it carries (the pre-clamp code spun the search a billion iterations per sample).
    let units = vec![
        UnitSpec::new(
            "IEnvGen",
            Rate::Audio,
            vec![
                c(0.5), // index: halfway up the one real segment
                c(0.0),
                c(0.0),
                c(1.0e9), // numSegments, absurdly oversized
                c(1.0),   // totalDur
                c(1.0),
                c(1.0),
                c(0.0),
                c(1.0), // seg0: dur=1, shape=lin, curve=0, endLevel=1
            ],
            1,
        ),
        out(0),
    ];
    let got = render(units, 1)[0];
    assert!(
        (got - 0.5).abs() < 1e-4,
        "clamped walk should still read the real segment, got {got}"
    );
}
