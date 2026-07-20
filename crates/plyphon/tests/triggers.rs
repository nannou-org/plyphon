//! Exercise the trigger / sample-hold / flip-flop units (`Trig`, `Trig1`, `TDelay`, `Latch`, `Gate`,
//! `ToggleFF`, `SetResetFF`, `Schmidt`) and the counting / timing units (`PulseCount`,
//! `PulseDivider`, `Stepper`, `ZeroCrossing`, `Timer`, `Sweep`, `Phasor`) driven by `Impulse` and
//! `SinOsc` sources.

use plyphon::{AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine};

const SR: f32 = 48_000.0;

fn render_units(units: Vec<UnitSpec>, frames: usize) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "t".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("t", ROOT_GROUP_ID, AddAction::Tail, &[])
        .expect("synth_new");
    let mut out = Vec::with_capacity(frames + 512);
    let mut buf = Vec::new();
    let sizes = [64usize, 100, 128, 480, 512, 333];
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

fn impulse(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "Impulse",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

fn sin(freq: f32) -> UnitSpec {
    UnitSpec::new(
        "SinOsc",
        Rate::Audio,
        vec![InputRef::Constant(freq), InputRef::Constant(0.0)],
        1,
    )
}

fn out(unit: u32) -> UnitSpec {
    UnitSpec::new(
        "Out",
        Rate::Audio,
        vec![InputRef::Constant(0.0), InputRef::Unit { unit, output: 0 }],
        0,
    )
}

fn transitions(o: &[f32]) -> usize {
    o.windows(2).filter(|w| w[0] != w[1]).count()
}

#[test]
fn trig1_holds_for_its_duration() {
    // Trig1(Impulse(10), 0.01): 1 for 0.01 s (~10% of each 0.1 s period), else 0.
    let out = render_units(
        vec![
            impulse(10.0),
            UnitSpec::new(
                "Trig1",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.01),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize,
    );
    assert!(
        out.iter().all(|&x| x == 0.0 || x == 1.0),
        "Trig1 must be 0 or 1"
    );
    let duty = out.iter().filter(|&&x| x == 1.0).count() as f32 / out.len() as f32;
    assert!(
        (duty - 0.1).abs() < 0.03,
        "Trig1 duty should be ~0.1, got {duty}"
    );
}

#[test]
fn tdelay_delays_single_sample_impulses() {
    // TDelay(Impulse(10), 0.02): ~10 delayed one-sample pulses per second.
    let out = render_units(
        vec![
            impulse(10.0),
            UnitSpec::new(
                "TDelay",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.02),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize,
    );
    assert!(
        out.iter().all(|&x| x == 0.0 || x == 1.0),
        "TDelay pulses must be 0 or 1"
    );
    let pulses = out.iter().filter(|&&x| x == 1.0).count();
    assert!(
        (8..=12).contains(&pulses),
        "TDelay(10Hz) should emit ~10 pulses, got {pulses}"
    );
}

#[test]
fn toggle_ff_halves_the_trigger_rate() {
    // ToggleFF(Impulse(20)): flips each impulse -> a square wave at 10 Hz (20 transitions/second).
    let out = render_units(
        vec![
            impulse(20.0),
            UnitSpec::new(
                "ToggleFF",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
        SR as usize,
    );
    assert!(
        out.iter().all(|&x| x == 0.0 || x == 1.0),
        "ToggleFF must be 0 or 1"
    );
    let high = out.iter().filter(|&&x| x == 1.0).count() as f32 / out.len() as f32;
    assert!(
        (high - 0.5).abs() < 0.1,
        "ToggleFF should be ~50% high, got {high}"
    );
    // 20 impulses -> ~20 flips (transitions) per second.
    assert!(
        (15..=25).contains(&transitions(&out)),
        "unexpected flip count {}",
        transitions(&out)
    );
}

#[test]
fn latch_holds_the_sample_between_triggers() {
    // Latch(SinOsc(100), Impulse(500)): sampled at 500 Hz, held for ~96 samples between triggers.
    let out = render_units(
        vec![
            sin(100.0),
            impulse(500.0),
            UnitSpec::new(
                "Latch",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                1,
            ),
            out(2),
        ],
        SR as usize / 10,
    );
    assert!(
        out.iter().all(|&x| (-1.1..=1.1).contains(&x)),
        "Latch out of range"
    );
    // Held between triggers, so the vast majority of adjacent samples are identical.
    // Over 0.1 s at 500 Hz there are ~50 triggers, so ~50 changes: it tracks the sine but holds far
    // more often than it changes.
    let changes = transitions(&out);
    assert!(
        changes < out.len() / 20,
        "Latch should hold between triggers, but changed {changes} of {} samples",
        out.len()
    );
    assert!(
        changes > 20,
        "Latch should still track the sine, only {changes} changes"
    );
}

#[test]
fn gate_passes_when_open_and_holds_when_shut() {
    // Gate(SinOsc(200), 1) passes the tone; Gate(SinOsc(200), 0) never opens, holding its initial 0.
    let open = render_units(
        vec![
            sin(200.0),
            UnitSpec::new(
                "Gate",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(1.0),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize / 4,
    );
    assert!(
        goertzel(&open, 200.0) > 0.3,
        "an open Gate should pass the tone"
    );

    let shut = render_units(
        vec![
            sin(200.0),
            UnitSpec::new(
                "Gate",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize / 4,
    );
    assert!(
        shut.iter().all(|&x| x == 0.0),
        "a shut Gate should hold its initial 0"
    );
}

#[test]
fn schmidt_squares_a_sine_with_hysteresis() {
    // Schmidt(SinOsc(100), -0.5, 0.5): a clean 0/1 square at the sine's frequency.
    let out = render_units(
        vec![
            sin(100.0),
            UnitSpec::new(
                "Schmidt",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(-0.5),
                    InputRef::Constant(0.5),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize,
    );
    assert!(
        out.iter().all(|&x| x == 0.0 || x == 1.0),
        "Schmidt must be 0 or 1"
    );
    let high = out.iter().filter(|&&x| x == 1.0).count() as f32 / out.len() as f32;
    assert!(
        (high - 0.5).abs() < 0.15,
        "a symmetric Schmidt should be ~50% high, got {high}"
    );
    // One rise + one fall per sine cycle -> ~200 transitions/second at 100 Hz.
    assert!(
        (150..=250).contains(&transitions(&out)),
        "unexpected edge count {}",
        transitions(&out)
    );
}

#[test]
fn set_reset_ff_latches_between_set_and_reset() {
    // SetResetFF(Impulse(11), Impulse(5)): set at 11 Hz, reset at 5 Hz -> stays in {0, 1}, both seen.
    let out = render_units(
        vec![
            impulse(11.0),
            impulse(5.0),
            UnitSpec::new(
                "SetResetFF",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                1,
            ),
            out(2),
        ],
        SR as usize,
    );
    assert!(
        out.iter().all(|&x| x == 0.0 || x == 1.0),
        "SetResetFF must be 0 or 1"
    );
    assert!(
        out.contains(&1.0) && out.contains(&0.0),
        "both states should occur"
    );
}

fn constant(v: f32) -> UnitSpec {
    UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(v)], 1)
}

#[test]
fn pulse_count_counts_rising_edges() {
    // PulseCount(Impulse(10), 0): counts up ~10 over 1 s, never decreasing.
    let out = render_units(
        vec![
            impulse(10.0),
            constant(0.0),
            UnitSpec::new(
                "PulseCount",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                1,
            ),
            out(2),
        ],
        SR as usize,
    );
    assert!(
        out.windows(2).all(|w| w[1] >= w[0]),
        "PulseCount must be monotonic without a reset"
    );
    let last = *out.last().unwrap();
    assert!(
        (9.0..=11.0).contains(&last),
        "PulseCount(10Hz) should reach ~10, got {last}"
    );
}

#[test]
fn pulse_divider_divides_the_rate() {
    // PulseDivider(Impulse(40), 4): fires every 4th impulse -> ~10 pulses/second.
    let out = render_units(
        vec![
            impulse(40.0),
            UnitSpec::new(
                "PulseDivider",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(4.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize,
    );
    let pulses = out.iter().filter(|&&x| x == 1.0).count();
    assert!(
        (8..=12).contains(&pulses),
        "PulseDivider(40Hz/4) should emit ~10, got {pulses}"
    );
}

#[test]
fn stepper_cycles_within_bounds() {
    // Stepper(Impulse(20), 0, min=0, max=4, step=1, resetval=0): steps 0..4 and wraps.
    let out = render_units(
        vec![
            impulse(20.0),
            constant(0.0),
            UnitSpec::new(
                "Stepper",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(4.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out(2),
        ],
        SR as usize,
    );
    assert!(
        out.iter()
            .all(|&x| (0.0..=4.0).contains(&x) && x.fract() == 0.0),
        "Stepper out of bounds"
    );
    assert!(
        out.contains(&0.0) && out.contains(&4.0),
        "Stepper should span its full range"
    );
}

#[test]
fn zero_crossing_estimates_frequency() {
    // ZeroCrossing(SinOsc(500)) should settle near 500 Hz.
    let out = render_units(
        vec![
            sin(500.0),
            UnitSpec::new(
                "ZeroCrossing",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
        SR as usize / 2,
    );
    let est = *out.last().unwrap();
    assert!(
        (450.0..=550.0).contains(&est),
        "ZeroCrossing should estimate ~500 Hz, got {est}"
    );
}

#[test]
fn timer_measures_the_trigger_period() {
    // Timer(Impulse(10)) should report ~0.1 s between triggers.
    let out = render_units(
        vec![
            impulse(10.0),
            UnitSpec::new(
                "Timer",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            out(1),
        ],
        SR as usize,
    );
    let last = *out.last().unwrap();
    assert!(
        (0.09..=0.11).contains(&last),
        "Timer(10Hz) should report ~0.1 s, got {last}"
    );
}

#[test]
fn sweep_ramps_and_resets() {
    // Sweep(Impulse(2), 100): ramps at 100/s, reset every 0.5 s -> spans ~[0, 50].
    let out = render_units(
        vec![
            impulse(2.0),
            UnitSpec::new(
                "Sweep",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(100.0),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize,
    );
    let max = out.iter().cloned().fold(f32::MIN, f32::max);
    let min = out.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        (40.0..=55.0).contains(&max),
        "Sweep should ramp to ~50, peak {max}"
    );
    assert!(min < 5.0, "Sweep should reset near 0, min {min}");
}

#[test]
fn phasor_ramps_and_wraps() {
    // Phasor(0, 0.001, 0, 1, 0): a 0..1 sawtooth wrapping every 1000 samples.
    let out = render_units(
        vec![
            constant(0.0),
            UnitSpec::new(
                "Phasor",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.001),
                    InputRef::Constant(0.0),
                    InputRef::Constant(1.0),
                    InputRef::Constant(0.0),
                ],
                1,
            ),
            out(1),
        ],
        SR as usize / 10,
    );
    assert!(
        out.iter().all(|&x| (0.0..1.0).contains(&x)),
        "Phasor should stay in [0, 1)"
    );
    assert!(
        out.iter().any(|&x| x < 0.05) && out.iter().any(|&x| x > 0.95),
        "Phasor should sweep the whole range"
    );
}

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
