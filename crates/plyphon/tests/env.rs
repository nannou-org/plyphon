//! The `EnvGen` multi-segment envelope generator: a one-shot percussive envelope that frees its own
//! synth via a done action, and a gated ADSR that sustains until the gate falls and then releases.
//!
//! The envelope is encoded exactly as SuperCollider unrolls an `Env`: the five control inputs
//! (`gate`, `levelScale`, `levelBias`, `timeScale`, `doneAction`) followed by `initialLevel`,
//! `numSegments`, `releaseNode`, `loopNode`, then four inputs per segment (`target`, `time`,
//! `curveType`, `curveValue`).

use plyphon::{
    AddAction, Event, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, engine,
};

const SR: f32 = 48_000.0;

fn render(world: &mut plyphon::World, frames: usize) -> Vec<f32> {
    let sizes = [64usize, 100, 128, 480, 512, 333];
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

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

/// A 440 Hz sine whose amplitude is the EnvGen output, routed to bus 0.
///
/// `params`/`gate` is the gate input (a `Param` so a test can close it via `set_control`, or a
/// `Constant` for a free-running one-shot); `env` is the flat envelope array spliced in after the
/// five control inputs.
fn enveloped_sine(name: &str, params: Vec<Param>, gate: InputRef, env: Vec<f32>) -> SynthDef {
    let mut env_inputs = vec![
        gate,                    // gate
        InputRef::Constant(1.0), // levelScale
        InputRef::Constant(0.0), // levelBias
        InputRef::Constant(1.0), // timeScale
        InputRef::Constant(2.0), // doneAction = 2 (free self)
    ];
    env_inputs.extend(env.into_iter().map(InputRef::Constant));
    SynthDef {
        name: name.to_string(),
        params,
        units: vec![
            // EnvGen.kr(env, ...): a control-rate amplitude envelope.
            UnitSpec::new("EnvGen", Rate::Control, env_inputs, 1),
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            // SinOsc * EnvGen (BinaryOpUGen, special index 2 = multiply).
            UnitSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Unit { unit: 1, output: 0 },
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                num_outputs: 1,
                special_index: 2,
            },
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

/// `initialLevel, numSegments, releaseNode, loopNode` followed by `(target, time, linear, 0)` per
/// segment - the tail of SuperCollider's unrolled `Env` array, with linear segments.
fn env_array(initial: f32, release_node: f32, segments: &[(f32, f32)]) -> Vec<f32> {
    let mut a = vec![initial, segments.len() as f32, release_node, -99.0];
    for &(target, time) in segments {
        a.extend_from_slice(&[target, time, 1.0, 0.0]); // curveType 1 = linear
    }
    a
}

/// `EnvGen.kr(env, gate) -> DC.ar -> Out.ar(0)`: the raw envelope value, zero-order-held per block,
/// so tests can read levels directly.
fn env_dc_def(name: &str, params: Vec<Param>, gate: InputRef, env: Vec<f32>) -> SynthDef {
    let mut env_inputs = vec![
        gate,
        InputRef::Constant(1.0), // levelScale
        InputRef::Constant(0.0), // levelBias
        InputRef::Constant(1.0), // timeScale
        InputRef::Constant(2.0), // doneAction = 2 (free self)
    ];
    env_inputs.extend(env.into_iter().map(InputRef::Constant));
    SynthDef {
        name: name.to_string(),
        params,
        units: vec![
            UnitSpec::new("EnvGen", Rate::Control, env_inputs, 1),
            UnitSpec::new(
                "DC",
                Rate::Audio,
                vec![InputRef::Unit { unit: 0, output: 0 }],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 1, output: 0 },
                ],
                0,
            ),
        ],
    }
}

#[test]
fn perc_envelope_shapes_a_note_then_frees_it() {
    // Env.perc(0.02, 0.2): rise to 1 over 20 ms, fall to 0 over 200 ms, no release node, then free.
    let env = env_array(0.0, -99.0, &[(1.0, 0.02), (0.0, 0.2)]);
    let def = enveloped_sine("perc", vec![], InputRef::Constant(1.0), env);

    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("perc", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // Early in the decay the note is clearly audible.
    let early = render(&mut world, (SR * 0.05) as usize);
    assert!(
        rms(&early) > 0.1,
        "the percussive note should be audible during its decay, got rms {}",
        rms(&early)
    );

    // Render well past the ~0.22 s envelope; the done action frees the synth (its state returns to
    // the rt-pool on the audio thread; only the notification flows to the NRT side).
    let _ = render(&mut world, (SR * 0.3) as usize);
    nrt.process();
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        if matches!(event, Event::NodeEnded(n) if n.node == node) {
            ended = true;
        }
    }
    assert!(ended, "the percussive envelope should free its synth");

    // With the synth gone, the bus is silent.
    let late = render(&mut world, (SR * 0.05) as usize);
    assert!(
        rms(&late) < 1e-6,
        "expected silence after the note freed itself"
    );
}

#[test]
fn adsr_sustains_until_the_gate_falls() {
    // Env.adsr(0.01, 0.1, 0.5, 0.1): attack to 1, decay to the 0.5 sustain (release node 2), hold,
    // then on gate release fall to 0 and free.
    let env = env_array(0.0, 2.0, &[(1.0, 0.01), (0.5, 0.1), (0.0, 0.1)]);
    let def = enveloped_sine(
        "adsr",
        vec![Param::control("gate", 1.0)],
        InputRef::Param(0),
        env,
    );

    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("adsr", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");

    // Let the attack+decay (~0.11 s) finish, then sample the sustain: 0.5 * a unit sine, rms ~0.35.
    let _ = render(&mut world, (SR * 0.2) as usize);
    let sustain = render(&mut world, (SR * 0.05) as usize);
    let sustain_rms = rms(&sustain);
    assert!(
        (0.25..=0.42).contains(&sustain_rms),
        "the envelope should hold at the 0.5 sustain level, got rms {sustain_rms}"
    );

    // The envelope holds indefinitely while gated: a later window matches the first.
    let _ = render(&mut world, (SR * 0.3) as usize);
    let still = render(&mut world, (SR * 0.05) as usize);
    assert!(
        (rms(&still) - sustain_rms).abs() < 0.05,
        "the sustain should hold steady while the gate is open"
    );

    // Close the gate; the release segment falls to 0 over ~0.1 s and the done action frees the synth.
    controller.set_control(node, 0, 0.0).expect("set gate");
    let _ = render(&mut world, (SR * 0.3) as usize);
    nrt.process();
    let mut ended = false;
    while let Some(event) = nrt.poll() {
        if matches!(event, Event::NodeEnded(n) if n.node == node) {
            ended = true;
        }
    }
    assert!(ended, "releasing the gate should free the synth");

    let silent = render(&mut world, (SR * 0.05) as usize);
    assert!(
        rms(&silent) < 1e-6,
        "expected silence after the release completed"
    );
}

#[test]
fn partial_gate_sustains() {
    // scsynth releases on `gate <= 0`, not below 0.5: a gate lowered to 0.4 must keep sustaining.
    let env = env_array(0.0, 2.0, &[(1.0, 0.01), (0.5, 0.05), (0.0, 0.1)]);
    let def = env_dc_def(
        "partial",
        vec![Param::control("gate", 1.0)],
        InputRef::Param(0),
        env,
    );
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("partial", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let _ = render(&mut world, (SR * 0.1) as usize); // reach the sustain
    controller.set_control(node, 0, 0.4).unwrap();
    let after = render(&mut world, (SR * 0.2) as usize);
    let last = *after.last().unwrap();
    assert!(
        (last - 0.5).abs() < 1e-3,
        "a 0.4 gate must keep sustaining at 0.5, got {last}"
    );
}

#[test]
fn rising_gate_retriggers_from_current_level() {
    let env = env_array(0.0, 2.0, &[(1.0, 0.01), (0.5, 0.05), (0.0, 0.2)]);
    let def = env_dc_def(
        "retrig",
        vec![Param::control("gate", 1.0)],
        InputRef::Param(0),
        env,
    );
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("retrig", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let _ = render(&mut world, (SR * 0.1) as usize); // sustain at 0.5
    controller.set_control(node, 0, 0.0).unwrap();
    let _ = render(&mut world, (SR * 0.1) as usize); // half-way down the 0.2 s release
    // Re-open the gate mid-release: the envelope must ramp back up from wherever it sits (no
    // jump), reach the sustain again, and the synth must not have been freed.
    controller.set_control(node, 0, 1.0).unwrap();
    let up = render(&mut world, (SR * 0.15) as usize);
    for pair in up.chunks(64).collect::<Vec<_>>().windows(2) {
        let (a, b) = (pair[0][0], pair[1][0]);
        assert!(
            (b - a).abs() < 0.25,
            "retrigger must ramp from the current level, found a {a} -> {b} jump"
        );
    }
    let last = *up.last().unwrap();
    assert!(
        (last - 0.5).abs() < 1e-3,
        "the retriggered envelope should sustain again at 0.5, got {last}"
    );
    nrt.process();
    assert!(
        !std::iter::from_fn(|| nrt.poll())
            .any(|e| matches!(e, Event::NodeEnded(n) if n.node == node)),
        "a retriggered envelope must not fire its done action"
    );
}

#[test]
fn gate_below_minus_one_forces_a_timed_release() {
    // A gate of -1.5 releases linearly over |gate|-1 = 0.5 s to the final level, regardless of the
    // release segment's own (here much shorter) time.
    let env = env_array(0.0, 2.0, &[(1.0, 0.01), (0.5, 0.05), (0.0, 0.01)]);
    let def = env_dc_def(
        "cutoff",
        vec![Param::control("gate", 1.0)],
        InputRef::Param(0),
        env,
    );
    let (mut controller, mut nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("cutoff", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let _ = render(&mut world, (SR * 0.1) as usize); // sustain at 0.5
    controller.set_control(node, 0, -1.5).unwrap();
    let down = render(&mut world, (SR * 0.25) as usize);
    let mid = down[(SR * 0.25) as usize - 1];
    assert!(
        (mid - 0.25).abs() < 0.02,
        "half-way through the forced 0.5 s release the level should be ~0.25, got {mid}"
    );
    let _ = render(&mut world, (SR * 0.3) as usize);
    nrt.process();
    assert!(
        std::iter::from_fn(|| nrt.poll())
            .any(|e| matches!(e, Event::NodeEnded(n) if n.node == node)),
        "the forced release should complete and free the synth"
    );
}

/// Render a single 0.1 s segment from 0.2 to 1.0 with the given curve, returning the envelope
/// value at half the segment.
fn curve_value_at_half(curve_type: f32, curve_value: f32) -> f32 {
    let env = vec![
        0.2,   // initialLevel
        1.0,   // numSegments
        -99.0, // releaseNode
        -99.0, // loopNode
        1.0,   // target
        0.1,   // time
        curve_type,
        curve_value,
    ];
    let def = env_dc_def("curve", vec![], InputRef::Constant(1.0), env);
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    controller
        .synth_new("curve", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    let out = render(&mut world, (SR * 0.05) as usize);
    *out.last().unwrap()
}

#[test]
fn curve_shapes_match_their_closed_forms() {
    let half = 0.5f32;
    // step(0): the whole segment sits at the target.
    assert!((curve_value_at_half(0.0, 0.0) - 1.0).abs() < 1e-6, "step");
    // linear(1).
    assert!((curve_value_at_half(1.0, 0.0) - 0.6).abs() < 0.02, "linear");
    // welch(4) rising: start + (end-start)*sin(pi/2 * t).
    let welch = 0.2 + 0.8 * (core::f32::consts::FRAC_PI_2 * half).sin();
    assert!(
        (curve_value_at_half(4.0, 0.0) - welch).abs() < 0.02,
        "welch"
    );
    // squared(6): linear in sqrt space.
    let sq = {
        let y = 0.2f32.sqrt() + (1.0 - 0.2f32.sqrt()) * half;
        y * y
    };
    assert!((curve_value_at_half(6.0, 0.0) - sq).abs() < 0.02, "squared");
    // cubed(7): linear in cube-root space.
    let cb = {
        let y = 0.2f32.powf(1.0 / 3.0) + (1.0 - 0.2f32.powf(1.0 / 3.0)) * half;
        y * y * y
    };
    assert!((curve_value_at_half(7.0, 0.0) - cb).abs() < 0.02, "cubed");
    // hold(8): the start holds for the whole segment.
    assert!((curve_value_at_half(8.0, 0.0) - 0.2).abs() < 1e-6, "hold");
}
