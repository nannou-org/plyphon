//! The utility UGens: Pan2 (equal-power stereo), MulAdd (scale/offset), Lag (smoothing), and
//! Amplitude (envelope following).

use plyphon::{
    AddAction, InputRef, Options, Param, ROOT_GROUP_ID, Rate, SynthDef, UgenSpec, World, engine,
};

const SR: f32 = 48_000.0;

fn engine_with(def: SynthDef, channels: usize) -> World {
    let (mut controller, _nrt, world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: channels,
        ..Options::default()
    });
    controller.add_synthdef(def);
    controller
        .synth_new("test", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    world
}

fn render(world: &mut World, frames: usize, channels: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * channels];
    world.fill(&mut out, channels);
    out
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
}

fn ugen(name: &str, inputs: Vec<InputRef>, outputs: usize) -> UgenSpec {
    UgenSpec::new(name, Rate::Audio, inputs, outputs)
}

#[test]
fn lag_seeds_to_its_input_on_the_first_block() {
    // Lag.ar(in, 1.0): a one-second smoother whose `in` is a parameter *set at runtime* to 5.0 - not
    // a build-time constant. Only a first-block `init` that reads the live input can start the output
    // at 5; without it the smoother ramps up from zero, an audible onset glitch.
    let def = SynthDef {
        name: "test".to_string(),
        params: vec![Param {
            name: "in".to_string(),
            default: 0.0,
        }],
        ugens: vec![
            ugen("Lag", vec![InputRef::Param(0), InputRef::Constant(1.0)], 1),
            ugen(
                "Out",
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 0, output: 0 },
                ],
                0,
            ),
        ],
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("test", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    controller.set_control(node, 0, 5.0).unwrap();
    // The very first block must already sit at the input value, not ramp from zero.
    let first = render(&mut world, 64, 1);
    assert!(
        first.iter().all(|&s| (s - 5.0).abs() < 0.01),
        "Lag should start seeded at its input (5.0); first sample was {}",
        first[0]
    );
}

#[test]
fn pan2_places_a_signal_in_the_stereo_field() {
    // Saw.ar(220) -> Pan2(in, pos = -1, level = 1) -> Out.ar(0, [left, right]).
    let def = SynthDef {
        name: "test".to_string(),
        params: vec![],
        ugens: vec![
            ugen("Saw", vec![InputRef::Constant(220.0)], 1),
            ugen(
                "Pan2",
                vec![
                    InputRef::Ugen { ugen: 0, output: 0 },
                    InputRef::Constant(-1.0), // hard left
                    InputRef::Constant(1.0),
                ],
                2,
            ),
            ugen(
                "Out",
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 1, output: 0 },
                    InputRef::Ugen { ugen: 1, output: 1 },
                ],
                0,
            ),
        ],
    };
    let mut world = engine_with(def, 2);
    let out = render(&mut world, SR as usize / 8, 2);
    let left: Vec<f32> = out.iter().step_by(2).copied().collect();
    let right: Vec<f32> = out.iter().skip(1).step_by(2).copied().collect();
    assert!(
        rms(&left) > 0.1,
        "panned-left signal should be in the left channel"
    );
    assert!(
        rms(&right) < 0.01 * rms(&left),
        "panned-left signal should be near-silent on the right"
    );
}

#[test]
fn mul_add_scales_and_offsets() {
    // MulAdd(SinOsc, mul = 0, add = 0.3): the oscillator is zeroed and offset to a constant 0.3.
    let def = SynthDef {
        name: "test".to_string(),
        params: vec![],
        ugens: vec![
            ugen(
                "SinOsc",
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            ugen(
                "MulAdd",
                vec![
                    InputRef::Ugen { ugen: 0, output: 0 },
                    InputRef::Constant(0.0),
                    InputRef::Constant(0.3),
                ],
                1,
            ),
            ugen(
                "Out",
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 1, output: 0 },
                ],
                0,
            ),
        ],
    };
    let mut world = engine_with(def, 1);
    let out = render(&mut world, 256, 1);
    assert!(
        out.iter().all(|s| (s - 0.3).abs() < 1e-4),
        "MulAdd(x, 0, 0.3) should be a constant 0.3"
    );
}

#[test]
fn lag_smooths_a_step() {
    // Lag.ar(in, 0.1) with `in` starting at 0: stepping it to 1 at runtime should ramp the output
    // toward 1 over ~0.1 s rather than jump - de-zippering the change. (A *constant* input, by
    // contrast, is held from the first sample - that case is covered by the seeding test above.)
    let def = SynthDef {
        name: "test".to_string(),
        params: vec![Param {
            name: "in".to_string(),
            default: 0.0,
        }],
        ugens: vec![
            ugen("Lag", vec![InputRef::Param(0), InputRef::Constant(0.1)], 1),
            ugen(
                "Out",
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 0, output: 0 },
                ],
                0,
            ),
        ],
    };
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(def);
    let node = controller
        .synth_new("test", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();

    // Seeded at the initial input (0): silent before the step.
    let before = render(&mut world, 64, 1);
    assert!(
        before.iter().all(|&s| s.abs() < 1e-6),
        "the lag should start seeded at 0"
    );

    // Step the input to 1; the block right after the step is still climbing, not jumped to 1.
    controller.set_control(node, 0, 1.0).unwrap();
    let stepping = render(&mut world, 64, 1);
    assert!(
        stepping[0] < 0.2 && rms(&stepping) < 0.5,
        "the step should ramp, not jump, got first {}",
        stepping[0]
    );

    // After ~0.3 s (a few lag times) it has settled at 1.
    let _ = render(&mut world, (SR * 0.3) as usize, 1);
    let late = render(&mut world, 64, 1);
    assert!(
        late.iter().all(|s| (s - 1.0).abs() < 0.05),
        "the lag should settle at 1"
    );
}

#[test]
fn amplitude_follows_the_envelope() {
    // Amplitude(SinOsc.ar(440) * 0.5): tracks the ~0.5 peak magnitude.
    let def = SynthDef {
        name: "test".to_string(),
        params: vec![],
        ugens: vec![
            ugen(
                "SinOsc",
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            ),
            UgenSpec {
                name: "BinaryOpUGen".to_string(),
                rate: Rate::Audio,
                inputs: vec![
                    InputRef::Ugen { ugen: 0, output: 0 },
                    InputRef::Constant(0.5),
                ],
                num_outputs: 1,
                special_index: 2, // multiply
            },
            ugen(
                "Amplitude",
                vec![
                    InputRef::Ugen { ugen: 1, output: 0 },
                    InputRef::Constant(0.01),
                    InputRef::Constant(0.05),
                ],
                1,
            ),
            ugen(
                "Out",
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Ugen { ugen: 2, output: 0 },
                ],
                0,
            ),
        ],
    };
    let mut world = engine_with(def, 1);
    let _ = render(&mut world, (SR * 0.2) as usize, 1); // settle the follower
    let out = render(&mut world, (SR * 0.1) as usize, 1);
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    assert!(
        out.iter().all(|&s| s >= 0.0),
        "an amplitude envelope is non-negative"
    );
    assert!(
        (0.35..=0.55).contains(&mean),
        "expected the follower to track ~0.5, got {mean}"
    );
}
