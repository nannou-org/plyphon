//! `Sum3`/`Sum4` (the optimised summers), and the `DynKlank` expansion pattern - a modulatable modal
//! bank built as a summed `Ringz` bank, which is exactly what scsynth compiles `DynKlank` into (it is a
//! class-library macro, not a server UGen).

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
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

fn dc(v: f32) -> UnitSpec {
    UnitSpec::new("DC", Rate::Audio, vec![InputRef::Constant(v)], 1)
}

fn out_unit(src: u32) -> UnitSpec {
    UnitSpec::new("Out", Rate::Audio, vec![InputRef::Constant(0.0), u(src)], 0)
}

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames + 256);
    let mut buf = vec![0.0f32; 256];
    while out.len() < frames {
        world.fill(&mut buf, 1);
        out.extend_from_slice(&buf);
    }
    out.truncate(frames);
    out
}

/// Render `frames` of a graph whose `out_src` unit is routed to `Out`.
fn render_synth(mut units: Vec<UnitSpec>, out_src: u32, frames: usize) -> Vec<f32> {
    units.push(out_unit(out_src));
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR as f64,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "s".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("s", ROOT_GROUP_ID, AddAction::Tail)
        .expect("synth_new");
    render(&mut world, frames)
}

#[test]
fn sum3_adds_three_inputs() {
    let out = render_synth(
        vec![
            dc(0.1),
            dc(0.2),
            dc(0.3),
            UnitSpec::new("Sum3", Rate::Audio, vec![u(0), u(1), u(2)], 1),
        ],
        3,
        64,
    );
    assert!((out[32] - 0.6).abs() < 1e-5, "Sum3 = 0.6, got {}", out[32]);
}

#[test]
fn sum4_adds_four_inputs() {
    let out = render_synth(
        vec![
            dc(0.1),
            dc(0.2),
            dc(0.3),
            dc(0.4),
            UnitSpec::new("Sum4", Rate::Audio, vec![u(0), u(1), u(2), u(3)], 1),
        ],
        4,
        64,
    );
    assert!((out[32] - 1.0).abs() < 1e-5, "Sum4 = 1.0, got {}", out[32]);
}

#[test]
fn dynklank_expansion_rings_at_its_modes() {
    // scsynth compiles `DynKlank([[300,700,1100], ...], Impulse.ar(4), ...)` into a summed Ringz bank;
    // build that expansion directly (one Ringz per mode on a shared impulse, mixed by Sum3) and confirm
    // it rings at each mode. plyphon's Ringz recomputes its coefficients on any freq change, so the same
    // graph is fully modulatable - which is all "DynKlank" means.
    let modes = [300.0f32, 700.0, 1100.0];
    let mut units = vec![UnitSpec::new(
        "Impulse",
        Rate::Audio,
        vec![InputRef::Constant(4.0), InputRef::Constant(0.0)],
        1,
    )];
    for &f in &modes {
        units.push(UnitSpec::new(
            "Ringz",
            Rate::Audio,
            vec![u(0), InputRef::Constant(f), InputRef::Constant(0.6)],
            1,
        ));
    }
    // units 1,2,3 are the Ringz; sum them (unit 4), then tame the level (unit 5).
    units.push(UnitSpec::new(
        "Sum3",
        Rate::Audio,
        vec![u(1), u(2), u(3)],
        1,
    ));
    units.push(UnitSpec {
        name: "BinaryOpUGen".to_string(),
        rate: Rate::Audio,
        inputs: vec![u(4), InputRef::Constant(0.15)],
        num_outputs: 1,
        special_index: 2, // multiply
    });

    let out = render_synth(units, 5, SR as usize / 2);
    assert!(out.iter().all(|s| s.is_finite()), "bank must stay finite");
    assert!(
        out.iter().all(|&s| s.abs() < 4.0),
        "bank should stay bounded"
    );
    let off = goertzel(&out, 500.0); // between the 300 and 700 modes
    for &f in &modes {
        assert!(
            goertzel(&out, f) > 5.0 * off,
            "the bank should ring at its {f} Hz mode"
        );
    }
}
