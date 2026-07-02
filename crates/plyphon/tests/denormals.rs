//! Denormal protection: recursive `f32` state (the Lag family's one-pole, comb/allpass feedback
//! lines) must flush decaying tails to exact zero (`zapgremlins`) rather than linger in the
//! subnormal range - plyphon has no hardware flush-to-zero (none exists on wasm), so a subnormal
//! tail is a large CPU cliff during silence.

use plyphon::{
    AddAction, InputRef, Options, ROOT_GROUP_ID, Rate, SynthDef, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;
const BLOCK: usize = 64;

fn render(world: &mut World, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames.next_multiple_of(BLOCK)];
    for chunk in out.chunks_mut(BLOCK) {
        world.fill(chunk, 1);
    }
    out.truncate(frames);
    out
}

fn assert_no_subnormals(out: &[f32], what: &str) {
    for (i, &s) in out.iter().enumerate() {
        assert!(
            s == 0.0 || s.abs() >= f32::MIN_POSITIVE,
            "{what}: subnormal sample {s:e} at frame {i}"
        );
    }
}

/// One impulse at t=0 through `unit` (built by `mk`), rendered for `secs`.
fn impulse_through(units: Vec<UnitSpec>, secs: f64) -> Vec<f32> {
    let (mut controller, _nrt, mut world) = engine(Options {
        sample_rate: SR,
        block_size: BLOCK,
        output_channels: 1,
        ..Options::default()
    });
    controller.add_synthdef(SynthDef {
        name: "tail".to_string(),
        params: vec![],
        units,
    });
    controller
        .synth_new("tail", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, (SR * secs) as usize)
}

#[test]
fn lag_tail_flushes_to_exact_zero() {
    // Impulse -> Lag(0.01): the one-pole decays ~1.4%/sample; without the zap it would sit in
    // subnormal range from ~0.13 s to ~0.19 s. With it, the tail snaps to 0 near 0.05 s.
    let out = impulse_through(
        vec![
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(0.5), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Lag",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.01),
                ],
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
        0.25,
    );
    assert!(out[0] > 0.5, "the impulse should pass through the lag");
    assert_no_subnormals(&out, "Lag tail");
    let tail = &out[(SR * 0.2) as usize..];
    assert!(
        tail.iter().all(|s| *s == 0.0),
        "the lag tail should have flushed to exact zero by 0.2 s"
    );
}

#[test]
fn comb_tail_flushes_to_exact_zero() {
    // Impulse -> CombC(delay 0.01, decay 0.2): -60 dB per 0.2 s puts the recirculation below the
    // 1e-15 zap threshold by ~1 s; without the zap the line would recirculate subnormals from
    // ~1.5 s onward (scsynth relies on hardware FTZ here).
    let out = impulse_through(
        vec![
            UnitSpec::new(
                "Impulse",
                Rate::Audio,
                vec![InputRef::Constant(0.1), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "CombC",
                Rate::Audio,
                vec![
                    InputRef::Unit { unit: 0, output: 0 },
                    InputRef::Constant(0.01), // maxdelaytime
                    InputRef::Constant(0.01), // delaytime
                    InputRef::Constant(0.2),  // decaytime
                ],
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
        1.4,
    );
    assert!(
        out.iter().any(|s| s.abs() > 0.5),
        "the impulse should ring the comb"
    );
    assert_no_subnormals(&out, "Comb tail");
    let tail = &out[(SR * 1.3) as usize..];
    assert!(
        tail.iter().all(|s| *s == 0.0),
        "the comb tail should have flushed to exact zero by 1.3 s"
    );
}
