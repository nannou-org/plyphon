//! Tests specific to the on-RT, pool-backed synth model: a freed synth's state returns to the
//! rt-pool (no leak), pool exhaustion is reported as `SynthFailed` rather than panicking, and a def
//! that would overflow the World-shared wire scratch fails to compile.

use plyphon::{
    AddAction, BuildError, Event, InputRef, Nrt, Options, Param, ROOT_GROUP_ID, Rate, SynthDef,
    SynthNewError, UnitSpec, World, engine,
};

const SR: f64 = 48_000.0;

/// `SinOsc.ar(freq) -> Out.ar(0, sig)`: one audio wire, one control parameter.
fn sine_def() -> SynthDef {
    SynthDef {
        name: "sine".to_string(),
        params: vec![Param::control("freq", 440.0)],
        units: vec![
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Param(0), InputRef::Constant(0.0)],
                1,
            ),
            UnitSpec::new(
                "Out",
                Rate::Audio,
                vec![
                    InputRef::Constant(0.0),
                    InputRef::Unit { unit: 0, output: 0 },
                ],
                0,
            ),
        ],
    }
}

/// A def with `n` audio-rate `SinOsc`s (hence `n` audio wires) summed to one `Out`.
fn wide_def(n: u32) -> SynthDef {
    let mut units: Vec<UnitSpec> = (0..n)
        .map(|_| {
            UnitSpec::new(
                "SinOsc",
                Rate::Audio,
                vec![InputRef::Constant(440.0), InputRef::Constant(0.0)],
                1,
            )
        })
        .collect();
    let out_inputs = std::iter::once(InputRef::Constant(0.0))
        .chain((0..n).map(|u| InputRef::Unit { unit: u, output: 0 }))
        .collect();
    units.push(UnitSpec::new("Out", Rate::Audio, out_inputs, 0));
    SynthDef {
        name: "wide".to_string(),
        params: vec![],
        units,
    }
}

/// Drive the engine for `frames` of mono audio so queued commands are applied and any events emitted.
fn render(world: &mut World, frames: usize) {
    let mut buf = vec![0.0f32; 64];
    let mut done = 0;
    while done < frames {
        world.fill(&mut buf, 1);
        done += buf.len();
    }
}

/// Drain and return all currently available notifications.
fn drain(nrt: &mut Nrt) -> Vec<Event> {
    nrt.process();
    std::iter::from_fn(|| nrt.poll()).collect()
}

fn opts() -> Options {
    Options {
        sample_rate: SR,
        output_channels: 1,
        ..Options::default()
    }
}

#[test]
fn freeing_a_synth_returns_its_state_to_the_pool() {
    let (mut controller, _nrt, mut world) = engine(opts());
    controller.add_synthdef(sine_def());

    assert_eq!(world.rt_memory_used(), 0, "no synths, no pool use");

    // Spawn and apply: the synth's state block is now allocated in the pool.
    let node = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    let used_live = world.rt_memory_used();
    assert!(used_live > 0, "a live synth should occupy pool memory");

    // Free and apply: the block returns to the pool.
    controller.free(node).unwrap();
    render(&mut world, 256);
    assert_eq!(
        world.rt_memory_used(),
        0,
        "freeing the synth should reclaim its pool block"
    );

    // Churn many spawn/free cycles and confirm steady-state use stays at zero (no leak, full
    // coalescing).
    for _ in 0..64 {
        let n = controller
            .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
            .unwrap();
        render(&mut world, 64);
        controller.free(n).unwrap();
        render(&mut world, 64);
    }
    assert_eq!(world.rt_memory_used(), 0, "no steady-state pool growth");
}

#[test]
fn pool_exhaustion_reports_synth_failed_then_recovers() {
    // A deliberately tiny pool: only a handful of synths fit.
    let (mut controller, mut nrt, mut world) = engine(Options {
        pool_bytes: 1024,
        ..opts()
    });
    controller.add_synthdef(sine_def());

    // Spawn far more than fit; `synth_new` always queues fine (failure is detected on the RT thread).
    let mut nodes = Vec::new();
    for _ in 0..64 {
        nodes.push(
            controller
                .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
                .unwrap(),
        );
    }
    render(&mut world, 1024);
    let events = drain(&mut nrt);
    let started = events
        .iter()
        .filter(|e| matches!(e, Event::NodeStarted(_)))
        .count();
    let failed = events
        .iter()
        .filter(|e| matches!(e, Event::SynthFailed { .. }))
        .count();
    assert!(started > 0, "some synths should fit the pool");
    assert!(
        failed > 0,
        "spawning past the tiny pool should report SynthFailed, not panic"
    );

    // Free everything; the pool empties.
    controller.free_all(ROOT_GROUP_ID).unwrap();
    render(&mut world, 1024);
    let _ = drain(&mut nrt);
    assert_eq!(world.rt_memory_used(), 0, "freeing all reclaims the pool");

    // There is room again: a fresh synth starts.
    let node = controller
        .synth_new("sine", ROOT_GROUP_ID, AddAction::Tail)
        .unwrap();
    render(&mut world, 256);
    assert!(
        drain(&mut nrt)
            .iter()
            .any(|e| matches!(e, Event::NodeStarted(n) if n.node == node)),
        "a synth should start once the pool has room again"
    );
}

#[test]
fn a_def_exceeding_the_wire_cap_fails_to_compile() {
    // Only two audio wires of shared scratch; a 3-wire def must be rejected at compile time.
    let (mut controller, _nrt, _world) = engine(Options {
        max_wire_bufs: 2,
        ..opts()
    });
    controller.add_synthdef(wide_def(3));

    let err = controller
        .synth_new("wide", ROOT_GROUP_ID, AddAction::Tail)
        .expect_err("a def needing 3 wires must fail under a 2-wire cap");
    assert!(
        matches!(
            err,
            SynthNewError::Build(BuildError::TooManyWires {
                needed: 3,
                limit: 2
            })
        ),
        "expected TooManyWires, got {err:?}"
    );
}
