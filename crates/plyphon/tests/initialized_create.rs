//! Atomic initialized synth creation and initial-control timing regressions.

use bytemuck::{Pod, Zeroable};
use plyphon::{
    AddAction, BuildContext, BuildError, BuiltUnit, CommandTime, DoneAction, Event, InitCtx,
    InputRef, MAX_INITIAL_CONTROLS, Options, Param, ProcessCtx, ROOT_GROUP_ID, Rate, SynthDef,
    SynthNewError, Unit, UnitDef, UnitSpec, engine, unit_spec,
};

/// One control block used by every focused engine test.
const BLOCK: usize = 64;

/// A custom unit whose first output proves the value observed by `Unit::init`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InitProbe {
    /// Value captured from input zero during the first-block init pass.
    initial: f32,
}

impl Unit for InitProbe {
    fn init(&mut self, ctx: &InitCtx<'_>) {
        self.initial = ctx.ins.control(0);
    }

    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction {
        ctx.outs.audio(0).fill(self.initial);
        DoneAction::Nothing
    }
}

/// Registers [`InitProbe`] without allocating or capturing host state.
struct InitProbeCtor;

impl UnitDef for InitProbeCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(InitProbe { initial: -1.0 }))
    }
}

/// A one-parameter synth that turns its control into an audio signal and writes output bus zero.
fn signal_def(name: &str, param: Param, source: &str) -> SynthDef {
    SynthDef {
        name: name.to_string(),
        params: vec![param],
        units: vec![
            UnitSpec::new(source, Rate::Audio, vec![InputRef::Param(0)], 1),
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

/// A constant-signal definition using the built-in `DC` unit.
fn dc_def(name: &str, param: Param) -> SynthDef {
    signal_def(name, param, "DC")
}

/// Single-output test options with explicit lifecycle headroom.
fn options() -> Options {
    Options {
        block_size: BLOCK,
        output_channels: 1,
        max_nodes: 32,
        critical_event_capacity: 128,
        ..Options::default()
    }
}

/// Render exactly one control block and return its mono output.
fn block(world: &mut plyphon::World) -> Vec<f32> {
    let mut output = vec![0.0; BLOCK];
    world.fill(&mut output, 1);
    output
}

/// Drain the merged lifecycle stream currently visible to the NRT side.
fn drain(nrt: &mut plyphon::Nrt) -> Vec<Event> {
    std::iter::from_fn(|| nrt.poll()).collect()
}

/// Queue rejection leaves both transport rings and the automatic id unchanged.
#[test]
fn synth_new_with_controls_is_all_or_none_and_preserves_id_on_queue_full() {
    let (mut controller, _nrt, mut world) = engine(Options {
        command_capacity: 1,
        ..options()
    });
    controller.add_synthdef(dc_def("signal", Param::control("value", 0.0)));
    controller.ensure_compiled("signal").unwrap();
    let _ = block(&mut world);

    controller.free(999_999).unwrap();
    assert!(matches!(
        controller.synth_new("signal", ROOT_GROUP_ID, AddAction::Tail, &[(0, 0.25)]),
        Err(SynthNewError::QueueFull)
    ));
    let _ = block(&mut world);
    let id = controller
        .synth_new("signal", ROOT_GROUP_ID, AddAction::Tail, &[(0, 0.25)])
        .unwrap();
    assert_eq!(id, 1000, "the failed publication must not consume an id");
}

/// Supplied values reach the first unit init pass and the immediately following process pass.
#[test]
fn initial_controls_are_visible_to_first_init_and_first_process() {
    let mut scalar = Param::control("value", 0.0);
    scalar.rate = Rate::Scalar;
    for (name, param) in [
        ("normal", Param::control("value", 0.0)),
        ("scalar", scalar),
        ("audio", Param::audio("value", 0.0)),
    ] {
        let (mut controller, _nrt, mut world) = engine(options());
        controller
            .registry_mut()
            .register("InitProbe", Box::new(InitProbeCtor));
        controller.add_synthdef(signal_def(name, param, "InitProbe"));
        controller
            .synth_new(name, ROOT_GROUP_ID, AddAction::Tail, &[(0, 0.625)])
            .unwrap();

        let output = block(&mut world);
        assert!(
            output.iter().all(|sample| *sample == 0.625),
            "{name} initial control must be visible to init and process"
        );
    }
}

/// A supplied trigger is visible for the first block and resets before the second.
#[test]
fn initial_trig_control_fires_once_on_first_block() {
    let (mut controller, _nrt, mut world) = engine(options());
    controller.add_synthdef(dc_def("trig", Param::trig("trigger", 0.0)));
    controller
        .synth_new("trig", ROOT_GROUP_ID, AddAction::Tail, &[(0, 1.0)])
        .unwrap();

    assert!(block(&mut world).iter().all(|sample| *sample == 1.0));
    assert!(block(&mut world).iter().all(|sample| *sample == 0.0));
}

/// Lag state starts at the supplied target rather than the authored default.
#[test]
fn initial_lag_control_starts_at_supplied_value_without_default_ramp() {
    let (mut controller, _nrt, mut world) = engine(options());
    controller.add_synthdef(dc_def("lag", Param::lag("value", 0.0, 1.0)));
    controller
        .synth_new("lag", ROOT_GROUP_ID, AddAction::Tail, &[(0, 0.875)])
        .unwrap();

    let output = block(&mut world);
    assert!(output.iter().all(|sample| *sample == 0.875));
}

/// Initialized creation bypasses, but does not close or mutate, an ambient schedule window.
#[test]
fn initialized_create_is_immediate_and_preserves_ambient_schedule_window() {
    let (mut controller, mut nrt, mut world) = engine(options());
    controller.add_synthdef(dc_def("signal", Param::control("value", 0.0)));
    controller.ensure_compiled("signal").unwrap();
    let _ = block(&mut world);

    controller.begin_scheduled(CommandTime::At(u64::MAX));
    let immediate = controller
        .synth_new("signal", ROOT_GROUP_ID, AddAction::Tail, &[(0, 0.5)])
        .unwrap();
    let scheduled = controller
        .synth_new("signal", ROOT_GROUP_ID, AddAction::Tail, &[])
        .unwrap();
    let _ = block(&mut world);
    let events = drain(&mut nrt);
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, Event::NodeStarted(info) if info.node == immediate) })
    );
    assert!(
        !events
            .iter()
            .any(|event| { matches!(event, Event::NodeStarted(info) if info.node == scheduled) })
    );
}

/// Duplicate writes retain slice order and invalid indices publish nothing.
#[test]
fn initial_control_duplicates_are_ordered_and_out_of_range_is_preflighted() {
    let (mut controller, _nrt, mut world) = engine(options());
    controller.add_synthdef(dc_def("signal", Param::control("value", 0.0)));
    let id = controller
        .synth_new(
            "signal",
            ROOT_GROUP_ID,
            AddAction::Tail,
            &[(0, 0.25), (0, 0.75)],
        )
        .unwrap();
    assert_eq!(id, 1000);
    assert!(block(&mut world).iter().all(|sample| *sample == 0.75));

    assert!(matches!(
        controller.synth_new("signal", ROOT_GROUP_ID, AddAction::Tail, &[(1, 1.0)]),
        Err(SynthNewError::InitialControlOutOfRange {
            param: 1,
            params: 1
        })
    ));
}

/// Oversized payloads fail before publication or automatic-id consumption.
#[test]
fn initialized_create_over_capacity_is_atomic_and_preserves_id() {
    let (mut controller, _nrt, _world) = engine(options());
    let params = (0..=MAX_INITIAL_CONTROLS)
        .map(|index| Param::control(format!("p{index}"), 0.0))
        .collect();
    controller.add_synthdef(SynthDef {
        name: "wide-controls".to_string(),
        params,
        units: vec![],
    });
    let controls = (0..=MAX_INITIAL_CONTROLS)
        .map(|index| (index, index as f32))
        .collect::<Vec<_>>();
    assert!(matches!(
        controller.synth_new(
            "wide-controls",
            ROOT_GROUP_ID,
            AddAction::Tail,
            &controls
        ),
        Err(SynthNewError::InitialControlsCapacityExceeded {
            controls,
            capacity: MAX_INITIAL_CONTROLS
        }) if controls == MAX_INITIAL_CONTROLS + 1
    ));
    assert_eq!(
        controller
            .synth_new(
                "wide-controls",
                ROOT_GROUP_ID,
                AddAction::Tail,
                &controls[..MAX_INITIAL_CONTROLS],
            )
            .unwrap(),
        1000
    );
}
