//! `/s_new`-with-controls parity on the existing command surface: a create batched with its
//! control writes via `Controller::try_send_batch` applies the values before the synth's first
//! `init`/`process` tick, exactly as scsynth's `/s_new` trailing `[control, value]` pairs (which
//! are set through the `/n_set` path on creation, before `Graph_FirstCalc` runs the unit ctors).

use bytemuck::{Pod, Zeroable};
use plyphon::{
    AddAction, BuildContext, BuildError, BuiltUnit, ControllerBatchCommand, DoneAction, InitCtx,
    InputRef, Options, Param, ProcessCtx, ROOT_GROUP_ID, Rate, SynthDef, Unit, UnitDef, UnitSpec,
    engine, unit_spec,
};

const BLOCK: usize = 64;

/// A unit whose steady output is the value its one-time `init` pass observed on input zero,
/// proving what the ctor-equivalent saw.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InitProbe {
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

struct InitProbeCtor;

impl UnitDef for InitProbeCtor {
    fn build(&self, _ctx: &BuildContext<'_>) -> Result<BuiltUnit, BuildError> {
        Ok(unit_spec(InitProbe { initial: -1.0 }))
    }
}

/// `source(param 0) -> Out.ar(0)`.
fn probe_def(name: &str, param: Param, source: &str) -> SynthDef {
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

fn options() -> Options {
    Options {
        block_size: BLOCK,
        output_channels: 1,
        ..Options::default()
    }
}

/// Render exactly one control block and return its mono output.
fn block(world: &mut plyphon::World) -> Vec<f32> {
    let mut output = vec![0.0; BLOCK];
    world.fill(&mut output, 1);
    output
}

/// Batch the create of `id` from `def_name` with one initial control write.
fn create_with_control(controller: &mut plyphon::Controller, def_name: &str, id: i32, value: f32) {
    let def_id = controller.ensure_compiled(def_name).unwrap();
    controller
        .try_send_batch(&[
            ControllerBatchCommand::AddSynth {
                id,
                def_id,
                target: ROOT_GROUP_ID,
                action: AddAction::Tail,
            },
            ControllerBatchCommand::SetControl {
                node: id,
                param: 0,
                value,
            },
        ])
        .unwrap();
}

/// The batched value reaches the first `init` pass and the first `process` tick, for every
/// parameter rate.
#[test]
fn batched_create_applies_controls_before_first_init_and_process() {
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
        controller.add_synthdef(probe_def(name, param, "InitProbe"));
        create_with_control(&mut controller, name, 2000, 0.625);

        let output = block(&mut world);
        assert!(
            output.iter().all(|sample| *sample == 0.625),
            "{name}: the batched value must be visible to init and process, got {:?}",
            &output[..2]
        );
    }
}

/// A batched trigger value is seen for the first block only, then resets - as an `/s_new` trig
/// pair does in scsynth.
#[test]
fn batched_trig_control_fires_once_on_first_block() {
    let (mut controller, _nrt, mut world) = engine(options());
    controller.add_synthdef(probe_def("trig", Param::trig("trigger", 0.0), "DC"));
    create_with_control(&mut controller, "trig", 2000, 1.0);

    assert!(block(&mut world).iter().all(|sample| *sample == 1.0));
    assert!(block(&mut world).iter().all(|sample| *sample == 0.0));
}

/// A batched lagged control starts at the supplied value with no ramp from the default: the lag
/// state seeds from the live value slot on the first tick (scsynth's `LagControl_Ctor`).
#[test]
fn batched_lag_control_starts_at_supplied_value() {
    let (mut controller, _nrt, mut world) = engine(options());
    controller.add_synthdef(probe_def("lag", Param::lag("value", 0.0, 1.0), "DC"));
    create_with_control(&mut controller, "lag", 2000, 0.875);

    assert!(block(&mut world).iter().all(|sample| *sample == 0.875));
    assert!(block(&mut world).iter().all(|sample| *sample == 0.875));
}
