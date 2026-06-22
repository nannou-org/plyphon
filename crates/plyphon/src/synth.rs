//! A live synth instance - plyphon's port of scsynth's `Graph`.
//!
//! A `Synth` owns its UGens and a flat wire arena. It is built off the audio thread (by
//! [`crate::synthdef::SynthDef::instantiate`]) and then processed on the audio thread, where it
//! must not allocate.
//!
//! The process loop avoids scsynth's aliasing raw `float*` wires while staying `unsafe`-free: each
//! UGen writes into a pre-allocated scratch buffer (disjoint from the input wires), then the loop
//! copies that scratch into the UGen's arena output wires (a full block for audio-rate outputs, a
//! single value for control-rate outputs). Inputs and outputs are therefore never borrowed both
//! mutably and immutably at once.

use crate::bus::AudioBus;
use crate::rate::Rate;
use crate::ugen::{DoneAction, InputSource, Inputs, Outputs, ProcessContext, Ugen};

/// Where a UGen output is published: an audio wire (a block) or a control wire (one value).
#[derive(Copy, Clone, Debug)]
pub(crate) struct OutputWire {
    /// The output's calculation rate.
    pub rate: Rate,
    /// Index into the synth's audio wires (audio rate) or control wires (control/scalar rate).
    pub wire: u32,
}

/// A live synth instance.
pub struct Synth {
    /// UGens in topological calc order.
    ugens: Vec<Box<dyn Ugen>>,
    /// Audio wires, flat: wire `w` occupies `audio_wires[w*bs .. (w+1)*bs]`.
    audio_wires: Vec<f32>,
    /// Control wires (one value each); the first entries back the control parameters, the rest hold
    /// control-rate UGen outputs.
    control_wires: Vec<f32>,
    /// Reused per-UGen output scratch, `max_outputs_per_ugen * block_size`.
    scratch: Vec<f32>,
    /// Per-UGen resolved input sources.
    inputs_plan: Vec<Box<[InputSource]>>,
    /// Per-UGen output wires (rate + index).
    outputs_plan: Vec<Box<[OutputWire]>>,
    /// Control-parameter index -> control wire index.
    param_wires: Vec<u32>,
    block_size: usize,
}

impl Synth {
    /// Assemble a synth from its already-allocated parts. Called by the SynthDef builder.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        ugens: Vec<Box<dyn Ugen>>,
        audio_wires: Vec<f32>,
        control_wires: Vec<f32>,
        scratch: Vec<f32>,
        inputs_plan: Vec<Box<[InputSource]>>,
        outputs_plan: Vec<Box<[OutputWire]>>,
        param_wires: Vec<u32>,
        block_size: usize,
    ) -> Self {
        Synth {
            ugens,
            audio_wires,
            control_wires,
            scratch,
            inputs_plan,
            outputs_plan,
            param_wires,
            block_size,
        }
    }

    /// Compute one control block, writing into `out_bus` via any `Out` UGens. Returns the strongest
    /// [`DoneAction`] any of its UGens requested this block (e.g. an envelope asking to free).
    #[must_use]
    pub fn process(&mut self, ctx: &ProcessContext<'_>, out_bus: &mut AudioBus) -> DoneAction {
        let Synth {
            ugens,
            audio_wires,
            control_wires,
            scratch,
            inputs_plan,
            outputs_plan,
            block_size,
            ..
        } = self;
        let bs = *block_size;
        let mut done = DoneAction::Nothing;
        for u in 0..ugens.len() {
            let ins = Inputs::new(
                &inputs_plan[u],
                audio_wires.as_slice(),
                control_wires.as_slice(),
                bs,
            );
            let mut outs = Outputs::new(scratch.as_mut_slice(), bs);
            done = done.max(ugens[u].process(ctx, ins, &mut outs, out_bus));
            // Publish this UGen's scratch outputs into the arena wires.
            for (k, output) in outputs_plan[u].iter().enumerate() {
                let src = k * bs;
                match output.rate {
                    Rate::Audio => {
                        let dst = output.wire as usize * bs;
                        audio_wires[dst..dst + bs].copy_from_slice(&scratch[src..src + bs]);
                    }
                    Rate::Control | Rate::Scalar => {
                        control_wires[output.wire as usize] = scratch[src];
                    }
                }
            }
        }
        done
    }

    /// Set control parameter `param` to `value`. No-op if out of range. Allocation-free (RT-safe).
    pub fn set_control(&mut self, param: usize, value: f32) {
        if let Some(&wire) = self.param_wires.get(param) {
            self.control_wires[wire as usize] = value;
        }
    }
}
