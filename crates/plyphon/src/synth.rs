//! A live synth instance - plyphon's port of scsynth's `Graph`.
//!
//! A `Synth` owns its UGens and a flat wire arena. It is built off the audio thread (by
//! [`crate::synthdef::SynthDef::instantiate`]) and then processed on the audio thread, where it
//! must not allocate.
//!
//! The process loop avoids scsynth's aliasing raw `float*` wires while staying `unsafe`-free: each
//! UGen writes into a pre-allocated scratch buffer (disjoint from the input wires), then the loop
//! copies that scratch into the UGen's arena output wires. Inputs and outputs are therefore never
//! borrowed both mutably and immutably at once.

use crate::bus::AudioBus;
use crate::ugen::{InputSource, Inputs, Outputs, ProcessContext, Ugen};

/// A live synth instance.
pub struct Synth {
    /// UGens in topological calc order.
    ugens: Vec<Box<dyn Ugen>>,
    /// Audio wires, flat: wire `w` occupies `audio_wires[w*bs .. (w+1)*bs]`.
    audio_wires: Vec<f32>,
    /// Control wires (one value each); the first entries back the control parameters.
    control_wires: Vec<f32>,
    /// Reused per-UGen output scratch, `max_outputs_per_ugen * block_size`.
    scratch: Vec<f32>,
    /// Per-UGen resolved input sources.
    inputs_plan: Vec<Box<[InputSource]>>,
    /// Per-UGen audio output wire indices.
    outputs_plan: Vec<Box<[u32]>>,
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
        outputs_plan: Vec<Box<[u32]>>,
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

    /// Compute one control block, writing into `out_bus` via any `Out` UGens.
    pub fn process(&mut self, ctx: &ProcessContext<'_>, out_bus: &mut AudioBus) {
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
        for u in 0..ugens.len() {
            let ins = Inputs::new(&inputs_plan[u], audio_wires.as_slice(), control_wires.as_slice(), bs);
            let mut outs = Outputs::new(scratch.as_mut_slice(), bs);
            ugens[u].process(ctx, ins, &mut outs, out_bus);
            // Publish this UGen's scratch outputs into the arena audio wires.
            for (k, &wire) in outputs_plan[u].iter().enumerate() {
                let dst = wire as usize * bs;
                let src = k * bs;
                audio_wires[dst..dst + bs].copy_from_slice(&scratch[src..src + bs]);
            }
        }
    }

    /// Set control parameter `param` to `value`. No-op if out of range. Allocation-free (RT-safe).
    pub fn set_control(&mut self, param: usize, value: f32) {
        if let Some(&wire) = self.param_wires.get(param) {
            self.control_wires[wire as usize] = value;
        }
    }
}
