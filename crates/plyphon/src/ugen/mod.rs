//! Unit generators - plyphon's port of scsynth's `Unit`/`UnitCalcFunc`.
//!
//! A [`Ugen`] is constructed off the audio thread (it may allocate) and then [`Ugen::process`]ed
//! once per control block on the audio thread, where it must not allocate or block. Everything a
//! UGen reads from the wider engine is passed by argument via [`ProcessContext`], [`Inputs`],
//! [`Outputs`] and the output bus - there is no global state.
//!
//! Inputs (read-only views into the synth's wires) and outputs (mutable scratch slices) are
//! deliberately *separate* arguments rather than one bundle, so a UGen can hold an input borrow and
//! an output borrow at the same time without aliasing - the safe equivalent of scsynth's raw
//! aliasing `float*` wires.

pub mod binary_op;
pub mod disk_in;
pub mod filter;
pub mod input;
pub mod line;
pub mod noise;
pub mod out;
pub mod play_buf;
pub mod registry;
pub mod sin_osc;
pub mod unary_op;

use crate::io::Io;
use crate::rate::{Rate, RateInfo};
use crate::wavetable::Wavetables;

/// What a UGen asks the engine to do with its enclosing synth when it finishes - plyphon's subset
/// of scsynth's done-action codes. Ordered so the strongest action wins when combined.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
pub enum DoneAction {
    /// Keep running (no action). scsynth code 0.
    #[default]
    Nothing,
    /// Pause the enclosing synth. scsynth code 1.
    Pause,
    /// Free the enclosing synth. scsynth code 2 (and, for now, the higher free-variant codes).
    FreeSelf,
}

impl DoneAction {
    /// Map a scsynth done-action code (carried as a float UGen input) to a [`DoneAction`].
    pub fn from_code(code: f32) -> DoneAction {
        match code as i32 {
            1 => DoneAction::Pause,
            n if n >= 2 => DoneAction::FreeSelf,
            _ => DoneAction::Nothing,
        }
    }
}

pub use binary_op::BinaryOp;
pub use disk_in::DiskIn;
pub use filter::Butter;
pub use input::In;
pub use line::Line;
pub use noise::WhiteNoise;
pub use out::Out;
pub use play_buf::PlayBuf;
pub use registry::{BuildContext, UgenCtor, UgenRegistry};
pub use sin_osc::SinOsc;
pub use unary_op::UnaryOp;

/// Immutable per-block context handed to every [`Ugen::process`] call.
///
/// The block counter and access to the World's shared buses and buffers live on the mutable
/// [`Io`] handle instead; this holds only the read-only per-block constants.
#[derive(Copy, Clone)]
pub struct ProcessContext<'a> {
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// Shared wavetables (sine, ...), owned by the engine.
    pub wavetables: &'a Wavetables,
}

/// How a single UGen input is sourced. Resolved once at build time from the SynthDef.
#[derive(Copy, Clone, Debug)]
pub enum InputSource {
    /// A constant baked into the SynthDef.
    Constant(f32),
    /// A control-rate wire (index into the synth's control wires).
    Control(u32),
    /// An audio-rate wire (index into the synth's audio wires).
    Audio(u32),
}

impl InputSource {
    /// The calculation rate this source presents to a consuming UGen.
    pub fn rate(self) -> Rate {
        match self {
            InputSource::Constant(_) => Rate::Scalar,
            InputSource::Control(_) => Rate::Control,
            InputSource::Audio(_) => Rate::Audio,
        }
    }
}

/// Read-only view of a UGen's inputs for one block.
///
/// A small bundle of borrows (hence `Copy`). Audio wires are stored flat; wire `w` occupies
/// `audio_wires[w*bs .. (w+1)*bs]`.
#[derive(Copy, Clone)]
pub struct Inputs<'a> {
    sources: &'a [InputSource],
    audio_wires: &'a [f32],
    control_wires: &'a [f32],
    block_size: usize,
}

impl<'a> Inputs<'a> {
    /// Construct an input view. Used by the synth process loop.
    pub fn new(
        sources: &'a [InputSource],
        audio_wires: &'a [f32],
        control_wires: &'a [f32],
        block_size: usize,
    ) -> Self {
        Inputs {
            sources,
            audio_wires,
            control_wires,
            block_size,
        }
    }

    /// Number of inputs.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether there are no inputs.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// The calculation rate of input `i`.
    pub fn rate(&self, i: usize) -> Rate {
        self.sources[i].rate()
    }

    /// Audio-rate input `i` as a `block_size` slice.
    ///
    /// Only meaningful when input `i` is audio-rate; UGens select by [`Inputs::rate`] (they chose
    /// their calc variant at build time from these same rates), so a correctly-built graph never
    /// calls this on a non-audio input. A non-audio input yields an empty slice rather than panic.
    pub fn audio(&self, i: usize) -> &'a [f32] {
        match self.sources[i] {
            InputSource::Audio(w) => {
                let start = w as usize * self.block_size;
                &self.audio_wires[start..start + self.block_size]
            }
            _ => &self.audio_wires[..0],
        }
    }

    /// The single value of a constant or control-rate input `i`.
    ///
    /// An audio-rate input collapses to its first sample (scsynth's `IN0`).
    pub fn control(&self, i: usize) -> f32 {
        match self.sources[i] {
            InputSource::Constant(v) => v,
            InputSource::Control(w) => self.control_wires[w as usize],
            InputSource::Audio(w) => self.audio_wires[w as usize * self.block_size],
        }
    }
}

/// Mutable view of a UGen's output wires for one block.
///
/// Outputs are written into pre-allocated scratch (disjoint from the input wires), then the synth
/// process loop copies them into the arena. Output `i` occupies `scratch[i*bs .. (i+1)*bs]`.
pub struct Outputs<'a> {
    scratch: &'a mut [f32],
    block_size: usize,
}

impl<'a> Outputs<'a> {
    /// Construct an output view over `scratch`. Used by the synth process loop.
    pub fn new(scratch: &'a mut [f32], block_size: usize) -> Self {
        Outputs {
            scratch,
            block_size,
        }
    }

    /// Audio-rate output `i` as a mutable `block_size` slice to write into.
    pub fn audio(&mut self, i: usize) -> &mut [f32] {
        let start = i * self.block_size;
        &mut self.scratch[start..start + self.block_size]
    }

    /// Control-rate output `i` as a single mutable value to write (the first scratch slot, which the
    /// synth process loop publishes to the output's control wire).
    pub fn control(&mut self, i: usize) -> &mut f32 {
        &mut self.scratch[i * self.block_size]
    }
}

/// A unit generator: constructed off the audio thread, processed on it.
pub trait Ugen: Send {
    /// Compute one control block.
    ///
    /// Reads `ins`, writes its outputs into `outs`, and (for I/O UGens like `In`/`Out`/`PlayBuf`)
    /// reads or writes the World's shared buses and buffers through `io`. Must not allocate, block,
    /// or take locks. Returns the [`DoneAction`] the UGen wants applied to its enclosing synth
    /// (almost always [`DoneAction::Nothing`]).
    #[must_use]
    fn process(
        &mut self,
        ctx: &ProcessContext<'_>,
        ins: Inputs<'_>,
        outs: &mut Outputs<'_>,
        io: &mut Io,
    ) -> DoneAction;
}
