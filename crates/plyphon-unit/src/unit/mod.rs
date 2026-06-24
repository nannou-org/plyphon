//! Unit generators - plyphon's port of scsynth's `Unit`/`UnitCalcFunc`.
//!
//! A [`Unit`] is constructed off the audio thread (it may allocate) and then [`Unit::process`]ed
//! once per control block on the audio thread, where it must not allocate or block. Everything a
//! unit reads from the wider engine arrives in one [`ProcessCtx`] argument - the read-only
//! [`Inputs`], the writable [`Outputs`], the engine constants, and the shared buses/buffers - so
//! there is no global state.
//!
//! `ProcessCtx` is a plain field aggregate, and the operations on the shared buses/buffers are free
//! fns in the [`io`] submodule that take only the field they need (e.g. `io::audio_in(&ctx.buses,
//! ..)`). That keeps them borrow-friendly: because `ins`, `outs`, and `buses` are disjoint fields, a
//! unit can read an input and write an output (or a bus) in the same expression - the safe
//! equivalent of scsynth's raw aliasing `float*` wires.

pub mod band_limited;
pub mod binary_op;
pub mod disk_in;
pub mod env;
pub mod filter;
pub mod input;
pub mod io;
pub mod lf;
pub mod line;
pub mod noise;
pub mod out;
pub mod pan;
pub mod play_buf;
pub mod registry;
pub mod sin_osc;
pub mod unary_op;
pub mod util;

use bytemuck::Pod;

use plyphon_dsp::buffer::BufferTable;
use plyphon_dsp::bus::Buses;
use plyphon_dsp::rate::{Rate, RateInfo};
use plyphon_dsp::wavetable::Wavetables;

/// What a unit asks the engine to do with its enclosing synth when it finishes - plyphon's subset
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
    /// Map a scsynth done-action code (carried as a float unit input) to a [`DoneAction`].
    pub fn from_code(code: f32) -> DoneAction {
        match code as i32 {
            1 => DoneAction::Pause,
            n if n >= 2 => DoneAction::FreeSelf,
            _ => DoneAction::Nothing,
        }
    }

    /// Encode as a small integer tag, so a unit can hold a `DoneAction` in its `Pod` state.
    pub fn to_tag(self) -> u32 {
        self as u32
    }

    /// Decode a tag produced by [`DoneAction::to_tag`] (any out-of-range tag maps to `FreeSelf`).
    pub fn from_tag(tag: u32) -> DoneAction {
        match tag {
            0 => DoneAction::Nothing,
            1 => DoneAction::Pause,
            _ => DoneAction::FreeSelf,
        }
    }
}

pub use band_limited::{Pulse, Saw};
pub use binary_op::BinaryOp;
pub use disk_in::DiskIn;
pub use env::EnvGen;
pub use filter::Butter;
pub use input::In;
pub use io::{audio_in, audio_out, buffer_at, control_in, control_out, stream_at_mut};
pub use lf::{Impulse, LFPulse, LFSaw};
pub use line::Line;
pub use noise::WhiteNoise;
pub use out::Out;
pub use pan::Pan2;
pub use play_buf::PlayBuf;
pub use registry::{BuildContext, UnitDef, UnitRegistry};
pub use sin_osc::SinOsc;
pub use unary_op::UnaryOp;
pub use util::{Amplitude, Lag, MulAdd};

/// Everything a unit touches while processing one control block - plyphon's safe decomposition of
/// scsynth's `unit` (which reaches inputs, outputs, and the world through one pointer).
///
/// The signal ports ([`ins`](Self::ins)/[`outs`](Self::outs)) and engine constants are plain fields.
/// The shared [`buses`](Self::buses)/[`buffers`](Self::buffers) are fields too, but their dangerous
/// mutators are crate-private - a unit touches them only through the audited free fns in
/// [`io`], so it cannot resize a bus or swap a buffer. Those fns take individual
/// fields rather than `&self`, so reading `ins` and writing `buses` in one expression borrows
/// disjoint fields.
pub struct ProcessCtx<'a> {
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// Shared wavetables (sine, ...), owned by the engine.
    pub wavetables: &'a Wavetables,
    /// This unit's inputs for the block (read-only).
    pub ins: Inputs<'a>,
    /// This unit's output scratch for the block.
    pub outs: Outputs<'a>,
    /// The World's shared buses, via the [`io`] free fns (`In`/`Out`).
    pub buses: &'a mut Buses,
    /// The World's shared buffer table, via the [`io`] free fns (`PlayBuf`/`DiskIn`).
    pub buffers: &'a mut BufferTable,
    /// The current block counter (stamps bus writes: the first writer clears, the rest sum).
    pub buf_counter: u64,
}

/// What a unit may touch while *seeding* state on the first block - see [`Unit::init`].
///
/// Like [`ProcessCtx`] but read-only on the world and without [`outs`](ProcessCtx::outs): `init`
/// seeds the unit's own state from live inputs; it does not produce output or mutate the world.
pub struct InitCtx<'a> {
    /// Audio-rate constants.
    pub audio: &'a RateInfo,
    /// Control-rate constants.
    pub control: &'a RateInfo,
    /// Shared wavetables.
    pub wavetables: &'a Wavetables,
    /// This unit's inputs for the block (read-only).
    pub ins: Inputs<'a>,
    /// The World's shared buses (read-only), via the [`io`] free fns.
    pub buses: &'a Buses,
    /// The World's shared buffer table (read-only), via the [`io`] free fns.
    pub buffers: &'a BufferTable,
    /// The current block counter.
    pub buf_counter: u64,
}

/// How a single unit input is sourced. Resolved once at build time from the SynthDef.
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
    /// The calculation rate this source presents to a consuming unit.
    pub fn rate(self) -> Rate {
        match self {
            InputSource::Constant(_) => Rate::Scalar,
            InputSource::Control(_) => Rate::Control,
            InputSource::Audio(_) => Rate::Audio,
        }
    }
}

/// Read-only view of a unit's inputs for one block.
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
    /// Only meaningful when input `i` is audio-rate; units select by [`Inputs::rate`] (they chose
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

/// Mutable view of a unit's output wires for one block.
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

/// A unit generator - plyphon's `Unit` is scsynth's server-side `Unit` (the language-side `UGen` has
/// no plyphon analogue; we consume compiled SynthDefs directly). Its state must be [`Pod`] so it can
/// live as bytes in the rt-pool and be reinterpreted without `unsafe`; behaviour is invoked through
/// the [`ProcessFn`]/[`InitFn`] vtable a [`UnitDef`] builds via [`unit_spec`].
pub trait Unit: Pod {
    /// Re-seed any per-instance randomness from `seed`, called once when the synth is constructed on
    /// the audio thread (before the first block). The default is a no-op; units with an
    /// [`Rng`](plyphon_dsp::rng::Rng) override it so that two instances of the same def decorrelate -
    /// plyphon's stand-in for scsynth seeding each `Graph`'s `RGen`. Must not allocate or block.
    fn reseed(&mut self, _seed: u64) {}

    /// Seed state from the unit's initial inputs.
    ///
    /// Called once, on the first control block, in topological order immediately before this unit's
    /// first [`Unit::process`] - on the audio thread, where inputs are live. By then every input is
    /// readable at its real starting value: constants, control parameters (including `/s_new` args
    /// and `/n_map`ped buses), and the first-block outputs of upstream units. Stateful units seed
    /// here so their first block is already correct - e.g. a smoother starts *at* its input rather
    /// than ramping up from zero - which is what avoids onset clicks.
    ///
    /// This mirrors the seeding an scsynth `*_Ctor` does at its first calc; *allocation*, by
    /// contrast, happens earlier and off the audio thread when the unit is built. Like
    /// [`Unit::process`] it must not allocate, block, or take locks. The default is a no-op.
    fn init(&mut self, _ctx: &InitCtx<'_>) {}

    /// Compute one control block.
    ///
    /// Reads `ctx.ins`, writes `ctx.outs`, and (for I/O units like `In`/`Out`/`PlayBuf`) reads or
    /// writes the World's shared buses and buffers via the [`io`] free fns. Must
    /// not allocate, block, or take locks. Returns the [`DoneAction`] the unit wants applied to its
    /// enclosing synth (almost always [`DoneAction::Nothing`]).
    #[must_use]
    fn process(&mut self, ctx: &mut ProcessCtx<'_>) -> DoneAction;
}

/// A type-erased per-block calc function over a unit's pool-resident state bytes - plyphon's
/// `UnitCalcFunc`/`mCalcFunc`. `state` is exactly `size_of::<T>()` bytes, aligned for `T`.
pub type ProcessFn = fn(&mut [u8], &mut ProcessCtx<'_>) -> DoneAction;

/// A type-erased one-time seeding function over a unit's pool-resident state bytes (see
/// [`Unit::init`]).
pub type InitFn = fn(&mut [u8], &InitCtx<'_>);

/// A type-erased per-instance re-seed function over a unit's pool-resident state bytes (see
/// [`Unit::reseed`]).
pub type ReseedFn = fn(&mut [u8], u64);

/// Reinterpret `bytes` as `T` and run its [`Unit::process`]. Monomorphised per `T` and coerced to a
/// [`ProcessFn`]; the cast cannot fail because the slot is sized and aligned for `T` by construction.
fn process_thunk<T: Unit>(bytes: &mut [u8], ctx: &mut ProcessCtx<'_>) -> DoneAction {
    bytemuck::from_bytes_mut::<T>(bytes).process(ctx)
}

/// As [`process_thunk`], for [`Unit::init`].
fn init_thunk<T: Unit>(bytes: &mut [u8], ctx: &InitCtx<'_>) {
    bytemuck::from_bytes_mut::<T>(bytes).init(ctx);
}

/// As [`process_thunk`], for [`Unit::reseed`].
fn reseed_thunk<T: Unit>(bytes: &mut [u8], seed: u64) {
    bytemuck::from_bytes_mut::<T>(bytes).reseed(seed);
}

/// A built unit: its calc/seed vtable plus the initial state image to copy into the pool. Produced
/// off the audio thread by a [`UnitDef`] (via [`unit_spec`]) and baked into a
/// [`GraphDef`](crate::graphdef::GraphDef).
pub struct BuiltUnit {
    /// Per-block calc function.
    pub process: ProcessFn,
    /// One-time first-block seeding function.
    pub init: InitFn,
    /// Per-instance re-seed function (no-op for units without randomness).
    pub reseed: ReseedFn,
    /// `size_of::<T>()` - the bytes this unit's state occupies in the arena.
    pub size: usize,
    /// `align_of::<T>()` - the alignment its state slot needs.
    pub align: usize,
    /// The initial state, as bytes to `copy_from_slice` into the slot when a synth is built on-RT.
    pub init_bytes: Box<[u8]>,
}

/// Build a [`BuiltUnit`] from an initial unit state. The thunks are monomorphised for `T` here, so a
/// [`UnitDef`] only constructs its initial state and hands it to this helper.
pub fn unit_spec<T: Unit>(state: T) -> BuiltUnit {
    BuiltUnit {
        process: process_thunk::<T>,
        init: init_thunk::<T>,
        reseed: reseed_thunk::<T>,
        size: core::mem::size_of::<T>(),
        align: core::mem::align_of::<T>(),
        init_bytes: bytemuck::bytes_of(&state).to_vec().into_boxed_slice(),
    }
}
