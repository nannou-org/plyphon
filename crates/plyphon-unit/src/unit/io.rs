//! Free functions for the audited operations a unit performs on the World's shared buses and
//! buffers - plyphon's safe-Rust answer to scsynth's "units reach everything through `mWorld`".
//!
//! [`ProcessCtx`]/[`InitCtx`] expose the buses and buffer table as plain fields, but those types'
//! dangerous mutators are crate-private; these free fns are the public, audited surface a unit uses
//! to touch shared storage. Taking only the field each needs - rather than `&self`/`&mut self` on
//! the whole context - keeps them borrow-friendly: a unit can read its inputs and write a bus in one
//! expression, because the borrows land on disjoint fields (`ctx.ins` vs `ctx.buses`).
//!
//! [`ProcessCtx`]: crate::unit::ProcessCtx
//! [`InitCtx`]: crate::unit::InitCtx

use plyphon_dsp::buffer::{Buffer, BufferTable};
use plyphon_dsp::bus::Buses;
use plyphon_dsp::stream::StreamPlayback;

/// Audio bus channel `ch` for this block (an empty slice if `ch` is out of range), for `In.ar`.
pub fn audio_in(buses: &Buses, ch: usize) -> &[f32] {
    let audio = buses.audio();
    if ch < audio.num_channels() {
        audio.channel(ch)
    } else {
        &[]
    }
}

/// Accumulate `src` into audio bus channel `ch` for this block (`Out.ar`). Out of range is a no-op.
pub fn audio_out(buses: &mut Buses, buf_counter: u64, ch: usize, src: &[f32]) {
    buses.audio_mut().write_accumulate(ch, buf_counter, src);
}

/// Control bus channel `ch`'s current value (0.0 if out of range), for `In.kr`.
pub fn control_in(buses: &Buses, ch: usize) -> f32 {
    buses.control().read(ch)
}

/// Accumulate `value` into control bus channel `ch` for this block (`Out.kr`). Out of range is a
/// no-op.
pub fn control_out(buses: &mut Buses, buf_counter: u64, ch: usize, value: f32) {
    buses.control_mut().write_accumulate(ch, buf_counter, value);
}

/// The (flat, in-memory) buffer at `index`, if one is installed, for `PlayBuf`.
pub fn buffer_at(buffers: &BufferTable, index: usize) -> Option<&Buffer> {
    buffers.get(index)
}

/// The streaming endpoint at `index`, mutably (to pull chunks), for `DiskIn`.
pub fn stream_at_mut(buffers: &mut BufferTable, index: usize) -> Option<&mut StreamPlayback> {
    buffers.stream_mut(index)
}
