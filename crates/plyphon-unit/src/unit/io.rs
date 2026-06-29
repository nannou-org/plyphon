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
use plyphon_dsp::stream::{StreamPlayback, StreamRecording};

use crate::unit::LocalBus;

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

/// Local feedback-bus channel `ch` for this block (read), for `LocalIn` - the value `LocalOut` wrote
/// last block. An empty slice if `ch` is out of range.
pub fn local_in<'a>(local: &'a LocalBus<'_>, ch: usize) -> &'a [f32] {
    local.channel(ch)
}

/// Overwrite local feedback-bus channel `ch` with `src` for this block, for `LocalOut`. Out of range
/// is a no-op; a shorter `src` leaves the channel's tail untouched.
pub fn local_out(local: &mut LocalBus<'_>, ch: usize, src: &[f32]) {
    if let Some(dst) = local.channel_mut(ch) {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
    }
}

/// Accumulate `value` into control bus channel `ch` for this block (`Out.kr`). Out of range is a
/// no-op.
pub fn control_out(buses: &mut Buses, buf_counter: u64, ch: usize, value: f32) {
    buses.control_mut().write_accumulate(ch, buf_counter, value);
}

/// Number of hardware output bus channels (the first audio bus channels), for `NumOutputBuses`.
pub fn num_output_buses(buses: &Buses) -> usize {
    buses.output_channels()
}

/// Number of hardware input bus channels (those following the outputs), for `NumInputBuses`.
pub fn num_input_buses(buses: &Buses) -> usize {
    buses.input_channels()
}

/// Total audio bus channels - output, input, and private - for `NumAudioBuses`.
pub fn num_audio_buses(buses: &Buses) -> usize {
    buses.audio().num_channels()
}

/// Total control bus channels, for `NumControlBuses`.
pub fn num_control_buses(buses: &Buses) -> usize {
    buses.control().num_channels()
}

/// The number of buffer slots (the table capacity), for `NumBuffers`.
pub fn num_buffers(buffers: &BufferTable) -> usize {
    buffers.capacity()
}

/// The (flat, in-memory) buffer at `index`, if one is installed, for `PlayBuf`.
pub fn buffer_at(buffers: &BufferTable, index: usize) -> Option<&Buffer> {
    buffers.get(index)
}

/// The (flat, in-memory) buffer at `index`, mutably, for units that write samples from the audio
/// thread (`RecordBuf`/`BufWr`). Wraps the RT-safe `BufferTable::get_mut`; a stream/empty/out-of-range
/// slot yields `None`.
pub fn buffer_at_mut(buffers: &mut BufferTable, index: usize) -> Option<&mut Buffer> {
    buffers.get_mut(index)
}

/// The streaming endpoint at `index`, mutably (to pull chunks), for `DiskIn`.
pub fn stream_at_mut(buffers: &mut BufferTable, index: usize) -> Option<&mut StreamPlayback> {
    buffers.stream_mut(index)
}

/// The recording endpoint at `index`, mutably (to push chunks), for `DiskOut`.
pub fn recording_at_mut(buffers: &mut BufferTable, index: usize) -> Option<&mut StreamRecording> {
    buffers.recording_mut(index)
}
