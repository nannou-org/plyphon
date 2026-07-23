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

use plyphon_dsp::buffer::{BufView, BufViewMut, BufferTable};
use plyphon_dsp::bus::Buses;
use plyphon_dsp::rate::Rate;
use plyphon_dsp::stream::{StreamPlayback, StreamRecording};

use crate::unit::{Inputs, LocalBufs, LocalBus};

/// Audio bus channel `ch` for this block (an empty slice if `ch` is out of range), for `In.ar`.
pub fn audio_in(buses: &Buses, ch: usize) -> &[f32] {
    let audio = buses.audio();
    if ch < audio.num_channels() {
        audio.channel(ch)
    } else {
        &[]
    }
}

/// Whether audio bus channel `ch` was written during block `buf_counter` (scsynth's
/// `touched[i] == bufCounter`). `In.ar` outputs zero for an untouched channel - a channel whose
/// writer freed or runs later in the tree holds *last* block's audio, which only `InFeedback`
/// deliberately reads. Out of range reads as untouched.
pub fn audio_in_touched(buses: &Buses, ch: usize, buf_counter: u64) -> bool {
    let audio = buses.audio();
    ch < audio.num_channels() && audio.is_touched(ch, buf_counter)
}

/// Accumulate `src` into audio bus channel `ch` for this block (`Out.ar`). Out of range is a no-op.
pub fn audio_out(buses: &mut Buses, buf_counter: u64, ch: usize, src: &[f32]) {
    buses.audio_mut().write_accumulate(ch, buf_counter, src);
}

/// Accumulate `src` into audio bus channel `ch` at sample `offset`, decimating by `factor` (`Out.ar`
/// from a reblocked/resampled graph: each sub-block tick writes its own slice of the World-block
/// channel, taking every `factor`-th oversampled sample). The first writer of the block clears the
/// whole channel; `offset == 0`, `factor == 1` reduces to [`audio_out`]. Out of range is a no-op.
pub fn audio_out_decimated(
    buses: &mut Buses,
    buf_counter: u64,
    ch: usize,
    offset: usize,
    src: &[f32],
    factor: usize,
) {
    buses
        .audio_mut()
        .write_accumulate_decimated(ch, buf_counter, offset, src, factor);
}

/// Overwrite audio bus channel `ch` at sample `offset` with `src` decimated by `factor`, marking it
/// touched (`ReplaceOut.ar`). `offset == 0`, `factor == 1` overwrites the whole channel. Out of range
/// is a no-op.
pub fn audio_replace_decimated(
    buses: &mut Buses,
    buf_counter: u64,
    ch: usize,
    offset: usize,
    src: &[f32],
    factor: usize,
) {
    buses
        .audio_mut()
        .write_replace_decimated(ch, buf_counter, offset, src, factor);
}

/// Crossfade `src` (decimated by `factor`) into audio bus channel `ch` at sample `offset`:
/// `dst = dst*(1-xfade) + src*xfade` (`XOut.ar`). The first writer of the block clears the whole
/// channel first (as `Out` does), so the mix is against this block's audio or silence.
/// `offset == 0`, `factor == 1` crossfades the whole channel. Out of range is a no-op.
pub fn audio_crossfade(
    buses: &mut Buses,
    buf_counter: u64,
    ch: usize,
    offset: usize,
    src: &[f32],
    factor: usize,
    xfade: f32,
) {
    buses
        .audio_mut()
        .write_crossfade_decimated(ch, buf_counter, offset, src, factor, xfade);
}

/// Crossfade `value` into control bus channel `ch`: `ch = ch*(1-xfade) + value*xfade` (`XOut.kr`).
/// The first writer of the block treats the existing value as zero. Out of range is a no-op.
pub fn control_crossfade(buses: &mut Buses, buf_counter: u64, ch: usize, value: f32, xfade: f32) {
    buses
        .control_mut()
        .write_crossfade(ch, buf_counter, value, xfade);
}

/// Overwrite control bus channel `ch` with `value` for this block (`ReplaceOut.kr`). Out of range is
/// a no-op.
pub fn control_replace(buses: &mut Buses, buf_counter: u64, ch: usize, value: f32) {
    buses.control_mut().write_replace(ch, buf_counter, value);
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
///
/// An `index` at or past the table's capacity is a graph-local buffer number (`capacity + i`, the
/// number a `LocalBuf` outputs - scsynth's `world->mNumSndBufs + i`) and resolves into `local`
/// instead, so every buffer consumer reads local buffers with no special-casing. Where scsynth's
/// `CTOR_GET_BUF` falls back to world buffer 0 for an out-of-range local number, plyphon treats it
/// as missing (`None`).
pub fn buffer_at<'a>(
    buffers: &'a BufferTable,
    local: &'a LocalBufs<'_>,
    index: usize,
) -> Option<BufView<'a>> {
    let capacity = buffers.capacity();
    if index >= capacity {
        local.view(index - capacity)
    } else {
        buffers.get(index).map(|buffer| buffer.view())
    }
}

/// The (flat, in-memory) buffer at `index`, mutably, for units that write samples from the audio
/// thread (`RecordBuf`/`BufWr`). A stream/empty/out-of-range slot yields `None`; a past-capacity
/// `index` resolves to the graph-local buffers, as in [`buffer_at`].
pub fn buffer_at_mut<'a>(
    buffers: &'a mut BufferTable,
    local: &'a mut LocalBufs<'_>,
    index: usize,
) -> Option<BufViewMut<'a>> {
    let capacity = buffers.capacity();
    if index >= capacity {
        local.view_mut(index - capacity)
    } else {
        buffers.get_mut(index).map(|buffer| buffer.view_mut())
    }
}

/// Buffer `a` mutably and buffer `b` immutably as disjoint borrows, for a two-buffer spectral op (a
/// `PV_*` unit reading `b` while rewriting `a`). `None` unless `a != b` and both resolve. Each side
/// resolves independently - world table or graph-local per [`buffer_at`] - so a chain may pair a
/// world buffer with a `LocalBuf` in either role.
pub fn buffer_pair_mut<'a>(
    buffers: &'a mut BufferTable,
    local: &'a mut LocalBufs<'_>,
    a: usize,
    b: usize,
) -> Option<(BufViewMut<'a>, BufView<'a>)> {
    let capacity = buffers.capacity();
    match (a < capacity, b < capacity) {
        (true, true) => buffers
            .pair_mut(a, b)
            .map(|(a, b)| (a.view_mut(), b.view())),
        (true, false) => {
            let b = local.view(b - capacity)?;
            Some((buffers.get_mut(a)?.view_mut(), b))
        }
        (false, true) => {
            let a = local.view_mut(a - capacity)?;
            Some((a, buffers.get(b)?.view()))
        }
        (false, false) => local.pair_mut(a - capacity, b - capacity),
    }
}

/// The streaming endpoint at `index`, mutably (to pull chunks), for `DiskIn`.
pub fn stream_at_mut(buffers: &mut BufferTable, index: usize) -> Option<&mut StreamPlayback> {
    buffers.stream_mut(index)
}

/// The recording endpoint at `index`, mutably (to push chunks), for `DiskOut`/`ScopeOut`.
pub fn recording_at_mut(buffers: &mut BufferTable, index: usize) -> Option<&mut StreamRecording> {
    buffers.recording_mut(index)
}

/// Sample channel input `i` at within-block index `k` - per sample at audio rate, or the single value
/// broadcast at control rate; `0.0` if `i` is past the unit's inputs. The shared input reader for the
/// channel-writing units (`RecordBuf`/`BufWr`/`DiskOut`/`ScopeOut`).
pub(crate) fn sample_channel(ins: &Inputs<'_>, i: usize, k: usize) -> f32 {
    if i >= ins.len() {
        0.0
    } else if ins.rate(i) == Rate::Audio {
        ins.audio(i)[k]
    } else {
        ins.control(i)
    }
}

/// Stream `num_channels` input channels (inputs `first_channel..first_channel + num_channels`) into the
/// recording cued at `bufnum`, one recording channel per input; a no-op if no recording is cued there.
/// Shared by `DiskOut` and `ScopeOut` - the audio thread only copies the block into the recording's
/// chunk queue (drained off-thread); a full queue drops the surplus (a bounded overrun), never blocking
/// or allocating. `ins` borrows the wires (not the buffer table), so it coexists with the `&mut`
/// recording.
pub(crate) fn stream_channels_to_recording(
    ins: &Inputs<'_>,
    buffers: &mut BufferTable,
    block: usize,
    bufnum: usize,
    first_channel: usize,
    num_channels: usize,
) {
    if let Some(rec) = recording_at_mut(buffers, bufnum) {
        rec.write(block, num_channels, |frame, ch| {
            sample_channel(ins, first_channel + ch, frame)
        });
    }
}
