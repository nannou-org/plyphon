# Feature parity with scsynth

A living checklist of where plyphon stands against [`scsynth`][scsynth], to gauge what works today
and what is left to port. plyphon is an early-stage research engine, so most of the surface is still
ahead of it. A checked box (`[x]`) is on par with scsynth; an unchecked box (`[ ]`) is not done yet -
partial items stay unchecked and spell out what is missing.

## Engine & architecture

- [x] Lock-free real-time / control split (`World` / `Controller` / `Nrt`); no locks, allocation, or blocking on the audio thread
- [x] No `unsafe` (`#![forbid(unsafe_code)]`) and no global mutable state - instances are passed by argument, so the engine is multi-instance and headless-testable (`World::fill`)
- [x] One engine across native and `wasm32-unknown-unknown` (runs in the browser)
- [x] Reblocks any host buffer size to the internal control block
- [x] Per-UGen RNG (Taus88) and engine-owned wavetables
- [x] Per-instance UGen auxiliary memory - a UGen can reserve a build-time-sized buffer (a delay line, sized from a scalar input like `maxdelaytime`) folded into the synth's single rt-pool block, so the whole delay/comb/allpass family is a pure UGen addition. Unlike scsynth's per-unit `RTAlloc` there is still one allocation and one free per synth; the arena is left uninitialised (no `/s_new` memset) and units guard their own cold start, as scsynth's `_z` calc variants do
- [x] Multi-channel output buses and audio input buses (duplex via `World::fill_duplex`)
- [x] Calc rates: scalar, control, audio
- [x] Demand rate - pull-based, like scsynth: a consumer (`Demand`/`Duty`) pulls a source's next value (or resets it) on the audio thread, recursing through nested sources; sources are single-output and emit `NaN` at exhaustion. Demand units are split out of the per-block calc list into a separate demand plan with their own block span, and the recursion is allocation-free (a bounded stack copy of `Pod` state); off-RT compilation rejects over-large state or over-deep nesting so the audio thread stays bounded
- [x] Non-real-time (score render) mode - `Render` drives the engine offline on a deterministic free-running clock (no DLL resync), feeding a time-tagged score lazily; the `plyphon-osc` `parse_score`/`render_osc_score` pair renders scsynth's binary OSC command file to audio (see `example-render`)
- [x] OSC bundle time-tag scheduling - bundle time tags schedule sample-accurately on the audio thread, against a drift-corrected clock (a DLL tracking the device rate); `OffsetOut` delays a scheduled synth's whole output by its creation offset for the synth's life (scsynth's `OffsetOut_next` delay-and-carry, the leading samples silent and each block's tail carried into the next via per-channel `aux`), so the onset lands on the exact sample and no input sample is dropped. The one divergence: plyphon units have no destructor, so the final `offset` samples in the carry are not flushed on free (scsynth's `OffsetOut_Dtor`) - ~0 for a voice ending in silence

Dynamic binary plugin loading (`.scx`) is intentionally out of scope: UGens are compiled into the
engine (pure Rust, no FFI), so there is nothing to load at runtime.

The **shared-memory / internal-server** surface is also intentionally out of scope, for the same
reason plyphon shares no memory with its clients: scsynth's **shm scope ring** (the transport behind
`ScopeOut2`, mapped and polled by the GUI), **shared controls** (the mmap'd shared control-bus region),
the `/late` lateness reply (a late bundle is resolved to the soonest sample, never reported), and the
legacy **binary integer OSC command form** (`inData[0] == 0` selecting `gCmdArray[int]`; plyphon
dispatches string addresses only). A host that wants scope/metering uses plyphon's own **`ScopeOut`**
instead (implemented - see the I/O category below), which streams every input sample over the RT-safe
chunk ring rather than shared memory. `LocalBuf` (SynthDef-side local buffers, e.g. for an FFT chain)
is **deferred**, not excluded - a chain buffer is `/b_alloc`'d instead, so it does not block the `PV_*`
family.

Intra-graph **reblock and resample** (scsynth's `kGraph_Reblock` / `kGraph_Resample`, per-SynthDef
`Reblock(n)` / `Resample(n)`) are **done**: a def can run its graph at a smaller power-of-two control
block than the World's (tighter envelope/trigger timing, lower-latency `LocalIn`/`LocalOut` feedback -
the local bus shrinks to the graph block) and/or oversampled by a power-of-two factor (anti-aliasing
for nonlinear units). The `GraphDef` carries its own `RateInfo` pair (smaller block and/or higher
sample rate), `Graph::process` runs the calc list `num_ticks = (world_block / block) * factor` times
per World block (collapsing to one pass, zero-cost, for an ordinary def), and the boundary
`In`/`Out`/`OffsetOut`/`AudioControl` cross the World-block-wide bus per tick: `In` reads its tick
slice and zero-order-holds it up to the graph rate, `Out` decimates the oversampled interior down
(`audio_out_decimated`, which clears the channel on the first writer then sums each tick's slice).
Interior DSP units are unaware, packing their wires at the graph block. Authored with
`Controller::add_synthdef_reblocked(def, block)` / `add_synthdef_resampled(def, factor)`; a reblocked
linear chain is byte-identical to the non-reblocked one and an oversampled band-limited chain matches
to within oscillator phase drift. (`LocalIn`/`LocalOut` need no decimate/ZOH - their bus is per-synth
and already graph-rate - and `OffsetOut`'s onset offset, World-block-relative, coarsens to per-tick, a
documented edge.)

Reblock/resample also load from a **scsynth version-3 binary def**: `scgf` parses the v3 framing (a
per-def `int32` size prefix and the trailing `blockSize`/`resampleFactor` fields), and
`plyphon::synthdef::read::parse` hands each def's setting to `Controller::add_synthdef_rate`, so a
`.scsyndef` compiled with `Reblock`/`Resample` is honoured on `/d_recv` and `/d_load` - not only via
the programmatic API. The **control-driven** forms (`blockSize -1` / `resampleFactor -1.0`, where the
value comes from a synth control at instantiation) are unsupported - plyphon bakes the graph block into
the per-synth layout at compile - and fall back to no reblock/resample.

## UGens (67 of scsynth's ~250, grouped by category)

- [ ] **I/O** - have Out, OffsetOut, In, LocalIn, LocalOut, InFeedback (a per-synth feedback bus with a one-block delay; `InFeedback` aliases `In`), and ScopeOut - a live monitoring/analysis tap that streams every sample of its (multichannel) input off the audio thread to the app, the shared-memory-free equivalent of scsynth's `ScopeOut2`. It reuses `DiskOut`'s bounded lock-free chunk-ring transport (`Controller::cue_scope` returns the `StreamConsumer` the app drains); scsynth's `ScopeOut` (an interleaved plain-`SndBuf` read only by the in-process internal server) and `ScopeOut2` (a shm ring) are both replaced by this one no-shm streaming path. Several `ScopeOut` units on distinct bufnums tap several graph points at once. Missing: ReplaceOut, XOut, SoundIn
- [ ] **Oscillators** - have SinOsc, Saw, Pulse, LFSaw, LFPulse, Impulse; missing Blip, VarSaw, SyncSaw, LFTri/LFPar/LFCub, Osc/OscN, COsc, FSinOsc, Klang, Klank
- [ ] **Noise** - have WhiteNoise; missing PinkNoise, BrownNoise, GrayNoise, ClipNoise, Dust/Dust2, LFNoise0/1/2, LFDNoise*, Crackle
- [ ] **Filters** - have LPF, HPF, Lag; missing BPF, BRF, RLPF, RHPF, Resonz, Ringz, OnePole/OneZero, TwoPole/TwoZero, Integrator, LeakDC, Slew, Decay/Decay2, Formlet, MoogFF, MidEQ
- [ ] **Envelopes** - have EnvGen, Line; missing XLine, Linen, IEnvGen, DemandEnvGen
- [ ] **Panning** - have Pan2; missing LinPan2, Pan4, Balance2, Rotate2, XFade2, LinXFade2, PanAz, Splay
- [ ] **Dynamics** - have Amplitude; missing Compander, Limiter, Normalizer, DetectSilence
- [ ] **Math / multichannel** - have BinaryOpUGen, UnaryOpUGen, MulAdd; missing Sum3/Sum4, Select, Index, Clip/Wrap/Fold, LinLin/LinExp
- [ ] **Buffer playback / recording** - have PlayBuf, DiskIn, RecordBuf (record into a buffer with overdub/run/loop/doneAction), BufWr (write channels at a phase index), DiskOut (stream channels out to a cued recording buffer, drained off the audio thread to a sink); missing BufRd, VDiskIn, TGrains, GrainBuf
- [ ] **Triggers / timing** - have SendTrig (fires `/tr` on a rising edge, at control or audio rate), SendReply (emits a custom OSC path with a bounded number of values, over a dedicated node-message ring), FreeSelf, PauseSelf, Done, FreeSelfWhenDone, PauseSelfWhenDone, Free, Pause; missing Trig/Trig1, TDelay, Latch, Gate, Phasor, Sweep, Timer, PulseCount, PulseDivider, Stepper, ToggleFF, Poll
- [ ] **Info** - have SampleRate, SampleDur, RadiansPerSample, ControlRate, ControlDur, NumOutputBuses, NumInputBuses, NumAudioBuses, NumControlBuses, NumRunningSynths, NumBuffers, BufFrames, BufChannels, BufSamples, BufSampleRate, BufRateScale, BufDur, and SubsampleOffset. `SubsampleOffset` needed a real engine feature: the scheduler now retains the *fractional* (sub-sample) part of a scheduled event's within-block position (scsynth's `mSubsampleOffset`) instead of flooring it away. `Clock::block_offset` returns the integer sample offset (clamped, round-to-nearest as before) alongside the unclamped fractional remainder in `[0, 1)`; it is threaded World -> Graph -> `ProcessCtx::subsample_offset` exactly parallel to the existing integer `sample_offset`, and the `SubsampleOffset` UGen snapshots it once on its first block (like `OffsetOut` captures the integer offset). `OffsetOut` is unchanged - scsynth's `OffsetOut` honours only the integer offset.
- [ ] **Delays / reverb** - have DelayN (the first UGen on per-instance aux memory); missing DelayL/C, CombN/L/C, AllpassN/L/C, FreeVerb, GVerb, Pluck, PitchShift
- [ ] **Demand-rate** - have Demand, Duty, Dseq, Dseries, Dwhite, Dbufrd/Dbufwr (demand-rate buffer read/write, via the buffer reach threaded into the demand pull), Dpoll (post a demanded value to the host); missing TDuty, Dser, Drand, Dxrand, Dwrand, Dgeom, Dbrown/Dibrown, Diwhite, Dswitch/Dswitch1, Dstutter, Dconst, Dreset
- [ ] **FFT / spectral** (behind the default-on `fft` feature; realfft, std-only) - have FFT, IFFT (short-time analysis/resynthesis over a packed-spectrum chain buffer, via the shared `FftTables`), PV_MagMul, PV_MagSquared. The bin-level seam is complete: buffers track a `coord` (Complex/Polar, scsynth's `SndBuf::coord`), `pv::to_polar`/`to_complex` convert in place idempotently (the `ToPolarApx`/`ToComplexApx` analogue, using exact `hypot`/`atan2`), `pv::pv_frame` is the `PV_GET_BUF` preamble, and `pv::Spectrum`/`Bin` are the typed `SCPolarBuf`/`SCComplexBuf` views - so the rest of the Cartesian and polar `PV_*` family is now a per-unit port. Missing: the remaining `PV_*` units, Pitch, Onsets, BeatTrack, and `LocalBuf` (SynthDef-side local FFT buffers; not needed - a chain buffer is `/b_alloc`'d)
- [ ] **Chaos / rate conversion** - have A2K, K2A, T2A, DC; missing the chaos set: Lorenz, LinCong, Henon, ...

## OSC server commands (65 of ~65)

The *getters* (`/status`, `/sync`, `/rtMemoryStatus`, `/n_query`, `/c_get`/`/c_getn`, `/s_get`/`/s_getn`,
`/b_get`/`/b_getn`, `/g_queryTree`) read live engine state over a third RT→NRT ring - a fixed-size
`Copy` `Reply` enum mirroring the events ring. They are *asynchronous*: the dispatcher issues one
query per element, the engine answers a block later, and the dispatcher reassembles the answers (in
the FIFO order the queries were issued) into one OSC reply message. The RT side returns only numeric
indices and values; all names are resolved control-side. `/g_dumpTree` reuses the tree walk but
formats to an optional host text sink. `/b_gen` fills buffers control-side (sine1/2/3, cheby, with
normalize) or via an engine-side copy.

A second group are **server/transport commands**: they concern the connection or the host process,
not the synthesis engine, so they live in the host/transport layer rather than the engine OSC
front-end. As in scsynth - where the audio thread always writes node notifications to FIFOs and the
comm layer alone decides delivery (iterating the registered reply-addresses, `mUsers`) - plyphon's
`OscDispatcher` always *emits* the node notifications, and who receives them is the host's call. The
`plyphon-cli` server implements these: `/notify` (a client's per-connection notification subscription,
its `notified` set mirroring `mUsers`), `/quit`, `/dumpOSC`, and `/version`. (The engine-state
queries `/status`/`/sync`/`/rtMemoryStatus` are *not* server commands - the server forwards them to
the dispatcher and routes each async answer back to the requester, alongside the other getters.)

The **host actions** need a plugin registry or filesystem the engine does not model, so they are
surfaced upward as typed capabilities on a `Host` trait the dispatcher drives in `run_pending` -
exactly as `/b_allocRead` already defers I/O to an app-provided `BufferSource`: `/cmd`/`/u_cmd` (a
`CommandHost`), `/d_load`/`/d_loadDir` (a `DefSource`), and `/b_write` + `/b_close` (a `BufferSink`,
the engine streaming the buffer out race-free - a whole-buffer snapshot, or `leaveOpen=1` left open
for `DiskOut`). The OSC surface is now complete; the one command that touches the audio thread,
`/n_trace`, streams each calc unit's I/O over the reply ring to a host text sink. (scsynth's `/n_cmd`
is unimplemented - commented out in its command table - so plyphon omits it too.) `/b_read` splices a
file region into an existing buffer at `bufStartFrame` (scsynth's `BufReadCmd`), and with `leaveOpen=1`
keeps the file open and streams it off disk into a `DiskIn` (the read counterpart to `/b_write
leaveOpen=1`); `/b_readChannel leaveOpen=1` streams only the selected channels through a deinterleaving
stream wrapper. The **buffer surface is now complete with no deferrals**. The CLI server now also
captures **live hardware input** for `In.ar` (`--input-channels` > 0): cpal has no duplex stream, so a
separate capture stream feeds an `rtrb` jitter/drift ring that the output callback drains, reblocking the
engine to exact control blocks (a carry-FIFO) so input stays sample-faithful on any host buffer size.
The remaining gap is non-OSC: **UGen breadth**.

**Server / top-level** (10/10)

- [x] /notify - server-owned (plyphon-cli): per-connection subscription to node notifications
- [x] /status - engine query: real ugen/synth/group/synthdef counts (avg/peak CPU reported as `0.0`)
- [x] /quit - server-owned (plyphon-cli)
- [x] /cmd - routes a plugin command to an app-provided `CommandHost` (plyphon ships none, so this is a seam for embedders); the host owns any reply, as scsynth's `PlugIn_DoCmd` does
- [x] /dumpOSC - server-owned (plyphon-cli)
- [x] /clearSched - engine front-end (clears the World scheduler)
- [x] /sync - engine query: a command-stream barrier answered with `/synced`
- [x] /error - engine front-end: permanent (`0`/`1`) and bundle-local (`-1`/`-2`) modes gate `/fail`
- [x] /version - server-owned (plyphon-cli)
- [x] /rtMemoryStatus - engine query: rt-pool free/largest-chunk bytes

**SynthDef** (5/5)

- [x] /d_recv
- [x] /d_load - loads a SynthDef file through an app-provided `DefSource`, registers each def, replies `/done /d_load` (plyphon reads one file; scsynth globs the path)
- [x] /d_loadDir - loads every def file under a directory through the `DefSource`
- [x] /d_free
- [x] /d_freeAll

**Synth** (4/4)

- [x] /s_new - all five add actions, including `addReplace` (4): the new synth takes the target
  node's exact slot and the target (with its subtree) is freed, the replaced node's `/n_end` firing
  before the new node's `/n_go` (scsynth's `Node_Replace`)
- [x] /s_get - getter; reply echoes the as-given control token (index or name)
- [x] /s_getn
- [x] /s_noid - partial: detaches control-name resolution; the node keeps running and stays reachable
  by control index (plyphon does not reassign a hidden negative id)

**Node** (14/15)

- [x] /n_set
- [x] /n_free
- [x] /n_map
- [x] /n_mapn
- [x] /n_before
- [x] /n_after
- [x] /n_order
- [x] /n_setn
- [x] /n_fill
- [x] /n_run
- [x] /n_query - getter; one `/n_info` per node (parent/prev/next/isGroup, head/tail for a group)
- [x] /n_trace - dumps each calc unit's inputs/outputs (first sample, scsynth's `ZIN0`/`ZOUT0`) for one block, streamed over the reply ring to a host text sink (headless, no OSC reply); unit names resolved control-side by calc index
- [x] /n_mapa - maps an `AudioControl` parameter to an audio bus (its audio wire takes the bus each block); a no-op on a control-rate param
- [x] /n_mapan - the range form of `/n_mapa`
- [ ] /n_cmd - not applicable: unimplemented in scsynth itself (commented out in its command table), so plyphon omits it

**Group** (8/8)

- [x] /g_new - all five add actions, including `addReplace` (4): a fresh empty group takes the target
  node's slot and the target subtree is freed; the root group cannot be replaced (a no-op, as scsynth
  guards with `kSCErr_ReplaceRootGroup`)
- [x] /g_head
- [x] /g_tail
- [x] /g_freeAll
- [x] /g_deepFree
- [x] /p_new - emulated by an ordinary group, as scsynth does
- [x] /g_dumpTree - getter; formats an indented tree to an optional host text sink (no OSC reply)
- [x] /g_queryTree - getter; pre-order tree stream with optional control values

**Unit** (1/1)

- [x] /u_cmd - routes a unit command (node id + unit index + name) to the `CommandHost`, mirroring scsynth's `Unit_DoCmd`

**Control bus** (5/5)

- [x] /c_set
- [x] /c_setn
- [x] /c_fill
- [x] /c_get - getter
- [x] /c_getn - getter

**Buffer** (17/17)

- [x] /b_alloc
- [x] /b_allocRead
- [x] /b_read - `leaveOpen=0` reads a file region into the already-allocated buffer at `bufStartFrame`, keeping its dimensions (scsynth's `BufReadCmd` - a region splice via a `WriteBufferRegion` command, not a whole-buffer replace; the file's channel count must match). `leaveOpen=1` keeps the file open and streams it off disk into a `DiskIn` (a `CueStream` slot fed each `run_pending` tick from a host `BufferStream`; the CLI streams a WAV from disk), ended by `/b_close`
- [x] /b_write - `leaveOpen=0` writes a whole-buffer snapshot (the engine streams the buffer's samples out race-free to an app-provided [`BufferSink`] - no shared buffer memory - driven across `run_pending` ticks); `leaveOpen=1` installs a `DiskOut` recording slot and leaves the sink open for streaming. Replies `/done /b_write <bufnum>`. Partial `numFrames`/`startFrame` ranges are deferred; header/sample formats are the sink's choice (the path)
- [x] /b_close - closes a `leaveOpen=1` stream: the engine flushes `DiskOut`'s final partial chunk (mirroring scsynth's `DiskOut_Dtor`, so every frame is written) and frees the recording slot, then the sink is closed and `/done /b_close <bufnum>` replied. The slot is freed afterwards (it holds no flat data, unlike scsynth's `SndBuf`)
- [x] /b_free
- [x] /b_zero
- [x] /b_query
- [x] /b_set
- [x] /b_setn
- [x] /b_fill
- [x] /b_gen - partial: sine1/2/3, cheby (+ normalize), copy; mono only; wavetable mode unsupported
  (no `Osc` UGen yet); always generates fresh (no accumulate-onto-existing)
- [x] /b_get - getter
- [x] /b_getn - getter
- [x] /b_allocReadChannel - reads only the selected file channels into a fresh buffer (control-side `CopyChannels` deinterleave; out-of-range channel reads as silence)
- [x] /b_readChannel - the channel-subset form of `/b_read`: deinterleaves the selected channels (to a width that must match the buffer) and splices them into the existing buffer at `bufStartFrame`. `leaveOpen=1` streams only the selected channels off disk into a `DiskIn`, via a `ChannelSelectStream` wrapper that deinterleaves the selection per chunk (the streaming analogue of the in-memory splice)
- [x] /b_setSampleRate

## Replies, notifications & done actions

- [x] Replies: /done, /fail, /b_info, and node notifications /n_go /n_end /n_off /n_on - each
  carries the full `/n_info`-shaped position (node, parent, prev, next, isGroup, plus head/tail for a
  group), captured at the moment of the event as scsynth's `Node_StateMsg` does (for `/n_end`, before
  the node leaves the tree, so a deep free reports each descendant with an already-removed predecessor
  reading back as `-1`)
- [x] Getter replies: /status.reply, /synced (the `/sync` barrier), /rtMemoryStatus.reply, /n_info
  (`/n_query`), /g_queryTree.reply, /c_set·/c_setn (`/c_get`·`/c_getn`), /n_set·/n_setn
  (`/s_get`·`/s_getn`), /b_set·/b_setn (`/b_get`·`/b_getn`)
- [x] /tr (SendTrig, over a dedicated best-effort trigger ring) and SendReply's custom `/<path>
  [nodeID, replyID, values...]` (over a parallel best-effort node-message ring; path + values are
  carried inline in a bounded `Copy` carrier, no audio-thread allocation) and /n_move from
  out-of-band node moves (broadcast in `/n_info`'s format when a move command relinks the tree)
- [x] Done actions beyond 0 (none), 1 (pause self), 2 (free self): codes 3-14, the free/pause variants that also touch neighbours or the enclosing group
- [x] Per-unit done flag (scsynth's `mDone`, kept in the RT-pool block): producers (`EnvGen`/`Line`/`PlayBuf`) mark completion independently of the done action, so the done-watching units (`Done`/`FreeSelfWhenDone`/`PauseSelfWhenDone`) can observe a source unit finishing

## SynthDefs & buffers

- [x] SCgf binary SynthDefs load via `/d_recv` (and the [`scgf`](crates/scgf) crate also encodes them); named parameters are folded from SC's `Control` UGens
- [x] Control family beyond plain `Control`: `AudioControl` (an audio-rate parameter, lifted to an audio wire each block and mappable with `/n_mapa`), `TrigControl` (a `/n_set` is seen for one block then resets to 0), and `LagControl` (a control-rate one-pole de-zipper, lag times from the folded UGen's inputs)
- [x] Buffers: allocate, free, zero, query, `b_gen` (sine/cheby/copy) fills, `b_get`/`b_set` element access, and asynchronous loading through an app-provided [`BufferSource`](crates/plyphon-buffers) (the I/O seam), plus chunk-streaming playback with `DiskIn` and chunk-streaming recording with `DiskOut` (drained off the audio thread through the mirror [`BufferSink`] write seam)
- [x] `/b_write` to disk through an app-provided [`BufferSink`]: the engine streams a buffer's samples out race-free (it shares no buffer memory with the host, unlike scsynth reading `SndBuf::data` from its NRT thread), driven across `run_pending` ticks; the CLI writes a float WAV. `leaveOpen=0` snapshots the whole buffer; `leaveOpen=1` leaves the sink open for `DiskOut` streaming, ended by `/b_close`. Channel-subset reads (`/b_allocReadChannel`/`/b_readChannel`) and `/n_trace` (a per-unit I/O dump to a text sink) complete the OSC surface. Partial frame ranges and `b_gen` wavetable fills (no `Osc` UGen to consume them yet) remain

[scsynth]: https://github.com/supercollider/supercollider
