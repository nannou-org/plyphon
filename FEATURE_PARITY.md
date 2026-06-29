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
- [x] OSC bundle time-tag scheduling - bundle time tags schedule sample-accurately on the audio thread, against a drift-corrected clock (a DLL tracking the device rate); `OffsetOut` places a scheduled synth's onset on the exact sample

Dynamic binary plugin loading (`.scx`) is intentionally out of scope: UGens are compiled into the
engine (pure Rust, no FFI), so there is nothing to load at runtime.

## UGens (64 of scsynth's ~250, grouped by category)

- [ ] **I/O** - have Out, OffsetOut, In, LocalIn, LocalOut, InFeedback (a per-synth feedback bus with a one-block delay; `InFeedback` aliases `In`); missing ReplaceOut, XOut, SoundIn
- [ ] **Oscillators** - have SinOsc, Saw, Pulse, LFSaw, LFPulse, Impulse; missing Blip, VarSaw, SyncSaw, LFTri/LFPar/LFCub, Osc/OscN, COsc, FSinOsc, Klang, Klank
- [ ] **Noise** - have WhiteNoise; missing PinkNoise, BrownNoise, GrayNoise, ClipNoise, Dust/Dust2, LFNoise0/1/2, LFDNoise*, Crackle
- [ ] **Filters** - have LPF, HPF, Lag; missing BPF, BRF, RLPF, RHPF, Resonz, Ringz, OnePole/OneZero, TwoPole/TwoZero, Integrator, LeakDC, Slew, Decay/Decay2, Formlet, MoogFF, MidEQ
- [ ] **Envelopes** - have EnvGen, Line; missing XLine, Linen, IEnvGen, DemandEnvGen
- [ ] **Panning** - have Pan2; missing LinPan2, Pan4, Balance2, Rotate2, XFade2, LinXFade2, PanAz, Splay
- [ ] **Dynamics** - have Amplitude; missing Compander, Limiter, Normalizer, DetectSilence
- [ ] **Math / multichannel** - have BinaryOpUGen, UnaryOpUGen, MulAdd; missing Sum3/Sum4, Select, Index, Clip/Wrap/Fold, LinLin/LinExp
- [ ] **Buffer playback / recording** - have PlayBuf, DiskIn, RecordBuf (record into a buffer with overdub/run/loop/doneAction), BufWr (write channels at a phase index), DiskOut (stream channels out to a cued recording buffer, drained off the audio thread to a sink); missing BufRd, VDiskIn, TGrains, GrainBuf
- [ ] **Triggers / timing** - have SendTrig (fires `/tr` on a rising edge, at control or audio rate), SendReply (emits a custom OSC path with a bounded number of values, over a dedicated node-message ring), FreeSelf, PauseSelf, Done, FreeSelfWhenDone, PauseSelfWhenDone, Free, Pause; missing Trig/Trig1, TDelay, Latch, Gate, Phasor, Sweep, Timer, PulseCount, PulseDivider, Stepper, ToggleFF, Poll
- [ ] **Info** - have SampleRate, SampleDur, RadiansPerSample, ControlRate, ControlDur, NumOutputBuses, NumInputBuses, NumAudioBuses, NumControlBuses, NumRunningSynths, NumBuffers, BufFrames, BufChannels, BufSamples, BufSampleRate, BufRateScale, BufDur; missing SubsampleOffset
- [ ] **Delays / reverb** - have DelayN (the first UGen on per-instance aux memory); missing DelayL/C, CombN/L/C, AllpassN/L/C, FreeVerb, GVerb, Pluck, PitchShift
- [ ] **Demand-rate** - have Demand, Duty, Dseq, Dseries, Dwhite, Dbufrd/Dbufwr (demand-rate buffer read/write, via the buffer reach threaded into the demand pull), Dpoll (post a demanded value to the host); missing TDuty, Dser, Drand, Dxrand, Dwrand, Dgeom, Dbrown/Dibrown, Diwhite, Dswitch/Dswitch1, Dstutter, Dconst, Dreset
- [ ] **FFT / spectral** - none yet: FFT/IFFT, the `PV_*` set, Pitch, Onsets, BeatTrack
- [ ] **Chaos / rate conversion** - have A2K, K2A, T2A, DC; missing the chaos set: Lorenz, LinCong, Henon, ...

## OSC server commands (55 of ~65)

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

The genuinely-deferred host actions - `/cmd`/`/u_cmd`/`/n_cmd`, `/d_load`/`/d_loadDir`,
`/b_write`/`/b_close`, and `/n_trace` - need a plugin registry or filesystem the engine does not
model; the intent is to surface them as typed higher-level actions for the embedding host, the way
`/b_allocRead` already defers I/O to an app-provided `BufferSource`.

**Server / top-level** (9/10)

- [x] /notify - server-owned (plyphon-cli): per-connection subscription to node notifications
- [x] /status - engine query: real ugen/synth/group/synthdef counts (avg/peak CPU reported as `0.0`)
- [x] /quit - server-owned (plyphon-cli)
- [ ] /cmd
- [x] /dumpOSC - server-owned (plyphon-cli)
- [x] /clearSched - engine front-end (clears the World scheduler)
- [x] /sync - engine query: a command-stream barrier answered with `/synced`
- [x] /error - engine front-end: permanent (`0`/`1`) and bundle-local (`-1`/`-2`) modes gate `/fail`
- [x] /version - server-owned (plyphon-cli)
- [x] /rtMemoryStatus - engine query: rt-pool free/largest-chunk bytes

**SynthDef** (3/5)

- [x] /d_recv
- [ ] /d_load
- [ ] /d_loadDir
- [x] /d_free
- [x] /d_freeAll

**Synth** (4/4)

- [x] /s_new
- [x] /s_get - getter; reply echoes the as-given control token (index or name)
- [x] /s_getn
- [x] /s_noid - partial: detaches control-name resolution; the node keeps running and stays reachable
  by control index (plyphon does not reassign a hidden negative id)

**Node** (13/15)

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
- [ ] /n_trace
- [x] /n_mapa - maps an `AudioControl` parameter to an audio bus (its audio wire takes the bus each block); a no-op on a control-rate param
- [x] /n_mapan - the range form of `/n_mapa`
- [ ] /n_cmd

**Group** (8/8)

- [x] /g_new
- [x] /g_head
- [x] /g_tail
- [x] /g_freeAll
- [x] /g_deepFree
- [x] /p_new - emulated by an ordinary group, as scsynth does
- [x] /g_dumpTree - getter; formats an indented tree to an optional host text sink (no OSC reply)
- [x] /g_queryTree - getter; pre-order tree stream with optional control values

**Unit** (0/1)

- [ ] /u_cmd

**Control bus** (5/5)

- [x] /c_set
- [x] /c_setn
- [x] /c_fill
- [x] /c_get - getter
- [x] /c_getn - getter

**Buffer** (13/17)

- [x] /b_alloc
- [x] /b_allocRead
- [x] /b_read
- [x] /b_free
- [x] /b_zero
- [x] /b_query
- [ ] /b_write
- [ ] /b_close
- [x] /b_set
- [x] /b_setn
- [x] /b_fill
- [x] /b_gen - partial: sine1/2/3, cheby (+ normalize), copy; mono only; wavetable mode unsupported
  (no `Osc` UGen yet); always generates fresh (no accumulate-onto-existing)
- [x] /b_get - getter
- [x] /b_getn - getter
- [ ] /b_allocReadChannel
- [ ] /b_readChannel
- [x] /b_setSampleRate

## Replies, notifications & done actions

- [x] Replies: /done, /fail, /b_info, and node notifications /n_go /n_end /n_off /n_on
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
- [ ] OSC `/b_write`/`/b_close` and non-streaming whole-buffer saves to disk (`DiskOut` covers streaming recording; the OSC surface and a one-shot buffer snapshot remain), and `b_gen` wavetable fills (no `Osc` UGen to consume them yet)

[scsynth]: https://github.com/supercollider/supercollider
