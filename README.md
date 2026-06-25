# plyphon

An embeddable implementation of SuperCollider's scsynth audio synthesis engine.

An scsynth-compatible synthesis core that can be driven by any pure-Rust audio backend (e.g. `cpal`, `bevy_audio`, etc) preserving scsynth's hard realtime performance - no locks, blocking, heap allocation or I/O on the audio thread.

## Goals

- **Embedded targets and `wasm32-unknown-unknown`** - plyphon is a `#![no_std]` Rust crate, making it compatible with a lot of embedded device targets, including the web. C++ bindings are typically hard to wrangle for Rust's most common web target. An attempt has been made in [`scsynth-rs`](https://github.com/mitchmindtree/scsynth-rs), but the approach requires many small patches to supercollider itself, and teasing out the raw state machines for the RT and NRT threads while providing a reliably safe API is non-trivial. A pure-Rust implementation simplifies this a lot.
- **Library-first, OSC optional** - plyphon is a library you embed and drive through a typed Controller API. The OSC layer is an optional front-end on top. In-process users can skip OSC entirely (no serialization, no socket) while still getting scsynth-compatible OSC when they want it.
- **Custom storage** - plyphon makes no assumptions about the availability of a filesystem. Storage is abstracted away with traits, allowing implementations to fetch sounds however they like (filesystem, web storage, network fetch, etc).
- **No global state, many engines per process** - scsynth keeps its UGen library, interface table, and audio arena in process-global statics, so a process is
  effectively a single server. plyphon owns all of that in an engine value passed by argument, so multiple independent engines can coexist in one process - useful for
  tests, multi-tenant/embedded hosts, plugin hosts, or per-voice sandboxing.

## Crates

| Crate | Description |
| --- | --- |
| [`plyphon-osc`](crates/plyphon-osc) | SuperCollider-compatible OSC command front-end. |
| [`plyphon-buffers`](crates/plyphon-buffers) | Async `BufferSource` traits for loading sample data, the app-provided I/O seam. |
| [`plyphon`](crates/plyphon) | Control-side facade - the `Controller`, `SynthDef` authoring and compilation, and the `engine()` builder. Re-exports the three crates below. |
| [`plyphon-rt`](crates/plyphon-rt) | Real-time audio driver - the `World` engine, node tree, command protocol, and NRT cleanup. |
| [`plyphon-unit`](crates/plyphon-unit) | Unit-generator abstraction - the `Unit` trait, the built-in units, and the compiled `GraphDef`. |
| [`plyphon-dsp`](crates/plyphon-dsp) | Shared DSP primitives - rates, RNG, wavetables, buses, buffers, and streams. |
| [`rt-alloc`](crates/rt-alloc) | Safe, `no_std` real-time memory pool - a port of scsynth's `AllocPool`. |
| [`scgf`](crates/scgf) | Parser and encoder for SuperCollider's binary SynthDef format (SCgf). |

## Examples

| Example | Description |
| --- | --- |
| [`example-motif`](crates/examples/motif) | A looping motif of self-freeing notes via `cpal` (the web demo). |
| [`example-sine`](crates/examples/sine) | The simplest example: a continuous sine. |
| [`example-custom-unit`](crates/examples/custom-unit) | Implement a custom unit generator (a `tanh` saturator) and register it alongside the base set. |
| [`example-routing`](crates/examples/routing) | Bus routing: an LFO-swept filter on noise, wired through audio and control buses. |
| [`example-control`](crates/examples/control) | Host-driven control buses: an arpeggio steered by `/n_map` + `/c_set`. |
| [`example-scgf`](crates/examples/scgf) | Loads a SuperCollider SCgf-compiled SynthDef and plays it. |
| [`example-sampler`](crates/examples/sampler) | Implements a `BufferSource` that loads a checked-in WAV (filesystem natively, `fetch` on the web) and loops it with `PlayBuf`. |
| [`example-stream`](crates/examples/stream) | Streams a WAV from storage in chunks via a `BufferStream`/`StreamFeeder` and plays it with `DiskIn`. |
| [`example-waveforms`](crates/examples/waveforms) | Cycles through the oscillators (`Saw`/`Pulse`/`LFSaw`/`LFPulse`/`Impulse`) through a filter. |
| [`example-pan`](crates/examples/pan) | A tone auto-panned across the stereo field with `Pan2`. |
| [`example-envelope`](crates/examples/envelope) | Percussive plucks shaped by multi-segment `EnvGen` envelopes that free their own synths. |
| [`example-osc`](crates/examples/osc) | Drives the engine through encoded SuperCollider OSC packets (no sockets) and prints the control commands and the replies/notifications that flow back. |

## Building

All dependencies (the Rust toolchain, `alsa`/`pkg-config` for the native `cpal` backend, and the
wasm tooling) are provided by the Nix flake:

```console
nix develop                  # or `direnv allow` (uses ./.envrc)
cargo build
cargo test
cargo run -p example-sine    # the simplest demo: a continuous sine
```

Each example is a `cargo run -p <name>` away - see the [Examples](#examples) table for the full set.

## The web demo

```console
nix run .#serve-plyphon-website
# or, for live reload during development:
trunk serve
```

Open `localhost:8088` and click once to start audio (browsers hold audio until a user gesture).

`cpal` is the audio backend on both targets: natively via ALSA/CoreAudio/WASAPI, on the web via
its WebAudio backend (the `wasm-bindgen` cpal feature). The `plyphon` engine that feeds it is
identical on both - the only platform-specific part of the demo is how its control plane is ticked.

## Feature parity with scsynth

A living checklist of where plyphon stands against [`scsynth`][scsynth], to gauge what works today
and what is left to port. plyphon is an early-stage research engine, so most of the surface is still
ahead of it. A checked box (`[x]`) is on par with scsynth; an unchecked box (`[ ]`) is not done yet -
partial items stay unchecked and spell out what is missing.

### Engine & architecture

- [x] Lock-free real-time / control split (`World` / `Controller` / `Nrt`); no locks, allocation, or blocking on the audio thread
- [x] No `unsafe` (`#![forbid(unsafe_code)]`) and no global mutable state - instances are passed by argument, so the engine is multi-instance and headless-testable (`World::fill`)
- [x] One engine across native and `wasm32-unknown-unknown` (runs in the browser)
- [x] Reblocks any host buffer size to the internal control block
- [x] Per-UGen RNG (Taus88) and engine-owned wavetables
- [x] Multi-channel output buses and audio input buses (duplex via `World::fill_duplex`)
- [x] Calc rates: scalar, control, audio
- [ ] Demand rate
- [ ] Non-real-time (score render) mode - real-time only so far
- [ ] OSC bundle time-tag scheduling - bundles are applied immediately, the time tag is ignored

Dynamic binary plugin loading (`.scx`) is intentionally out of scope: UGens are compiled into the
engine (pure Rust, no FFI), so there is nothing to load at runtime.

### UGens (21 of scsynth's ~250, grouped by category)

- [ ] **I/O** - have Out, In; missing ReplaceOut, OffsetOut, XOut, LocalIn/LocalOut, InFeedback, SoundIn
- [ ] **Oscillators** - have SinOsc, Saw, Pulse, LFSaw, LFPulse, Impulse; missing Blip, VarSaw, SyncSaw, LFTri/LFPar/LFCub, Osc/OscN, COsc, FSinOsc, Klang, Klank
- [ ] **Noise** - have WhiteNoise; missing PinkNoise, BrownNoise, GrayNoise, ClipNoise, Dust/Dust2, LFNoise0/1/2, LFDNoise*, Crackle
- [ ] **Filters** - have LPF, HPF, Lag; missing BPF, BRF, RLPF, RHPF, Resonz, Ringz, OnePole/OneZero, TwoPole/TwoZero, Integrator, LeakDC, Slew, Decay/Decay2, Formlet, MoogFF, MidEQ
- [ ] **Envelopes** - have EnvGen, Line; missing XLine, Linen, IEnvGen, DemandEnvGen
- [ ] **Panning** - have Pan2; missing LinPan2, Pan4, Balance2, Rotate2, XFade2, LinXFade2, PanAz, Splay
- [ ] **Dynamics** - have Amplitude; missing Compander, Limiter, Normalizer, DetectSilence
- [ ] **Math / multichannel** - have BinaryOpUGen, UnaryOpUGen, MulAdd; missing Sum3/Sum4, Select, Index, Clip/Wrap/Fold, LinLin/LinExp
- [ ] **Buffer playback** - have PlayBuf, DiskIn; missing BufRd, BufWr, RecordBuf, DiskOut, VDiskIn, TGrains, GrainBuf
- [ ] **Triggers / timing** - none yet: Trig/Trig1, TDelay, Latch, Gate, Phasor, Sweep, Timer, PulseCount, PulseDivider, Stepper, ToggleFF, SendTrig, SendReply, Done, FreeSelf, Pause
- [ ] **Info** - none yet: SampleRate, SampleDur, ControlRate, BufFrames, BufDur, NumChannels, RadiansPerSample
- [ ] **Delays / reverb** - none yet: DelayN/L/C, CombN/L/C, AllpassN/L/C, FreeVerb, GVerb, Pluck, PitchShift
- [ ] **Demand-rate** - none yet: Demand, Duty/TDuty, Dseq, Dser, Drand, Dwhite, Dseries, Dgeom (needs demand rate)
- [ ] **FFT / spectral** - none yet: FFT/IFFT, the `PV_*` set, Pitch, Onsets, BeatTrack
- [ ] **Chaos / rate conversion** - none yet: Lorenz, LinCong, Henon, ... and A2K/K2A/T2A/DC

### OSC server commands (22 of ~65)

**Server / top-level** (0/10)

- [ ] /notify
- [ ] /status
- [ ] /quit
- [ ] /cmd
- [ ] /dumpOSC
- [ ] /clearSched
- [ ] /sync
- [ ] /error
- [ ] /version
- [ ] /rtMemoryStatus

**SynthDef** (1/5)

- [x] /d_recv
- [ ] /d_load
- [ ] /d_loadDir
- [ ] /d_free
- [ ] /d_freeAll

**Synth** (1/4)

- [x] /s_new
- [ ] /s_get
- [ ] /s_getn
- [ ] /s_noid

**Node** (7/15)

- [x] /n_set
- [x] /n_free
- [x] /n_map
- [x] /n_mapn
- [x] /n_before
- [x] /n_after
- [x] /n_order
- [ ] /n_setn
- [ ] /n_fill
- [ ] /n_run - the engine already pauses/resumes nodes (`Controller::node_run`); just not wired to OSC
- [ ] /n_query
- [ ] /n_trace
- [ ] /n_mapa
- [ ] /n_mapan
- [ ] /n_cmd

**Group** (5/8)

- [x] /g_new
- [x] /g_head
- [x] /g_tail
- [x] /g_freeAll
- [x] /g_deepFree
- [ ] /p_new
- [ ] /g_dumpTree
- [ ] /g_queryTree

**Unit** (0/1)

- [ ] /u_cmd

**Control bus** (2/5)

- [x] /c_set
- [x] /c_setn
- [ ] /c_fill
- [ ] /c_get
- [ ] /c_getn

**Buffer** (6/17)

- [x] /b_alloc
- [x] /b_allocRead
- [x] /b_read
- [x] /b_free
- [x] /b_zero
- [x] /b_query
- [ ] /b_write
- [ ] /b_close
- [ ] /b_set
- [ ] /b_setn
- [ ] /b_fill
- [ ] /b_gen
- [ ] /b_get
- [ ] /b_getn
- [ ] /b_allocReadChannel
- [ ] /b_readChannel
- [ ] /b_setSampleRate

### Replies, notifications & done actions

- [x] Replies: /done, /fail, /b_info, and node notifications /n_go /n_end /n_off /n_on
- [ ] /status.reply, /synced (the `/sync` barrier), /tr (SendTrig), /n_info (`/n_query`), /g_queryTree.reply, and the `/c_get` / `/b_get` getters
- [ ] Done actions beyond 0 (none), 1 (pause self), 2 (free self): codes 3-14, the free/pause variants that also touch neighbours or the enclosing group

### SynthDefs & buffers

- [x] SCgf binary SynthDefs load via `/d_recv` (and the [`scgf`](crates/scgf) crate also encodes them); named parameters are folded from SC's `Control` UGens
- [ ] Control family beyond plain `Control`: `TrigControl`/`LagControl` parse but behave as plain controls; SynthDef variants
- [x] Buffers: allocate, free, zero, query, and asynchronous loading through an app-provided [`BufferSource`](crates/plyphon-buffers) (the I/O seam), plus chunk-streaming playback with `DiskIn`
- [ ] Writing/recording buffers to disk, `b_gen` wavetable fills, and `b_get`/`b_set` element access

## License

Licensed under [GPL-3.0-or-later](LICENSE), matching SuperCollider's license.

[scsynth]: https://github.com/supercollider/supercollider
[cpal]: https://github.com/RustAudio/cpal
