# plyphon

An embeddable implementation of SuperCollider's scsynth audio synthesis engine.

An scsynth-compatible synthesis core that can be driven by any pure-Rust audio backend (e.g. `cpal`, `bevy_audio`, etc) preserving scsynth's hard realtime performance - no locks, blocking, heap allocation or I/O on the audio thread. <sup>Named after [*this*](https://www.youtube.com/watch?v=vsNbUlyERV0).</sup>

## Goals

- **Embedded targets and `wasm32-unknown-unknown`** - plyphon is a `#![no_std]` Rust crate, making it compatible with a lot of embedded device targets, including the web. C++ bindings are typically hard to wrangle for Rust's most common web target. An attempt has been made in [`scsynth-rs`](https://github.com/mitchmindtree/scsynth-rs), but the approach requires many small patches to supercollider itself, and teasing out the raw state machines for the RT and NRT threads while providing a reliably safe API is non-trivial. A pure-Rust implementation simplifies this a lot.
- **Library-first, OSC optional** - plyphon is a library you embed and drive through a typed Controller API. The OSC layer is an optional front-end on top. In-process users can skip OSC entirely (no serialization, no socket) while still getting scsynth-compatible OSC when they want it.
- **Custom storage** - plyphon makes no assumptions about the availability of a filesystem. Storage is abstracted away with traits, allowing implementations to fetch sounds however they like (filesystem, web storage, network fetch, etc).
- **No global state, many engines per process** - scsynth keeps its UGen library, interface table, and audio arena in process-global statics, so a process is
  effectively a single server. plyphon owns all of that in an engine value passed by argument, so multiple independent engines can coexist in one process - useful for
  tests, multi-tenant/embedded hosts, plugin hosts, or per-voice sandboxing.

## Crates

Listed in reverse-topological order - dependents first, their dependencies below.

| Crate | Description |
| --- | --- |
| [`plyphon-cli`](crates/plyphon-cli) | The `plyphon` binary - an scsynth-compatible OSC synthesis server (UDP/TCP) and offline renderer, built on the crates below. |
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
| [`motif`](crates/examples/motif) | A looping motif of self-freeing notes via `cpal`. |
| [`sine`](crates/examples/sine) | The simplest example: a continuous sine. |
| [`custom-unit`](crates/examples/custom-unit) | Implement a custom unit generator (a `tanh` saturator) and register it alongside the base set. |
| [`routing`](crates/examples/routing) | Bus routing: an LFO-swept filter on noise, wired through audio and control buses. |
| [`control`](crates/examples/control) | Host-driven control buses: an arpeggio steered by `/n_map` + `/c_set`. |
| [`scgf`](crates/examples/scgf) | Loads a SuperCollider SCgf-compiled SynthDef and plays it. |
| [`sampler`](crates/examples/sampler) | Implements a `BufferSource` that loads a checked-in WAV (filesystem natively, `fetch` on the web) and loops it with `PlayBuf`. |
| [`stream`](crates/examples/stream) | Streams a WAV from storage in chunks via a `BufferStream`/`StreamFeeder` and plays it with `DiskIn`. |
| [`waveforms`](crates/examples/waveforms) | Cycles through the oscillators (`Saw`/`Pulse`/`LFSaw`/`LFPulse`/`Impulse`) through a filter. |
| [`operators`](crates/examples/operators) | A ring-modulated, soft-clipped bell tone built from `BinaryOpUGen`/`UnaryOpUGen` math operators (`midicps`, `midiratio`, ring modulation, `softclip`). |
| [`filters`](crates/examples/filters) | A classic resonant low-pass sweep: a saw through an LFO-swept `RLPF` (one of the resonant biquads `RLPF`/`RHPF`/`BPF`/`BRF`/`Resonz`/`Ringz`). |
| [`moog`](crates/examples/moog) | A resonant acid bassline: a saw through a `MoogFF` Moog-ladder filter with an LFO-swept cutoff and high feedback resonance (the filter additions `MoogFF`/`Formlet`/`MidEQ`/`SOS`/`Median`/`Lag2`/`Hilbert`/`FreqShift`). |
| [`noise`](crates/examples/noise) | Metallic rain: `Dust2` impulses ring a `Ringz` resonator over a quiet `PinkNoise` bed (the noise family `WhiteNoise`/`ClipNoise`/`GrayNoise`/`PinkNoise`/`BrownNoise`/`Dust`/`Dust2`). |
| [`wandering`](crates/examples/wandering) | A generative burble driven by the low-frequency/dynamic noise family: `LFNoise1` wanders the pitch, `LFNoise2` sweeps the filter, `LFDNoise3` shimmers the amplitude (`LFNoise0/1/2`/`LFClipNoise`/`LFDNoise0/1/3`/`LFDClipNoise`). |
| [`sample-hold`](crates/examples/sample-hold) | A self-playing sample-and-hold sequence: an `Impulse` clock latches a pitch contour with `Latch` into a `Decay2`-plucked saw (the in-graph trigger units `Trig`/`Latch`/`Gate`/`ToggleFF`/`Stepper`/`Phasor`/...). |
| [`auto-wah`](crates/examples/auto-wah) | An envelope-following auto-wah: a `PeakFollower` tracks a plucked source's amplitude and drives an `RLPF` cutoff (the signal-measurement units `Peak`/`RunningMin`/`RunningMax`/`PeakFollower`/`MostChange`/`LeastChange`/`LastValue`). |
| [`scale-walk`](crates/examples/scale-walk) | A scale-quantised melodic walk: an `LFNoise1` indexes a scale buffer through `Index`, so a smooth wander always lands on in-scale notes (the selection/lookup units `Select`/`Index`/`IndexL`/`WrapIndex`/`FoldIndex`). |
| [`bell`](crates/examples/bell) | Modal bell synthesis: an `Impulse` strikes a `Klank` bank of inharmonic resonators (tubular-bell ratios), ringing a struck-metal bell (the additive/modal banks `Klang`/`Klank`). |
| [`wavetable`](crates/examples/wavetable) | A morphing wavetable drone: a `VOsc` sweeps its position through a bank of increasingly-bright wavetables, crossfading the timbre from a pure sine to a rich saw (the wavetable oscillators `Osc`/`OscN`/`COsc`/`VOsc`/`VOsc3`). |
| [`vowel`](crates/examples/vowel) | A synthesized sung vowel: three `Formant` oscillators voice the formants of an "ah" over a vibrato'd fundamental, forming a vocal-like drone (`Formant`, alongside the batch's `SinOscFB`/`Blip`/`LFGauss`). |
| [`granular`](crates/examples/granular) | A granular time-stretch: `Warp1` smears a short synthesized arpeggio into a slowly-morphing stereo pad, holding pitch while a slow pointer stretches time (the granular family `GrainSin`/`GrainFM`/`GrainIn`/`GrainBuf`/`TGrains`/`Warp1`). |
| [`reverb`](crates/examples/reverb) | A cavernous space: two Karplus-Strong plucked strings on drifting clocks smear through a large `GVerb` FDN reverb (the delay/reverb family `BufDelay*`/`BufComb*`/`BufAllpass*`/`DelTapWr`/`DelTapRd`/`Pluck`/`PitchShift`/`FreeVerb`/`FreeVerb2`/`GVerb`). |
| [`waveshaping`](crates/examples/waveshaping) | A wavefolder: a sine driven hard through `Fold` with an LFO-swept drive (the range shapers `Clip`/`Wrap`/`Fold`/`LinExp`/...). |
| [`chaos`](crates/examples/chaos) | A chaotic drone: a `CuspN` oscillator through a resonant filter swept by a slow `LatoocarfianN` map (the chaotic generators `CuspN`/`QuadN`/`GbmanN`/`StandardN`/`LatoocarfianN`/`LinCongN`). |
| [`hard-sync`](crates/examples/hard-sync) | A hard-sync lead: a `SyncSaw` whose saw frequency is swept by an LFO over a fixed pitch (the oscillators `LFTri`/`LFPar`/`LFCub`/`VarSaw`/`SyncSaw`/`FSinOsc`). |
| [`bouncing-ball`](crates/examples/bouncing-ball) | Physical-model percussion: a `TBall` bouncing on an oscillating floor rings a `Ringz` resonator (the physical models `Spring`/`Ball`/`TBall`). |
| [`comb-string`](crates/examples/comb-string) | Karplus-Strong plucked strings: periodic noise bursts excite a tuned `CombL` resonator (delay = one pitch period), diffused by an `AllpassC` (the recirculating delays `CombN/L/C`/`AllpassN/L/C` and interpolating `DelayL/C`). |
| [`drunk-melody`](crates/examples/drunk-melody) | A self-driving drunk-walk melody: `Duty.kr` clocks a `Dibrown` integer random walk over MIDI notes (`midicps` + `Lag` glide), showcasing the generator demand sources `Dgeom`/`Diwhite`/`Dbrown`/`Dibrown`. |
| [`pan`](crates/examples/pan) | A tone auto-panned across the stereo field with `Pan2`. |
| [`envelope`](crates/examples/envelope) | Percussive plucks shaped by multi-segment `EnvGen` envelopes that free their own synths. |
| [`osc`](crates/examples/osc) | Drives the engine through encoded SuperCollider OSC packets (no sockets) and prints the control commands and the replies/notifications that flow back. |
| [`schedule`](crates/examples/schedule) | Sample-accurate rhythm: schedules time-tagged OSC bundles up front, each note onsetting on its exact sample via the engine's drift-corrected scheduler and `OffsetOut`. |
| [`render`](crates/examples/render) | Offline (non-real-time) rendering: reads a binary OSC score and renders it to a WAV faster than real time and deterministically - plyphon's `scsynth -N`. |

## Building

All dependencies (the Rust toolchain, `alsa`/`pkg-config` for the native `cpal` backend, and the
wasm tooling) are provided by the Nix flake:

```console
nix develop                  # or `direnv allow` (uses ./.envrc)
cargo build
cargo test
cargo run --release -p example-sine    # the simplest demo: a continuous sine
```

Each example is a `cargo run -p <name>` away - see the [Examples](#examples) table for the full set.

## The web demo

Every example also runs in the browser - the same engine, compiled to `wasm32-unknown-unknown`. The
web build is one site: a landing page linking to a page per example, each running that example's
wasm. It is built by `nix build .#plyphon-website` and auto-deployed to
[GitHub Pages](https://mitchmindtree.github.io/plyphon/) on every push to `main`.

```console
nix run .#serve-plyphon-website     # build the whole site and serve it on localhost:8088
# or, for live-reloading a single example during development:
trunk serve web/<name>.html
```

Open `localhost:8088` and click once to start audio (browsers hold audio until a user gesture).

`cpal` is the audio backend on both targets: natively via ALSA/CoreAudio/WASAPI, on the web via its
WebAudio backend (the `wasm-bindgen` cpal feature). The `plyphon` engine that feeds it is identical
on both - only how the control plane is ticked differs by platform.

## Feature parity with scsynth

plyphon is an early-stage research engine. See [`FEATURE_PARITY.md`](FEATURE_PARITY.md) for a living
checklist of where it stands against scsynth - engine architecture, UGens, OSC commands, replies,
and SynthDef/buffer support.

## License

Licensed under [GPL-3.0-or-later](LICENSE), matching SuperCollider's license.
