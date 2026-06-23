# plyphon

A **pure-Rust** rewrite of the core of SuperCollider's [`scsynth`][scsynth] audio synthesis
engine. No C++, no FFI, no submodules - the entire engine is Rust, so it builds for native
targets and `wasm32-unknown-unknown` alike, and runs in the browser. The goal is an
scsynth-compatible synthesis core that can be driven by any pure-Rust audio backend (e.g.
[`cpal`][cpal]), preserving scsynth's hard real-time guarantees (no locks, blocking, or
allocation on the audio thread).

This is early-stage research, developed in parallel with [`scsynth-rs`][scsynth-rs] (which
embeds the C++ engine via FFI). plyphon already runs a lock-free `World`/`Controller`/`Nrt`
engine with a growing set of UGens, loads SuperCollider SynthDefs (SCgf), accepts OSC commands,
and plays both natively and in the browser.

## Crates

| Crate | Description |
| --- | --- |
| [`plyphon`](crates/plyphon) | The pure-Rust engine core. |
| [`scgf`](crates/scgf) | Parser and encoder for SuperCollider's binary SynthDef format (SCgf). |
| [`plyphon-osc`](crates/plyphon-osc) | SuperCollider-compatible OSC command front-end. |
| [`plyphon-buffers`](crates/plyphon-buffers) | Async `BufferSource` traits for loading sample data (the I/O seam; impls are the app's). |
| [`plyphon-example-motif`](crates/plyphon-example-motif) | A looping motif of self-freeing notes via `cpal` (the web demo). |
| [`plyphon-example-sine`](crates/plyphon-example-sine) | The simplest example: a continuous sine. |
| [`plyphon-example-routing`](crates/plyphon-example-routing) | Bus routing: an LFO-swept filter on noise, wired through audio and control buses. |
| [`plyphon-example-control`](crates/plyphon-example-control) | Host-driven control buses: an arpeggio steered by `/n_map` + `/c_set`. |
| [`plyphon-example-scgf`](crates/plyphon-example-scgf) | Loads a SuperCollider SCgf-compiled SynthDef and plays it. |
| [`plyphon-example-sampler`](crates/plyphon-example-sampler) | Implements a `BufferSource` that loads a checked-in WAV (filesystem natively, `fetch` on the web) and loops it with `PlayBuf`. |
| [`plyphon-example-stream`](crates/plyphon-example-stream) | Streams a WAV from storage in chunks via a `BufferStream`/`StreamFeeder` and plays it with `DiskIn`. |
| [`plyphon-example-waveforms`](crates/plyphon-example-waveforms) | Cycles through the oscillators (`Saw`/`Pulse`/`LFSaw`/`LFPulse`/`Impulse`) through a filter. |
| [`plyphon-example-pan`](crates/plyphon-example-pan) | A tone auto-panned across the stereo field with `Pan2`. |
| [`plyphon-example-envelope`](crates/plyphon-example-envelope) | Percussive plucks shaped by multi-segment `EnvGen` envelopes that free their own synths. |
| [`plyphon-example-osc`](crates/plyphon-example-osc) | Drives the engine through encoded SuperCollider OSC packets (no sockets) and prints the control commands and the replies/notifications that flow back. |

## Building

All dependencies (the Rust toolchain, `alsa`/`pkg-config` for the native `cpal` backend, and the
wasm tooling) are provided by the Nix flake:

```console
nix develop            # or `direnv allow` (uses ./.envrc)
cargo build
cargo test
cargo run -p plyphon-example-sine      # the simplest demo: a continuous sine
cargo run -p plyphon-example-routing   # bus routing: an LFO-swept filter on noise
cargo run -p plyphon-example-control   # host-driven control buses: a bus-steered arpeggio
cargo run -p plyphon-example-scgf      # load and play a SuperCollider SCgf SynthDef
cargo run -p plyphon-example-sampler   # implement a BufferSource and play a loaded sample
cargo run -p plyphon-example-stream    # stream a WAV in chunks and play it with DiskIn
cargo run -p plyphon-example-waveforms # cycle through the oscillators through a filter
cargo run -p plyphon-example-pan       # a tone auto-panned across the stereo field
cargo run -p plyphon-example-envelope  # percussive plucks shaped by EnvGen envelopes
cargo run -p plyphon-example-osc       # drive the engine over OSC packets and print the replies
cargo build --target wasm32-unknown-unknown -p plyphon-example-motif
```

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

## License

Licensed under [GPL-3.0-or-later](LICENSE), matching SuperCollider's license.

[scsynth]: https://github.com/supercollider/supercollider
[scsynth-rs]: https://github.com/mitchmindtree/scsynth-rs
[cpal]: https://github.com/RustAudio/cpal
