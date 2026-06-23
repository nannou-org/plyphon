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
| [`plyphon-example-motif`](crates/plyphon-example-motif) | A looping motif of self-freeing notes via `cpal` (the web demo). |
| [`plyphon-example-sine`](crates/plyphon-example-sine) | The simplest example: a continuous sine. |

## Building

All dependencies (the Rust toolchain, `alsa`/`pkg-config` for the native `cpal` backend, and the
wasm tooling) are provided by the Nix flake:

```console
nix develop            # or `direnv allow` (uses ./.envrc)
cargo build
cargo test
cargo run -p plyphon-example-sine   # the simplest demo: a continuous sine
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
