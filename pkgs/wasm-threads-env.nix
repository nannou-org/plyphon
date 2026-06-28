# Build environment for cpal's AudioWorklet backend on wasm32. The worklet runs on a real audio
# thread, which needs WASM threads: a `std` recompiled with atomics (`-Z build-std`, hence
# `CARGO_UNSTABLE_BUILD_STD`) and the `+atomics`/`+bulk-memory`/`+mutable-globals` target features.
# Both are nightly-only. RUSTFLAGS applies only to the wasm target (cargo excludes host build
# scripts when `--target` is set), so host tooling is untouched.
#
# Only the target features are set: with `+atomics`, rustc/wasm-ld automatically configure shared,
# imported memory *and* export `__heap_base` (which wasm-bindgen's threading transform needs to
# inject per-thread state). Passing the shared-/import-memory link args by hand overrides those
# defaults and drops the `__heap_base` export, breaking `wasm-bindgen`.
#
# Shared by `pkgs/plyphon-website-worklet.nix` and the `plyphon-web` dev shell so the flags can't
# drift between the Nix build and local `trunk serve`.
{
  RUSTFLAGS = "-C target-feature=+atomics,+bulk-memory,+mutable-globals";
  CARGO_UNSTABLE_BUILD_STD = "std,panic_abort";
}
