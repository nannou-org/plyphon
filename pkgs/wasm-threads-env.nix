# Build environment for cpal's AudioWorklet backend on wasm32. The worklet runs on a real audio
# thread, which needs WASM threads: a `std` recompiled with atomics (`-Z build-std`, hence
# `CARGO_UNSTABLE_BUILD_STD`) and a *shared* linear memory the worklet thread imports. Both are
# nightly-only. RUSTFLAGS applies only to the wasm target (cargo excludes host build scripts when
# `--target` is set), so host tooling is untouched.
#
# These are cpal's documented audioworklet flags (examples/audioworklet-beep/.cargo/config.toml):
# `--shared-memory`/`--max-memory`/`--import-memory` make the memory shared and imported so the
# worklet can be handed the *same* memory, and the `__tls_*` exports let wasm-bindgen's threading
# transform set up per-thread state (without them it fails with "failed to find `__heap_base`").
#
# Shared by `pkgs/plyphon-website-worklet.nix` and the `plyphon-web` dev shell so the flags can't
# drift between the Nix build and local `trunk serve`.
{
  RUSTFLAGS = builtins.concatStringsSep " " [
    # SIMD128 vectorizes the per-sample DSP loops (broad browser support alongside threads).
    "-C target-feature=+atomics,+simd128"
    "-C link-arg=--shared-memory"
    "-C link-arg=--max-memory=1073741824"
    "-C link-arg=--import-memory"
    "-C link-arg=--export=__heap_base"
    "-C link-arg=--export=__wasm_init_tls"
    "-C link-arg=--export=__tls_size"
    "-C link-arg=--export=__tls_align"
    "-C link-arg=--export=__tls_base"
  ];
  CARGO_UNSTABLE_BUILD_STD = "std,panic_abort";
}
