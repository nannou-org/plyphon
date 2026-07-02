# Dev shell for the AudioWorklet web build (`trunk serve web/<name>.html`). It uses the
# nightly toolchain and the WASM-threads build flags, so cargo here recompiles `std` with atomics
# via `-Z build-std`. Native `cargo` commands belong in the default `plyphon-dev` shell instead -
# the build-std flags here would make a host build fail.
{
  binaryen,
  lib,
  lld,
  miniserve,
  mkShell,
  rustToolchainWasmNightly,
  trunk,
  wasm-bindgen-cli,
}:
mkShell (
  {
    name = "plyphon-web-dev";
    nativeBuildInputs = [
      rustToolchainWasmNightly
      binaryen
      lld
      trunk
      wasm-bindgen-cli
      miniserve
    ];
  }
  // (import ./pkgs/wasm-threads-env.nix)
)
