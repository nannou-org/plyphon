{
  alsa-lib,
  binaryen,
  lib,
  lld,
  miniserve,
  mkShell,
  pkg-config,
  rustToolchain,
  stdenv,
  trunk,
  wasm-bindgen-cli,
}:
let
  # Runtime libs the native demo links/dlopens against (cpal -> ALSA on Linux).
  runtimeLibs = [ alsa-lib ];
in
mkShell {
  name = "plyphon-dev";
  nativeBuildInputs = [
    rustToolchain
    # cpal's ALSA backend builds via the `alsa-sys` crate, which uses pkg-config.
    pkg-config
    # Web demo tooling.
    binaryen
    lld
    trunk
    wasm-bindgen-cli
    miniserve
  ];
  buildInputs = runtimeLibs;
  env = lib.optionalAttrs stdenv.hostPlatform.isLinux {
    LD_LIBRARY_PATH = lib.makeLibraryPath runtimeLibs;
  };
}
