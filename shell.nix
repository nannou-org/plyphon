{
  alsa-lib,
  binaryen,
  cargo-semver-checks,
  lib,
  lld,
  miniserve,
  mkShell,
  pkg-config,
  release-plz,
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
    # `release-plz` drives the release process (see .github/workflows/release.yml);
    # running it from this shell reuses the native build deps that `cargo publish`'s
    # verify build needs, and lets maintainers preview a release with
    # `nix develop -c release-plz update`. `cargo-semver-checks` is the binary
    # release-plz shells out to for `semver_check` (release-plz.toml).
    cargo-semver-checks
    release-plz
  ];
  buildInputs = runtimeLibs;
  env = lib.optionalAttrs stdenv.hostPlatform.isLinux {
    LD_LIBRARY_PATH = lib.makeLibraryPath runtimeLibs;
  };
}
