# The `plyphon` binary: an scsynth-compatible OSC synthesis server and offline renderer, built
# natively from the workspace's `plyphon-cli` crate. The engine is pure Rust; the only system
# dependency is ALSA for `cpal`'s Linux audio backend (darwin links CoreAudio from the SDK stdenv).
{
  lib,
  stdenv,
  rustPlatform,
  pkg-config,
  alsa-lib,
}:
let
  # Everything except build artifacts (mirrors pkgs/plyphon-website.nix).
  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      let
        base = baseNameOf (toString path);
      in
      !(builtins.elem base [
        "target"
        "result"
        "dist"
        ".direnv"
      ])
      && lib.cleanSourceFilter path type;
  };
in
rustPlatform.buildRustPackage {
  pname = "plyphon";
  version = "0.1.0";
  inherit src;
  cargoLock.lockFile = ../Cargo.lock;

  # Build and test only the CLI crate (its `plyphon` binary), not the whole workspace.
  cargoBuildFlags = [
    "-p"
    "plyphon-cli"
  ];
  cargoTestFlags = [
    "-p"
    "plyphon-cli"
  ];

  nativeBuildInputs = [ pkg-config ];
  buildInputs = lib.optionals stdenv.hostPlatform.isLinux [ alsa-lib ];

  meta = {
    description = "An scsynth-compatible OSC synthesis server and offline renderer";
    homepage = "https://github.com/nannou-org/plyphon";
    license = lib.licenses.gpl3Plus;
    mainProgram = "plyphon";
  };
}
