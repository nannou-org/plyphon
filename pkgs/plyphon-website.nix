# The web demo: builds `plyphon-example` for `wasm32-unknown-unknown` via `trunk`. The whole
# engine is pure Rust, so this build needs no C++ toolchain and no submodules.
{
  binaryen,
  lib,
  lld,
  rustPlatformWasm,
  trunk,
  wasm-bindgen-cli,
}:
let
  # Everything except build artifacts.
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
rustPlatformWasm.buildRustPackage {
  pname = "plyphon-website";
  version = "0.1.0";
  inherit src;
  cargoLock.lockFile = ../Cargo.lock;
  doCheck = false;
  dontFixup = true;

  nativeBuildInputs = [
    binaryen
    lld
    trunk
    wasm-bindgen-cli
  ];

  # Tell trunk to use Nix-provided tools, not download its own.
  TRUNK_SKIP_VERSION_CHECK = "true";

  buildPhase = ''
    # trunk (via wasm-bindgen) needs a writable cache/home dir in the sandbox.
    export HOME=$(mktemp -d)
    trunk build --release --dist $out
  '';

  installPhase = "true";
}
