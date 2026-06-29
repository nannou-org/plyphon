# The legacy/fallback web demo: every example for `wasm32-unknown-unknown` via `trunk` on cpal's
# default (deprecated `ScriptProcessorNode`) backend - stable toolchain, no cross-origin isolation
# needed. Kept as a backup; the default `plyphon-website` is the AudioWorklet build. The whole
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
  pname = "plyphon-website-legacy";
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

    # Each example is its own wasm binary built to its own page under $out/<name>/. trunk emits the
    # entry HTML as index.html plus the bin's assets, and public_url="./" keeps asset URLs relative
    # to the page. The engine compiles once into the shared target dir, so only the per-example
    # link/bindgen/opt steps repeat.
    for name in \
      sine motif waveforms envelope pan feedback delay custom-unit duty-seq \
      routing control node-control glide osc schedule triggers send-reply scgf sampler stream record; do
      trunk build --release --dist $out/$name web/$name.html
    done

    # The landing page is static (it only links to the example pages); copy it and the shared
    # stylesheet to the site root. (A `trunk build --dist $out` here would wipe the example dirs.)
    cp web/index.html $out/index.html
    cp web/style.css $out/style.css
  '';

  installPhase = "true";
}
