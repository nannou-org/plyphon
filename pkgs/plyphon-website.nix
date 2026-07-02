# The plyphon web demo (the site that ships): every example built for cpal's AudioWorklet
# backend, which runs audio on a dedicated Web Audio thread via WASM threads (SharedArrayBuffer).
# It needs a *nightly* toolchain (`-Z build-std` to recompile `std` with atomics) and the
# shared-memory build flags from `wasm-threads-env.nix`.
#
# This is a plain `mkDerivation` rather than `buildRustPackage` because `-Z build-std` recompiles
# `std` from the rust-src component, so `std`'s own crates.io deps must be vendored alongside the
# app's - the sandbox has no network and `buildRustPackage` only vendors the workspace lock.
{
  binaryen,
  lib,
  lld,
  runCommand,
  rustPlatform,
  rustToolchainWasmNightly,
  stdenv,
  trunk,
  wasm-bindgen-cli,
}:
let
  # Everything except build artifacts (mirrors the src filter in pkgs/plyphon.nix).
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

  # Vendor both the workspace deps and `std`'s deps (from the rust-src component the nightly
  # toolchain ships), then merge them into one source-replacement tree so build-std resolves
  # everything offline. Neither lock has git deps, so no `outputHashes` are needed.
  appDeps = rustPlatform.importCargoLock { lockFile = ../Cargo.lock; };
  stdDeps = rustPlatform.importCargoLock {
    lockFile = "${rustToolchainWasmNightly}/lib/rustlib/src/rust/library/Cargo.lock";
  };
  cargoVendor = runCommand "plyphon-website-vendor" { } ''
    mkdir -p $out
    cp -r ${appDeps}/. $out/
    # Shared crate+version dirs are byte-identical, so skipping collisions is safe.
    cp -rn ${stdDeps}/. $out/
  '';
in
stdenv.mkDerivation (
  {
    pname = "plyphon-website";
    version = "0.1.0";
    inherit src;

    nativeBuildInputs = [
      rustToolchainWasmNightly
      binaryen
      lld
      trunk
      wasm-bindgen-cli
    ];

    # Tell trunk to use Nix-provided tools, not download its own; resolve everything from the vendor.
    TRUNK_SKIP_VERSION_CHECK = "true";
    CARGO_NET_OFFLINE = "true";

    configurePhase = ''
      runHook preConfigure
      # trunk (via wasm-bindgen) and cargo need writable home/cache dirs in the sandbox.
      export HOME=$(mktemp -d)
      export CARGO_HOME=$HOME/.cargo
      mkdir -p $CARGO_HOME
      cat > $CARGO_HOME/config.toml <<EOF
      [source.crates-io]
      replace-with = "vendored-sources"

      [source.vendored-sources]
      directory = "${cargoVendor}"
      EOF
      runHook postConfigure
    '';

    buildPhase = ''
      runHook preBuild
      mkdir -p $out
      # Each example is its own wasm binary built to its own page under $out/<name>/. The example
      # pages build on cpal's AudioWorklet backend; the shared build flags (atomics + build-std) come
      # from the derivation env below. The set is the pages themselves, so a new example (added as a
      # web/*.html page) builds and links here automatically - no list to keep in sync.
      for page in web/*.html; do
        name=$(basename "$page" .html)
        [ "$name" = index ] && continue
        trunk build --release --dist $out/$name web/$name.html
      done

      # Static landing page + stylesheet at the site root (a `trunk build --dist $out` would wipe
      # the example dirs). coi-serviceworker is copied next to each page by trunk.
      cp web/index.html $out/index.html
      cp web/style.css $out/style.css
      runHook postBuild
    '';

    dontInstall = true;
    dontFixup = true;
  }
  // (import ./wasm-threads-env.nix)
)
